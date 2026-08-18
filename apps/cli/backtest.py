"""Backtest CLI — MASTER_PLAN §M3.

    python -m apps.cli.backtest --strategy momentum --top 30
    python -m apps.cli.backtest --strategy hold --cost-multiple 3

Runs against the NSE panel ingested by `apps.cli.ingest_nse`. `--cost-multiple`
runs gauntlet check 7 (§5.4): a strategy that dies at 3x modelled costs was
never viable.

**Costs are the Indian schedule by default**, because that is where live capital
goes (§0.1). A strategy validated against US costs and pointed at NSE will be
destroyed by the ~50x difference (§7.3).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from decimal import Decimal

import polars as pl

from core.clock import UTC, as_decision_time, utc_now
from core.config import settings
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from data.corpactions.actions import CorporateActionBook
from data.feeds.yahoo import YahooActionsLoader
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from data.universe.pit import UniverseBuilder, UniverseSpec
from engine.backtest import BacktestConfig, BacktestEngine, MarketModel, NextOpenFill
from engine.costs.india import NseEquityCostModel
from engine.costs.model import ScaledCostModel
from quant.strategies.base import Strategy
from quant.strategies.baselines import BuyAndHold, CrossSectionalMomentum, SmaCrossover

#: NSE cash equities trade in 5-paisa ticks.
NSE_TICK = Decimal("0.05")


def nse_instrument(instrument_id: str, symbol: str) -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(instrument_id),
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange=Exchange.NSE,
        currency=Currency.INR,
        tick_size=NSE_TICK,
    )


def load_panel(store: PanelStore) -> pl.DataFrame:
    """Everything in the panel, long format.

    A whole-history read is correct here: the engine re-filters per bar on
    `receive_time`, which is where point-in-time discipline actually lives
    (§14.1.4).
    """
    return store.view(as_of=as_decision_time(utc_now()))


def build_universe(store: PanelStore, top_n: int) -> tuple[InstrumentId, ...]:
    """Point-in-time universe as of the panel's most recent session.

    **Anchored to the last session, not to wall-clock now.** A panel covering
    2024 queried in 2026 has nothing in the trailing liquidity window, and the
    universe comes back empty — which looks like a configuration problem rather
    than the date arithmetic it actually is.

    Note this is a single universe applied across the whole backtest. A
    production run rebalances membership on a schedule via
    `UniverseBuilder.build_schedule`; this CLI is a single-shot inspection tool
    and says so rather than pretending otherwise.
    """
    sessions = store.sessions()
    if not sessions:
        return ()

    # End of the last session's publication window, so that session is
    # observable. Bhavcopy publishes ~2.5h after the 10:00 UTC close.
    last = sessions[-1]
    anchor = datetime(last.year, last.month, last.day, 23, 59, tzinfo=UTC)

    universe = UniverseBuilder(store).build(
        as_decision_time(anchor),
        UniverseSpec(top_n=top_n, min_sessions=20, lookback_days=60),
    )
    return tuple(InstrumentId(m) for m in universe.members)


def load_actions(
    instruments: dict[InstrumentId, Instrument],
    symbols: dict[str, str],
    enabled: bool = True,
) -> CorporateActionBook:
    """Corporate actions for the universe, from the free Yahoo source.

    **Without this the backtester treats a 2:1 split as a -50% day**, and the
    error is invisible — it produces a plausible return series rather than an
    exception (§9). `--no-actions` exists to measure exactly how much that
    matters on your data, not as a convenience.
    """
    if not enabled:
        print("corporate actions DISABLED — splits will read as price crashes")
        return CorporateActionBook([])

    wanted = {iid: symbols.get(iid, iid) for iid in instruments}
    result = YahooActionsLoader().fetch_book(wanted)
    book = result.book
    print(f"corporate actions: {len(book)} events, {len(book.instruments)}/{len(wanted)} names")

    # An uncovered name backtests unadjusted and reports nothing, so it is
    # printed rather than logged. Failure is not the same as "no actions":
    # names with genuinely none are silent by design.
    if result.failures:
        print(f"  UNCOVERED (fetch failed): {', '.join(result.failures)}")
        print("  those names run UNADJUSTED — any split in them is a fake crash")
    return book


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backtest on the NSE panel")
    parser.add_argument("--strategy", default="momentum", choices=["momentum", "sma", "hold"])
    parser.add_argument("--top", type=int, default=30, help="Universe size")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=60, help="Momentum lookback in bars")
    parser.add_argument("--skip", type=int, default=5, help="Momentum skip window in bars")
    parser.add_argument("--cash", type=Decimal, default=Decimal(1_000_000))
    parser.add_argument("--cost-multiple", type=Decimal, default=Decimal(1))
    parser.add_argument("--lake", default=None)
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Skip corporate actions. Splits will read as price crashes (9).",
    )
    return parser.parse_args(argv)


def build_strategy(args: argparse.Namespace) -> Strategy:
    if args.strategy == "hold":
        return BuyAndHold()
    if args.strategy == "sma":
        return SmaCrossover(fast=args.fast, slow=args.slow)
    return CrossSectionalMomentum(
        lookback_bars=args.lookback,
        skip_bars=args.skip,
        top_fraction=Decimal("0.3"),
    )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")

    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}\nRun: python -m apps.cli.ingest_nse --start 2024-01-01")
        return 1

    if history.is_empty():
        print("panel is empty")
        return 1

    universe = build_universe(store, args.top)
    if not universe:
        print("universe is empty — ingest more sessions, or lower --top")
        return 1

    symbols = dict(history.select("instrument_id", "symbol").unique().iter_rows())
    instruments = {InstrumentId(i): nse_instrument(i, symbols.get(i, i)) for i in universe}

    strategy = build_strategy(args)
    base_costs = NseEquityCostModel()
    costs = (
        base_costs if args.cost_multiple == 1 else ScaledCostModel(base_costs, args.cost_multiple)
    )

    actions = load_actions(instruments, symbols, enabled=not args.no_actions)

    engine = BacktestEngine(
        strategy=strategy,
        market=MarketModel(
            cost_model=costs,
            fill_model=NextOpenFill(costs),
            instruments=instruments,
            actions=actions,
        ),
        config=BacktestConfig(initial_cash=args.cash),
    )
    result = engine.run(history, universe=universe)

    if result.equity_curve.is_empty():
        print("no equity curve produced — not enough history for the lookback")
        return 1

    curve = result.equity_curve
    start_equity = float(curve["equity"][0])
    end_equity = float(curve["equity"][-1])
    fees = float(curve["fees_paid"][-1])

    print(f"strategy      : {strategy.name} {strategy.spec.parameters}")
    print(f"cost model    : {costs.name}")
    print(f"universe      : {len(universe)} NSE names")
    print(f"period        : {curve['event_time'][0]} -> {curve['event_time'][-1]}")
    print(f"sessions      : {result.bars_processed}")
    print()
    print(f"start equity  : {start_equity:>15,.2f}")
    print(f"end equity    : {end_equity:>15,.2f}")
    print(f"total return  : {result.total_return:>15.2%}")
    print(f"max drawdown  : {result.max_drawdown:>15.2%}")
    print(f"fees paid     : {fees:>15,.2f}  ({fees / start_equity:.2%} of start)")
    print()
    print(
        f"orders        : {result.orders_generated} generated, "
        f"{result.orders_filled} filled, {result.orders_rejected} rejected, "
        f"{result.orders_no_market} no market, {result.orders_unfunded} unfunded"
    )
    print(f"trades        : {result.trades.height}")
    if result.liquidity_failures:
        print(f"liquidity     : {result.liquidity_failures} order(s) refused by volume")
    return 0


if __name__ == "__main__":
    sys.exit(run())
