"""Factor research lab — MASTER_PLAN §6.

    python -m apps.cli.factor momentum_12_1
    python -m apps.cli.factor --all --sessions 1000
    python -m apps.cli.factor reversal_5d --horizon 5

**The fast loop.** Scores a signal directly against forward returns instead of
inferring its value from an equity curve. A study takes under a second; a
backtest takes minutes and the gauntlet takes roughly forty-eight of them. Most
ideas should die here.

What the output answers, in order:

    IC          does the signal rank names in the order they later perform
    decay       at which horizon — which sets the holding period
    quantiles   is it monotonic, or one lucky extreme bucket
    turnover    can the edge survive being traded that often (§7.1)

**A strong IC is not a strategy.** It is permission to build one and send it to
the gauntlet. The cost line at the bottom is the first thing that kills a
signal with a real but tiny edge.
"""

from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal

import polars as pl

from apps.cli.backtest import load_panel
from core.config import settings
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from core.orders import Side
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from engine.costs.india import NseEquityCostModel
from engine.costs.model import TradeContext
from quant.research.composite import CompositeSpec, combine_factors, factor_correlations
from quant.research.factors import FORWARD_HORIZONS, Factor, FactorSpec, build_factor
from quant.research.ic import FactorReport, analyse_factor, rolling_ic

RULE = "─" * 78

#: The order this screen prices against: a 30-name book on ₹1,000,000, which is
#: what `--top 30` and the default capital actually produce. Named because the
#: round trip is *not* scale-free — the depository fee is a flat ₹15.34 per
#: scrip per sell-day, so it is 4.6bp on a ₹33,000 order and 0.15bp on a ₹1M
#: one, and market impact pushes the other way as size grows.
SCREEN_NOTIONAL = Decimal(1_000_000) / 30
SCREEN_PRICE = Decimal(1_000)


def _round_trip_cost() -> float:
    """Buy-plus-sell cost as a fraction, from the model the backtester uses.

    Derived rather than hardcoded. The constant here was 0.0022 against a model
    that charges 0.0033 on this order — the statutory legs alone come to about
    0.0024, so the old figure was roughly the fee floor with market impact left
    out entirely. It drifted because nothing tied it to the model, and it is a
    verdict ("DIES ON COSTS"), not a footnote: understating it by a third moves
    marginal signals from dead to alive.
    """
    instrument = Instrument(
        instrument_id=InstrumentId("NSE:INE000000000"),
        symbol="SCREEN",
        asset_class=AssetClass.EQUITY,
        exchange=Exchange.NSE,
        currency=Currency.INR,
        tick_size=Decimal("0.01"),
    )
    model = NseEquityCostModel()
    quantity = SCREEN_NOTIONAL / SCREEN_PRICE
    total = sum(
        model.cost(
            TradeContext(
                instrument=instrument,
                side=side,
                quantity=quantity,
                price=SCREEN_PRICE,
                # Deep enough that impact reflects a liquid name rather than
                # the screen's own thinness; the min_adv filter enforces this.
                adv_value=Decimal(10**9),
            )
        ).total
        for side in (Side.BUY, Side.SELL)
    )
    return float(total / SCREEN_NOTIONAL)


ROUND_TRIP_COST = _round_trip_cost()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a signal against forward returns")
    parser.add_argument(
        "factor",
        nargs="?",
        choices=[f.value for f in Factor],
        help="Signal to score. Omit with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Score every factor in the library")
    parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Forward horizon for the quantile study, in sessions",
    )
    parser.add_argument(
        "--min-adv",
        type=float,
        default=1e7,
        help="Liquidity floor. Illiquid names manufacture IC from stale prices.",
    )
    parser.add_argument("--sessions", type=int, default=0, help="Trailing window. 0 uses all.")
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument(
        "--combine",
        default=None,
        help=(
            "Comma-separated factors to combine into one composite signal. "
            "Order matters: orthogonalisation is sequential, so the first "
            "factor keeps the variance it shares with later ones."
        ),
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Report how independent the factor library actually is",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Rolling one-year IC — has the factor stopped working?",
    )
    parser.add_argument("--no-orthogonalise", action="store_true")
    parser.add_argument("--equal-weight", action="store_true")
    parser.add_argument("--lake", default=None)
    return parser.parse_args(argv)


def cost_verdict(report: FactorReport) -> str:
    """Whether the quantile spread survives trading at the signal's turnover.

    Crude by construction and useful anyway: it is the difference between a
    signal with a real edge and one with an edge smaller than its own
    transaction costs, which is most of them.
    """
    if report.turnover <= 0 or not report.quantiles:
        return "  cost check: not applicable"

    # Turnover is per session; a holding period of `horizon` sessions trades
    # roughly that fraction of the book each rebalance.
    per_rebalance = min(1.0, report.turnover * report.quantile_horizon)
    charge = per_rebalance * ROUND_TRIP_COST
    net = report.spread - charge
    verdict = "survives" if net > 0 else "DIES ON COSTS"
    return (
        f"  cost check: spread {report.spread:+.3%} - "
        f"{per_rebalance:.0%} turnover x {ROUND_TRIP_COST:.2%} round trip "
        f"= {net:+.3%} net  -> {verdict}"
    )


def study(history: pl.DataFrame, factor: Factor, args: argparse.Namespace) -> FactorReport:
    spec = FactorSpec(factor=factor, min_adv=args.min_adv, window=args.sessions)
    horizons = tuple(sorted({*FORWARD_HORIZONS, args.horizon}))
    scored = build_factor(history, spec, horizons)
    return analyse_factor(scored, factor.value, horizons, args.horizon, args.buckets)


def run_composite(history: pl.DataFrame, args: argparse.Namespace) -> int:
    """Score a combination of factors as one signal."""
    factors = tuple(Factor(f.strip()) for f in args.combine.split(",") if f.strip())
    spec = CompositeSpec(
        factors=factors,
        min_adv=args.min_adv,
        window=args.sessions,
        orthogonalise=not args.no_orthogonalise,
        ic_weight=not args.equal_weight,
        horizon=args.horizon,
    )
    horizons = tuple(sorted({*FORWARD_HORIZONS, args.horizon}))

    started = time.perf_counter()
    scored, weights = combine_factors(history, spec, horizons)
    if scored.is_empty():
        print("no names survived the liquidity filter and lookback")
        return 1

    report = analyse_factor(scored, "composite", horizons, args.horizon, args.buckets)
    print()
    print(RULE)
    print("COMPOSITE  " + " + ".join(f.value for f in factors))
    print("  weights   " + "   ".join(f"{k} {v:.1%}" for k, v in weights.items()))
    print(f"  orthogonalised {not args.no_orthogonalise}   ic-weighted {not args.equal_weight}")
    print()
    print(report.format())
    print(cost_verdict(report))
    print(f"  [{time.perf_counter() - started:.2f}s]")

    print()
    print("  Weights are fitted on the same sample they are scored against.")
    print("  That is in-sample by construction — this is a candidate, and the")
    print("  gauntlet's walk-forward and PBO checks exist to test it (§5.4).")
    return 0


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.factor or args.all or args.combine or args.overlap):
        print("give a factor name, or --all, --combine a,b,c, or --overlap")
        return 1

    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")
    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}\nRun: python -m apps.cli.ingest_nse --start 2019-01-01")
        return 1

    if args.overlap:
        chosen = (
            tuple(Factor(f) for f in args.combine.split(",")) if args.combine else tuple(Factor)
        )
        print()
        print(RULE)
        print("FACTOR OVERLAP")
        print(
            factor_correlations(
                history, chosen, min_adv=args.min_adv, window=args.sessions or 750
            ).format()
        )
        print()
        print("  A library of sixteen worth six independent bets is not a library")
        print("  of sixteen. Combining duplicates counts the same effect twice and")
        print("  calls it diversification.")
        return 0

    if args.combine:
        return run_composite(history, args)

    wanted = list(Factor) if args.all else [Factor(args.factor)]
    for factor in wanted:
        started = time.perf_counter()
        report = study(history, factor, args)
        elapsed = time.perf_counter() - started

        print()
        print(RULE)
        print(report.format())
        print(cost_verdict(report))
        if args.rolling:
            windows = rolling_ic(
                build_factor(
                    history,
                    FactorSpec(factor, min_adv=args.min_adv, window=args.sessions),
                    (args.horizon,),
                ),
                args.horizon,
            )
            if windows:
                step = max(1, len(windows) // 8)
                print("  ROLLING 1y IC")
                for w in windows[::step]:
                    bar = "#" * max(0, int(abs(w.ic) * 300))
                    print(f"    {w.end}  {w.ic:>+7.4f}  {bar[:30]}")
            else:
                print("  ROLLING 1y IC: not enough sessions")
        print(f"  {factor.description}")
        print(f"  [{elapsed:.2f}s]")

    print()
    print("A strong IC is permission to build a strategy, not evidence one works.")
    print("Send survivors to: python -m apps.cli.validate")
    return 0


if __name__ == "__main__":
    sys.exit(run())
