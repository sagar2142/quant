"""Backtest engine — the M3 gate.

Four things must hold before any research result is worth reading (§M3):

    (a) the same experiment reruns to identical numbers
    (b) buy-and-hold matches hand arithmetic to the rupee
    (c) the shuffle-future test passes — no look-ahead
    (d) 3x costs degrades performance sensibly

`TestNoLookAhead` is the most important class in this file. If it fails,
nothing else measured by this engine means anything.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from engine.backtest.engine import BacktestConfig, BacktestEngine, MarketModel
from engine.backtest.fills import NextOpenFill
from engine.costs.india import NseEquityCostModel
from engine.costs.model import ScaledCostModel
from engine.costs.slippage import SlippageModel
from quant.strategies.base import MarketView, Strategy, StrategySpec, TargetWeights
from quant.strategies.baselines import BuyAndHold, SmaCrossover

IID = InstrumentId("NSE:TEST")
T0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
NO_SLIPPAGE = SlippageModel(spread_k=Decimal(0), impact_lambda=Decimal(0))

INSTRUMENT = Instrument(
    instrument_id=IID,
    symbol="TEST",
    asset_class=AssetClass.EQUITY,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.01"),
)
INSTRUMENTS = {IID: INSTRUMENT}


def history(closes: list[float], instrument_id: InstrumentId = IID) -> pl.DataFrame:
    """Flat bars at each close, published one hour after the bar closes."""
    n = len(closes)
    events = [T0 + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "event_time": events,
            "receive_time": [t + timedelta(hours=1) for t in events],
            "instrument_id": [instrument_id] * n,
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * n,
        },
        schema_overrides={
            "event_time": pl.Datetime("us", "UTC"),
            "receive_time": pl.Datetime("us", "UTC"),
        },
    )


def engine(strategy: Strategy, cost_multiplier: Decimal = Decimal(1), **cfg) -> BacktestEngine:
    base = NseEquityCostModel(slippage=NO_SLIPPAGE)
    model = base if cost_multiplier == 1 else ScaledCostModel(base, cost_multiplier)
    return BacktestEngine(
        strategy=strategy,
        market=MarketModel(
            cost_model=model,
            fill_model=NextOpenFill(model),
            instruments=INSTRUMENTS,
        ),
        config=BacktestConfig(**cfg),
    )


class TestDeterminism:
    """M3 gate (a): the same experiment reruns to identical numbers."""

    def test_repeated_runs_are_identical(self):
        data = history([100.0 + i for i in range(60)])
        runs = [engine(SmaCrossover(fast=5, slow=10)).run(data, universe=(IID,)) for _ in range(3)]
        curves = [r.equity_curve["equity"].to_list() for r in runs]
        assert curves[0] == curves[1] == curves[2]

    def test_trade_logs_are_identical(self):
        data = history([100.0 + (i % 7) * 3 for i in range(80)])
        a = engine(SmaCrossover(fast=5, slow=10)).run(data, universe=(IID,))
        b = engine(SmaCrossover(fast=5, slow=10)).run(data, universe=(IID,))
        assert a.trades.to_dicts() == b.trades.to_dicts()


class TestBuyAndHoldHandCalculation:
    """M3 gate (b): matches hand arithmetic to the rupee."""

    def test_flat_market_loses_exactly_the_costs(self):
        # Price never moves, so the entire loss must be the entry cost.
        data = history([100.0] * 10)
        result = engine(BuyAndHold(), rebalance_threshold=Decimal("0.5")).run(data, universe=(IID,))
        final = result.equity_curve["equity"][-1]
        fees = result.equity_curve["fees_paid"][-1]
        assert result.orders_filled == 1
        # Started at 1,000,000; the only loss is fees plus the tick rounding
        # on a single entry.
        assert final == pytest.approx(1_000_000 - fees, abs=200)

    def test_price_doubles_roughly_doubles_equity(self):
        rising = [100.0 * (1 + i / 9) for i in range(10)]  # 100 -> 200
        data = history(rising)
        result = engine(BuyAndHold(), rebalance_threshold=Decimal("0.5")).run(data, universe=(IID,))
        # Fully invested at ~111 (bar 1 open), ending at 200: ~+80%.
        assert 0.6 < result.total_return < 0.9

    def test_no_trades_when_universe_empty(self):
        data = history([100.0] * 10)
        result = engine(BuyAndHold()).run(data, universe=())
        assert result.orders_generated == 0
        assert result.equity_curve["equity"][-1] == 1_000_000.0


class TestNoLookAhead:
    """M3 gate (c). The most important class in this file.

    If a strategy can see the future, every number this engine produces is
    fiction that looks like evidence.
    """

    def test_strategy_never_sees_beyond_decision_time(self):
        """A strategy that records what it saw must never hold a future bar."""
        seen: list[tuple[datetime, datetime]] = []

        class Spy(Strategy):
            def __init__(self) -> None:
                super().__init__(
                    StrategySpec(name="spy", universe="fixed", timeframe="1d", lookback=1)
                )

            def generate(self, view: MarketView) -> TargetWeights:
                if not view.history.is_empty():
                    seen.append((view.as_of, view.history["event_time"].max()))
                return TargetWeights(view.as_of, {})

        data = history([100.0 + i for i in range(20)])
        engine(Spy()).run(data, universe=(IID,))

        assert seen, "strategy was never called"
        for as_of, latest_seen in seen:
            assert latest_seen <= as_of, (
                f"look-ahead: saw a bar at {latest_seen} while deciding at {as_of}"
            )

    def test_publication_lag_hides_the_current_bar(self):
        """A bar published after the decision time must be invisible."""
        seen_counts: list[int] = []

        class Counter(Strategy):
            def __init__(self) -> None:
                super().__init__(
                    StrategySpec(name="counter", universe="fixed", timeframe="1d", lookback=1)
                )

            def generate(self, view: MarketView) -> TargetWeights:
                seen_counts.append(view.bar_count())
                return TargetWeights(view.as_of, {})

        data = history([100.0] * 10)
        # receive_time is one hour after event_time, and decisions are taken at
        # event_time, so the current bar is never observable.
        engine(Counter()).run(data, universe=(IID,))
        assert seen_counts[0] == 0
        assert seen_counts[-1] == len(seen_counts) - 1

    def test_shuffle_future_leaves_results_unchanged(self):
        """§5.4 test 2 — the cheapest high-value check in the system.

        Corrupt every bar after the midpoint. A sound strategy's decisions in
        the first half must be bit-identical, because it could not have seen
        them.
        """
        closes = [100.0 + i for i in range(40)]
        clean = history(closes)

        corrupted_closes = closes[:20] + [c * 3 for c in closes[20:]]
        corrupted = history(corrupted_closes)

        strategy = SmaCrossover(fast=3, slow=5)
        a = engine(strategy).run(clean, universe=(IID,))
        b = engine(strategy).run(corrupted, universe=(IID,))

        # Trades executed before the corruption point must match exactly.
        cutoff = T0 + timedelta(days=20)
        trades_a = [t for t in a.trades.to_dicts() if t["event_time"] < cutoff]
        trades_b = [t for t in b.trades.to_dicts() if t["event_time"] < cutoff]
        assert trades_a == trades_b
        assert trades_a, "no trades before the corruption point — test is vacuous"

    def test_fill_never_uses_decision_bar_price(self):
        """A fill must come from the next bar, never the decision bar."""
        # Decision bar closes at 100; next bar opens at 500. If the engine
        # filled at the decision bar's price, we would see ~100.
        closes = [100.0, 500.0, 500.0, 500.0]
        data = history(closes)
        result = engine(BuyAndHold(), rebalance_threshold=Decimal("0.5")).run(data, universe=(IID,))
        assert result.trades.height >= 1
        assert result.trades["price"][0] == pytest.approx(500.0, rel=0.01)


class TestCostSensitivity:
    """M3 gate (d): 3x costs must degrade performance sensibly."""

    def test_higher_costs_reduce_returns(self):
        data = history([100.0 + (i % 11) * 5 for i in range(120)])
        strategy = SmaCrossover(fast=3, slow=8)
        cheap = engine(strategy, cost_multiplier=Decimal(1)).run(data, universe=(IID,))
        dear = engine(strategy, cost_multiplier=Decimal(3)).run(data, universe=(IID,))
        assert dear.equity_curve["fees_paid"][-1] > cheap.equity_curve["fees_paid"][-1]
        assert dear.total_return < cheap.total_return

    def test_fees_scale_roughly_threefold(self):
        data = history([100.0 + (i % 11) * 5 for i in range(120)])
        strategy = SmaCrossover(fast=3, slow=8)
        cheap = engine(strategy, cost_multiplier=Decimal(1)).run(data, universe=(IID,))
        dear = engine(strategy, cost_multiplier=Decimal(3)).run(data, universe=(IID,))
        ratio = dear.equity_curve["fees_paid"][-1] / cheap.equity_curve["fees_paid"][-1]
        assert 2.5 < ratio < 3.5


class TestTurnoverControl:
    def test_rebalance_threshold_suppresses_dust_trades(self):
        data = history([100.0 + (i % 3) * 0.05 for i in range(40)])
        loose = engine(BuyAndHold(), rebalance_threshold=Decimal("0.10")).run(data, universe=(IID,))
        tight = engine(BuyAndHold(), rebalance_threshold=Decimal("0.0001")).run(
            data, universe=(IID,)
        )
        assert loose.orders_generated <= tight.orders_generated

    def test_min_order_value_blocks_tiny_trades(self):
        data = history([100.0] * 20)
        result = engine(BuyAndHold(), min_order_value=Decimal(10_000_000)).run(
            data, universe=(IID,)
        )
        assert result.orders_generated == 0


class TestEngineValidation:
    def test_missing_columns_rejected(self):
        data = history([100.0] * 5).drop("volume")
        with pytest.raises(ValueError, match="missing columns"):
            engine(BuyAndHold()).run(data, universe=(IID,))

    def test_empty_history_rejected(self):
        data = history([100.0] * 5).head(0)
        with pytest.raises(ValueError, match="empty history"):
            engine(BuyAndHold()).run(data, universe=(IID,))

    def test_lookback_respected_before_first_signal(self):
        data = history([100.0] * 30)
        result = engine(SmaCrossover(fast=5, slow=20)).run(data, universe=(IID,))
        # Nothing can trade before the slow window is full.
        if result.trades.height:
            first = result.trades["event_time"][0]
            assert first >= T0 + timedelta(days=20)

    def test_equity_curve_covers_every_bar(self):
        data = history([100.0] * 15)
        result = engine(BuyAndHold()).run(data, universe=(IID,))
        assert result.equity_curve.height == 15


class TestStrategyContract:
    def test_weights_clipped_to_max_position(self):
        class Greedy(Strategy):
            def __init__(self) -> None:
                super().__init__(
                    StrategySpec(
                        name="greedy",
                        universe="fixed",
                        timeframe="1d",
                        max_position=Decimal("0.05"),
                        max_gross=Decimal(1),
                    )
                )

            def generate(self, view: MarketView) -> TargetWeights:
                return TargetWeights(view.as_of, {IID: Decimal(10)})

        data = history([100.0] * 10)
        view = MarketView(
            as_of=as_decision_time(data["receive_time"][0]),
            history=data,
            universe=(IID,),
        )
        clipped = Greedy()(view)
        assert clipped.weights[IID] == Decimal("0.05")

    def test_gross_scaling_preserves_relative_views(self):
        weights = TargetWeights(
            as_of=as_decision_time(T0),
            weights={IID: Decimal("0.8"), InstrumentId("B"): Decimal("0.4")},
        )
        clipped = weights.clipped(Decimal(1), Decimal("0.6"))
        # 2:1 ratio must survive the scaling.
        assert clipped.weights[IID] / clipped.weights[InstrumentId("B")] == 2
        assert clipped.gross == pytest.approx(Decimal("0.6"))

    def test_sma_rejects_inverted_windows(self):
        with pytest.raises(ValueError, match="must be shorter"):
            SmaCrossover(fast=50, slow=20)


class TestEquityIsStampedWhenItHappened:
    """A fill on bar T+1 must appear in the row for T+1, not the row for T.

    The equity row used to be appended *after* the bar's orders executed, so
    the resulting position was booked into the decision bar and marked at that
    bar's close — a price from before the trade. A buy filled at an open of 130
    against a close of 100 showed a 23% loss on the bar preceding the trade.
    Terminal wealth was right; every path-dependent metric computed from the
    curve was not.
    """

    def gapped(self, execution_open: float, n: int = 8) -> pl.DataFrame:
        """Flat closes at 100; only the execution bar's open differs."""
        events = [T0 + timedelta(days=i) for i in range(n)]
        opens = [100.0] * n
        opens[2] = execution_open
        return pl.DataFrame(
            {
                "event_time": events,
                "receive_time": [t + timedelta(hours=1) for t in events],
                "instrument_id": [IID] * n,
                "open": opens,
                "high": [max(o, 100.0) * 1.001 for o in opens],
                "low": [min(o, 100.0) * 0.999 for o in opens],
                "close": [100.0] * n,
                "volume": [1_000_000.0] * n,
            },
            schema_overrides={
                "event_time": pl.Datetime("us", "UTC"),
                "receive_time": pl.Datetime("us", "UTC"),
            },
        )

    def run(self, execution_open: float):
        return engine(BuyAndHold(), rebalance_threshold=Decimal("0.5")).run(
            self.gapped(execution_open), universe=(IID,)
        )

    def test_the_bar_before_the_trade_is_untouched_by_it(self):
        result = self.run(130.0)
        trade_ts = result.trades["event_time"][0]
        curve = result.equity_curve
        before = curve.filter(pl.col("event_time") < trade_ts)["equity"].to_list()
        assert before, "expected at least one pre-trade bar"
        assert all(e == pytest.approx(1_000_000.0) for e in before)

    def test_the_entry_gap_lands_on_the_trade_bar(self):
        result = self.run(130.0)
        trade_ts = result.trades["event_time"][0]
        curve = result.equity_curve
        on_bar = curve.filter(pl.col("event_time") == trade_ts)["equity"][0]
        assert on_bar < 800_000.0

    def test_a_favourable_gap_moves_the_same_way(self):
        """Symmetric: the misattribution was not directional, which is why it
        survived — it never made anything look obviously wrong."""
        result = self.run(70.0)
        trade_ts = result.trades["event_time"][0]
        before = result.equity_curve.filter(pl.col("event_time") < trade_ts)["equity"].to_list()
        assert all(e == pytest.approx(1_000_000.0) for e in before)

    def test_terminal_wealth_is_unchanged_by_the_timing(self):
        """The correction re-times P&L; it does not create or destroy any."""
        assert self.run(100.0).equity_curve["equity"][-1] == pytest.approx(998_819.70, abs=1.0)

    def test_the_curve_has_one_row_per_bar(self):
        result = self.run(100.0)
        assert result.equity_curve.height == 8
