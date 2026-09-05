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

from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from engine.accounting import Portfolio
from engine.backtest.engine import BacktestConfig, BacktestEngine, MarketModel
from engine.backtest.fills import NextOpenFill
from engine.costs.borrow import BorrowModel
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


def engine(
    strategy: Strategy,
    cost_multiplier: Decimal = Decimal(1),
    borrow: BorrowModel | None = None,
    **cfg,
) -> BacktestEngine:
    base = NseEquityCostModel(slippage=NO_SLIPPAGE)
    model = base if cost_multiplier == 1 else ScaledCostModel(base, cost_multiplier)
    return BacktestEngine(
        strategy=strategy,
        market=MarketModel(
            cost_model=model,
            fill_model=NextOpenFill(model),
            instruments=INSTRUMENTS,
            borrow=borrow or BorrowModel(),
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


class TestRebalanceFrequency:
    """Costs are paid per trade; information arrives at the horizon the signal
    measures. The engine could only re-decide every bar, so a signal whose IC
    peaks at 63 days was traded daily — paying transaction costs at many times
    the rate its information refreshed.
    """

    def data(self, sessions: int = 200) -> pl.DataFrame:
        rng = np.random.default_rng(4)
        frames = []
        for i in range(6):
            closes = list(100.0 * np.exp(np.cumsum(rng.normal((i - 3) * 0.001, 0.012, sessions))))
            frames.append(history(closes, InstrumentId(f"NSE:INE{i:03d}")))
        return pl.concat(frames)

    def run(self, every: int):
        instruments = {
            InstrumentId(f"NSE:INE{i:03d}"): INSTRUMENT.model_copy(
                update={"instrument_id": InstrumentId(f"NSE:INE{i:03d}"), "symbol": f"N{i}"}
            )
            for i in range(6)
        }
        model = NseEquityCostModel(slippage=NO_SLIPPAGE)
        return BacktestEngine(
            strategy=SmaCrossover(fast=5, slow=10),
            market=MarketModel(
                cost_model=model, fill_model=NextOpenFill(model), instruments=instruments
            ),
            config=BacktestConfig(rebalance_every=every),
        ).run(self.data(), universe=tuple(instruments))

    def test_rebalancing_less_often_trades_less(self):
        assert self.run(21).orders_filled < self.run(1).orders_filled

    def test_rebalancing_less_often_costs_less(self):
        daily = self.run(1).equity_curve["fees_paid"][-1]
        monthly = self.run(21).equity_curve["fees_paid"][-1]
        assert monthly < daily

    def test_daily_is_the_default(self):
        """The behaviour every existing result was produced under."""
        assert BacktestConfig().rebalance_every == 1

    def test_the_curve_still_covers_every_bar(self):
        """Holding between rebalances is not the same as stopping: the book is
        still marked every session, or a drawdown between decisions would be
        invisible."""
        held = self.run(21)
        daily = self.run(1)
        assert held.equity_curve.height == daily.equity_curve.height

    def test_zero_is_treated_as_daily_rather_than_dividing_by_it(self):
        assert self.run(0).orders_filled == self.run(1).orders_filled


class TestBorrowIsChargedForTime:
    """A short pays to be held, not to be traded — MASTER_PLAN §7.3.

    The engine could build market-neutral books before this existed, and every
    one of them financed its short leg for free. That is not a small error in a
    known direction; it is an unknown one, because the rate depends on names
    the signal happened to pick.
    """

    def test_a_long_only_run_is_bit_for_bit_unchanged(self):
        """Load-bearing. Every result already measured is long-only, so if this
        fails, adding borrow costs silently invalidated the entire record."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        free = engine(BuyAndHold(), borrow=BorrowModel(annual_rate=Decimal(0)))
        charged = engine(BuyAndHold(), borrow=BorrowModel(annual_rate=Decimal("0.25")))

        one = free.run(history(prices))
        two = charged.run(history(prices))

        assert one.equity_curve["equity"].to_list() == two.equity_curve["equity"].to_list()
        assert one.final_portfolio.fees_paid == two.final_portfolio.fees_paid

    def test_a_short_pays_every_session_it_is_held(self):
        held = Portfolio(cash=Decimal(1_000_000))
        held.positions[IID] = replace(held.position(IID), quantity=Decimal(-100))
        model = BorrowModel(annual_rate=Decimal("0.252"))  # 0.1% per session
        engine_ = engine(BuyAndHold(), borrow=model)

        marks = {IID: Decimal(1000)}
        before = held.cash
        engine_._charge_borrow(held, marks)
        one_session = before - held.cash

        engine_._charge_borrow(held, marks)
        engine_._charge_borrow(held, marks)
        assert before - held.cash == one_session * 3

    def test_the_charge_lands_in_fees_not_in_thin_air(self):
        """Cash out must equal fees recorded, or the equity curve and the fee
        total tell two different stories about the same run."""
        held = Portfolio(cash=Decimal(1_000_000))
        held.positions[IID] = replace(held.position(IID), quantity=Decimal(-100))
        engine_ = engine(BuyAndHold(), borrow=BorrowModel(annual_rate=Decimal("0.252")))

        before_cash, before_fees = held.cash, held.fees_paid
        engine_._charge_borrow(held, {IID: Decimal(1000)})

        assert before_cash - held.cash == held.fees_paid - before_fees
        assert held.fees_paid > before_fees

    def test_a_name_with_no_mark_is_skipped_not_guessed(self):
        """A short in a suspended name still has to be borrowed, but pricing it
        off nothing would invent the number (§14.1.5)."""
        held = Portfolio(cash=Decimal(1_000_000))
        held.positions[IID] = replace(held.position(IID), quantity=Decimal(-100))
        engine_ = engine(BuyAndHold(), borrow=BorrowModel(annual_rate=Decimal("0.25")))

        before = held.cash
        engine_._charge_borrow(held, {})
        assert held.cash == before

    def test_a_flat_book_pays_nothing(self):
        flat = Portfolio(cash=Decimal(1_000_000))
        engine_ = engine(BuyAndHold(), borrow=BorrowModel(annual_rate=Decimal("0.25")))
        engine_._charge_borrow(flat, {IID: Decimal(1000)})
        assert flat.cash == Decimal(1_000_000)


class TestLatestCloseCannotSeeTheDecisionBar:
    """The look-ahead this nearly shipped as a speed-up.

    `latest_close()` is now seeded by the engine instead of being recomputed
    from a growing prefix on every bar — 92% of a backtest's runtime. The first
    version seeded it from `last_marks`, the *valuation* mark, which includes
    the decision session's own close because the ledger values the book at
    today's price. What a strategy may see is `receive_time <= as_of`, which
    excludes a bar that closed at 15:30 and published at 18:00 — today's among
    them.

    The run's return went from 3.61% to 4.64% and nothing failed. These pin the
    distinction so the next person optimising this cannot make the trade
    silently.
    """

    def _market(self):
        model = NseEquityCostModel(slippage=NO_SLIPPAGE)
        return MarketModel(
            cost_model=model,
            fill_model=NextOpenFill(model),
            instruments=self._instruments(),
        )

    def _instruments(self):
        return {
            InstrumentId(i): Instrument(
                instrument_id=InstrumentId(i),
                symbol=i[-1],
                asset_class=AssetClass.EQUITY,
                exchange=Exchange.NSE,
                currency=Currency.INR,
                tick_size=Decimal("0.01"),
            )
            for i in ("NSE:A", "NSE:B")
        }

    def _panel(self):
        """Two names, three sessions, published 2.5h after each close."""
        import polars as pl

        rows = []
        for day, closes in enumerate([(100.0, 200.0), (110.0, 190.0), (121.0, 180.0)], start=1):
            event = datetime(2024, 1, day, 10, 0, tzinfo=UTC)
            for instrument_id, close in zip(("NSE:A", "NSE:B"), closes, strict=True):
                rows.append(
                    {
                        "event_time": event,
                        "receive_time": event + timedelta(hours=2, minutes=30),
                        "instrument_id": instrument_id,
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close,
                        "volume": 1e7,
                    }
                )
        return pl.DataFrame(rows)

    def test_the_view_never_carries_the_decision_session_close(self):
        """The decision bar publishes after the decision, so it is not there."""
        seen: list[dict] = []

        class Recorder(BuyAndHold):
            def generate(self, view):
                seen.append(dict(view.latest_close()))
                return super().generate(view)

        engine = BacktestEngine(
            strategy=Recorder(),
            market=self._market(),
            config=BacktestConfig(initial_cash=Decimal(100_000)),
        )
        engine.run(self._panel(), universe=("NSE:A", "NSE:B"))

        assert len(seen) >= 2, "the strategy never ran twice"
        # At the first decision nothing has published at all: session 1 closed
        # at 10:00 and publishes at 12:30, so there is no observable close yet.
        assert seen[0] == {}
        # At the second, session 1 is out and session 2 is not — the decision
        # bar's own close is still unpublished.
        assert seen[1] == {"NSE:A": 100.0, "NSE:B": 200.0}
        assert 110.0 not in seen[1].values()

    def test_it_matches_what_the_view_history_says(self):
        """The seeded answer and the one computed from the view's own rows are
        the same dictionary. If they ever differ, the seed is a second source
        of truth rather than a faster route to the first."""
        import polars as pl

        checked: list[bool] = []

        class Recorder(BuyAndHold):
            def generate(self, view):
                from_rows = (
                    view.history.sort("event_time")
                    .group_by("instrument_id")
                    .agg(pl.col("close").last())
                )
                expected = dict(
                    zip(
                        from_rows["instrument_id"].to_list(),
                        from_rows["close"].to_list(),
                        strict=True,
                    )
                )
                checked.append(view.latest_close() == expected)
                return super().generate(view)

        engine = BacktestEngine(
            strategy=Recorder(),
            market=self._market(),
            config=BacktestConfig(initial_cash=Decimal(100_000)),
        )
        engine.run(self._panel(), universe=("NSE:A", "NSE:B"))
        assert checked and all(checked)
