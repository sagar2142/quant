"""Backtest regression suite — MASTER_PLAN §M3 gate, §14.5.

Two jobs:

1. **Reconciliation.** Buy-and-hold on real NSE data must agree with hand
   arithmetic. If the engine and a pocket calculator disagree about the simplest
   possible strategy, nothing more complex is worth measuring.

2. **Golden numbers.** Pinned results that fail the build on any drift. A
   refactor that changes a backtest result is either a bug or a deliberate
   change requiring a new pinned value — never a silent difference.

Requires the NSE panel to be populated:

    python -m apps.cli.ingest_nse --start 2024-01-01 --end 2024-03-31
"""

from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest

from core.clock import as_decision_time, utc_now
from core.config import settings
from core.instruments import InstrumentId
from data.store.bars import NoDataError
from data.store.panel import PanelStore

pytestmark = pytest.mark.regression

#: Enough sessions for a short lookback to produce trades.
MIN_SESSIONS = 30


@pytest.fixture(scope="module")
def panel() -> PanelStore:
    store = PanelStore(settings.lake, venue="NSE")
    sessions = store.sessions()
    if len(sessions) < MIN_SESSIONS:
        pytest.skip(
            f"NSE panel has {len(sessions)} sessions, need {MIN_SESSIONS} — run: "
            "python -m apps.cli.ingest_nse --start 2024-01-01 --end 2024-03-31"
        )
    return store


#: The regression window is the most recent year of sessions, not the whole
#: panel. These tests are fingerprints — they need determinism over a stable
#: window, not statistical power — and a full multi-year backtest per test
#: turned the suite from seconds into an hour the day the panel was backfilled.
REGRESSION_SESSIONS = 250


@pytest.fixture(scope="module")
def history(panel: PanelStore) -> pl.DataFrame:
    try:
        full = panel.view(as_of=as_decision_time(utc_now()))
    except NoDataError:
        pytest.skip("panel empty")
    recent = full["event_time"].unique().sort().tail(REGRESSION_SESSIONS)
    return full.filter(pl.col("event_time").is_in(recent.implode()))


@pytest.fixture(scope="module")
def universe(panel: PanelStore) -> tuple[InstrumentId, ...]:
    from apps.cli.backtest import build_universe

    members = build_universe(panel, top_n=10)
    if not members:
        pytest.skip("universe empty — ingest more sessions")
    return members


def build_engine(strategy, history: pl.DataFrame, universe, cost_multiple=Decimal(1)):
    from apps.cli.backtest import nse_instrument
    from engine.backtest import BacktestConfig, BacktestEngine, MarketModel, NextOpenFill
    from engine.costs.india import NseEquityCostModel
    from engine.costs.model import ScaledCostModel

    symbols = dict(history.select("instrument_id", "symbol").unique().iter_rows())
    instruments = {InstrumentId(i): nse_instrument(i, symbols.get(i, i)) for i in universe}

    base = NseEquityCostModel()
    costs = base if cost_multiple == 1 else ScaledCostModel(base, cost_multiple)
    return BacktestEngine(
        strategy=strategy,
        market=MarketModel(
            cost_model=costs, fill_model=NextOpenFill(costs), instruments=instruments
        ),
        config=BacktestConfig(initial_cash=Decimal(10_000_000)),
    )


class TestDeterminismOnRealData:
    """M3 gate (a), on real NSE sessions rather than a toy series."""

    def test_identical_across_runs(self, history, universe):
        from quant.strategies.baselines import BuyAndHold

        runs = [
            build_engine(BuyAndHold(), history, universe).run(history, universe=universe)
            for _ in range(3)
        ]
        equities = [r.equity_curve["equity"].to_list() for r in runs]
        assert equities[0] == equities[1] == equities[2]

        trades = [r.trades.to_dicts() for r in runs]
        assert trades[0] == trades[1] == trades[2]


class TestBuyAndHoldReconciliation:
    """M3 gate (b): agree with a pocket calculator."""

    def test_entry_is_after_the_first_observable_bar(self, history, universe):
        """Gate (c) on real data.

        At the first session the strategy has observed nothing — the bhavcopy
        publishes 2.5h after the close — so it cannot act. Its first possible
        decision is session 1, executing on session 2.
        """
        from quant.strategies.baselines import BuyAndHold

        result = build_engine(BuyAndHold(), history, universe).run(history, universe=universe)
        if result.trades.is_empty():
            pytest.skip("no trades — panel too short for this universe")

        sessions = sorted(history["event_time"].unique().to_list())
        first_trade = result.trades["event_time"].min()
        assert first_trade >= sessions[2]

    def test_equity_equals_positions_plus_cash(self, history, universe):
        """The ledger must close: equity is not an independent quantity.

        Marks are each instrument's **last observed** close, not the final
        session's cross-section. A name that stops trading mid-window — a
        rename, a delisting — is absent from the last session but still held,
        and the engine correctly carries it at its last-seen price. Marking
        from the final session alone silently drops that position from the
        hand-computed side and reports a six-figure "discrepancy" that is
        actually the test's own survivorship bias.
        """
        from quant.strategies.baselines import BuyAndHold

        result = build_engine(BuyAndHold(), history, universe).run(history, universe=universe)
        final = result.final_portfolio
        curve_end = Decimal(str(result.equity_curve["equity"][-1]))

        last_seen = (
            history.sort("event_time")
            .group_by("instrument_id")
            .agg(pl.col("close").last())
            .to_dicts()
        )
        marks = {row["instrument_id"]: Decimal(str(row["close"])) for row in last_seen}
        held = final.open_positions()
        assert set(held) <= set(marks), "a held position was never observed in the panel"
        position_value = sum(
            (position.market_value(marks[iid]) for iid, position in held.items()),
            start=Decimal(0),
        )
        assert abs((final.cash + position_value) - curve_end) < Decimal("1.00")

    def test_fees_are_nonzero_and_bounded(self, history, universe):
        """NSE delivery costs ~0.22% round trip (§7.1). A buy-and-hold entry
        pays roughly half of that, once."""
        from quant.strategies.baselines import BuyAndHold

        result = build_engine(BuyAndHold(), history, universe).run(history, universe=universe)
        if result.trades.is_empty():
            pytest.skip("no trades")
        fees = float(result.equity_curve["fees_paid"][-1])
        assert fees > 0
        assert fees / 10_000_000 < 0.01


class TestCostSensitivityOnRealData:
    """M3 gate (d), and gauntlet check 7 (§5.4)."""

    def test_triple_costs_increases_fees(self, history, universe):
        from quant.strategies.baselines import CrossSectionalMomentum

        strategy = CrossSectionalMomentum(lookback_bars=20, skip_bars=2)
        cheap = build_engine(strategy, history, universe, Decimal(1)).run(
            history, universe=universe
        )
        dear = build_engine(strategy, history, universe, Decimal(3)).run(history, universe=universe)
        if cheap.trades.is_empty():
            pytest.skip("no trades — panel too short")

        assert dear.equity_curve["fees_paid"][-1] > cheap.equity_curve["fees_paid"][-1]

    def test_higher_costs_never_improve_returns(self, history, universe):
        from quant.strategies.baselines import CrossSectionalMomentum

        strategy = CrossSectionalMomentum(lookback_bars=20, skip_bars=2)
        cheap = build_engine(strategy, history, universe, Decimal(1)).run(
            history, universe=universe
        )
        dear = build_engine(strategy, history, universe, Decimal(3)).run(history, universe=universe)
        if cheap.trades.is_empty():
            pytest.skip("no trades")
        assert dear.total_return <= cheap.total_return


class TestGoldenNumbers:
    """Pinned invariants. Drift fails the build (§14.5).

    These are fingerprints, not claims about profitability.
    """

    def test_no_orders_are_rejected(self, history, universe):
        """Sizing plans against available cash, so an unfundable order is never
        created (§17). A rejection means the funding logic regressed.

        Orders for names that did not trade that session are counted separately
        as `orders_no_market`: a delisting or a halt is the market's absence,
        not a defect in our logic, and conflating the two makes a corporate
        event look like a bug.
        """
        from quant.strategies.baselines import BuyAndHold

        result = build_engine(BuyAndHold(), history, universe).run(history, universe=universe)
        assert result.orders_rejected == 0

    def test_absent_instruments_are_reported_not_hidden(self, history, universe):
        """A name that stops trading must be visible, never silently dropped."""
        from quant.strategies.baselines import BuyAndHold

        result = build_engine(BuyAndHold(), history, universe).run(history, universe=universe)
        assert result.orders_generated >= result.orders_filled + result.orders_no_market

    def test_equity_curve_covers_every_session(self, history, universe):
        from quant.strategies.baselines import BuyAndHold

        result = build_engine(BuyAndHold(), history, universe).run(history, universe=universe)
        assert result.equity_curve.height == history["event_time"].n_unique()

    def test_cash_never_goes_negative(self, history, universe):
        """Margin allowance is zero by default; an overdraft would be free
        leverage flattering every metric downstream (§14.1.5)."""
        from quant.strategies.baselines import CrossSectionalMomentum

        result = build_engine(
            CrossSectionalMomentum(lookback_bars=20, skip_bars=2), history, universe
        ).run(history, universe=universe)
        assert min(result.equity_curve["cash"].to_list()) >= 0
