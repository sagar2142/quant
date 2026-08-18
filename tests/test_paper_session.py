"""The paper trading cycle (§20, §M9).

What must survive here is not "orders get submitted" — it is the properties
that make six weeks of paper evidence about the live system: fills come from
the broker rather than from our own assumptions, risk blocks are enforced
mid-cycle against the moving book, reconciliation runs every cycle, and state
round-trips to the paisa.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal

import pytest

from core.clock import utc_now
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from engine.accounting import Portfolio
from engine.costs.india import NseEquityCostModel
from trading.execution.broker import BrokerPosition, PaperBroker
from trading.paper.session import CycleInputs, PaperSession
from trading.paper.state import PaperState, PaperStateStore, StateCorruptError
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits

A = InstrumentId("NSE:AAA")
B = InstrumentId("NSE:BBB")


def instrument(instrument_id: InstrumentId) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=str(instrument_id).split(":")[1],
        asset_class=AssetClass.EQUITY,
        exchange=Exchange.NSE,
        currency=Currency.INR,
        tick_size=Decimal("0.05"),
    )


INSTRUMENTS = {A: instrument(A), B: instrument(B)}

#: Generous ADV so the liquidity check never interferes unless a test wants it.
DEEP_ADV = {A: Decimal(500_000_000), B: Decimal(500_000_000)}


#: The default limits cap any single name at 10% of NAV (§8) — correct for a
#: thirty-name book, and guaranteed to block a two-name test portfolio. Tests
#: that are not about limits declare wide ones; the default stays strict and
#: `test_default_limits_block_a_concentrated_book` proves it.
WIDE_LIMITS = RiskLimits(max_position_pct=Decimal("0.95"))


def session_for(broker: PaperBroker, limits: RiskLimits | None = None) -> PaperSession:
    return PaperSession(
        instruments=INSTRUMENTS,
        cost_model=NseEquityCostModel(),
        risk=RiskEngine(limits=limits or WIDE_LIMITS),
        broker=broker,
        strategy_id="test",
    )


def inputs_for(
    weights: dict[InstrumentId, Decimal],
    marks: dict[InstrumentId, Decimal] | None = None,
) -> CycleInputs:
    return CycleInputs(
        session=date(2026, 8, 14),
        weights=weights,
        marks=marks or {A: Decimal(100), B: Decimal(200)},
        adv=DEEP_ADV,
    )


class TestCycleHappyPath:
    def test_a_rebalance_reaches_the_book(self):
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = session_for(PaperBroker(instruments=INSTRUMENTS)).run_cycle(
            portfolio,
            Decimal(1_000_000),
            inputs_for({A: Decimal("0.5"), B: Decimal("0.5")}),
        )
        assert report.submitted == 2
        assert report.fills_applied == 2
        assert not portfolio.position(A).is_flat
        assert not portfolio.position(B).is_flat
        assert report.reconciliation is not None and report.reconciliation.is_clean

    def test_fees_come_from_the_cost_model_not_the_broker(self):
        """PaperBroker reports zero fees; the NSE schedule is applied here.

        Same model as the backtest, so paper-versus-backtest drift is a market
        difference, not an accounting one.
        """
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = session_for(PaperBroker(instruments=INSTRUMENTS)).run_cycle(
            portfolio,
            Decimal(1_000_000),
            # 0.45 of NAV: inside the ₹5L single-order cap, which stays at its
            # default here precisely so a fat order would still be refused.
            inputs_for({A: Decimal("0.45")}),
        )
        assert report.fills_applied == 1
        assert report.fees_paid > 0
        assert portfolio.fees_paid == report.fees_paid

    def test_dropped_names_are_closed(self):
        """A name the strategy no longer wants is sold, not silently retained."""
        broker = PaperBroker(instruments=INSTRUMENTS)
        portfolio = Portfolio(cash=Decimal(1_000_000))
        cycle = session_for(broker)

        cycle.run_cycle(portfolio, Decimal(1_000_000), inputs_for({A: Decimal("0.5")}))
        marker_after_first = broker.fills_since(None)[-1].broker_fill_id

        report = cycle.run_cycle(
            portfolio,
            Decimal(1_000_000),
            inputs_for({B: Decimal("0.5")}),
            fill_marker=marker_after_first,
        )
        assert portfolio.position(A).is_flat
        assert not portfolio.position(B).is_flat
        assert report.reconciliation is not None and report.reconciliation.is_clean


class TestWholeShares:
    def test_cash_equity_quantities_are_integers(self):
        """A lot size of one means whole shares, not "no rounding".

        Found by the first real paper cycle, which planned an order for
        148.0198 shares. No exchange accepts that, and a backtest that fills it
        flatters itself by the fractional remainder on every position.
        """
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = session_for(PaperBroker(instruments=INSTRUMENTS)).run_cycle(
            portfolio,
            Decimal(1_000_000),
            # 100/3 forces a non-terminating decimal if nothing rounds.
            inputs_for({A: Decimal("0.45")}, marks={A: Decimal(100) / 3, B: Decimal(200)}),
        )
        assert report.fills_applied == 1
        quantity = portfolio.position(A).quantity
        assert quantity == quantity.to_integral_value()
        assert quantity > 0


class TestRiskEnforcement:
    def test_default_limits_block_a_concentrated_book(self):
        """Two names at 50% each versus the stock 10% per-name cap.

        This is the configuration every other test has to opt out of, and it
        must keep blocking: a paper loop that quietly ran with loosened limits
        would produce six weeks of evidence about the wrong system.
        """
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = session_for(PaperBroker(instruments=INSTRUMENTS), RiskLimits()).run_cycle(
            portfolio,
            Decimal(1_000_000),
            inputs_for({A: Decimal("0.5"), B: Decimal("0.5")}),
        )
        assert report.submitted == 0
        assert len(report.blocked) == 2
        for blocked in report.blocked:
            assert any(c.name == "position_size" for c in blocked.verdict.breaches)

    def test_gross_exposure_is_judged_against_the_moving_book(self):
        """Each order sees the book as already-submitted orders left it.

        Thirty orders judged against one stale snapshot are each individually
        fine while their sum breaches gross exposure; the second order here
        must be blocked because the first was submitted.
        """
        limits = RiskLimits(
            max_position_pct=Decimal("0.95"), max_gross_exposure_pct=Decimal("0.45")
        )
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = session_for(PaperBroker(instruments=INSTRUMENTS), limits).run_cycle(
            portfolio,
            Decimal(1_000_000),
            inputs_for({A: Decimal("0.4"), B: Decimal("0.4")}),
        )
        assert report.submitted == 1
        assert len(report.blocked) == 1
        blocked = report.blocked[0]
        assert any(c.name == "gross_exposure" for c in blocked.verdict.breaches)

    def test_kill_switch_blocks_everything(self):
        risk = RiskEngine()
        risk.engage_kill("test halt", "pytest")
        cycle = PaperSession(
            instruments=INSTRUMENTS,
            cost_model=NseEquityCostModel(),
            risk=risk,
            broker=PaperBroker(instruments=INSTRUMENTS),
        )
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = cycle.run_cycle(portfolio, Decimal(1_000_000), inputs_for({A: Decimal("0.5")}))
        assert report.submitted == 0
        assert portfolio.cash == Decimal(1_000_000)

    def test_a_blocked_order_does_not_strand_the_rest(self):
        """One blocked name must not abort the whole rebalance."""
        limits = RiskLimits(max_position_pct=Decimal("0.3"))  # A at 0.5 breaches, B at 0.2 fits
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = session_for(PaperBroker(instruments=INSTRUMENTS), limits).run_cycle(
            portfolio,
            Decimal(1_000_000),
            # A breaches the per-name cap; B is fine.
            inputs_for({A: Decimal("0.5"), B: Decimal("0.2")}),
        )
        assert len(report.blocked) == 1
        assert report.submitted == 1
        assert not portfolio.position(B).is_flat


class TestReconciliation:
    def test_an_injected_discrepancy_halts(self):
        """The broker holding something we do not know about is a halt (§9)."""
        broker = PaperBroker(instruments=INSTRUMENTS)
        broker.inject_position(BrokerPosition(B, Decimal(100), Decimal(200)))
        portfolio = Portfolio(cash=Decimal(1_000_000))
        report = session_for(broker).run_cycle(
            portfolio, Decimal(1_000_000), inputs_for({A: Decimal("0.5")})
        )
        assert report.reconciliation is not None
        assert not report.reconciliation.is_clean
        assert report.should_halt

    def test_marker_prevents_double_application(self):
        """Re-running with the persisted marker must not re-apply old fills."""
        broker = PaperBroker(instruments=INSTRUMENTS)
        portfolio = Portfolio(cash=Decimal(1_000_000))
        cycle = session_for(broker)

        first = cycle.run_cycle(portfolio, Decimal(1_000_000), inputs_for({A: Decimal("0.5")}))
        quantity_after_first = portfolio.position(A).quantity

        # Same weights: the planner sees the target met and plans nothing; the
        # marker keeps the old fills from being replayed on top.
        second = cycle.run_cycle(
            portfolio,
            Decimal(1_000_000),
            inputs_for({A: Decimal("0.5")}),
            fill_marker=first.fill_marker,
        )
        assert second.fills_applied == 0
        assert portfolio.position(A).quantity == quantity_after_first


class TestStateRoundTrip:
    def state(self) -> PaperState:
        portfolio = Portfolio(cash=Decimal("993211.47"))
        broker = PaperBroker(instruments=INSTRUMENTS)
        session_for(broker).run_cycle(
            portfolio, Decimal(1_000_000), inputs_for({A: Decimal("0.5")})
        )
        return PaperState(
            strategy_id="test",
            portfolio=portfolio,
            peak_equity=Decimal("1000123.45"),
            fill_marker="pf-abc",
            cycles=3,
            last_cycle_at=utc_now(),
            last_session=date(2026, 8, 14),
        )

    def test_round_trip_is_exact(self, tmp_path):
        store = PaperStateStore(tmp_path)
        original = self.state()
        store.save(original)
        loaded = store.restore()

        assert loaded.portfolio.cash == original.portfolio.cash
        assert loaded.peak_equity == original.peak_equity
        assert loaded.cycles == 3
        assert loaded.last_session == date(2026, 8, 14)
        position = loaded.portfolio.position(A)
        assert position.quantity == original.portfolio.position(A).quantity
        assert position.average_price == original.portfolio.position(A).average_price

    def test_decimals_survive_as_strings(self, tmp_path):
        """The file must never contain JSON numbers for money."""
        store = PaperStateStore(tmp_path)
        store.save(self.state())
        raw = json.loads(store.state_path.read_text(encoding="utf-8"))
        assert isinstance(raw["cash"], str)
        assert all(isinstance(p["quantity"], str) for p in raw["positions"])

    def test_wrong_version_refuses_to_load(self, tmp_path):
        store = PaperStateStore(tmp_path)
        store.save(self.state())
        raw = json.loads(store.state_path.read_text(encoding="utf-8"))
        raw["version"] = 999
        store.state_path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(StateCorruptError, match="schema version"):
            store.restore()

    def test_truncated_file_refuses_to_load(self, tmp_path):
        store = PaperStateStore(tmp_path)
        store.save(self.state())
        content = store.state_path.read_text(encoding="utf-8")
        store.state_path.write_text(content[: len(content) // 2], encoding="utf-8")
        with pytest.raises(StateCorruptError):
            store.restore()

    def test_halt_survives_the_round_trip(self, tmp_path):
        """A halt that a restart resets is not a halt."""
        store = PaperStateStore(tmp_path)
        state = self.state()
        state.engage_halt("phantom position in NSE:BBB")
        store.save(state)
        loaded = store.restore()
        assert loaded.halted
        assert "phantom" in loaded.halt_reason

    def test_equity_log_appends(self, tmp_path):
        store = PaperStateStore(tmp_path)
        for day in range(3):
            store.append_equity(
                date(2026, 8, 10) + timedelta(days=day),
                Decimal(1_000_000) + day,
                Decimal(500_000),
                Decimal(100),
            )
        history = store.equity_history()
        assert len(history) == 3
        assert history[0]["session"] == "2026-08-10"
        assert history[2]["equity"] == "1000002"
