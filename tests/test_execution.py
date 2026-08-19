"""Order lifecycle, paper broker, reconciliation and alerting (§9, §18, §19).

`TestUnknownState` and `TestReconciliation` are the classes that matter: they
cover the two failure modes that turn a working system into one that quietly
holds a position it does not know about.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from core.clock import UTC, utc_now
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from core.orders import OrderState, OrderType, Side
from engine.accounting import Fill, Portfolio
from engine.costs.model import CostBreakdown
from ops.alerts import Alert, AlertRouter, ConsoleSink, Severity
from trading.execution.broker import BrokerError, BrokerPosition, PaperBroker
from trading.execution.orders import IllegalTransitionError, Order, OrderTransition, TradingMode
from trading.reconcile.positions import BreakKind, reconcile_positions

A = InstrumentId("NSE:A")
B = InstrumentId("NSE:B")
T0 = datetime(2024, 6, 3, 10, 0, tzinfo=UTC)

INSTRUMENT_A = Instrument(
    instrument_id=A,
    symbol="A",
    asset_class=AssetClass.EQUITY,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.05"),
)
INSTRUMENT_B = Instrument(
    instrument_id=B,
    symbol="B",
    asset_class=AssetClass.EQUITY,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.05"),
)
INSTRUMENTS = {A: INSTRUMENT_A, B: INSTRUMENT_B}


def order(**overrides) -> Order:
    defaults = dict(
        strategy_id="s1",
        instrument_id=A,
        side=Side.BUY,
        quantity=Decimal(100),
        order_type=OrderType.MARKET,
        mode=TradingMode.PAPER,
        decision_time=T0,
    )
    return Order(**{**defaults, **overrides})


class TestOrderConstruction:
    def test_starts_created_with_history(self):
        o = order()
        assert o.state is OrderState.CREATED
        assert len(o.history) == 1

    def test_idempotency_key_is_unique(self):
        assert order().idempotency_key != order().idempotency_key

    def test_zero_quantity_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            order(quantity=Decimal(0))

    def test_limit_order_needs_a_price(self):
        with pytest.raises(ValueError, match="limit price"):
            order(order_type=OrderType.LIMIT)

    def test_stop_order_needs_a_stop(self):
        with pytest.raises(ValueError, match="stop price"):
            order(order_type=OrderType.STOP)

    def test_naive_decision_time_rejected(self):
        with pytest.raises(ValueError, match="naive"):
            order(decision_time=datetime(2024, 6, 3, 10, 0))


class TestStateMachine:
    def test_legal_path_to_filled(self):
        o = order()
        for state in (OrderState.RISK_CHECKED, OrderState.SUBMITTED, OrderState.ACKNOWLEDGED):
            o.transition(state)
        o.apply_fill(Decimal(100), Decimal(500))
        assert o.state is OrderState.FILLED

    def test_illegal_transition_raises(self):
        # A state machine that tolerates an undefined transition has stopped
        # describing reality.
        with pytest.raises(IllegalTransitionError, match="cannot move"):
            order().transition(OrderState.FILLED)

    def test_terminal_states_accept_nothing(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.REJECTED)
        with pytest.raises(IllegalTransitionError, match="terminal"):
            o.transition(OrderState.SUBMITTED)

    def test_history_is_appended_not_rewritten(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        assert [t.to_state for t in o.history] == [
            OrderState.CREATED,
            OrderState.RISK_CHECKED,
            OrderState.SUBMITTED,
        ]

    def test_format_history_renders(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        assert "RISK_CHECKED" in o.format_history()


class TestPartialFills:
    def test_partial_then_complete(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        o.apply_fill(Decimal(40), Decimal(100))
        assert o.state is OrderState.PARTIALLY_FILLED
        assert o.remaining == 60
        o.apply_fill(Decimal(60), Decimal(110))
        assert o.state is OrderState.FILLED

    def test_average_price_is_weighted(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        o.apply_fill(Decimal(50), Decimal(100))
        o.apply_fill(Decimal(50), Decimal(200))
        assert o.average_fill_price == Decimal(150)

    def test_overfill_rejected(self):
        """A venue or adapter bug that would corrupt the position permanently."""
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        with pytest.raises(ValueError, match="exceeds remaining"):
            o.apply_fill(Decimal(150), Decimal(100))

    def test_non_positive_fill_rejected(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        with pytest.raises(ValueError, match="positive"):
            o.apply_fill(Decimal(0), Decimal(100))

    @pytest.mark.parametrize("price", [Decimal(0), Decimal(-1)])
    def test_non_positive_fill_price_rejected(self, price):
        """A zero-price fill would divide the weighted average by a free trade
        and corrupt cost basis permanently."""
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        with pytest.raises(ValueError, match="fill price must be positive"):
            o.apply_fill(Decimal(10), price)

    def test_supplied_history_is_not_overwritten(self):
        """Reconstructing an order from the database must keep its real
        history, not stamp a fresh CREATED over it."""
        existing = [
            OrderTransition(None, OrderState.CREATED, T0, "created"),
            OrderTransition(OrderState.CREATED, OrderState.RISK_CHECKED, T0, "checked"),
        ]
        rebuilt = order(history=existing, state=OrderState.RISK_CHECKED)
        assert rebuilt.history == existing


class TestUnknownState:
    """The load-bearing state (§19)."""

    def test_submitted_can_become_unknown(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        o.mark_unknown("submit timed out")
        assert o.state is OrderState.UNKNOWN
        assert o.needs_reconciliation

    def test_unknown_is_not_terminal(self):
        o = order()
        o.transition(OrderState.RISK_CHECKED)
        o.transition(OrderState.SUBMITTED)
        o.mark_unknown("timeout")
        assert not o.is_terminal
        # It still consumes risk budget: the venue may hold it.
        assert o.is_live

    def test_reconciliation_resolves_it_either_way(self):
        for resolution in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
            o = order()
            o.transition(OrderState.RISK_CHECKED)
            o.transition(OrderState.SUBMITTED)
            o.mark_unknown("timeout")
            o.transition(resolution, "resolved by reconciliation")
            assert o.state is resolution


class TestPaperBroker:
    def test_submit_returns_an_id_and_fills(self):
        broker = PaperBroker(INSTRUMENTS)
        broker_id = broker.submit(order(), Decimal(500))
        assert broker_id.startswith("paper-")
        assert len(broker.fills_since(None)) == 1

    def test_slippage_moves_against_the_trader(self):
        broker = PaperBroker(INSTRUMENTS, slippage_bps=Decimal(100))
        broker.submit(order(side=Side.BUY), Decimal(500))
        buy_price = broker.fills_since(None)[0].price
        assert buy_price > Decimal(500)

        broker = PaperBroker(INSTRUMENTS, slippage_bps=Decimal(100))
        broker.submit(order(side=Side.SELL), Decimal(500))
        assert broker.fills_since(None)[0].price < Decimal(500)

    def test_fill_price_respects_tick_size(self):
        broker = PaperBroker(INSTRUMENTS, slippage_bps=Decimal(7))
        broker.submit(order(), Decimal("500.03"))
        price = broker.fills_since(None)[0].price
        assert price % Decimal("0.05") == 0

    def test_live_order_refused(self):
        """A mode mismatch is how a live order reaches a simulator."""
        broker = PaperBroker(INSTRUMENTS)
        with pytest.raises(BrokerError, match="refuses a LIVE"):
            broker.submit(order(mode=TradingMode.LIVE), Decimal(500))

    def test_unknown_instrument_refused(self):
        broker = PaperBroker({})
        with pytest.raises(BrokerError, match="unknown instrument"):
            broker.submit(order(), Decimal(500))

    def test_non_positive_reference_refused(self):
        with pytest.raises(BrokerError, match="positive"):
            PaperBroker(INSTRUMENTS).submit(order(), Decimal(0))

    def test_positions_accumulate(self):
        broker = PaperBroker(INSTRUMENTS, slippage_bps=Decimal(0))
        broker.submit(order(quantity=Decimal(100)), Decimal(500))
        broker.submit(order(quantity=Decimal(100)), Decimal(700))
        position = broker.positions()[0]
        assert position.quantity == 200
        assert position.average_price == Decimal(600)

    def test_closing_removes_the_position(self):
        broker = PaperBroker(INSTRUMENTS, slippage_bps=Decimal(0))
        broker.submit(order(side=Side.BUY), Decimal(500))
        broker.submit(order(side=Side.SELL), Decimal(600))
        assert broker.positions() == []

    def test_fills_since_marker(self):
        broker = PaperBroker(INSTRUMENTS)
        broker.submit(order(), Decimal(500))
        first = broker.fills_since(None)[0].broker_fill_id
        broker.submit(order(), Decimal(510))
        assert len(broker.fills_since(first)) == 1

    def test_cancel_unknown_order_refused(self):
        with pytest.raises(BrokerError, match="unknown order"):
            PaperBroker(INSTRUMENTS).cancel("nope")

    def test_mode_is_paper(self):
        assert PaperBroker(INSTRUMENTS).mode is TradingMode.PAPER
        assert not TradingMode.PAPER.touches_real_money
        assert TradingMode.LIVE.touches_real_money

    def test_submission_walks_the_real_state_machine(self):
        """Paper must exercise the lifecycle live depends on (§19, §M9).

        A simulator that fills an order while leaving it in CREATED means six
        weeks of paper trading prove nothing about the state machine — and the
        first live order takes an untested path.
        """
        submitted = order()
        broker = PaperBroker(INSTRUMENTS)
        broker_id = broker.submit(submitted, Decimal(500))
        assert submitted.state is OrderState.ACKNOWLEDGED
        assert submitted.broker_order_id == broker_id
        assert [t.to_state for t in submitted.history] == [
            OrderState.CREATED,
            OrderState.RISK_CHECKED,
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
        ]

    def test_resubmitting_does_not_double_transition(self):
        """A retry must not walk an already-acknowledged order through the
        machine again — that would raise on an illegal transition and turn a
        harmless duplicate into a crash mid-rebalance."""
        broker = PaperBroker(INSTRUMENTS)
        retried = order()
        broker.submit(retried, Decimal(500))
        broker.submit(retried, Decimal(500))
        assert retried.state is OrderState.ACKNOWLEDGED
        assert [t.to_state for t in retried.history].count(OrderState.ACKNOWLEDGED) == 1

    def test_reducing_a_position_keeps_the_entry_price(self):
        """Selling half does not re-average the remainder.

        The remaining shares were bought at the original average; moving it on
        an exit would misstate cost basis, and therefore realised P&L, for the
        rest of the position's life.
        """
        broker = PaperBroker(INSTRUMENTS, slippage_bps=Decimal(0))
        broker.submit(order(quantity=Decimal(200)), Decimal(500))
        broker.submit(order(quantity=Decimal(50), side=Side.SELL), Decimal(900))
        position = broker.positions()[0]
        assert position.quantity == 150
        assert position.average_price == Decimal(500)

    def test_flipping_through_zero_opens_at_the_new_price(self):
        broker = PaperBroker(INSTRUMENTS, slippage_bps=Decimal(0))
        broker.submit(order(quantity=Decimal(100), side=Side.BUY), Decimal(500))
        broker.submit(order(quantity=Decimal(150), side=Side.SELL), Decimal(600))
        position = broker.positions()[0]
        assert position.quantity == -50

    def test_cancel_marks_the_order_cancelled(self):
        broker = PaperBroker(INSTRUMENTS)
        pending = order()
        broker_id = broker.submit(pending, Decimal(500))
        assert pending.state is OrderState.ACKNOWLEDGED
        broker.cancel(broker_id)
        assert pending.state is OrderState.CANCELLED

    def test_cancelling_twice_is_refused(self):
        """A terminal order cannot be cancelled again. Tolerating it would mean
        the state machine has stopped describing reality (§19)."""
        broker = PaperBroker(INSTRUMENTS)
        broker_id = broker.submit(order(), Decimal(500))
        broker.cancel(broker_id)
        with pytest.raises(BrokerError, match="already CANCELLED"):
            broker.cancel(broker_id)

    def test_an_unrecognised_marker_returns_everything(self):
        """Fail safe, not silent: a marker from a different broker session must
        not cause fills to be skipped. Replaying a known fill is caught by the
        order state machine; skipping one is invisible.
        """
        broker = PaperBroker(INSTRUMENTS)
        broker.submit(order(), Decimal(500))
        broker.submit(order(), Decimal(510))
        assert len(broker.fills_since("pf-does-not-exist")) == 2

    def test_injected_position_is_visible_to_reconciliation(self):
        """The hook that simulates the discrepancies reconciliation catches."""
        broker = PaperBroker(INSTRUMENTS)
        broker.inject_position(BrokerPosition(A, Decimal(42), Decimal(500)))
        assert broker.positions()[0].quantity == 42


class TestReconciliation:
    """An unexplained break is a system-down event, not a rounding issue (§9)."""

    def portfolio_with(self, quantity: Decimal, price: Decimal = Decimal(500)) -> Portfolio:
        p = Portfolio(cash=Decimal(10_000_000))
        p.apply_fill(
            Fill(
                instrument_id=A,
                side=Side.BUY,
                quantity=quantity,
                price=price,
                costs=CostBreakdown(),
                event_time=utc_now(),
            )
        )
        return p

    def test_matching_positions_are_clean(self):
        report = reconcile_positions(
            self.portfolio_with(Decimal(100)),
            [BrokerPosition(A, Decimal(100), Decimal(500))],
        )
        assert report.is_clean
        assert not report.should_halt

    def test_quantity_mismatch_halts(self):
        """The broker thinks 100, we think 90. One of those sizes the next order."""
        report = reconcile_positions(
            self.portfolio_with(Decimal(90)),
            [BrokerPosition(A, Decimal(100), Decimal(500))],
        )
        assert report.should_halt
        assert report.breaks[0].kind is BreakKind.QUANTITY
        assert report.breaks[0].difference == Decimal(-10)

    def test_phantom_position_halts(self):
        report = reconcile_positions(self.portfolio_with(Decimal(100)), [])
        assert report.breaks[0].kind is BreakKind.PHANTOM
        assert report.should_halt

    def test_unrecorded_position_halts(self):
        report = reconcile_positions(
            Portfolio(cash=Decimal(1_000_000)),
            [BrokerPosition(A, Decimal(50), Decimal(500))],
        )
        assert report.breaks[0].kind is BreakKind.UNRECORDED
        assert report.should_halt

    def test_price_difference_is_reported_but_does_not_halt(self):
        # Misstates P&L, not risk. Worth knowing, not worth halting.
        report = reconcile_positions(
            self.portfolio_with(Decimal(100), Decimal(500)),
            [BrokerPosition(A, Decimal(100), Decimal("500.10"))],
        )
        assert report.breaks[0].kind is BreakKind.PRICE
        assert not report.should_halt

    def test_zero_tolerance_by_default(self):
        report = reconcile_positions(
            self.portfolio_with(Decimal(100)),
            [BrokerPosition(A, Decimal("99.999"), Decimal(500))],
        )
        assert not report.is_clean

    def test_tolerance_can_be_widened_explicitly(self):
        report = reconcile_positions(
            self.portfolio_with(Decimal(100)),
            [BrokerPosition(A, Decimal("99.999"), Decimal(500))],
            quantity_tolerance=Decimal("0.01"),
            price_tolerance=Decimal("0.01"),
        )
        assert report.is_clean

    def test_report_names_the_next_step(self):
        report = reconcile_positions(self.portfolio_with(Decimal(100)), [])
        assert "system-down event" in report.format()

    def test_clean_report_formats(self):
        assert "CLEAN" in reconcile_positions(Portfolio(cash=Decimal(0)), []).format()

    def test_multiple_instruments(self):
        p = self.portfolio_with(Decimal(100))
        report = reconcile_positions(
            p,
            [
                BrokerPosition(A, Decimal(100), Decimal(500)),
                BrokerPosition(B, Decimal(20), Decimal(300)),
            ],
        )
        assert report.instruments_checked == 2
        assert len(report.breaks) == 1


class TestAlerting:
    """Alert construction and routing, against a recording sink.

    Push channels were removed; the console is the only sink. These test the
    *content* of an alert — severity escalation, the runbook pointer — which is
    what a runbook-following operator actually reads.
    """

    def stub(self) -> tuple[object, list[Alert]]:
        captured: list[Alert] = []

        class Recording:
            def send(self, alert: Alert) -> bool:
                captured.append(alert)
                return True

        return Recording(), captured

    def test_router_tries_every_sink(self):
        """A sink that fails must not stop the ones after it — the point of a
        second channel is that it works when the first does not."""

        class Failing:
            def send(self, alert: Alert) -> bool:
                return False

        sink, captured = self.stub()
        assert AlertRouter([Failing(), sink]).send(Alert(Severity.CRITICAL, "test"))
        assert len(captured) == 1

    def test_router_without_sinks_reports_failure(self):
        assert AlertRouter([]).send(Alert(Severity.INFO, "test")) is False

    def test_console_sink_accepts(self):
        assert ConsoleSink().send(Alert(Severity.INFO, "test"))

    def test_severity_escalates_with_staleness(self):
        """A feed 30x past its threshold is not the same event as one 1.5x
        past it, and the severity has to say so."""
        sink, captured = self.stub()
        router = AlertRouter([sink])
        router.data_stale("nse", seconds=3, threshold=2)
        router.data_stale("nse", seconds=60, threshold=2)
        assert captured[0].severity is Severity.WARN
        assert captured[1].severity is Severity.CRITICAL

    def test_alerts_carry_a_runbook(self):
        """An alert without a runbook is a problem statement at 3am."""
        sink, captured = self.stub()
        AlertRouter([sink]).reconciliation_break(2, "detail")
        assert captured[0].runbook
        assert "runbook:" in captured[0].format()

    def test_critical_wakes_the_operator(self):
        assert Severity.CRITICAL.wakes_the_operator
        assert not Severity.WARN.wakes_the_operator

    def test_daily_summary_includes_drift(self):
        sink, captured = self.stub()
        AlertRouter([sink]).daily_summary(Decimal(1234), 5, drift=-0.012)
        assert "drift" in captured[0].body.lower()

    def test_kill_switch_alert_names_the_operator(self):
        sink, captured = self.stub()
        AlertRouter([sink]).kill_switch(engaged=True, by="sagar", reason="test halt")
        assert "sagar" in captured[0].body
        assert captured[0].severity is Severity.CRITICAL
