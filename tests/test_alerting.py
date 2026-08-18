"""Alert routing and the paper loop's use of it (§M9, §12.7).

The M9 gate requires having been woken by an alert at least once. That is only
possible if two things hold, and both are tested here: a configured channel is
actually built, and the paper loop actually raises on the events that matter.

**Delivery must never change the outcome.** An unreachable alerting API cannot
turn a clean session into a failure, and — far more important — cannot turn a
halted one into a success.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from apps.cli.paper import announce, persist_outcome
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from engine.accounting import Portfolio
from ops.alerts import Alert, AlertRouter, Severity
from ops.routing import build_router, describe_channels
from trading.paper.session import BlockedOrder, CycleReport
from trading.paper.state import PaperState, PaperStateStore
from trading.reconcile.positions import (
    BreakKind,
    ReconciliationBreak,
    ReconciliationReport,
)
from trading.risk.engine import RiskCheck, RiskVerdict
from trading.risk.limits import RiskDecision

A = InstrumentId("NSE:AAA")
INSTRUMENT = Instrument(
    instrument_id=A,
    symbol="AAA",
    asset_class=AssetClass.EQUITY,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.05"),
)


class Capture:
    """A sink that records instead of sending."""

    def __init__(self, accept: bool = True) -> None:
        self.sent: list[Alert] = []
        self.accept = accept

    def send(self, alert: Alert) -> bool:
        self.sent.append(alert)
        return self.accept


class Exploding:
    """A sink that fails the way an unreachable API does."""

    def send(self, alert: Alert) -> bool:
        raise RuntimeError("network down")


def report_for(
    *,
    breaks: list[ReconciliationBreak] | None = None,
    blocked: list[BlockedOrder] | None = None,
    opening: Decimal = Decimal(1_000_000),
    closing: Decimal = Decimal(1_000_000),
) -> CycleReport:
    return CycleReport(
        session=date(2026, 8, 18),
        opening_equity=opening,
        closing_equity=closing,
        peak_equity=opening,
        blocked=blocked or [],
        reconciliation=ReconciliationReport(breaks=breaks or [], instruments_checked=1),
    )


def blocked_order() -> BlockedOrder:
    return BlockedOrder(
        instrument_id=A,
        quantity=Decimal(100),
        verdict=RiskVerdict(
            decision=RiskDecision.BLOCK,
            checks=(RiskCheck("position_size", passed=False, message="too large"),),
        ),
    )


class TestRouterConstruction:
    def test_console_is_the_channel(self):
        """Push channels were removed deliberately. A halt is recorded in state,
        printed, and returned as a non-zero exit code — three records that
        survive the process."""
        assert len(build_router().sinks) == 1

    def test_channels_are_described(self):
        assert describe_channels() == "alerts: console"


class TestPaperLoopRaisesAlerts:
    def test_a_reconciliation_break_pages(self):
        """§9 — an unexplained break is a system-down event."""
        sink = Capture()
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        report = report_for(
            breaks=[ReconciliationBreak(A, BreakKind.UNRECORDED, Decimal(0), Decimal(500))]
        )
        announce(AlertRouter([sink]), state, report)

        assert len(sink.sent) == 1
        alert = sink.sent[0]
        assert alert.severity is Severity.CRITICAL
        assert alert.runbook == "ops/runbooks/reconciliation_break.md"
        assert "HALTED" in alert.body

    def test_a_halt_suppresses_the_routine_summary(self):
        """One CRITICAL page, not a page plus a cheerful daily summary."""
        sink = Capture()
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        announce(
            AlertRouter([sink]),
            state,
            report_for(
                breaks=[ReconciliationBreak(A, BreakKind.QUANTITY, Decimal(10), Decimal(20))]
            ),
        )
        assert [a.severity for a in sink.sent] == [Severity.CRITICAL]

    def test_a_clean_session_files_a_summary(self):
        sink = Capture()
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        announce(AlertRouter([sink]), state, report_for(closing=Decimal(1_010_000)))

        assert len(sink.sent) == 1
        assert sink.sent[0].severity is Severity.INFO
        assert "+10,000.00" in sink.sent[0].body

    def test_blocked_orders_warn_alongside_the_summary(self):
        sink = Capture()
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        announce(AlertRouter([sink]), state, report_for(blocked=[blocked_order()]))

        severities = [a.severity for a in sink.sent]
        assert Severity.CRITICAL in severities  # risk breach
        assert Severity.INFO in severities  # summary still filed


class TestDeliveryNeverChangesTheOutcome:
    def test_a_failing_sink_does_not_mask_a_halt(self, tmp_path):
        """The single most important property here.

        If an unreachable Telegram could swallow the halt, the account would
        resume trading against a book known to be wrong.
        """
        store = PaperStateStore(tmp_path)
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        report = report_for(
            breaks=[ReconciliationBreak(A, BreakKind.UNRECORDED, Decimal(0), Decimal(500))]
        )

        with pytest.raises(RuntimeError):
            persist_outcome(store, state, report, AlertRouter([Exploding()]))

        # State was written *before* the alert was attempted, so the halt
        # survives even though the send blew up.
        assert store.restore().halted

    def test_state_is_saved_before_alerting(self, tmp_path):
        """Ordering is the guarantee: a network failure must not lose a halt."""
        store = PaperStateStore(tmp_path)
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        saw_state_on_disk: list[bool] = []

        class Nosy:
            def send(self, alert: Alert) -> bool:
                saw_state_on_disk.append(store.exists())
                return True

        persist_outcome(store, state, report_for(), AlertRouter([Nosy()]))
        assert saw_state_on_disk == [True]

    def test_a_rejected_alert_still_returns_the_halt_code(self, tmp_path):
        store = PaperStateStore(tmp_path)
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        report = report_for(
            breaks=[ReconciliationBreak(A, BreakKind.PHANTOM, Decimal(500), Decimal(0))]
        )
        code = persist_outcome(store, state, report, AlertRouter([Capture(accept=False)]))
        assert code == 2

    def test_no_sinks_configured_does_not_crash(self, tmp_path):
        """An empty router logs and continues. Losing an alert must never lose
        a cycle."""
        store = PaperStateStore(tmp_path)
        state = PaperState("t", Portfolio(cash=Decimal(1_000_000)), Decimal(1_000_000))
        assert persist_outcome(store, state, report_for(), AlertRouter([])) == 0
