"""Operator alerting — MASTER_PLAN §12.7, §24.

**The UI is never the only alarm.** You will be asleep, at work, or on a train
when the feed goes stale, and a dashboard nobody is looking at has not alerted
anyone. Telegram is the primary channel precisely because it pushes.

**Alerting must never break trading.** A failed send is logged and swallowed —
the one place in this system where swallowing an exception is correct, because
the alternative is that an unreachable Telegram API halts a working strategy.
Every other silent failure in this codebase is a bug; this one is the design.

**Severity is not decoration.** It decides whether the message can wait. INFO
is a daily summary, WARN wants attention today, CRITICAL means capital is at
risk right now and the operator should be woken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from core.clock import utc_now

__all__ = ["Alert", "AlertRouter", "AlertSink", "ConsoleSink", "Severity"]

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"

    @property
    def icon(self) -> str:
        return {Severity.INFO: "•", Severity.WARN: "▲", Severity.CRITICAL: "■"}[self]

    @property
    def wakes_the_operator(self) -> bool:
        return self is Severity.CRITICAL


@dataclass(frozen=True)
class Alert:
    """One operator-facing message."""

    severity: Severity
    title: str
    body: str = ""
    #: Where to look. An alert without a next step wastes the reader's time.
    runbook: str = ""

    def format(self) -> str:
        stamp = utc_now().strftime("%Y-%m-%d %H:%M:%SZ")
        lines = [f"{self.severity.icon} [{self.severity.value}] {self.title}", stamp]
        if self.body:
            lines.extend(["", self.body])
        if self.runbook:
            lines.extend(["", f"runbook: {self.runbook}"])
        return "\n".join(lines)


@runtime_checkable
class AlertSink(Protocol):
    def send(self, alert: Alert) -> bool:
        """Deliver an alert. Returns whether it succeeded. Never raises."""


@dataclass
class ConsoleSink:
    """Prints alerts. The fallback that always works."""

    def send(self, alert: Alert) -> bool:
        logger.warning("ALERT %s: %s", alert.severity.value, alert.title)
        return True


@dataclass
class AlertRouter:
    """Fans an alert out to every sink.

    Tries all of them even after one fails: the point of a second channel is
    that it works when the first does not.
    """

    sinks: list[AlertSink]

    def send(self, alert: Alert) -> bool:
        """Returns True if at least one sink accepted the alert."""
        if not self.sinks:
            logger.error("no alert sinks configured; alert lost: %s", alert.title)
            return False
        return any([sink.send(alert) for sink in self.sinks])  # noqa: C419 — all must run

    # ── the alerts the system actually raises ───────────────────────────────

    def data_stale(self, feed: str, seconds: float, threshold: float) -> bool:
        """Feed staleness (§12.7). CRITICAL because a stale price means every
        position is being valued and risk-checked against fiction."""
        return self.send(
            Alert(
                severity=Severity.CRITICAL if seconds > threshold * 5 else Severity.WARN,
                title=f"{feed} data stale: {seconds:.0f}s",
                body=f"No update for {seconds:.0f}s (threshold {threshold:.0f}s).",
                runbook="ops/runbooks/data_outage.md",
            )
        )

    def reconciliation_break(self, count: int, detail: str) -> bool:
        """An unexplained break is a system-down event (§9)."""
        return self.send(
            Alert(
                severity=Severity.CRITICAL,
                title=f"reconciliation: {count} break(s)",
                body=detail,
                runbook="ops/runbooks/reconciliation_break.md",
            )
        )

    def risk_breach(self, limit_name: str, observed: Decimal, threshold: Decimal) -> bool:
        return self.send(
            Alert(
                severity=Severity.CRITICAL,
                title=f"risk breach: {limit_name}",
                body=f"Observed {observed} against limit {threshold}.",
                runbook="ops/runbooks/risk_breach.md",
            )
        )

    def kill_switch(self, engaged: bool, by: str, reason: str) -> bool:
        state = "ENGAGED" if engaged else "RELEASED"
        return self.send(
            Alert(
                severity=Severity.CRITICAL,
                title=f"kill switch {state}",
                body=f"By {by}: {reason}",
                runbook="ops/runbooks/kill_switch.md",
            )
        )

    def drawdown_rung(self, drawdown: Decimal, scale: Decimal) -> bool:
        return self.send(
            Alert(
                severity=Severity.WARN,
                title=f"drawdown ladder engaged at {drawdown:.2%}",
                body=f"Positions scaled to {scale:.0%} of normal size.",
                runbook="ops/runbooks/drawdown.md",
            )
        )

    def daily_summary(self, pnl: Decimal, positions: int, drift: float | None) -> bool:
        body = f"P&L {pnl:+,.2f} across {positions} position(s)."
        if drift is not None:
            body += f"\nPaper-vs-backtest drift: {drift:+.2%}"
        return self.send(Alert(Severity.INFO, "daily summary", body))
