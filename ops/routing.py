"""Alert routing — MASTER_PLAN §12.7.

One place that decides which channels an alert reaches, so no caller has to
remember. `ops.alerts` knows *how* to send; this knows *where*.

**Console only, by design.** Push notification channels were removed: this is a
research and analysis system, and an alert that interrupts you is the wrong
shape for work you do at a desk. Anything that halts the account is already
persisted to state, printed, and returned as a non-zero exit code — three
records that survive the process, which a notification does not.

The router remains because the shape is right: a break must be *reported*
through a single path rather than printed from wherever it was noticed. Adding
a channel later means adding a sink here and nothing else.
"""

from __future__ import annotations

from ops.alerts import AlertRouter, AlertSink, ConsoleSink

__all__ = ["build_router", "describe_channels"]


def build_router() -> AlertRouter:
    """Every channel an alert reaches."""
    sinks: list[AlertSink] = [ConsoleSink()]
    return AlertRouter(sinks)


def describe_channels() -> str:
    """One line naming the live channels, for a CLI to print at startup."""
    return "alerts: console"
