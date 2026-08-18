"""The three clocks — MASTER_PLAN §3.3.

Most look-ahead bias enters through the gap between when something *happened*
and when the system could *know* it. Making the three timestamps distinct types
turns that whole bug class into a type error:

    event_time    when it happened at the exchange
    receive_time  when this system learned about it
    decision_time when a strategy is permitted to act on it

Invariant enforced by the backtester and the data layer:

    a strategy at decision_time = T may only observe data whose
    receive_time <= T                      (NOT event_time <= T)

A quarterly result "for Q2" is not usable on the last day of Q2. It is usable
on its filing date. Index membership as of 2019 is not today's membership.

All timestamps are timezone-aware UTC. Naive datetimes are rejected at
construction and banned by lint (§14.1.3). Conversion to market-local time
happens only at the display layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import NewType

__all__ = [
    "UTC",
    "DecisionTime",
    "EventTime",
    "ReceiveTime",
    "as_decision_time",
    "as_event_time",
    "as_receive_time",
    "assert_observable",
    "elapsed",
    "is_observable",
    "require_utc",
    "utc_now",
]

UTC = timezone.utc

#: When the event occurred at the venue.
EventTime = NewType("EventTime", datetime)
#: When this system received the event. The only clock that gates observation.
ReceiveTime = NewType("ReceiveTime", datetime)
#: The moment a strategy is allowed to act.
DecisionTime = NewType("DecisionTime", datetime)


class NaiveDatetimeError(ValueError):
    """Raised when a timestamp arrives without timezone information.

    Deliberately fatal rather than coerced: silently assuming a timezone is how
    an entire backtest ends up shifted by 5.5 hours without anyone noticing.
    """

    def __init__(self, value: datetime) -> None:
        super().__init__(
            f"naive datetime {value!r} has no timezone. Every timestamp in this "
            "system is timezone-aware UTC (MASTER_PLAN §14.1.3)."
        )


def utc_now() -> datetime:
    """Current time, timezone-aware UTC. The only sanctioned wall-clock call."""
    return datetime.now(UTC)


def require_utc(value: datetime) -> datetime:
    """Return `value` normalised to UTC, rejecting naive datetimes.

    Raises:
        NaiveDatetimeError: if `value` carries no timezone.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise NaiveDatetimeError(value)
    return value.astimezone(UTC)


def as_event_time(value: datetime) -> EventTime:
    return EventTime(require_utc(value))


def as_receive_time(value: datetime) -> ReceiveTime:
    return ReceiveTime(require_utc(value))


def as_decision_time(value: datetime) -> DecisionTime:
    return DecisionTime(require_utc(value))


def elapsed(later: datetime, earlier: datetime) -> timedelta:
    """Signed gap between two instants.

    Takes plain `datetime` deliberately. The clock NewTypes are assignable to
    `datetime`, but subtracting them directly fails type-checking because
    `datetime.__sub__` is overloaded — which is the type system doing its job.
    Crossing the boundary happens here, once, in a function that only measures.
    """
    return require_utc(later) - require_utc(earlier)


def is_observable(receive: ReceiveTime, decision: DecisionTime) -> bool:
    """True when data received at `receive` may be used at `decision`.

    Strictly ``receive <= decision``. Equality is permitted because a decision
    taken *on* an event is legitimate; anything later is the future.
    """
    return receive <= decision


def assert_observable(receive: ReceiveTime, decision: DecisionTime, what: str) -> None:
    """Fail loudly when data from the future is about to be used (§14.1.5).

    Raises:
        LookAheadError: if `receive` is after `decision`.
    """
    if not is_observable(receive, decision):
        raise LookAheadError(what, receive, decision)


class LookAheadError(AssertionError):
    """Data from the future reached a decision point.

    This is never recoverable and never suppressed: a backtest that continues
    after look-ahead produces a number that looks like evidence and is not.
    """

    def __init__(self, what: str, receive: ReceiveTime, decision: DecisionTime) -> None:
        ahead = elapsed(receive, decision)
        super().__init__(
            f"look-ahead on {what}: data received {receive.isoformat()} is "
            f"{ahead} after decision time {decision.isoformat()}"
        )
        self.what = what
        self.receive = receive
        self.decision = decision
