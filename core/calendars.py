"""Market calendars — MASTER_PLAN §1.2.

The system trades stock exchanges (§0.0), so every market here is *sessioned*:
a fixed open and close, weekends off, holidays off.

    NSE   09:15-15:30 IST  (UTC+05:30, no DST, so 03:45-10:00 UTC year-round)
    US    09:30-16:00 ET   (DST applies, so the UTC offset moves twice a year)

**The US daylight-saving shift is a real trap.** The NSE session sits at a fixed
UTC offset; the US session does not. Storing a US session as fixed UTC hours is
correct for roughly half the year and silently wrong for the other half, which
shows up as bars landing in the wrong session rather than as an error. Sessions
are therefore always defined in *local* exchange time and converted, never
stored as UTC constants.

**The authoritative trading calendar is empirical, not assumed.** A hardcoded
holiday list is wrong the moment an exchange announces an unscheduled closure or
a special session (NSE's Muhurat trading, a US half-day, an emergency halt).
``SessionCalendar.from_observed`` derives sessions from the dates actually
present in the data, which is self-consistent with whatever the backtest
replays.

The declared holiday list is kept only as a *cross-check*: M2's data-quality
suite flags any date where the declared calendar and the observed data
disagree. That is how a missing holiday surfaces as an alert instead of as a
silently wrong backtest.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from core.clock import UTC, require_utc

__all__ = [
    "ET",
    "IST",
    "Calendar",
    "MarketClosedError",
    "SessionCalendar",
    "nse_equity_calendar",
    "us_equity_calendar",
]

IST = ZoneInfo("Asia/Kolkata")
#: US exchanges observe daylight saving; this must stay a zone, never an offset.
ET = ZoneInfo("America/New_York")

WEEKDAYS = frozenset({0, 1, 2, 3, 4})  # Mon-Fri


class MarketClosedError(LookupError):
    """Asked for a session bound on a date the market never opened."""

    def __init__(self, day: date, market: str) -> None:
        super().__init__(f"{market} has no session on {day.isoformat()}")


@runtime_checkable
class Calendar(Protocol):
    """Every market answers these four questions."""

    name: str

    def is_open(self, ts: datetime) -> bool:
        """Whether the market is accepting trades at this instant."""

    def session_date(self, ts: datetime) -> date | None:
        """The trading session `ts` belongs to, or None if closed."""

    def sessions_between(self, start: date, end: date) -> list[date]:
        """Every trading session in [start, end], inclusive."""

    def session_bounds(self, day: date) -> tuple[datetime, datetime]:
        """(open, close) in UTC for a session date.

        Raises:
            MarketClosedError: if `day` is not a session.
        """


@dataclass(frozen=True)
class SessionCalendar:
    """Markets with explicit open/close times, weekends and holidays.

    Frozen: a calendar is configuration, and a calendar that mutates midway
    through a backtest silently changes which bars existed.
    """

    name: str
    tz: ZoneInfo
    open_local: time
    close_local: time
    weekmask: frozenset[int] = WEEKDAYS
    holidays: frozenset[date] = frozenset()
    #: When present this is the authority over declared rules — see module docstring.
    observed_sessions: frozenset[date] | None = None

    def __post_init__(self) -> None:
        if self.close_local <= self.open_local:
            raise ValueError(
                f"{self.name}: close {self.close_local} must be after open {self.open_local}"
            )

    @classmethod
    def from_observed(
        cls,
        base: SessionCalendar,
        sessions: Iterable[date],
    ) -> SessionCalendar:
        """Rebind a calendar to the sessions actually present in the data."""
        return dataclasses.replace(base, observed_sessions=frozenset(sessions))

    def is_session(self, day: date) -> bool:
        if self.observed_sessions is not None:
            return day in self.observed_sessions
        return day.weekday() in self.weekmask and day not in self.holidays

    def declared_is_session(self, day: date) -> bool:
        """Session per the declared rules, ignoring observed data.

        Used by the M2 quality suite to reconcile declared against observed.
        """
        return day.weekday() in self.weekmask and day not in self.holidays

    def is_open(self, ts: datetime) -> bool:
        local = require_utc(ts).astimezone(self.tz)
        if not self.is_session(local.date()):
            return False
        return self.open_local <= local.time() < self.close_local

    def session_date(self, ts: datetime) -> date | None:
        local = require_utc(ts).astimezone(self.tz)
        day = local.date()
        if not self.is_session(day):
            return None
        return day if self.open_local <= local.time() < self.close_local else None

    def sessions_between(self, start: date, end: date) -> list[date]:
        if end < start:
            return []
        days = (start + timedelta(days=i) for i in range((end - start).days + 1))
        return [d for d in days if self.is_session(d)]

    def session_bounds(self, day: date) -> tuple[datetime, datetime]:
        if not self.is_session(day):
            raise MarketClosedError(day, self.name)
        open_local = datetime.combine(day, self.open_local, tzinfo=self.tz)
        close_local = datetime.combine(day, self.close_local, tzinfo=self.tz)
        return open_local.astimezone(UTC), close_local.astimezone(UTC)

    def next_session(self, day: date, limit_days: int = 30) -> date:
        """The next trading session strictly after `day`.

        Raises:
            MarketClosedError: if none is found within `limit_days`.
        """
        for i in range(1, limit_days + 1):
            candidate = day + timedelta(days=i)
            if self.is_session(candidate):
                return candidate
        raise MarketClosedError(day, f"{self.name} (no session within {limit_days}d)")


def us_equity_calendar(holidays: Iterable[date] = ()) -> SessionCalendar:
    """US equity regular session: 09:30-16:00 America/New_York, Mon-Fri.

    Defined in *local* exchange time, not UTC. The US observes daylight saving,
    so the session sits at 13:30-20:00 UTC in summer and 14:30-21:00 UTC in
    winter. Pinning it to either would be silently wrong for half the year.

    Half-days (the day after Thanksgiving, Christmas Eve) close at 13:00 ET and
    are not modelled here — they arrive through `from_observed` once the data
    shows them, which is the same mechanism that handles unscheduled closures.
    """
    return SessionCalendar(
        name="US_EQUITY",
        tz=ET,
        open_local=time(9, 30),
        close_local=time(16, 0),
        weekmask=WEEKDAYS,
        holidays=frozenset(holidays),
    )


def nse_equity_calendar(holidays: Iterable[date] = ()) -> SessionCalendar:
    """NSE equity cash market: 09:15-15:30 IST, Mon-Fri.

    IST is UTC+05:30 year-round with no DST, so the session is 03:45-10:00 UTC.

    `holidays` is a *declared* list used only for cross-checking. Real sessions
    come from observed data via ``SessionCalendar.from_observed`` once the M2
    bhavcopy loader has run.
    """
    return SessionCalendar(
        name="NSE_EQUITY",
        tz=IST,
        open_local=time(9, 15),
        close_local=time(15, 30),
        weekmask=WEEKDAYS,
        holidays=frozenset(holidays),
    )
