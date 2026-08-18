"""Market calendars (§1.2).

Session boundaries in UTC are the load-bearing detail: get them wrong and every
bar lands in the wrong session. NSE sits at a fixed UTC offset; the US session
does not, and `TestDaylightSaving` is the class that guards that difference.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from core.calendars import (
    ET,
    IST,
    MarketClosedError,
    SessionCalendar,
    nse_equity_calendar,
    us_equity_calendar,
)
from core.clock import UTC

# 2024-01-15 was a Monday; 2024-01-20/21 a Saturday/Sunday.
MONDAY = date(2024, 1, 15)
SATURDAY = date(2024, 1, 20)
SUNDAY = date(2024, 1, 21)
REPUBLIC_DAY = date(2024, 1, 26)  # NSE holiday, a Friday


class TestNSECalendar:
    @pytest.fixture
    def nse(self) -> SessionCalendar:
        return nse_equity_calendar(holidays=[REPUBLIC_DAY])

    def test_session_bounds_in_utc(self, nse):
        # IST is UTC+05:30 with no DST: 09:15-15:30 IST == 03:45-10:00 UTC.
        open_utc, close_utc = nse.session_bounds(MONDAY)
        assert (open_utc.hour, open_utc.minute) == (3, 45)
        assert (close_utc.hour, close_utc.minute) == (10, 0)
        assert open_utc.tzinfo is UTC

    def test_open_during_session(self, nse):
        assert nse.is_open(datetime(2024, 1, 15, 10, 0, tzinfo=IST))

    def test_closed_before_open(self, nse):
        assert not nse.is_open(datetime(2024, 1, 15, 9, 14, tzinfo=IST))

    def test_open_exactly_at_bell(self, nse):
        assert nse.is_open(datetime(2024, 1, 15, 9, 15, tzinfo=IST))

    def test_closed_exactly_at_close(self, nse):
        # Half-open interval: 15:30 is closed, matching bar-stamping convention.
        assert not nse.is_open(datetime(2024, 1, 15, 15, 30, tzinfo=IST))

    def test_closed_on_weekend(self, nse):
        assert not nse.is_open(datetime(2024, 1, 20, 11, 0, tzinfo=IST))
        assert not nse.is_open(datetime(2024, 1, 21, 11, 0, tzinfo=IST))

    def test_closed_on_declared_holiday(self, nse):
        assert not nse.is_open(datetime(2024, 1, 26, 11, 0, tzinfo=IST))

    def test_accepts_utc_input(self, nse):
        # 04:00 UTC == 09:30 IST, mid-session.
        assert nse.is_open(datetime(2024, 1, 15, 4, 0, tzinfo=UTC))

    def test_session_date_none_when_closed(self, nse):
        assert nse.session_date(datetime(2024, 1, 20, 11, 0, tzinfo=IST)) is None
        assert nse.session_date(datetime(2024, 1, 15, 11, 0, tzinfo=IST)) == MONDAY

    def test_sessions_between_excludes_weekends_and_holidays(self, nse):
        sessions = nse.sessions_between(date(2024, 1, 22), date(2024, 1, 28))
        assert sessions == [
            date(2024, 1, 22),
            date(2024, 1, 23),
            date(2024, 1, 24),
            date(2024, 1, 25),
        ]  # 26th holiday, 27-28 weekend

    def test_session_bounds_raises_on_non_session(self, nse):
        with pytest.raises(MarketClosedError):
            nse.session_bounds(SATURDAY)

    def test_next_session_skips_weekend(self, nse):
        assert nse.next_session(date(2024, 1, 19)) == date(2024, 1, 22)

    def test_naive_datetime_rejected(self, nse):
        with pytest.raises(ValueError, match="naive"):
            nse.is_open(datetime(2024, 1, 15, 10, 0))

    def test_close_before_open_rejected(self):
        with pytest.raises(ValueError, match="must be after"):
            SessionCalendar(
                name="BROKEN",
                tz=IST,
                open_local=time(15, 30),
                close_local=time(9, 15),
            )


class TestUSCalendar:
    @pytest.fixture
    def us(self) -> SessionCalendar:
        return us_equity_calendar()

    def test_open_during_session(self, us):
        assert us.is_open(datetime(2024, 6, 3, 10, 0, tzinfo=ET))

    def test_closed_before_open(self, us):
        assert not us.is_open(datetime(2024, 6, 3, 9, 29, tzinfo=ET))

    def test_closed_exactly_at_close(self, us):
        assert not us.is_open(datetime(2024, 6, 3, 16, 0, tzinfo=ET))

    def test_closed_on_weekend(self, us):
        assert not us.is_open(datetime(2024, 6, 1, 11, 0, tzinfo=ET))

    def test_naive_datetime_rejected(self, us):
        with pytest.raises(ValueError, match="naive"):
            us.is_open(datetime(2024, 6, 3, 10, 0))


class TestDaylightSaving:
    """The US session moves in UTC twice a year. NSE never does.

    Pinning a US session to fixed UTC hours is correct for roughly half the
    year and silently wrong for the other half — bars land in the wrong session
    rather than raising an error.
    """

    def test_us_session_shifts_with_dst(self):
        us = us_equity_calendar()
        summer_open, summer_close = us.session_bounds(date(2024, 6, 3))
        winter_open, winter_close = us.session_bounds(date(2024, 12, 3))

        # EDT: 09:30 ET == 13:30 UTC.  EST: 09:30 ET == 14:30 UTC.
        assert (summer_open.hour, summer_open.minute) == (13, 30)
        assert (summer_close.hour, summer_close.minute) == (20, 0)
        assert (winter_open.hour, winter_open.minute) == (14, 30)
        assert (winter_close.hour, winter_close.minute) == (21, 0)

    def test_us_session_length_is_constant(self):
        us = us_equity_calendar()
        for day in (date(2024, 6, 3), date(2024, 12, 3)):
            open_utc, close_utc = us.session_bounds(day)
            assert close_utc - open_utc == (close_utc - open_utc).__class__(hours=6, minutes=30)

    def test_nse_session_never_shifts(self):
        nse = nse_equity_calendar()
        for day in (date(2024, 6, 3), date(2024, 12, 3)):
            open_utc, _ = nse.session_bounds(day)
            assert (open_utc.hour, open_utc.minute) == (3, 45)

    def test_the_two_markets_do_not_overlap(self):
        """NSE closes at 10:00 UTC; the US opens at 13:30 or 14:30 UTC.

        Useful operationally: a single daemon can serve both without contention,
        and a data-staleness alarm can distinguish "market closed" from "feed
        down" by knowing which session should be live.
        """
        nse = nse_equity_calendar()
        us = us_equity_calendar()
        _, nse_close = nse.session_bounds(date(2024, 6, 3))
        us_open, _ = us.session_bounds(date(2024, 6, 3))
        assert nse_close < us_open


class TestObservedSessions:
    """The declared calendar is a cross-check; observed data is the authority."""

    def test_observed_overrides_declared_rules(self):
        base = nse_equity_calendar()
        # A special Saturday session (NSE has run these) is not in the weekmask,
        # but it is in the data, so it must be a session.
        cal = SessionCalendar.from_observed(base, [MONDAY, SATURDAY])
        assert cal.is_session(SATURDAY)
        assert cal.is_open(datetime(2024, 1, 20, 11, 0, tzinfo=IST))

    def test_observed_excludes_undeclared_closure(self):
        base = nse_equity_calendar()
        cal = SessionCalendar.from_observed(base, [MONDAY])
        # Tuesday looks like a session by rule but has no data — an unscheduled
        # closure. Observed wins.
        assert cal.declared_is_session(date(2024, 1, 16))
        assert not cal.is_session(date(2024, 1, 16))

    def test_from_observed_preserves_session_times(self):
        base = nse_equity_calendar()
        cal = SessionCalendar.from_observed(base, [MONDAY])
        assert cal.session_bounds(MONDAY) == base.session_bounds(MONDAY)

    def test_declared_is_session_ignores_observed(self):
        cal = SessionCalendar.from_observed(nse_equity_calendar(), [])
        assert cal.declared_is_session(MONDAY)
        assert not cal.is_session(MONDAY)

    def test_us_half_day_arrives_through_observed(self):
        """Half-days are not modelled declaratively; they arrive with the data."""
        base = us_equity_calendar()
        cal = SessionCalendar.from_observed(base, [date(2024, 11, 29)])
        assert cal.is_session(date(2024, 11, 29))
        assert not cal.is_session(date(2024, 11, 28))  # Thanksgiving


def test_reversed_range_is_empty():
    for cal in (nse_equity_calendar(), us_equity_calendar()):
        assert cal.sessions_between(date(2024, 2, 1), date(2024, 1, 1)) == []
