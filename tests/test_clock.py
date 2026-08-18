"""The three clocks (§3.3) — the look-ahead firewall."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.clock import (
    UTC,
    LookAheadError,
    NaiveDatetimeError,
    as_decision_time,
    as_event_time,
    as_receive_time,
    assert_observable,
    elapsed,
    is_observable,
    require_utc,
    utc_now,
)

IST = timezone(timedelta(hours=5, minutes=30))


def test_naive_datetime_is_rejected_not_coerced():
    # Silently assuming a timezone is how a backtest ends up shifted by 5.5h.
    with pytest.raises(NaiveDatetimeError):
        require_utc(datetime(2024, 1, 1, 9, 15))


def test_aware_datetime_normalised_to_utc():
    ist_ts = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    utc_ts = require_utc(ist_ts)
    assert utc_ts.tzinfo is UTC
    assert utc_ts.hour == 3
    assert utc_ts.minute == 45


def test_utc_now_is_aware():
    assert utc_now().tzinfo is UTC


def test_constructors_reject_naive():
    naive = datetime(2024, 1, 1)
    for ctor in (as_event_time, as_receive_time, as_decision_time):
        with pytest.raises(NaiveDatetimeError):
            ctor(naive)


class TestObservability:
    """receive_time <= decision_time. Not event_time (§3.3)."""

    def test_past_data_is_observable(self):
        receive = as_receive_time(datetime(2024, 1, 1, 9, 0, tzinfo=UTC))
        decision = as_decision_time(datetime(2024, 1, 1, 10, 0, tzinfo=UTC))
        assert is_observable(receive, decision)

    def test_simultaneous_is_observable(self):
        # Acting on an event at the instant it arrives is legitimate.
        ts = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        assert is_observable(as_receive_time(ts), as_decision_time(ts))

    def test_future_data_is_not_observable(self):
        receive = as_receive_time(datetime(2024, 1, 1, 11, 0, tzinfo=UTC))
        decision = as_decision_time(datetime(2024, 1, 1, 10, 0, tzinfo=UTC))
        assert not is_observable(receive, decision)

    def test_assert_observable_raises_with_diagnostic(self):
        receive = as_receive_time(datetime(2024, 1, 1, 11, 0, tzinfo=UTC))
        decision = as_decision_time(datetime(2024, 1, 1, 10, 0, tzinfo=UTC))
        with pytest.raises(LookAheadError) as exc:
            assert_observable(receive, decision, "RELIANCE close")
        assert "RELIANCE close" in str(exc.value)
        assert "1:00:00" in str(exc.value)

    def test_assert_observable_silent_when_sound(self):
        ts = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        assert_observable(as_receive_time(ts), as_decision_time(ts), "ok")


def test_elapsed_measures_across_clock_types():
    receive = as_receive_time(datetime(2024, 1, 1, 11, 0, tzinfo=UTC))
    decision = as_decision_time(datetime(2024, 1, 1, 10, 0, tzinfo=UTC))
    assert elapsed(receive, decision) == timedelta(hours=1)


def test_elapsed_normalises_timezones():
    # Same instant, different representations.
    a = datetime(2024, 1, 1, 9, 15, tzinfo=IST)
    b = datetime(2024, 1, 1, 3, 45, tzinfo=UTC)
    assert elapsed(a, b) == timedelta(0)
