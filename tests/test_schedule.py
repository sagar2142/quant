"""The daily ingest planner — MASTER_PLAN §13.4, §M2.

Every test here is a way the lake ends up quietly wrong rather than loudly
broken: a session asked for before it was published and recorded as
unavailable, a fresh clone launching a seven-year backfill because someone ran
the daily task, a BSE request before 2024 that answers 200 with the homepage.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from core.calendars import IST, SessionCalendar, nse_equity_calendar
from ops.schedule import (
    FEEDS,
    MAX_CATCHUP_DAYS,
    Feed,
    feed_named,
    latest_due_session,
    plan_feed,
)

NSE = feed_named("nse")
BSE = feed_named("bse")


def ist(day: date, hour: int, minute: int = 0) -> datetime:
    """An instant in the timezone the exchange publishes on."""
    return datetime.combine(day, time(hour, minute), tzinfo=IST)


#: A Wednesday, with the week around it clear.
WEDNESDAY = date(2026, 9, 2)
MONDAY = date(2026, 9, 7)


class TestFeeds:
    def test_every_feed_has_a_fetcher(self) -> None:
        """`daily.run_plan` dispatches on the name; a feed added to `FEEDS`
        without a branch there would plan work nobody performs."""
        from apps.cli.daily import BACKFILL_COMMAND

        assert {f.name for f in FEEDS} == set(BACKFILL_COMMAND)

    def test_bse_does_not_reach_before_udiff(self) -> None:
        """Earlier dates return the homepage with HTTP 200, not a 404, so the
        boundary has to be declared rather than discovered per request."""
        assert BSE.first_session >= date(2024, 1, 1)

    def test_an_unknown_feed_names_the_real_ones(self) -> None:
        with pytest.raises(KeyError, match="nse"):
            feed_named("nifty")


class TestLatestDueSession:
    def test_before_publication_the_session_is_not_due(self) -> None:
        """The failure this exists to prevent.

        A job at 15:45 IST asks for a file that does not exist yet, gets a
        404, and records the day as unavailable. Nothing retries, because
        "unavailable" and "not published yet" look identical afterwards.
        """
        assert latest_due_session(NSE, ist(WEDNESDAY, 15, 45)) == WEDNESDAY - timedelta(days=1)

    def test_after_publication_today_is_due(self) -> None:
        assert latest_due_session(NSE, ist(WEDNESDAY, 19, 0)) == WEDNESDAY

    def test_monday_morning_looks_back_to_friday(self) -> None:
        """Not "yesterday" — yesterday was Sunday and has no bhavcopy."""
        assert latest_due_session(NSE, ist(MONDAY, 9, 0)) == date(2026, 9, 4)

    def test_it_walks_back_over_a_holiday_cluster(self) -> None:
        """The morning after a four-day break, the newest session is the one
        before the break, not the day before."""
        closed = {date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4)}
        calendar = nse_equity_calendar(holidays=closed)
        assert latest_due_session(NSE, ist(MONDAY, 9, 0), calendar) == date(2026, 8, 31)

    def test_a_clock_before_the_archive_begins_yields_nothing(self) -> None:
        assert latest_due_session(BSE, ist(date(2023, 6, 1), 19, 0)) is None

    def test_the_publication_time_is_per_feed(self) -> None:
        """F&O and the index file appear later than the equity bhavcopy, and a
        run between the two must fetch one and defer the other rather than
        marking the later one unavailable."""
        early = ist(WEDNESDAY, 18, 45)
        assert latest_due_session(NSE, early) == WEDNESDAY
        assert latest_due_session(feed_named("fo"), early) == WEDNESDAY - timedelta(days=1)


class TestPlan:
    def test_a_current_lake_fetches_nothing(self) -> None:
        plan = plan_feed(NSE, [WEDNESDAY], ist(WEDNESDAY, 19, 0))
        assert not plan.due
        assert "current through" in plan.reason

    def test_a_lake_ahead_of_the_due_session_still_fetches_nothing(self) -> None:
        """A manual backfill may have already fetched tonight's file. That is
        not a reason to fetch it again."""
        plan = plan_feed(NSE, [WEDNESDAY], ist(WEDNESDAY, 16, 0))
        assert not plan.due

    def test_it_resumes_from_the_day_after_the_newest_held(self) -> None:
        plan = plan_feed(NSE, [date(2026, 8, 28)], ist(WEDNESDAY, 19, 0))
        assert (plan.start, plan.end) == (date(2026, 8, 29), WEDNESDAY)
        assert plan.deferred == 0

    def test_gaps_in_the_middle_are_not_refetched(self) -> None:
        """Deliberate: the fetchers skip sessions already held, so the range
        is bounded by the newest. A hole in the middle is a backfill's job,
        and re-walking six years of dates every evening to find one is not."""
        plan = plan_feed(NSE, [date(2026, 1, 5), date(2026, 8, 31)], ist(WEDNESDAY, 19, 0))
        assert plan.start == date(2026, 9, 1)

    def test_a_long_gap_is_capped_and_the_rest_reported(self) -> None:
        """A fortnight away is caught up; a year away is a backfill.

        The number that matters is `deferred`: a run that silently fetched
        half a gap and reported success would leave a hole nothing mentions.
        """
        plan = plan_feed(NSE, [date(2025, 1, 2)], ist(WEDNESDAY, 19, 0))
        assert plan.due
        assert plan.sessions <= MAX_CATCHUP_DAYS
        assert plan.deferred > 300

    def test_an_empty_lake_fetches_a_window_not_the_archive(self) -> None:
        """The failure mode: someone clones the repo, runs the daily task, and
        it begins a seven-year backfill nobody asked for."""
        plan = plan_feed(NSE, [], ist(WEDNESDAY, 19, 0))
        assert plan.due
        assert plan.sessions <= MAX_CATCHUP_DAYS
        assert plan.deferred > 0
        assert plan.held == 0

    def test_an_empty_lake_never_reaches_before_the_archive(self) -> None:
        feed = Feed("test", time(19, 0), WEDNESDAY - timedelta(days=3), "t")
        plan = plan_feed(feed, [], ist(WEDNESDAY, 19, 0))
        assert plan.start == feed.first_session
        assert plan.deferred == 0

    def test_nothing_published_yet_is_not_a_failure(self) -> None:
        plan = plan_feed(BSE, [], ist(date(2023, 6, 1), 19, 0))
        assert not plan.due
        assert plan.reason == "nothing published yet"

    def test_the_cap_is_configurable(self) -> None:
        plan = plan_feed(NSE, [date(2026, 1, 2)], ist(WEDNESDAY, 19, 0), max_catchup_days=90)
        assert plan.sessions <= 90
        assert plan.sessions > MAX_CATCHUP_DAYS

    def test_a_plan_that_is_not_due_counts_no_sessions(self) -> None:
        assert plan_feed(NSE, [WEDNESDAY], ist(WEDNESDAY, 19, 0)).sessions == 0

    def test_a_closed_market_does_not_make_the_run_look_behind(self) -> None:
        """During a holiday cluster the newest held session *is* current, and
        the summary must say so rather than reporting a growing gap that no
        amount of fetching will close."""
        closed = {date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4)}
        calendar: SessionCalendar = nse_equity_calendar(holidays=closed)
        plan = plan_feed(NSE, [date(2026, 8, 31)], ist(date(2026, 9, 3), 19, 0), calendar)
        assert not plan.due
        assert "current through 2026-08-31" in plan.reason
