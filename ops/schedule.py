"""What the lake is missing, and whether it can be fetched yet — §13.4, §M2.

**Why this is a module and not a shell script.** The daily ingest has three
decisions in it that are easy to get subtly wrong, and each one fails by
producing a lake that *looks* fine:

1. **Is the file published yet?** NSE posts the bhavcopy after the close, not
   at it. A job that runs at 15:45 IST asks for a session that does not exist
   yet, gets a 404, and records the day as unavailable. Run it again tomorrow
   and nothing retries, because "unavailable" and "not yet published" are
   indistinguishable to a naive runner. So a session is only *due* once its
   publication time has passed.

2. **How far back to reach.** After a fortnight away the job should fill the
   gap, and after a fresh clone it should not silently start a seven-year
   backfill because someone ran the daily task. Catch-up is capped and the
   overflow is reported, not attempted.

3. **Where each feed's archive begins.** BSE's UDiFF files start 2024-01-01
   and earlier dates return the homepage with HTTP 200 — not a 404. A planner
   that asks for 2019 gets two thousand pages of HTML that parse to nothing.

Everything here is a pure function of (what is held, what time it is). No
network, no filesystem, so the awkward cases are testable rather than
discovered at 18:30 on a Tuesday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from core.calendars import IST, SessionCalendar, nse_equity_calendar
from core.clock import require_utc

__all__ = [
    "FEEDS",
    "MAX_CATCHUP_DAYS",
    "Feed",
    "Plan",
    "feed_named",
    "latest_due_session",
    "plan_feed",
]

#: Most sessions one daily run will reach back for. A fortnight covers a
#: holiday away and a long weekend. Past that the gap is a backfill — a
#: different job, run deliberately, with its own pacing — and the plan says so
#: rather than quietly starting one.
MAX_CATCHUP_DAYS = 15


@dataclass(frozen=True)
class Feed:
    """One source, and the two facts that decide when it can be asked."""

    name: str
    #: Local IST time the session's file appears. Deliberately later than the
    #: exchange's own claim: being an hour early costs a whole day of data,
    #: being an hour late costs nothing.
    publishes_local: time
    #: First session the archive actually serves. Asking earlier does not 404 —
    #: for BSE it returns the homepage with HTTP 200 — so the boundary has to
    #: be known here rather than discovered per request.
    first_session: date
    description: str


FEEDS: tuple[Feed, ...] = (
    Feed(
        name="nse",
        publishes_local=time(18, 30),
        first_session=date(2016, 1, 1),
        description="NSE equity bhavcopy",
    ),
    Feed(
        name="bse",
        publishes_local=time(19, 0),
        # UDiFF only. Earlier dates answer 200 with the BSE homepage, which is
        # why the boundary is declared rather than probed.
        first_session=date(2024, 1, 1),
        description="BSE equity bhavcopy (UDiFF)",
    ),
    Feed(
        name="fo",
        publishes_local=time(19, 0),
        first_session=date(2024, 1, 2),
        description="NSE F&O bhavcopy",
    ),
    Feed(
        name="indices",
        publishes_local=time(19, 0),
        first_session=date(2019, 1, 1),
        description="NSE index closes",
    ),
)


def feed_named(name: str) -> Feed:
    for feed in FEEDS:
        if feed.name == name:
            return feed
    raise KeyError(f"unknown feed {name!r}; expected {[f.name for f in FEEDS]}")


@dataclass(frozen=True)
class Plan:
    """What to fetch for one feed, or why nothing."""

    feed: Feed
    start: date | None
    end: date | None
    #: Sessions already held. Reported so a run that fetches nothing still says
    #: whether that is because the lake is current or because it is empty.
    held: int
    reason: str
    #: Sessions older than the catch-up cap that this run will not attempt.
    #: Non-zero means a backfill is owed, and the runner says which command.
    deferred: int = 0

    @property
    def due(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def sessions(self) -> int:
        """Calendar days in the range — an upper bound, not a count of files.

        Weekends and holidays are inside it and will be skipped by the fetcher,
        so this over-states the work. Over-stating is the safe direction for a
        line that exists to tell you roughly how long the run will take.
        """
        if self.start is None or self.end is None:
            return 0
        return (self.end - self.start).days + 1


def latest_due_session(
    feed: Feed, now: datetime, calendar: SessionCalendar | None = None
) -> date | None:
    """The newest session whose file should exist by `now`.

    Walks back from today rather than assuming "yesterday": on a Monday
    morning the answer is Friday, and on the morning after a four-day Diwali
    break it is the session before that.

    Returns:
        None when nothing has been published yet — a brand new archive, or a
        clock earlier than the feed's first session.
    """
    market = calendar or nse_equity_calendar()
    local = require_utc(now).astimezone(IST)

    # Up to a fortnight of consecutive non-sessions covers any Indian market
    # holiday cluster; beyond that something is wrong with the calendar, and
    # returning None is better than inventing a session date.
    for back in range(MAX_CATCHUP_DAYS):
        day = local.date() - timedelta(days=back)
        if day < feed.first_session or not market.is_session(day):
            continue
        published = datetime.combine(day, feed.publishes_local, tzinfo=IST)
        if local >= published:
            return day
    return None


def plan_feed(
    feed: Feed,
    held: list[date],
    now: datetime,
    calendar: SessionCalendar | None = None,
    max_catchup_days: int = MAX_CATCHUP_DAYS,
) -> Plan:
    """Decide what one feed should fetch on this run.

    Args:
        feed: The source.
        held: Sessions already in the lake, in any order.
        now: The current instant, timezone-aware.
        calendar: Trading calendar; NSE equity by default.
        max_catchup_days: Cap on how far back one run reaches.

    Returns:
        A `Plan`. `due` is False when there is nothing to fetch, and `reason`
        says which kind of nothing: current, not yet published, or an archive
        that has not begun.
    """
    latest = latest_due_session(feed, now, calendar)
    if latest is None:
        return Plan(feed, None, None, len(held), "nothing published yet")

    if not held:
        # A fresh lake is a backfill, not a daily run. Fetch the recent window
        # so the console has something, and name the rest as owed.
        start = max(feed.first_session, latest - timedelta(days=max_catchup_days - 1))
        deferred = max(0, (start - feed.first_session).days)
        return Plan(feed, start, latest, 0, "empty — fetching the recent window", deferred)

    newest = max(held)
    if newest >= latest:
        return Plan(feed, None, None, len(held), f"current through {newest}")

    start = max(newest + timedelta(days=1), feed.first_session)
    capped = max(start, latest - timedelta(days=max_catchup_days - 1))
    deferred = (capped - start).days
    return Plan(
        feed,
        capped,
        latest,
        len(held),
        f"behind by {(latest - newest).days} day(s)",
        deferred,
    )
