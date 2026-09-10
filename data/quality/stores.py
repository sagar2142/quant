"""Checks for the feeds that are not bars — MASTER_PLAN §M2, §9.

`check_bars` has guarded the price panel since the panel existed. Nothing
guarded anything else, and by now nearly every branch of the research consumes
something else: the risk model reads industry labels, the options screen reads
derivatives, a value factor will read quarterly results, and the alert rules
read the announced calendar. A quality gate that covers one of five inputs
reports on the input least likely to be wrong, because it is the one that has
been checked longest.

**These are store-level checks, not row-level ones.** `check_bars` asks whether
a series is internally sound — ordered, causal, no duplicates. The questions
here are different in kind: is this feed *current*, does it cover the names
anything actually trades, and does what it holds contradict itself. A sector
file can be perfectly well-formed and eighteen months old.

**Staleness is a finding, not an error.** Every one of these feeds is allowed to
be behind — NSE publishes constituents on its own schedule and companies report
when they report. What is not allowed is being behind without anyone knowing,
which is the entire distinction this module encodes.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from core.clock import utc_now
from data.quality.checks import Finding, Severity

__all__ = [
    "CALENDAR_STALE_DAYS",
    "SECTOR_STALE_DAYS",
    "StoreReport",
    "check_events",
    "check_fundamentals",
    "check_sectors",
]

#: Industry labels older than this are carried further than intended.
#:
#: NSE reviews index membership twice a year and an industry label almost never
#: moves, so this is generous — it catches an ingest that stopped rather than a
#: file that is merely not the newest.
SECTOR_STALE_DAYS = 45

#: The announced calendar is served forward only and has no archive, so an old
#: observation is not history, it is a gap nobody can fill. Companies announce
#: meetings days ahead, which is what makes a week the useful boundary.
CALENDAR_STALE_DAYS = 7

#: Below this share of the traded universe, a classification is not describing
#: the book — it is describing whichever names happened to be in one list.
MIN_SECTOR_COVERAGE = 0.60


class StoreReport:
    """Findings for one feed, in the shape the quality CLI already prints."""

    def __init__(self, name: str, held: str = "") -> None:
        self.name = name
        self.held = held
        self.findings: list[Finding] = []

    def add(self, check: str, severity: Severity, count: int, detail: str) -> None:
        self.findings.append(Finding(check, severity, count, detail))

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity is Severity.CRITICAL)

    @property
    def is_clean(self) -> bool:
        return self.critical_count == 0

    def format(self) -> str:
        status = "CLEAN" if self.is_clean else "CRITICAL FINDINGS"
        head = f"{self.name} — {self.held} — {status}" if self.held else f"{self.name} — {status}"
        lines = [head]
        lines.extend(f.format() for f in self.findings)
        if not self.findings:
            lines.append("           no findings")
        return "\n".join(lines)


def _age_days(observed: date, at: date | None = None) -> int:
    return ((at or utc_now().date()) - observed).days


def check_sectors(lake: Path, universe: set[str] | None = None) -> StoreReport:
    """Whether the industry classification is current and covers the book.

    Args:
        lake: Lake root.
        universe: ISINs the classification is expected to cover. Coverage is
            reported against what is actually traded rather than against the
            whole panel, most of which is too illiquid to hold.
    """
    from data.store.sectors import SectorStore  # noqa: PLC0415

    report = StoreReport("sectors")
    store = SectorStore(lake)
    observations = store.observations()
    if not observations:
        report.held = "nothing"
        report.add("absent", Severity.CRITICAL, 0, "no classification; run apps.cli.ingest_sectors")
        return report

    newest = observations[-1]
    # Present-tense by design: the question is whether the feed is current
    # *now*, so the newest observation is the right one. A dated read here
    # would be checking the freshness of a moment nobody asked about.
    view = store.view()  # lint: allow-unbounded-read
    report.held = f"{len(view.industries):,} names, observed {newest}"

    age = _age_days(newest)
    if age >= SECTOR_STALE_DAYS:
        report.add(
            "stale", Severity.WARN, age, f"observed {age} days ago; labels are being carried"
        )

    blank = sum(1 for industry in view.industries.values() if not industry.strip())
    if blank:
        # A blank label is worse than a missing name: it joins successfully and
        # then groups every one of them together as a single sector.
        report.add("blank_industry", Severity.CRITICAL, blank, "names carry an empty industry")

    if universe:
        covered = sum(1 for isin in universe if isin in view.industries)
        share = covered / len(universe)
        if share < MIN_SECTOR_COVERAGE:
            report.add(
                "coverage",
                Severity.WARN,
                len(universe) - covered,
                f"only {share:.0%} of the traded universe is classified",
            )
    return report


def check_fundamentals(lake: Path) -> StoreReport:
    """Whether quarterly results are readable, timestamped and populated."""
    from data.store.fundamentals import FundamentalStore  # noqa: PLC0415

    report = StoreReport("fundamentals")
    store = FundamentalStore(lake)
    # Present-tense, as above: this asks what the store holds, not what was
    # readable at some past decision.
    rows = store.view().rows  # lint: allow-unbounded-read
    if rows.is_empty():
        report.held = "nothing"
        report.add("absent", Severity.WARN, 0, "no filings; run apps.cli.ingest_results")
        return report

    with_numbers = rows.filter(rows["revenue"].is_not_null()).height
    report.held = f"{rows.height:,} filings, {with_numbers:,} with numbers"

    missing_time = int(rows["receive_time"].null_count())
    if missing_time:
        # The only column that may be compared against a decision time. A
        # filing without one is unreadable point-in-time, which makes it worse
        # than absent: it would be visible at every decision or at none.
        report.add(
            "no_receive_time",
            Severity.CRITICAL,
            missing_time,
            "filings cannot be read point-in-time",
        )

    published_early = rows.filter(rows["receive_time"].dt.date() < rows["period_end"]).height
    if published_early:
        # A result disseminated before the quarter it describes has ended is
        # either a parsing error or a date this system should not trust.
        report.add(
            "published_before_period_end",
            Severity.CRITICAL,
            published_early,
            "filings claim publication before the period they report",
        )

    # Multiplied rather than divided: `height // 2` is zero for a single
    # filing, so the integer division silently never fires on a small store —
    # which is exactly the store a fresh install has.
    if with_numbers * 2 < rows.height:
        report.add(
            "numbers_missing",
            Severity.WARN,
            rows.height - with_numbers,
            "filings carry dates but no XBRL figures; a value factor cannot read them",
        )
    return report


def check_events(lake: Path, at: datetime | None = None) -> StoreReport:
    """Whether the announced calendar is fresh enough to be worth reading."""
    from data.store.events import EventStore  # noqa: PLC0415

    report = StoreReport("events")
    store = EventStore(lake)
    observations = store.observations()
    if not observations:
        report.held = "nothing"
        report.add("absent", Severity.WARN, 0, "no calendar; run apps.cli.ingest_events")
        return report

    today = (at or utc_now()).date()
    newest = observations[-1]
    window = store.upcoming(today, within_days=CALENDAR_STALE_DAYS)
    report.held = f"observed {newest}, {window.rows.height} meeting(s) in the next week"

    age = _age_days(newest, today)
    if age > CALENDAR_STALE_DAYS:
        # Critical rather than a warning: this feed has no archive, so a stale
        # calendar reports quiet it cannot vouch for, and an alert rule reading
        # it would give a false all-clear on a name about to report.
        report.add(
            "stale",
            Severity.CRITICAL,
            age,
            f"observed {age} days ago; NSE keeps no archive, so the gap is unrecoverable",
        )

    unresolved = int(window.rows["instrument_id"].null_count()) if not window.rows.is_empty() else 0
    if unresolved:
        report.add(
            "unresolved",
            Severity.WARN,
            unresolved,
            "events could not be tied to an instrument and cannot reach a position",
        )
    return report
