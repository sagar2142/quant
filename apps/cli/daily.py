"""The daily ingest — MASTER_PLAN §13.4, §M2.

    python -m apps.cli.daily              # fetch what is missing
    python -m apps.cli.daily --dry-run    # say what it would fetch
    python -m apps.cli.daily --only nse indices

**The console was only correct when someone remembered to feed it.** Four
separate commands, each with its own date arguments, run by hand after the
close. Miss an evening and the staleness light goes amber; miss a fortnight and
the risk model is estimating a covariance from a market that stopped moving.

This runs all four, works out what each one is missing, and refuses to ask for
files that are not published yet. It is safe to run repeatedly and safe to run
at the wrong time of day — the planner's whole job is to make "too early" a
no-op instead of a day recorded as unavailable.

**One feed's failure does not stop the others.** BSE goes down more often than
NSE does, and an evening where BSE fails but the equity panel, F&O and the
index all land is a good evening, not a failed run. The exit code reflects
whether anything that was *due* went unfetched, and the summary says which.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from apps.cli import ingest_fo, ingest_indices, ingest_nse
from core.clock import utc_now
from core.config import settings
from data.store.derivatives import DerivativesStore
from data.store.indices import IndexStore
from data.store.panel import PanelStore
from ops.schedule import FEEDS, MAX_CATCHUP_DAYS, Feed, Plan, plan_feed

__all__ = ["Outcome", "held_sessions", "run", "run_plan"]

#: Slower than a backfill's pacing. A daily run fetches a handful of sessions
#: and has all evening to do it; there is nothing to gain by hurrying an
#: exchange that rate-limits.
DEFAULT_PAUSE = 1.0

BACKFILL_COMMAND = {
    "nse": "python -m apps.cli.ingest_nse --start {start}",
    "bse": "python -m apps.cli.ingest_nse --venue BSE --start {start}",
    "fo": "python -m apps.cli.ingest_fo --start {start}",
    "indices": "python -m apps.cli.ingest_indices --start {start}",
}


@dataclass(frozen=True)
class Outcome:
    """What one feed's run actually did."""

    plan: Plan
    fetched: int = 0
    unavailable: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the run did what it set out to.

        A plan that was not due is fine. A plan that was due and fetched
        nothing is not — that is the case worth an exit code, because it is
        the one that leaves the console quietly stale.
        """
        if self.error:
            return False
        if not self.plan.due:
            return True
        return self.fetched > 0


def held_sessions(feed: Feed, lake: Path) -> list[date]:
    """Sessions this feed already has in the lake.

    A directory glob, not a Parquet read: cheap enough to do for every feed on
    every run, which is what lets the planner be re-derived from the lake
    rather than from a state file that can drift out of step with it.
    """
    try:
        if feed.name == "nse":
            return PanelStore(lake, venue="NSE").sessions()
        if feed.name == "bse":
            return PanelStore(lake, venue="BSE").sessions()
        if feed.name == "fo":
            return DerivativesStore(lake).sessions()
        if feed.name == "indices":
            return IndexStore(lake).sessions()
    except OSError:
        return []
    raise KeyError(f"no store for feed {feed.name!r}")


def run_plan(plan: Plan, lake: Path, pause: float) -> Outcome:
    """Execute one plan by calling the feed's own fetcher.

    The fetchers are reused rather than reimplemented: a second copy of "walk
    these dates and skip what is held" would eventually disagree with the
    first, and the disagreement would be a silently missing session.
    """
    if not plan.due or plan.start is None or plan.end is None:
        return Outcome(plan)

    try:
        if plan.feed.name in ("nse", "bse"):
            venue = plan.feed.name.upper()
            ok, failed = ingest_nse.fetch_range(
                PanelStore(lake, venue=venue), plan.start, plan.end, pause, False, venue
            )
        elif plan.feed.name == "fo":
            ok, failed = ingest_fo.fetch_range(DerivativesStore(lake), plan.start, plan.end, pause)
        elif plan.feed.name == "indices":
            ok, failed = ingest_indices.fetch_range(IndexStore(lake), plan.start, plan.end, pause)
        else:  # pragma: no cover - FEEDS and this function are edited together
            return Outcome(plan, error=f"no fetcher for feed {plan.feed.name!r}")
    except Exception as exc:  # noqa: BLE001 - one feed must not take down the run
        return Outcome(plan, error=f"{type(exc).__name__}: {exc}")

    return Outcome(plan, fetched=ok, unavailable=failed)


def _report(outcomes: list[Outcome]) -> None:
    print("\n" + "=" * 68)
    print("  DAILY INGEST")
    print("=" * 68)
    for outcome in outcomes:
        plan = outcome.plan
        if outcome.error:
            status = f"FAILED  {outcome.error}"
        elif not plan.due:
            status = plan.reason
        elif outcome.fetched:
            status = f"{outcome.fetched} session(s) ingested"
            if outcome.unavailable:
                # Holidays live here too, so this is not an error on its own.
                status += f", {outcome.unavailable} not served"
        else:
            status = f"nothing fetched ({outcome.unavailable} not served)"
        print(f"  {plan.feed.name:<9}{plan.held:>6} held   {status}")

    owed = [o.plan for o in outcomes if o.plan.deferred > 0]
    if owed:
        print("\n  Older sessions this run did not reach back for:")
        for plan in owed:
            command = BACKFILL_COMMAND[plan.feed.name].format(start=plan.feed.first_session)
            print(f"    {plan.feed.name:<9}{plan.deferred:>6} day(s)   {command}")


def _classification_is_stale(lake: Path, max_age_days: int) -> bool:
    """Whether the industry classification is old enough to refetch.

    Stale rather than missing, because an absent classification is also stale —
    both mean the next read would be reaching further back than intended.
    """
    if max_age_days <= 0:
        return False
    from data.store.sectors import SectorStore  # noqa: PLC0415

    held = SectorStore(lake).observations()
    if not held:
        return True
    return (utc_now().date() - held[-1]).days >= max_age_days


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch every feed's missing sessions")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[f.name for f in FEEDS],
        default=None,
        help="Limit to these feeds. Default: all of them.",
    )
    parser.add_argument("--lake", default=None)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument(
        "--max-catchup-days",
        type=int,
        default=MAX_CATCHUP_DAYS,
        help="How far back one run reaches before calling it a backfill",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without fetching anything",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the data-quality suite afterwards",
    )
    parser.add_argument(
        "--alerts",
        action="store_true",
        help="Evaluate the alert rules against the lake this run produced",
    )
    parser.add_argument(
        "--classify-after-days",
        type=int,
        default=7,
        help=(
            "Refetch the industry classification when it is this old. 0 never "
            "fetches it. NSE changes index membership at reviews, so a daily "
            "fetch would learn nothing."
        ),
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lake = Path(args.lake) if args.lake else settings.lake
    now = utc_now()

    feeds = [f for f in FEEDS if args.only is None or f.name in args.only]
    plans = [
        plan_feed(feed, held_sessions(feed, lake), now, max_catchup_days=args.max_catchup_days)
        for feed in feeds
    ]

    if args.dry_run:
        print(f"  as of {now:%Y-%m-%d %H:%M} UTC\n")
        for plan in plans:
            window = f"{plan.start} -> {plan.end}" if plan.due else "—"
            print(f"  {plan.feed.name:<9}{plan.held:>6} held   {window:<26}{plan.reason}")
        return 0

    outcomes: list[Outcome] = []
    for plan in plans:
        if plan.due:
            print(f"\n[{plan.feed.name}] {plan.start} -> {plan.end}  ({plan.reason})")
        outcomes.append(run_plan(plan, lake, args.pause))

    _report(outcomes)

    # Industry classification, on a cadence rather than every run: NSE changes
    # index membership at reviews and an industry label almost never moves, so
    # a daily fetch would be a request a day to learn nothing.
    if _classification_is_stale(lake, args.classify_after_days):
        print(f"\n[sectors] classification older than {args.classify_after_days} days")
        from apps.cli import ingest_sectors  # noqa: PLC0415 - only on the cadence

        ingest_sectors.run(["--lake", str(lake)] if args.lake else [])

    # Alerts last, on the lake this run produced. Evaluating before the ingest
    # would test yesterday's data and report it as today's.
    if args.alerts:
        from apps.cli import watch  # noqa: PLC0415 - only when asked

        print("\n" + "=" * 68)
        print("  ALERTS")
        print("=" * 68)
        # Its exit code is not the ingest's: a rule that could not be evaluated
        # is worth saying and is not a failed download.
        watch.run(["--lake", str(lake)] if args.lake else [])

    if args.check:
        # After, not before: the check should judge the lake this run produced.
        # Its exit code is reported but does not fail the ingest — an ingest
        # that worked and a panel with a suspect bar are different problems,
        # and conflating them makes the daily job cry wolf.
        from apps.cli import quality  # noqa: PLC0415 - only needed with --check

        print("\n" + "=" * 68)
        print("  DATA QUALITY")
        print("=" * 68)
        if quality.run(["--lake", str(lake)] if args.lake else []):
            print("\n  Quality check reported problems (the ingest itself was fine).")

    stale = [o for o in outcomes if not o.ok]
    if stale:
        print(f"\n  {len(stale)} feed(s) due but not ingested: {[o.plan.feed.name for o in stale]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
