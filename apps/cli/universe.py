"""Point-in-time universe inspection — MASTER_PLAN §M2 gate.

    python -m apps.cli.universe --dates 2019-04-15 2024-03-15 --top 50

Prints membership at each date and, when two or more dates are given, the
survivorship report: which names were in the earlier universe and had stopped
trading by the later one.

That report *is* the M2 gate. If the "dropped out" list is empty across a
multi-year gap, the universe is being built from today's survivors and every
backtest downstream is fiction.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from itertools import pairwise

from core.clock import UTC, as_decision_time
from core.config import settings
from data.store.panel import PanelStore
from data.universe.pit import Universe, UniverseBuilder, UniverseSpec

#: 13:00 UTC = 18:30 IST, after the ~18:00 bhavcopy publication.
DECISION_HOUR_UTC = 13

#: Cap on how many dropped/added names to enumerate before summarising.
MAX_LISTED = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect point-in-time universes")
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--min-sessions", type=int, default=20)
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--min-value", type=float, default=1_000_000.0)
    parser.add_argument("--show", type=int, default=10, help="Members to list")
    parser.add_argument("--lake", default=None)
    return parser.parse_args(argv)


def describe(universe: Universe, show: int) -> None:
    print(f"\n{universe.as_of.date()} — {len(universe)} members")
    if not universe.members:
        print("  (empty: no panel data received by this date)")
        return
    for rank, member in enumerate(universe.members[:show], start=1):
        value_cr = universe.liquidity[member] / 1e7  # ₹ crore
        print(f"  {rank:>3}. {member:<28} median traded value ₹{value_cr:>8.2f} cr")
    if len(universe) > show:
        print(f"  ... {len(universe) - show} more")


def survivorship_report(early: Universe, late: Universe) -> None:
    dropped = sorted(set(early.members) - set(late.members))
    added = sorted(set(late.members) - set(early.members))

    print(f"\n── survivorship: {early.as_of.date()} → {late.as_of.date()}")
    print(f"   membership turnover : {early.turnover_against(late):.1%}")
    print(f"   dropped out         : {len(dropped)}")
    print(f"   newly qualified     : {len(added)}")

    for member in dropped[:MAX_LISTED]:
        print(f"     - {member}")
    if len(dropped) > MAX_LISTED:
        print(f"     ... {len(dropped) - MAX_LISTED} more")

    if dropped:
        print(
            "\n   These were tradable then and are not now. A survivorship-biased\n"
            "   universe would have silently omitted every one of them."
        )
    else:
        print(
            "\n   WARNING: nothing dropped out across this gap. Either the window is\n"
            "   too short, or the universe is being built from today's survivors."
        )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    panel = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")

    sessions = panel.sessions()
    if not sessions:
        print("no NSE panel data. Run: python -m apps.cli.ingest_nse --from-dir <dir>")
        return 1
    print(f"panel: {sessions[0]} → {sessions[-1]} ({len(sessions)} sessions)")

    spec = UniverseSpec(
        top_n=args.top,
        lookback_days=args.lookback,
        min_sessions=args.min_sessions,
        min_price=args.min_price,
        min_median_value=args.min_value,
    )
    builder = UniverseBuilder(panel)

    universes: list[Universe] = []
    for day in sorted(args.dates):
        stamp = datetime(day.year, day.month, day.day, DECISION_HOUR_UTC, tzinfo=UTC)
        universe = builder.build(as_decision_time(stamp), spec)
        universes.append(universe)
        describe(universe, args.show)

    for earlier, later in pairwise(universes):
        survivorship_report(earlier, later)

    return 0


if __name__ == "__main__":
    sys.exit(run())
