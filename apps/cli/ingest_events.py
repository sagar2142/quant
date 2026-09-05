"""Ingest the announced board-meeting calendar — MASTER_PLAN §3.3.

    python -m apps.cli.ingest_events
    python -m apps.cli.ingest_events --show

**Daily, and only forward.** NSE serves the meetings that have been announced
and keeps no archive, so this is the sector classification's shape rather than
a bhavcopy's: today's fetch is today's observation, and a run missed is a day
of the calendar nobody can recover.

Symbols are resolved to instruments through the recent panel, because the
endpoint carries no ISIN. Whatever cannot be resolved is printed. A calendar
that quietly covers four fifths of what it claims is worse than one that says
which fifth is missing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from core.clock import as_decision_time, utc_now
from core.config import settings
from data.feeds.nse_events import EventFormatError, event_calendar_url, parse_events, resolve
from data.store.events import EventStore
from data.store.panel import PanelStore

__all__ = ["fetch_events", "panel_symbols", "run"]

#: Sessions of panel read to build the symbol map. A name that has not traded
#: in a week cannot be resolved, which is the right answer — an event on a
#: suspended listing is not something this book can act on.
RESOLUTION_SESSIONS = 5

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar",
}


def panel_symbols(lake: Path, venue: str = "NSE") -> dict[str, str]:
    """Trading symbol -> instrument id, from the recent panel."""
    store = PanelStore(lake, venue=venue)
    sessions = store.sessions()
    if not sessions:
        return {}
    recent = store.view(
        as_of=as_decision_time(utc_now()),
        start=sessions[max(0, len(sessions) - RESOLUTION_SESSIONS)],
    )
    if recent.is_empty():
        return {}
    return {
        str(symbol).upper(): str(instrument_id)
        for symbol, instrument_id in zip(recent["symbol"], recent["instrument_id"], strict=True)
    }


def fetch_events(client: httpx.Client, observed: date) -> pl.DataFrame:
    """The announced calendar, parsed."""
    response = client.get(event_calendar_url())
    if response.status_code != httpx.codes.OK:
        raise EventFormatError(f"HTTP {response.status_code}")
    return parse_events(response.content, observed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NSE's announced event calendar")
    parser.add_argument("--lake", default=None)
    parser.add_argument("--venue", choices=["NSE", "BSE"], default="NSE")
    parser.add_argument("--within", type=int, default=14, help="Days ahead to print")
    parser.add_argument(
        "--show", action="store_true", help="Print the stored calendar without fetching"
    )
    return parser.parse_args(argv)


def _print_window(store: EventStore, today: date, within: int) -> None:
    window = store.upcoming(today, within_days=within)
    if window.observed_at is None:
        print("no calendar held")
        return
    stale = f"  ({window.age_days}d old)" if window.age_days else ""
    print(f"\nannounced for the next {within} days, observed {window.observed_at}{stale}")
    results = window.rows.filter(window.rows["is_results"])
    print(f"  {window.rows.height} meeting(s), {results.height} to consider results\n")
    for row in window.rows.head(30).iter_rows(named=True):
        mark = "RESULTS" if row["is_results"] else "       "
        print(f"  {row['event_date']}  {mark}  {row['symbol']:<14}{row['purpose'][:48]}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lake = Path(args.lake) if args.lake else settings.lake
    store = EventStore(lake)
    today = utc_now().date()

    if args.show:
        _print_window(store, today, args.within)
        return 0

    try:
        with httpx.Client(headers=NSE_HEADERS, timeout=60.0, follow_redirects=True) as client:
            events = fetch_events(client, today)
    except (httpx.HTTPError, EventFormatError) as exc:
        print(f"calendar unavailable: {exc}")
        return 1

    if events.is_empty():
        print("no events announced")
        return 0

    resolution = resolve(events, panel_symbols(lake, args.venue))
    written = store.write(today, resolution.rows)
    print(f"{written} event(s) stored, observed {today}")
    print(f"resolved to instruments: {resolution.coverage:.1%}")

    if resolution.unresolved:
        # Never silent: an event that cannot be joined to a position is still
        # an event, and a calendar that hides its gaps stops being trusted.
        print(f"\n{len(resolution.unresolved)} symbol(s) NOT in the recent panel:")
        print(f"  {', '.join(resolution.unresolved)}")
        print("  Usually SME-board or suspended listings the EQ bhavcopy omits.")

    _print_window(store, today, args.within)
    return 0


if __name__ == "__main__":
    sys.exit(run())
