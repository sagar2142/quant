"""Ingest NSE derivatives — MASTER_PLAN §13.4.

    python -m apps.cli.ingest_fo --start 2026-08-01
    python -m apps.cli.ingest_fo --start 2026-08-01 --end 2026-08-28

Deliberately a separate command from `ingest_nse`. The two write different
stores with different identities, and a single `--segment` flag on one command
would invite the assumption that a derivatives session and an equity session
are the same kind of thing arriving through the same door.

**Holidays 404 and are not errors.** A weekday with no session simply has no
file; the loader reports it and continues, because a backfill that stops at
every Diwali never finishes.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

from core.clock import utc_now
from core.config import settings
from data.feeds.nse_fo import DerivativesFormatError, nse_fo_url, parse_fo_bhavcopy
from data.store.derivatives import DerivativesStore

__all__ = ["fetch_range", "run"]

SATURDAY = 5
DEFAULT_PAUSE = 0.6

#: NSE rejects archive requests that do not look like a browser session.
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_range(
    store: DerivativesStore,
    start: date,
    end: date,
    pause: float = DEFAULT_PAUSE,
    refetch: bool = False,
) -> tuple[int, int]:
    """Fetch and ingest every session in [start, end].

    Sessions already held are skipped unless `refetch`, so an interrupted
    backfill resumes instead of starting over.
    """
    have = set() if refetch else set(store.sessions())
    ok = failed = skipped = 0

    with httpx.Client(headers=NSE_HEADERS, timeout=60.0, follow_redirects=True) as client:
        try:
            client.get("https://www.nseindia.com/")
        except httpx.HTTPError as exc:
            print(f"warning: could not prime the NSE session ({exc})")

        day = start
        while day <= end:
            if day.weekday() >= SATURDAY:
                day += timedelta(days=1)
                continue
            if day in have:
                skipped += 1
                day += timedelta(days=1)
                continue

            try:
                response = client.get(nse_fo_url(day))
                if response.status_code != httpx.codes.OK:
                    print(f"{day}  HTTP {response.status_code} (holiday or unavailable)")
                    failed += 1
                else:
                    session = parse_fo_bhavcopy(response.content, day)
                    count = store.write_session(day, session.contracts)
                    print(f"{day}  {count:>7,} contracts")
                    ok += 1
            except (httpx.HTTPError, DerivativesFormatError, ValueError) as exc:
                print(f"{day}  FAIL  {exc}")
                failed += 1
            time.sleep(pause)
            day += timedelta(days=1)

    if skipped:
        print(f"({skipped} session(s) already held, not refetched)")
    return ok, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NSE F&O bhavcopy")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--lake", default=None)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--refetch", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.lake) if args.lake else settings.lake
    store = DerivativesStore(root)

    # `utc_now` rather than `date.today()`: every clock read in this system
    # goes through one place, so a test can move time and a local timezone
    # cannot quietly shift which session is 'today'.
    end = args.end or utc_now().date()
    ok, failed = fetch_range(store, args.start, end, args.pause, refetch=args.refetch)

    print(f"\n{ok} session(s) ingested, {failed} unavailable")
    held = store.sessions()
    if held:
        print(f"derivatives now cover {held[0]} -> {held[-1]} ({len(held)} sessions)")
    return 0 if ok or not failed else 1


if __name__ == "__main__":
    sys.exit(run())
