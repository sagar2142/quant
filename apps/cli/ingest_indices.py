"""Ingest NSE index closes — MASTER_PLAN §6, §13.4.

    python -m apps.cli.ingest_indices --start 2019-01-01

Gives the risk model a real market to regress against. Until this runs, the
market factor is a column of ones, and every beta measured against it is an
approximation nothing on screen admits to.
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
from data.feeds.nse_indices import IndexFormatError, nse_index_url, parse_index_close
from data.store.indices import IndexStore

__all__ = ["fetch_range", "run"]

SATURDAY = 5
DEFAULT_PAUSE = 0.4

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_range(
    store: IndexStore,
    start: date,
    end: date,
    pause: float = DEFAULT_PAUSE,
    refetch: bool = False,
) -> tuple[int, int]:
    """Fetch and ingest every session in [start, end].

    Sessions already held are skipped unless `refetch`, so an interrupted
    backfill resumes rather than starting over.
    """
    have = set() if refetch else set(store.sessions())
    ok = failed = skipped = 0

    with httpx.Client(headers=NSE_HEADERS, timeout=45.0, follow_redirects=True) as client:
        try:
            client.get("https://www.nseindia.com/")
        except httpx.HTTPError as exc:
            print(f"warning: could not prime the NSE session ({exc})")

        day = start
        while day <= end:
            if day.weekday() >= SATURDAY or day in have:
                skipped += day in have
                day += timedelta(days=1)
                continue

            try:
                response = client.get(nse_index_url(day))
                if response.status_code != httpx.codes.OK:
                    # A holiday simply has no file. Reported, not raised: a
                    # backfill that stops at every Diwali never finishes.
                    failed += 1
                else:
                    session = parse_index_close(response.content, day)
                    count = store.write_session(day, session.rows)
                    print(f"{day}  {count:>4} indices")
                    ok += 1
            except (httpx.HTTPError, IndexFormatError, ValueError) as exc:
                print(f"{day}  FAIL  {exc}")
                failed += 1
            time.sleep(pause)
            day += timedelta(days=1)

    if skipped:
        print(f"({skipped} session(s) already held)")
    return ok, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NSE index closes")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--lake", default=None)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--refetch", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = IndexStore(Path(args.lake) if args.lake else settings.lake)

    end = args.end or utc_now().date()
    ok, failed = fetch_range(store, args.start, end, args.pause, refetch=args.refetch)

    print(f"\n{ok} session(s) ingested, {failed} unavailable")
    held = store.sessions()
    if held:
        print(f"indices now cover {held[0]} -> {held[-1]} ({len(held)} sessions)")
    return 0 if ok or not failed else 1


if __name__ == "__main__":
    sys.exit(run())
