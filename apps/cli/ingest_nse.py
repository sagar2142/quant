"""NSE bhavcopy ingest — MASTER_PLAN §M2.

    # from a directory of already-downloaded files
    python -m apps.cli.ingest_nse --from-dir .\\downloads

    # fetch a date range directly (see access note below)
    python -m apps.cli.ingest_nse --start 2024-01-01 --end 2024-03-31

**Access note.** NSE serves bhavcopy files for personal use, rate-limits
aggressively, and requires browser-like headers plus a session cookie. Review
NSE's terms before automating bulk downloads (§32). `--from-dir` exists so the
ingest path works regardless of how the files were obtained, and it is the
recommended route for a large historical backfill.

Filenames under `--from-dir` must contain a parseable date: either `YYYYMMDD`
or the legacy `DDMONYYYY` form.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

from core.clock import utc_now
from core.config import settings
from data.feeds.bse import BSE_SERIES, bse_udiff_url
from data.feeds.nse import (
    DEFAULT_SERIES,
    BhavcopyFormatError,
    legacy_url,
    parse_bhavcopy,
    udiff_url,
)
from data.store.panel import PanelStore

SATURDAY = 5

# Without a browser-like User-Agent, NSE returns 403.
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

_ISO = re.compile(r"(20\d{2})(\d{2})(\d{2})")
_LEGACY = re.compile(r"(\d{2})([A-Z]{3})(20\d{2})", re.IGNORECASE)

#: UDiFF replaced the legacy layout in mid-2024.
UDIFF_FROM = date(2024, 7, 8)


def date_from_filename(name: str) -> date | None:
    if (m := _ISO.search(name)) is not None:
        return date(int(m[1]), int(m[2]), int(m[3]))
    if (m := _LEGACY.search(name)) is not None:
        month = _MONTHS.index(m[2].upper()) + 1
        return date(int(m[3]), month, int(m[1]))
    return None


#: BSE serves the file only to requests that look like they came from its own
#: site; a bare fetch gets an HTML error page with a 200 status, which would
#: otherwise parse as an empty session rather than fail.
BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (neutron)",
    "Referer": "https://www.bseindia.com/",
}


def series_for(venue: str) -> tuple[str, ...]:
    """Which security groups count as equity on this exchange."""
    return BSE_SERIES if venue == "BSE" else DEFAULT_SERIES


def source_url(session_date: date, venue: str) -> str:
    """Where that exchange publishes the session's bhavcopy.

    NSE changed layout in July 2024 and keeps both archives, so a backfill
    crosses the boundary; BSE serves the current layout only, which is why a
    BSE backfill cannot reach as far back.
    """
    if venue == "BSE":
        return bse_udiff_url(session_date)
    return udiff_url(session_date) if session_date >= UDIFF_FROM else legacy_url(session_date)


def ingest_payload(
    panel: PanelStore,
    payload: bytes,
    session: date,
    venue: str = "NSE",
    series: tuple[str, ...] = DEFAULT_SERIES,
) -> int:
    """Parse one bhavcopy and write it as a panel session.

    Args:
        venue: Prefixes the instrument id. A name listed on both exchanges
            shares an ISIN and trades at two different prices, so the venue is
            part of identity rather than a label — `BSE:INE002A01018` and
            `NSE:INE002A01018` are different things to hold.
        series: Which groups count as equity. NSE says `EQ`; BSE groups by
            settlement history instead (A, B, T and the SME platforms), so the
            NSE filter matches nothing in a BSE file.
    """
    day = parse_bhavcopy(payload, session, series=series)
    frame = day.bars.with_columns(
        [
            (
                f"{venue}:"
                + day.bars["isin"].zip_with(
                    day.bars["isin"].str.len_chars() > 0, day.bars["symbol"]
                )
            ).alias("instrument_id")
        ]
    ).select(
        "event_time",
        "receive_time",
        "instrument_id",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
    )
    return panel.write_session(session, frame)


def ingest_directory(panel: PanelStore, directory: Path) -> tuple[int, int]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".csv", ".zip"})
    if not files:
        print(f"no .csv or .zip files under {directory}")
        return 0, 0

    ok = failed = 0
    for path in files:
        session = date_from_filename(path.name)
        if session is None:
            print(f"{path.name:<44} SKIP  no date in filename")
            failed += 1
            continue
        try:
            count = ingest_payload(panel, path.read_bytes(), session)
        except BhavcopyFormatError as exc:
            print(f"{path.name:<44} FAIL  {exc}")
            failed += 1
            continue
        print(f"{path.name:<44} {session}  {count:>5} rows")
        ok += 1
    return ok, failed


def fetch_range(  # noqa: PLR0913, PLR0917 - a date range, a venue, and how to pace it
    panel: PanelStore,
    start: date,
    end: date,
    pause: float,
    refetch: bool = False,
    venue: str = "NSE",
) -> tuple[int, int]:
    """Fetch and ingest every session in [start, end].

    Sessions already in the panel are skipped unless `refetch`. A multi-year
    backfill is thousands of rate-limited requests and *will* be interrupted;
    without this the only way to resume is to start over, which is how a
    backfill never finishes.
    """
    have = set() if refetch else set(panel.sessions())
    ok = failed = skipped = 0
    headers = BSE_HEADERS if venue == "BSE" else NSE_HEADERS
    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        # Prime the session cookie; NSE rejects bare archive requests.
        try:
            client.get("https://www.nseindia.com/")
        except httpx.HTTPError as exc:
            print(f"warning: could not prime NSE session ({exc})")

        day = start
        while day <= end:
            if day.weekday() >= SATURDAY:  # weekends are never sessions
                day += timedelta(days=1)
                continue
            if day in have:
                skipped += 1
                day += timedelta(days=1)
                continue  # no sleep: nothing was requested
            url = source_url(day, venue)
            try:
                response = client.get(url)
                if response.status_code != httpx.codes.OK:
                    # Holidays legitimately 404 — not an error, just no session.
                    print(f"{day}  HTTP {response.status_code} (holiday or unavailable)")
                    failed += 1
                else:
                    count = ingest_payload(panel, response.content, day, venue, series_for(venue))
                    print(f"{day}  {count:>5} rows")
                    ok += 1
            except (httpx.HTTPError, BhavcopyFormatError) as exc:
                print(f"{day}  FAIL  {exc}")
                failed += 1
            time.sleep(pause)
            day += timedelta(days=1)
    if skipped:
        print(f"({skipped} session(s) already in the panel, not refetched)")
    return ok, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NSE bhavcopy into the panel")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-dir", type=Path, help="Directory of downloaded files")
    source.add_argument("--start", type=date.fromisoformat, help="Fetch from this date")
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--venue",
        choices=["NSE", "BSE"],
        default="NSE",
        help=(
            "Which exchange to ingest. Written to a separate panel: the two "
            "are different venues with different prices, not one merged tape."
        ),
    )
    parser.add_argument("--lake", default=None)
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds between requests")
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Re-download sessions already in the panel (default: skip, so a run resumes)",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    panel = PanelStore(args.lake if args.lake is not None else settings.lake, venue=args.venue)

    if args.from_dir is not None:
        if not args.from_dir.is_dir():
            print(f"not a directory: {args.from_dir}")
            return 1
        ok, failed = ingest_directory(panel, args.from_dir)
        # A directory that produced nothing is a mistake worth an exit code.
        status = 0 if ok else 1
    else:
        end = args.end or utc_now().date()
        ok, failed = fetch_range(
            panel, args.start, end, args.pause, refetch=args.refetch, venue=args.venue
        )
        # ok == 0 with no failures means the range was already covered — a
        # resumed run that finds nothing left to do has succeeded, not failed.
        status = 1 if failed and not ok else 0

    sessions = panel.sessions()
    print(f"\n{ok} session(s) ingested, {failed} skipped")
    if sessions:
        print(f"panel now covers {sessions[0]} → {sessions[-1]} ({len(sessions)} sessions)")
    return status


if __name__ == "__main__":
    sys.exit(run())
