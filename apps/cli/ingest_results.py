"""Ingest quarterly results — MASTER_PLAN §3.3, §9.

    python -m apps.cli.ingest_results                    # the current window
    python -m apps.cli.ingest_results --start 2019 --end 2026
    python -m apps.cli.ingest_results --no-xbrl          # filing dates only

**Two requests deep.** The filings endpoint gives metadata and a link; the
numbers live in a separate XBRL document per filing. Fetching every document
for a year is thousands of requests, so the default fetches the filing list —
which alone is a usable earnings calendar and carries the dissemination times —
and pulls XBRL for as many as `--xbrl-limit` allows, newest first.

**Newest first is deliberate.** An interrupted run should leave the most
recently published results in the store, because those are what a present-tense
screen reads. A run that filled 2019 before 2025 would be least useful exactly
where it stopped.

Windows overlap by design: the store merges on the filing key and keeps the
latest `receive_time`, so re-running a window costs requests and changes
nothing. A company that refiles a correction is picked up the same way.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import polars as pl

from core.clock import utc_now
from core.config import settings
from data.feeds.nse_results import (
    FUNDAMENTAL_FACTS,
    ResultsFormatError,
    XbrlFacts,
    filings_url,
    parse_filings,
    parse_xbrl,
    xbrl_document_expr,
    xbrl_is_plausible,
)
from data.store.fundamentals import FundamentalStore

__all__ = ["fetch_filings", "fetch_xbrl", "run"]

DEFAULT_PAUSE = 0.4

#: How many XBRL documents one run fetches by default.
#:
#: A full archive is tens of thousands. This is a polite ceiling that makes the
#: default run finish in minutes; `--xbrl-limit 0` lifts it for a deliberate
#: backfill.
DEFAULT_XBRL_LIMIT = 400

#: The archive reaches this far back — measured, and it happens to line up with
#: the first NSE panel session, so a factor built here spans the same history.
FIRST_YEAR = 2019

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
}


def fetch_filings(client: httpx.Client, year: int) -> pl.DataFrame:
    """Every quarterly filing whose window falls in one calendar year.

    Returns:
        The filings, or an empty frame. A year that fails is reported by the
        caller rather than raised — losing one year should cost coverage, not
        the whole ingest.
    """
    url = filings_url(f"01-01-{year}", f"31-12-{year}")
    response = client.get(url)
    if response.status_code != httpx.codes.OK:
        raise ResultsFormatError(f"HTTP {response.status_code}")
    return parse_filings(response.content)


def _document(response: httpx.Response) -> XbrlFacts:
    """The facts in one XBRL response, or the reason there are none."""
    if response.status_code != httpx.codes.OK:
        raise ResultsFormatError(f"HTTP {response.status_code}")
    return parse_xbrl(response.content)


def fetch_xbrl(
    client: httpx.Client, filings: pl.DataFrame, limit: int, pause: float
) -> tuple[pl.DataFrame, int, int]:
    """Attach the numbers to as many filings as the limit allows.

    Returns:
        The frame with fact columns added, how many documents were read, and
        how many were refused. A refused document leaves its row with null
        facts rather than removing it: the filing genuinely happened, and the
        earnings calendar still wants its date.
    """
    columns = list(FUNDAMENTAL_FACTS.values())
    if filings.is_empty():
        return filings, 0, 0

    # Not a scheme check: NSE writes the filename as a dash when nothing was
    # filed, so `.../xbrl/-` is a well-formed URL on the right host that 404s.
    with_url = filings.filter(xbrl_document_expr())
    ordered = with_url.sort("receive_time", descending=True)
    if limit > 0:
        ordered = ordered.head(limit)

    facts: dict[str, list[float | None]] = {c: [] for c in columns}
    urls: list[str] = []
    read = refused = 0

    for url in ordered["xbrl_url"]:
        urls.append(str(url))
        values: dict[str, float | None] = dict.fromkeys(columns)
        try:
            response = client.get(str(url))
            parsed = _document(response)
            problem = xbrl_is_plausible(parsed)
            if problem:
                # Loud, and the row keeps null facts. A units error looks like
                # spectacular growth to any ranking factor downstream.
                print(f"  refused {url.rsplit('/', 1)[-1]}: {problem}")
                refused += 1
            else:
                values = {c: float(v) for c, v in parsed.values.items()}
                read += 1
        except (httpx.HTTPError, ResultsFormatError) as exc:
            print(f"  no numbers for {url.rsplit('/', 1)[-1]}: {exc}")
            refused += 1
        for column in columns:
            facts[column].append(values.get(column))
        time.sleep(pause)

    attached = pl.DataFrame({"xbrl_url": urls, **facts}) if urls else None
    if attached is None:
        return filings, 0, 0
    return filings.join(attached, on="xbrl_url", how="left"), read, refused


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NSE quarterly results")
    parser.add_argument("--lake", default=None)
    parser.add_argument(
        "--start", type=int, default=None, help=f"First year. Archive reaches {FIRST_YEAR}."
    )
    parser.add_argument("--end", type=int, default=None, help="Last year. Defaults to this one.")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument(
        "--xbrl-limit",
        type=int,
        default=DEFAULT_XBRL_LIMIT,
        help="XBRL documents per year. 0 fetches every one — slow, for a backfill.",
    )
    parser.add_argument(
        "--no-xbrl",
        action="store_true",
        help="Filing dates only. Enough for an earnings calendar, no numbers.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = FundamentalStore(Path(args.lake) if args.lake else settings.lake)

    this_year = utc_now().date().year
    end = args.end or this_year
    start = args.start or end
    if start < FIRST_YEAR:
        print(f"note: the archive begins {FIRST_YEAR}; earlier years return nothing")

    total_filings = total_read = total_refused = 0
    failed: list[int] = []

    # Newest first: an interrupted backfill should leave the most recent
    # results in the store, since that is what a present-tense screen reads.
    with httpx.Client(headers=NSE_HEADERS, timeout=90.0, follow_redirects=True) as client:
        for year in range(end, start - 1, -1):
            try:
                filings = fetch_filings(client, year)
            except (httpx.HTTPError, ResultsFormatError) as exc:
                print(f"{year}  FAILED  {exc}")
                failed.append(year)
                continue

            if filings.is_empty():
                print(f"{year}  no filings")
                continue

            names = filings["isin"].n_unique()
            print(f"{year}  {filings.height:>6} filings, {names:>5} names")

            read = refused = 0
            if not args.no_xbrl:
                filings, read, refused = fetch_xbrl(client, filings, args.xbrl_limit, args.pause)
                print(f"       {read:>6} with numbers, {refused} refused")

            written = store.write(filings)
            total_filings += filings.height
            total_read += read
            total_refused += refused
            print(f"       {written:>6} rows in {year}.parquet after merge")
            time.sleep(args.pause)

    if total_filings == 0:
        print("\nnothing ingested")
        return 1

    print(f"\n{total_filings:,} filings, {total_read:,} with numbers, {total_refused} refused")
    print("Read them point-in-time: FundamentalStore(lake).view(as_of)")
    if args.no_xbrl:
        print("No numbers fetched (--no-xbrl). The filing dates are a calendar, not a factor.")
    if failed:
        print(f"{len(failed)} year(s) unavailable: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
