"""Ingest NSE industry classification — MASTER_PLAN §1.1, §8.

    python -m apps.cli.ingest_sectors

Fetches several index constituent lists and merges them into one ISIN ->
industry map. Nothing in this system knew what a company does before this ran.

**Weekly, not daily.** Membership changes at index reviews and an industry
label almost never does, so refetching every session would be five requests a
day to learn nothing. `apps.cli.daily` calls this on a cadence rather than
every run.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from core.clock import utc_now
from core.config import settings
from data.feeds.nse_sectors import (
    BROAD_LISTS,
    PRIMARY_LIST,
    SectorFormatError,
    constituent_url,
    parse_constituents,
)
from data.store.sectors import SectorStore

__all__ = ["fetch_classification", "run"]

DEFAULT_PAUSE = 0.5

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_classification(
    lists: tuple[str, ...] | None = None, pause: float = DEFAULT_PAUSE
) -> tuple[pl.DataFrame, list[str]]:
    """Fetch the classification, falling back only if the primary fails.

    Args:
        lists: Override the list sequence. `None` fetches the total-market list
            and stops there unless it fails.
        pause: Seconds between requests.

    Returns:
        The merged frame and the names of lists that could not be read. A list
        that fails is reported rather than raised — losing one file should cost
        coverage, not the whole classification.

    Note:
        The fallbacks are not fetched on the happy path. Every one of them was
        measured to be a strict subset of the total-market list, contributing
        zero unique names, so fetching them routinely was four requests an
        ingest for nothing. They exist for the day the primary changes.

        Where several are fetched they merge broadest-first, de-duplicated on
        ISIN keeping the first, so precedence is fixed rather than incidental.
    """
    sequence = lists if lists is not None else BROAD_LISTS
    frames: list[pl.DataFrame] = []
    failures: list[str] = []

    with httpx.Client(headers=NSE_HEADERS, timeout=45.0, follow_redirects=True) as client:
        try:
            client.get("https://www.nseindia.com/")
        except httpx.HTTPError as exc:
            print(f"warning: could not prime the NSE session ({exc})")

        for name in sequence:
            try:
                response = client.get(constituent_url(name))
                if response.status_code != httpx.codes.OK:
                    print(f"  {name:<28} HTTP {response.status_code}")
                    failures.append(name)
                else:
                    parsed = parse_constituents(response.content, name)
                    print(f"  {name:<28} {parsed.rows.height:>4} names")
                    frames.append(parsed.rows)
            except (httpx.HTTPError, SectorFormatError) as exc:
                print(f"  {name:<28} FAILED  {exc}")
                failures.append(name)

            # The primary is a superset of every fallback, so once it lands
            # there is nothing left to learn.
            if frames and name == PRIMARY_LIST and lists is None:
                break
            time.sleep(pause)

    if not frames:
        return pl.DataFrame(), failures
    merged = pl.concat(frames).unique(subset=["isin"], keep="first").sort("isin")
    return merged, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NSE industry classification")
    parser.add_argument("--lake", default=None)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument(
        "--observed",
        type=date.fromisoformat,
        default=None,
        help="Observation date to stamp. Defaults to today; NSE serves no archive.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = SectorStore(Path(args.lake) if args.lake else settings.lake)

    merged, failures = fetch_classification(pause=args.pause)
    if merged.is_empty():
        print("\nno classification fetched; nothing written")
        return 1

    observed = args.observed or utc_now().date()
    written = store.write(observed, merged)

    industries = merged["industry"].unique().sort().to_list()
    print(f"\n{written:,} names across {len(industries)} industries, observed {observed}")
    for industry in industries:
        count = merged.filter(pl.col("industry") == industry).height
        print(f"  {industry:<40}{count:>5}")

    if failures:
        print(f"\n{len(failures)} list(s) unavailable: {', '.join(failures)}")
        print("Coverage is lower than it could be; re-run to pick them up.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
