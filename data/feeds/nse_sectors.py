"""Industry classification from NSE index constituents — MASTER_PLAN §1.1, §8.

**Nothing in this system knew what a company does.** Every factor, every
cluster, every exposure was computed from price. That is defensible for
concentration — `quant.analytics.clusters` argues, correctly, that two banks
which do not move together are two bets whatever the label says — but it
leaves questions nobody could answer: how much of the book is banks, whether a
drawdown was one sector unwinding, what a beta looks like against the industry
a name actually belongs to.

NSE publishes its index constituent lists as CSV, and each row carries the
company's `Industry` alongside its ISIN. That is the classification, from the
exchange, keyed on the identity this system already uses (§1.1).

**Industry, not index membership.** The two are different in a way that
matters. Membership of Nifty 500 changes every review, so using today's list to
describe 2019 would be survivorship. An industry label is a property of the
company — a bank was a bank in 2019 — so carrying it backwards is a far weaker
assumption. It is still an assumption, and `data.store.sectors` records when
each label was observed so a caller can see how far it is reaching.

**What this cannot tell you.** A company delisted before today appears in no
current constituent list and therefore has no industry here. Sector-aware
analysis over history is missing exactly the names that failed, which is the
direction that flatters. Coverage is reported per read for that reason.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import polars as pl

__all__ = [
    "BROAD_LISTS",
    "FALLBACK_LISTS",
    "PRIMARY_LIST",
    "SECTOR_SCHEMA",
    "SectorFormatError",
    "constituent_url",
    "parse_constituents",
]

#: The widest list NSE publishes, and the only one normally fetched.
#:
#: Measured: 755 names, and every other list is a strict subset of it — the 500,
#: the midcap 150, the smallcap and microcap 250s contributed *zero* unique
#: names between them. Fetching all five was four requests an ingest to learn
#: nothing.
PRIMARY_LIST = "ind_niftytotalmarket_list"

#: Tried only when the primary fails, in descending breadth.
#:
#: Kept rather than deleted because the measurement above is of one day's
#: files: if NSE renames or narrows the total-market list, a classification
#: covering the top few hundred names beats no classification at all. They cost
#: nothing while the primary works.
FALLBACK_LISTS: tuple[str, ...] = (
    "ind_nifty500list",
    "ind_niftymidcap150list",
    "ind_niftysmallcap250list",
    "ind_niftymicrocap250_list",
)

#: Every list this feed knows, primary first.
BROAD_LISTS: tuple[str, ...] = (PRIMARY_LIST, *FALLBACK_LISTS)

#: Columns the file must carry. Named so a layout change fails loudly rather
#: than producing a frame of nulls that classifies nothing.
REQUIRED_COLUMNS = ("Company Name", "Industry", "Symbol", "ISIN Code")

SECTOR_SCHEMA: dict[str, pl.DataType] = {
    "isin": pl.String(),
    "symbol": pl.String(),
    "company": pl.String(),
    "industry": pl.String(),
    #: Which list this row came from. Kept so a coverage gap can be traced to a
    #: list that failed rather than to a name that has no industry.
    "source": pl.String(),
}


class SectorFormatError(ValueError):
    """The file was not an NSE constituent list. Never guessed at."""


@dataclass(frozen=True)
class ConstituentList:
    """One index's members, with their industries."""

    source: str
    rows: pl.DataFrame


def constituent_url(list_name: str) -> str:
    """Where NSE publishes one index's constituents.

    Unlike the bhavcopy and the index closes, this carries no date: NSE serves
    the *current* membership at a fixed address and publishes no archive. That
    is the reason `data.store.sectors` stamps observations with the date they
    were fetched rather than with a session date — there is no session date to
    use.
    """
    return f"https://nsearchives.nseindia.com/content/indices/{list_name}.csv"


def parse_constituents(payload: bytes, source: str) -> ConstituentList:
    """Parse one constituent list.

    Args:
        payload: The CSV as downloaded.
        source: The list it came from, recorded on every row.

    Raises:
        SectorFormatError: if the layout is not the expected one, or the file
            is a web page rather than data.
    """
    head = payload[:512].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html")):
        raise SectorFormatError(f"NSE returned a web page rather than {source}")

    try:
        frame = pl.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise SectorFormatError(f"could not read {source} as CSV: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SectorFormatError(f"{source} is missing columns {missing}")

    rows = (
        frame.select(
            pl.col("ISIN Code").str.strip_chars().alias("isin"),
            pl.col("Symbol").str.strip_chars().str.to_uppercase().alias("symbol"),
            pl.col("Company Name").str.strip_chars().alias("company"),
            pl.col("Industry").str.strip_chars().alias("industry"),
            pl.lit(source).alias("source"),
        )
        # A row with no ISIN cannot be joined to anything this system holds, and
        # a row with no industry is not a classification. Both are dropped
        # rather than stored as blanks that would read as a real category.
        .filter((pl.col("isin").str.len_chars() > 0) & (pl.col("industry").str.len_chars() > 0))
        .unique(subset=["isin"], keep="first")
        .sort("isin")
    )
    return ConstituentList(source=source, rows=rows)
