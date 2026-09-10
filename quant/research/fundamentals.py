"""Attaching quarterly results to the panel — MASTER_PLAN §3.3, §6.

**Every one of the twenty-eight factors in this library is price and volume.**
So was every hypothesis the register has rejected. `factors.py` said why, and
said it honestly: *"there is no clean free source for Indian fundamentals, so
value and quality factors are absent rather than approximated badly."* That was
true when it was written. NSE's corporate-filings API is that source, and this
module is the join that makes it usable.

**The join is the entire difficulty, and it is a point-in-time join.** A vendor
fundamentals table gives you the quarter and the numbers, so the obvious join —
match the March quarter to March — reads a filing published in July from a
decision made in April. The distortion is enormous and invisible: it looks like
skill. Here each session is matched to the newest filing whose *dissemination*
preceded it, which is the only join that describes what a decision could have
known.

**Matched on dissemination date, not period.** A filing published on session D
is available to a decision made on D's close, because this system decides after
the close and fills on D+1. It was not in D's closing price — the market shut at
15:30 and the filing landed at 17:18 — so using it on D+1 is trading on it, not
peeking at it.

**Backward as-of, per instrument.** A name that has not reported in two years
carries its last filing forward and is reported as stale rather than dropped:
the staleness is a fact about the company, and the factor's user should see it
rather than have the name silently vanish from the cross-section.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

__all__ = [
    "FUNDAMENTAL_COLUMNS",
    "MAX_FILING_AGE_DAYS",
    "attach_fundamentals",
]

#: Columns lifted from the filing onto every session that could see it.
FUNDAMENTAL_COLUMNS = ("revenue", "net_profit", "eps_basic", "period_end", "receive_time")

#: Beyond this, the last filing is too old to describe the company.
#:
#: A quarterly filer is 30-120 days stale in normal operation; the tail beyond
#: two quarters is a company that has stopped reporting, and carrying its
#: numbers forward would score it on figures nobody has confirmed since. Named
#: rather than dropped silently: `filing_age_days` is on every row.
MAX_FILING_AGE_DAYS = 250


def attach_fundamentals(
    panel: pl.DataFrame,
    lake: Path,
    max_age_days: int = MAX_FILING_AGE_DAYS,
) -> pl.DataFrame:
    """Panel with each session's newest *knowable* filing joined on.

    Args:
        panel: Rows carrying `instrument_id` and `event_time`.
        lake: Lake root, for the fundamentals store.
        max_age_days: Filings older than this are attached but marked; set 0 to
            keep every one regardless of age.

    Returns:
        The panel plus `revenue`, `net_profit`, `eps_basic`, `period_end`,
        `filing_age_days` and `filing_stale`. A name with no filing readable at
        that session gets nulls, which is the honest answer — it is not a zero
        and it is not last quarter's.

    Consolidated wins where a company filed both on the same period: that is
    the group's economics, and mixing the two would put a parent's revenue
    beside a group's in the same cross-section.
    """
    from data.store.fundamentals import FundamentalStore  # noqa: PLC0415 - data layer

    filings = FundamentalStore(lake).view().rows  # lint: allow-unbounded-read
    if filings.is_empty() or panel.is_empty():
        return _empty_columns(panel)

    # The stores key differently and deliberately so: filings arrive keyed on
    # ISIN, which is the identity (§1.1), while the panel addresses a listing
    # on a venue. The prefix is taken from the panel rather than assumed, so
    # this works for BSE without a second code path.
    prefix = str(panel["instrument_id"][0]).split(":", 1)[0]

    # One row per (instrument, dissemination). Consolidated first so the
    # de-duplication keeps it, and the later filing of a corrected pair wins.
    usable = (
        filings.filter(pl.col("eps_basic").is_not_null() | pl.col("revenue").is_not_null())
        .sort(["isin", "receive_time", "consolidated"], descending=[False, False, True])
        .unique(subset=["isin", "receive_time"], keep="first", maintain_order=True)
        .with_columns((pl.lit(prefix + ":") + pl.col("isin")).alias("instrument_id"))
        .select("instrument_id", "receive_time", *FUNDAMENTAL_COLUMNS[:-1])
    )
    if usable.is_empty():
        return _empty_columns(panel)

    # The as-of key is a date on both sides. A filing disseminated on session D
    # is knowable for a decision taken on D's close, which is after the
    # exchange has shut and before the D+1 fill this system always uses.
    left = panel.with_columns(pl.col("event_time").dt.date().alias("_asof")).sort(
        ["_asof", "instrument_id"]
    )
    right = usable.with_columns(pl.col("receive_time").dt.date().alias("_asof")).sort(
        ["_asof", "instrument_id"]
    )

    joined = left.join_asof(
        right,
        on="_asof",
        by="instrument_id",
        strategy="backward",
    )

    age = (pl.col("_asof") - pl.col("receive_time").dt.date()).dt.total_days()
    return joined.with_columns(
        age.alias("filing_age_days"),
        (age > max_age_days).alias("filing_stale") if max_age_days > 0 else pl.lit(False),
    ).drop("_asof")


def _empty_columns(panel: pl.DataFrame) -> pl.DataFrame:
    """The same shape, all nulls.

    Returned rather than raising when nothing has been ingested, so a factor
    built on this scores nothing instead of failing — and nulls propagate to a
    dropped row rather than to a zero that would rank as the cheapest name in
    the market.
    """
    return panel.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("revenue"),
        pl.lit(None, dtype=pl.Float64).alias("net_profit"),
        pl.lit(None, dtype=pl.Float64).alias("eps_basic"),
        pl.lit(None, dtype=pl.Date).alias("period_end"),
        pl.lit(None, dtype=pl.Datetime(time_unit="us", time_zone="UTC")).alias("receive_time"),
        pl.lit(None, dtype=pl.Int64).alias("filing_age_days"),
        pl.lit(True).alias("filing_stale"),
    )
