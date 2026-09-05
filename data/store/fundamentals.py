"""Quarterly results, readable only from when they were published — §3.3, §9.

One Parquet file per period year: ``<root>/fundamentals/<year>.parquet``.

**The whole value of this store is the `receive_time` filter.** Fundamentals
are the easiest data in finance to use dishonestly, because the shape that
feels natural — a table of quarters, each with its numbers — invites reading
March's revenue in April. In reality that filing lands in July. A backtest that
ranks on it from April is not a good strategy; it is a strategy that knows the
future, and it will show a Sharpe to match.

So there is no accessor here that returns a quarter by its period. Every read
goes through `view(as_of)`, which admits a filing only once
`receive_time <= as_of` — the same rule the panel uses (§3.3), enforced in the
one place rather than trusted to each caller.

**Partitioned by period year, not receive year.** A filing's period never
changes, so a row never moves between files; a late filing would otherwise
migrate and break reproducibility for a run that had already read it. Reads
scan every file and filter on `receive_time` regardless, which is cheap — the
whole archive is a few hundred thousand rows.

**Consolidated and standalone are both kept.** A group and its parent are
different companies with different numbers, and silently collapsing them mixes
the two. `view` prefers consolidated where a name filed both, because that is
what a group's economics actually are, and says so rather than assuming the
caller knows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from core.clock import DecisionTime, require_utc
from data.feeds.nse_results import FILING_SCHEMA, FUNDAMENTAL_FACTS

__all__ = ["FUNDAMENTAL_SCHEMA", "FundamentalStore", "FundamentalView"]

#: The filing, plus whatever was extracted from its XBRL.
FUNDAMENTAL_SCHEMA: dict[str, pl.DataType] = {
    **FILING_SCHEMA,
    **{column: pl.Float64() for column in FUNDAMENTAL_FACTS.values()},
}

#: Columns that identify one filing. A company refiling a corrected result
#: produces the same key with a later `receive_time`, and the later one wins.
FILING_KEY = ("isin", "period_start", "period_end", "consolidated")


@dataclass(frozen=True)
class FundamentalView:
    """Filings observable at a decision time, and what that cost in coverage."""

    rows: pl.DataFrame
    as_of: datetime | None
    #: Filings held but not yet published at `as_of`. Reported rather than
    #: dropped in silence: a large number means the read is early, not that the
    #: companies did not report.
    withheld: int = 0

    @property
    def names(self) -> int:
        return int(self.rows["isin"].n_unique()) if not self.rows.is_empty() else 0

    def latest(self) -> pl.DataFrame:
        """The most recently published filing per name.

        What a cross-sectional factor reads: for each company, the newest
        result the market had actually seen by `as_of`. Consolidated wins a tie
        on the same period, being the group's real economics.
        """
        if self.rows.is_empty():
            return self.rows
        return (
            self.rows.sort(["receive_time", "consolidated"], descending=[True, True])
            .unique(subset=["isin"], keep="first")
            .sort("isin")
        )

    def age_days(self, at: datetime | None = None) -> pl.DataFrame:
        """`latest` with how stale each name's newest published result is.

        A quarterly filer is normally 30-120 days old. Much beyond that is a
        company that has stopped reporting, which is a fact about the company
        and not a reason to treat its last known numbers as current.
        """
        reference = require_utc(at) if at is not None else self.as_of
        latest = self.latest()
        if latest.is_empty() or reference is None:
            return latest
        return latest.with_columns(
            ((pl.lit(reference) - pl.col("receive_time")).dt.total_days()).alias("age_days")
        )


class FundamentalStore:
    """Quarterly results on disk, queryable only point-in-time."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self) -> Path:
        return self.root / "fundamentals"

    def _path(self, year: int) -> Path:
        return self._dir() / f"{year}.parquet"

    def years(self) -> list[int]:
        """Every period year held, ascending."""
        base = self._dir()
        if not base.exists():
            return []
        found = []
        for path in base.glob("*.parquet"):
            try:
                found.append(int(path.stem))
            except ValueError:
                continue
        return sorted(found)

    def write(self, frame: pl.DataFrame) -> int:
        """Merge filings into the store, newest filing of each key winning.

        Raises:
            ValueError: if a required column is absent.

        Merging rather than replacing, because one fetch covers a date window
        and windows overlap. A company that refiles a corrected result appears
        twice with the same key; the later `receive_time` is the one that
        survives, which is what a reader at a later date would have seen.

        **Ties are broken toward the row that has numbers.** The ordinary case
        is not a correction at all — it is the same filing fetched twice, once
        with its XBRL read and once without, because a run's document budget
        stopped short of it. Those rows carry an identical `receive_time`, and
        `unique` gives no order guarantee, so without an explicit preference
        the survivor is arbitrary: re-running the ingest would discard numbers
        it had already fetched, at random, with the row count unchanged and
        nothing to show anything had been lost.
        """
        missing = [c for c in FILING_SCHEMA if c not in frame.columns]
        if missing:
            raise ValueError(f"filing frame is missing columns {missing}")

        prepared = frame
        for column, dtype in FUNDAMENTAL_SCHEMA.items():
            if column not in prepared.columns:
                prepared = prepared.with_columns(pl.lit(None, dtype=dtype).alias(column))
        prepared = prepared.select(list(FUNDAMENTAL_SCHEMA))

        written = 0
        self._dir().mkdir(parents=True, exist_ok=True)
        for (year,), part in prepared.group_by(
            [pl.col("period_end").dt.year()], maintain_order=True
        ):
            path = self._path(int(year))
            combined = (
                pl.concat([pl.read_parquet(path), part], how="vertical_relaxed")
                if path.exists()
                else part
            )
            facts = list(FUNDAMENTAL_FACTS.values())
            merged = (
                combined.with_columns(
                    pl.sum_horizontal(pl.col(c).is_not_null() for c in facts).alias("_facts")
                )
                .sort(["receive_time", "_facts"], descending=[True, True], maintain_order=True)
                .unique(subset=list(FILING_KEY), keep="first", maintain_order=True)
                .drop("_facts")
                .sort(["period_end", "isin"])
            )
            merged.write_parquet(path, compression="zstd")
            written += merged.height
        return written

    def _all(self) -> pl.DataFrame:
        paths = [self._path(y) for y in self.years()]
        if not paths:
            return pl.DataFrame(schema=FUNDAMENTAL_SCHEMA)
        return pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")

    def newest_filing(self) -> datetime | None:
        """When the most recent filing held was published, or `None` if empty.

        A scheduling question, not a research one: it asks how current the
        store is, which is why it does not go through `view` — reading every
        row to take one maximum would be wasteful, and asking for a view
        without a decision time is the shape of an accidental future read.
        """
        newest: datetime | None = None
        for year in self.years():
            found = pl.read_parquet(self._path(year), columns=["receive_time"])[
                "receive_time"
            ].max()
            if isinstance(found, datetime) and (newest is None or found > newest):
                newest = found
        return newest

    def view(self, as_of: DecisionTime | None = None) -> FundamentalView:
        """Filings the market had published by `as_of`.

        Args:
            as_of: The decision point. `None` reads everything held, which is
                correct only for present-tense questions — a screen of what is
                known now. Any historical read must pass one.

        The filter is on `receive_time`, never on the period. A March quarter
        published in July is invisible to a June decision, which is the entire
        reason this store exists.
        """
        rows = self._all()
        if rows.is_empty():
            return FundamentalView(rows=rows, as_of=None)
        if as_of is None:
            return FundamentalView(rows=rows.sort("receive_time"), as_of=None)

        moment = require_utc(as_of)
        observable = rows.filter(pl.col("receive_time") <= moment)
        return FundamentalView(
            rows=observable.sort("receive_time"),
            as_of=moment,
            withheld=rows.height - observable.height,
        )
