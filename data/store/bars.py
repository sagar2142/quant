"""Bar storage — Parquet lake with point-in-time reads.

MASTER_PLAN §14.1.4: **the future must be unreachable by construction.**

There is deliberately no method that returns "all bars". The only read path is
``view(..., as_of=decision_time)``, which filters on ``receive_time <= as_of``.
A slice bug therefore cannot produce look-ahead, because the future was never
in the returned frame to begin with.

**Float64 here is correct, not a lapse (§14.1.2).** This is the research path:
bulk statistics over millions of rows, read into Polars columns. The exact
`Decimal` representation lives in `core.events` objects on the live path, where
volume is low and a broker might disagree with you about the number.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from core.clock import DecisionTime, require_utc
from core.events import Timeframe
from core.instruments import InstrumentId

__all__ = ["BAR_SCHEMA", "BarStore", "NoDataError"]

#: Canonical column layout. Every feed normalises into exactly this.
BAR_SCHEMA: dict[str, pl.DataType] = {
    # Bar CLOSE time, never open — a bar stamped with its open can be observed
    # before it has finished forming, which is look-ahead (core.events.Bar).
    "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    # When this system could first have known the bar. The only column that
    # gates observation.
    "receive_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    # Instantiated, not bare classes: Polars accepts both, but only instances
    # satisfy the DataType annotation.
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "trades": pl.Int64(),
}


class NoDataError(LookupError):
    """Nothing has been written for the requested series.

    Raised rather than returning an empty frame (§14.1.5): an empty frame
    flows silently through a strategy and becomes a silently skipped universe
    member, which looks exactly like a strategy that chose not to trade.

    Takes a plain message so every store can describe its own shape — coupling
    this to one store's identifiers is what previously required a `type: ignore`
    at the other call site, which then hid a real bug.
    """

    def __init__(self, what: str, root: Path) -> None:
        super().__init__(f"no data for {what} under {root}. Run the loader before reading.")


def _safe_key(value: str) -> str:
    """Filesystem-safe form of an instrument id.

    Instrument ids look like ``NSE:INE002A01018``; a colon is illegal in Windows
    paths, so it is folded to an underscore for directory names only. The id
    inside the data is never rewritten.
    """
    return value.replace(":", "_").replace("/", "_")


class BarStore:
    """Partitioned Parquet store: ``<root>/bars/<timeframe>/<instrument>/<year>.parquet``.

    Partitioning by year keeps individual files small enough to rewrite when a
    backfill corrects history, without needing a compaction step.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, instrument_id: InstrumentId, timeframe: Timeframe) -> Path:
        return self.root / "bars" / timeframe.value / _safe_key(instrument_id)

    def write(
        self,
        instrument_id: InstrumentId,
        timeframe: Timeframe,
        frame: pl.DataFrame,
    ) -> int:
        """Write bars, replacing any existing rows with the same `event_time`.

        Idempotent: re-running a loader over an overlapping window converges to
        one row per timestamp rather than accumulating duplicates.

        Returns:
            Total row count for this instrument/timeframe after the write.
        """
        frame = self._conform(frame)
        target = self._dir(instrument_id, timeframe)
        target.mkdir(parents=True, exist_ok=True)

        written = 0
        partitioned = frame.with_columns(pl.col("event_time").dt.year().alias("year"))
        for (year,), group in partitioned.group_by("year", maintain_order=True):
            path = target / f"{year}.parquet"
            part = group.drop("year")
            if path.exists():
                existing = pl.read_parquet(path)
                part = (
                    pl.concat([existing, part])
                    .unique(subset=["event_time"], keep="last")
                    .sort("event_time")
                )
            part.write_parquet(path, compression="zstd")
            written += part.height
        return written

    def view(
        self,
        instrument_id: InstrumentId,
        timeframe: Timeframe,
        *,
        as_of: DecisionTime,
        start: datetime | None = None,
    ) -> pl.DataFrame:
        """Bars observable at `as_of`, sorted by event_time.

        Filters ``receive_time <= as_of``. This is the whole point of the class:
        there is no way to ask for data the system had not yet received.

        Raises:
            NoDataError: if this instrument/timeframe has never been written.
        """
        cutoff = require_utc(as_of)
        directory = self._dir(instrument_id, timeframe)
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise NoDataError(f"{instrument_id} @ {timeframe.value}", self.root)

        lazy = pl.scan_parquet(files).filter(pl.col("receive_time") <= cutoff)
        if start is not None:
            lazy = lazy.filter(pl.col("event_time") >= require_utc(start))
        return lazy.sort("event_time").collect()

    def coverage(
        self, instrument_id: InstrumentId, timeframe: Timeframe
    ) -> tuple[datetime, datetime, int]:
        """(first_event_time, last_event_time, row_count) across all partitions.

        A metadata question, not a data read, so it is exempt from `as_of`.

        Raises:
            NoDataError: if nothing has been written.
        """
        directory = self._dir(instrument_id, timeframe)
        files = sorted(directory.glob("*.parquet"))
        if not files:
            raise NoDataError(f"{instrument_id} @ {timeframe.value}", self.root)

        stats = (
            pl.scan_parquet(files)
            .select(
                pl.col("event_time").min().alias("first"),
                pl.col("event_time").max().alias("last"),
                pl.len().alias("n"),
            )
            .collect()
        )
        return stats["first"][0], stats["last"][0], int(stats["n"][0])

    def instruments(self, timeframe: Timeframe) -> list[str]:
        """Filesystem-safe keys present for a timeframe."""
        directory = self.root / "bars" / timeframe.value
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_dir())

    @staticmethod
    def _conform(frame: pl.DataFrame) -> pl.DataFrame:
        """Validate and coerce an incoming frame to BAR_SCHEMA.

        Raises:
            ValueError: on missing columns, or on rows that violate the
                causality or OHLC invariants.
        """
        missing = set(BAR_SCHEMA) - set(frame.columns)
        if missing:
            raise ValueError(f"bar frame missing columns: {sorted(missing)}")

        out = frame.select([pl.col(name).cast(dtype) for name, dtype in BAR_SCHEMA.items()]).sort(
            "event_time"
        )

        if out.is_empty():
            raise ValueError("refusing to write an empty bar frame")

        bad_causality = out.filter(pl.col("receive_time") < pl.col("event_time")).height
        if bad_causality:
            raise ValueError(
                f"{bad_causality} bar(s) received before they occurred — "
                "clock drift or a feed adapter bug (core.events.CausalityError)"
            )

        bad_ohlc = out.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("open") > pl.col("high"))
            | (pl.col("open") < pl.col("low"))
            | (pl.col("close") > pl.col("high"))
            | (pl.col("close") < pl.col("low"))
            | (pl.col("low") <= 0)
            | (pl.col("volume") < 0)
        ).height
        if bad_ohlc:
            raise ValueError(f"{bad_ohlc} bar(s) violate OHLC invariants")

        return out
