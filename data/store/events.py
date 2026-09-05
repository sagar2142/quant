"""The announced calendar, and when it was announced — MASTER_PLAN §3.3.

One Parquet file per observation date: ``<root>/events/<date>.parquet``.

Dated for the same reason the sector classification is: NSE serves what is
ahead and keeps no archive, so an observation belongs to the day it was
fetched rather than to any session. Recording that is what stops a calendar
fetched today from presenting itself as one that was known last month.

**This is a present-tense feed and says so.** `upcoming()` answers "what is
coming, as far as the newest calendar knows". There is deliberately no accessor
that reconstructs what the calendar looked like on a past date, because the
data to do that honestly does not exist — a company that announced a meeting
and then moved it leaves no trace of the original date. Historical earnings
dates, properly timestamped, live in `data.store.fundamentals` instead, and
that is what a backtest should read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from data.feeds.nse_events import EVENT_SCHEMA

__all__ = ["EventStore", "EventWindow"]

#: Columns a stored calendar carries: the parsed event plus its resolution.
STORED_SCHEMA: dict[str, pl.DataType] = {
    **EVENT_SCHEMA,
    "instrument_id": pl.String(),
}


@dataclass(frozen=True)
class EventWindow:
    """Announced events in a forward window, and how current the calendar is."""

    rows: pl.DataFrame
    #: When the calendar was fetched. `None` when nothing is held.
    observed_at: date | None
    #: Days since it was fetched. A calendar is only as good as its age, and a
    #: stale one is worse than none — it reports quiet where there is none.
    age_days: int = 0

    @property
    def names(self) -> int:
        return int(self.rows["symbol"].n_unique()) if not self.rows.is_empty() else 0

    def for_instruments(self, instrument_ids: set[str]) -> pl.DataFrame:
        """Events for a set of held instruments.

        The book's question: is anything I own about to report?
        """
        if self.rows.is_empty():
            return self.rows
        return self.rows.filter(pl.col("instrument_id").is_in(list(instrument_ids)))


class EventStore:
    """Announced board meetings on disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self) -> Path:
        return self.root / "events"

    def _path(self, observed: date) -> Path:
        return self._dir() / f"{observed.isoformat()}.parquet"

    def observations(self) -> list[date]:
        """Every observation date held, ascending."""
        base = self._dir()
        if not base.exists():
            return []
        found = []
        for path in base.glob("*.parquet"):
            try:
                found.append(date.fromisoformat(path.stem))
            except ValueError:
                continue
        return sorted(found)

    def write(self, observed: date, frame: pl.DataFrame) -> int:
        """Store one observation, replacing any for the same date."""
        missing = [c for c in EVENT_SCHEMA if c not in frame.columns]
        if missing:
            raise ValueError(f"event frame is missing columns {missing}")
        prepared = frame
        if "instrument_id" not in prepared.columns:
            prepared = prepared.with_columns(pl.lit(None, pl.String).alias("instrument_id"))
        ordered = prepared.select(list(STORED_SCHEMA)).unique(
            subset=["symbol", "event_date", "purpose"], keep="first"
        )
        self._dir().mkdir(parents=True, exist_ok=True)
        ordered.write_parquet(self._path(observed), compression="zstd")
        return ordered.height

    def upcoming(
        self, today: date, within_days: int = 7, results_only: bool = False
    ) -> EventWindow:
        """Announced events between `today` and `within_days` ahead.

        Args:
            today: The day to look forward from. Inclusive.
            within_days: How far ahead. Inclusive.
            results_only: Keep only meetings called to consider results.

        Reads the newest calendar held and reports its age. Nothing here
        reconstructs a past calendar — see the module docstring.
        """
        held = self.observations()
        if not held:
            return EventWindow(rows=pl.DataFrame(schema=STORED_SCHEMA), observed_at=None)

        newest = held[-1]
        rows = pl.read_parquet(self._path(newest))
        horizon = today + timedelta(days=within_days)
        rows = rows.filter((pl.col("event_date") >= today) & (pl.col("event_date") <= horizon))
        if results_only:
            rows = rows.filter(pl.col("is_results"))
        return EventWindow(
            rows=rows.sort(["event_date", "symbol"]),
            observed_at=newest,
            age_days=(today - newest).days,
        )
