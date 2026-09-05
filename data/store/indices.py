"""Index series store — MASTER_PLAN §13.4.

One Parquet file per session, mirroring `PanelStore` in layout and discipline:
whole-file replacement so a corrected file needs no merge, and a `receive_time`
cutoff on every read so a decision cannot see a file published after it.

**Separate from the equity panel, deliberately.** An index is not a security:
it has no ISIN, it cannot be held, and a cross-section that contained it would
let a factor rank NIFTY 50 alongside the fifty stocks it is computed from.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from core.clock import DecisionTime, require_utc
from data.store.bars import NoDataError

__all__ = ["INDEX_SCHEMA", "IndexStore"]

INDEX_SCHEMA: dict[str, pl.DataType] = {
    "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "receive_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "index_name": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
}


class IndexStore:
    """``<root>/indices/<venue>/<year>/<date>.parquet``."""

    def __init__(self, root: Path, venue: str = "NSE") -> None:
        self.root = Path(root)
        self.venue = venue

    def _dir(self, session_date: date) -> Path:
        return self.root / "indices" / self.venue / str(session_date.year)

    def _path(self, session_date: date) -> Path:
        return self._dir(session_date) / f"{session_date.isoformat()}.parquet"

    def write_session(self, session_date: date, frame: pl.DataFrame) -> int:
        """Write one session, replacing any existing file for that date."""
        missing = [c for c in INDEX_SCHEMA if c not in frame.columns]
        if missing:
            raise ValueError(f"index session is missing columns {missing}")
        ordered = frame.select(list(INDEX_SCHEMA))
        directory = self._dir(session_date)
        directory.mkdir(parents=True, exist_ok=True)
        ordered.write_parquet(self._path(session_date), compression="zstd")
        return ordered.height

    def sessions(self) -> list[date]:
        """Every session present, ascending."""
        base = self.root / "indices" / self.venue
        if not base.exists():
            return []
        return sorted(
            date.fromisoformat(p.stem)
            for year_dir in base.iterdir()
            if year_dir.is_dir()
            for p in year_dir.glob("*.parquet")
        )

    def series(
        self,
        index_name: str,
        *,
        as_of: DecisionTime,
        start: date | None = None,
    ) -> pl.DataFrame:
        """One index's history, observable at `as_of`.

        Raises:
            NoDataError: if nothing has been ingested at all.
        """
        cutoff = require_utc(as_of)
        available = [
            d for d in self.sessions() if d <= cutoff.date() and (start is None or d >= start)
        ]
        if not available:
            if not self.sessions():
                raise NoDataError(f"{self.venue} indices", self.root)
            return pl.DataFrame(schema=INDEX_SCHEMA)

        return (
            pl.scan_parquet([self._path(d) for d in available])
            .filter((pl.col("receive_time") <= cutoff) & (pl.col("index_name") == index_name))
            .sort("event_time")
            .collect()
        )

    def names(self, *, as_of: DecisionTime) -> list[str]:
        """Every index held, from the newest observable session."""
        cutoff = require_utc(as_of)
        available = [d for d in self.sessions() if d <= cutoff.date()]
        if not available:
            return []
        return sorted(
            pl.scan_parquet(self._path(available[-1]))
            .select("index_name")
            .unique()
            .collect()["index_name"]
            .to_list()
        )
