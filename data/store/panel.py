"""Cross-sectional daily panel — MASTER_PLAN §M2.

`BarStore` holds one instrument's history; that is the right shape for
simulating a single name. Universe construction asks the opposite question —
*"of every stock that traded on this date, which 100 were most liquid?"* —
and answering it from 2,000 per-instrument files is 2,000 file opens.

A bhavcopy already arrives as a cross-section, so it is stored as one. Same
point-in-time discipline: the only read path takes `as_of` and filters on
`receive_time`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from core.clock import DecisionTime, require_utc
from data.store.bars import NoDataError

__all__ = ["PANEL_SCHEMA", "PanelStore"]

PANEL_SCHEMA: dict[str, pl.DataType] = {
    "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "receive_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "instrument_id": pl.String(),
    "symbol": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "trades": pl.Int64(),
}


class PanelStore:
    """One Parquet file per session: ``<root>/panel/<venue>/<year>/<date>.parquet``."""

    def __init__(self, root: Path, venue: str = "NSE") -> None:
        self.root = Path(root)
        self.venue = venue

    def _dir(self, session_date: date) -> Path:
        return self.root / "panel" / self.venue / str(session_date.year)

    def _path(self, session_date: date) -> Path:
        return self._dir(session_date) / f"{session_date.isoformat()}.parquet"

    def write_session(self, session_date: date, frame: pl.DataFrame) -> int:
        """Write one session, replacing any existing file for that date.

        Whole-file replacement makes re-ingesting a corrected bhavcopy trivially
        correct — there is no partial state to merge.
        """
        conformed = self._conform(frame)
        directory = self._dir(session_date)
        directory.mkdir(parents=True, exist_ok=True)
        conformed.write_parquet(self._path(session_date), compression="zstd")
        return conformed.height

    def sessions(self) -> list[date]:
        """Every session present, ascending."""
        base = self.root / "panel" / self.venue
        if not base.exists():
            return []
        return sorted(
            date.fromisoformat(p.stem)
            for year_dir in base.iterdir()
            if year_dir.is_dir()
            for p in year_dir.glob("*.parquet")
        )

    def view(
        self,
        *,
        as_of: DecisionTime,
        start: date | None = None,
    ) -> pl.DataFrame:
        """Every panel row observable at `as_of`.

        Raises:
            NoDataError: if no sessions have been written.
        """
        cutoff = require_utc(as_of)
        available = [d for d in self.sessions() if start is None or d >= start]
        # Cheap pre-filter on filename before touching contents; the
        # receive_time filter below is the authoritative one.
        candidates = [self._path(d) for d in available if d <= cutoff.date()]
        if not candidates:
            if not self.sessions():
                raise NoDataError(f"{self.venue} panel", self.root)
            return pl.DataFrame(schema=PANEL_SCHEMA)

        return (
            pl.scan_parquet(candidates)
            .filter(pl.col("receive_time") <= cutoff)
            .sort(["event_time", "instrument_id"])
            .collect()
        )

    def session_view(self, session_date: date, *, as_of: DecisionTime) -> pl.DataFrame:
        """A single session's cross-section, if observable at `as_of`."""
        path = self._path(session_date)
        if not path.exists():
            return pl.DataFrame(schema=PANEL_SCHEMA)
        return (
            pl.scan_parquet(path)
            .filter(pl.col("receive_time") <= require_utc(as_of))
            .sort("instrument_id")
            .collect()
        )

    @staticmethod
    def _conform(frame: pl.DataFrame) -> pl.DataFrame:
        missing = set(PANEL_SCHEMA) - set(frame.columns)
        if missing:
            raise ValueError(f"panel frame missing columns: {sorted(missing)}")

        out = frame.select([pl.col(name).cast(dtype) for name, dtype in PANEL_SCHEMA.items()])
        if out.is_empty():
            raise ValueError("refusing to write an empty panel session")

        if out.filter(pl.col("receive_time") < pl.col("event_time")).height:
            raise ValueError("panel rows received before they occurred")

        duplicates = out.height - out["instrument_id"].n_unique()
        if duplicates:
            raise ValueError(
                f"{duplicates} duplicate instrument_id rows in one session; "
                "a cross-section must carry each instrument once"
            )
        return out.sort("instrument_id")
