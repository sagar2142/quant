"""Derivatives store — MASTER_PLAN §13.4.

One Parquet file per session, mirroring `PanelStore` in layout and in
discipline: whole-file replacement so a corrected bhavcopy needs no merge, and
a `receive_time` cutoff on every read so a decision cannot see a file published
after it was made.

**Separate from the equity panel on purpose.** A single NSE session carries
about thirty thousand contracts against three thousand equities, and a contract
is identified by underlying, expiry, strike and right rather than by ISIN.
Writing them into the same store would put every strike of every expiry into
the cross-section the factor library scores, which is not a universe anyone
means to rank.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from core.clock import DecisionTime, require_utc
from data.store.bars import NoDataError

__all__ = ["CONTRACT_SCHEMA", "DerivativesStore"]

#: The columns every session file carries. Enforced on write so a schema drift
#: fails at the boundary rather than at the first read that needs the column.
CONTRACT_SCHEMA: dict[str, pl.DataType] = {
    "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "receive_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    "contract_id": pl.String(),
    "underlying": pl.String(),
    "instrument_type": pl.String(),
    "expiry": pl.Date(),
    "strike": pl.Float64(),
    "right": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "settlement": pl.Float64(),
    "underlying_price": pl.Float64(),
    "open_interest": pl.Float64(),
    "oi_change": pl.Float64(),
    "volume": pl.Float64(),
    "trades": pl.Float64(),
    "lot_size": pl.Float64(),
}


class DerivativesStore:
    """``<root>/derivatives/<segment>/<year>/<date>.parquet``."""

    def __init__(self, root: Path, segment: str = "NFO") -> None:
        self.root = Path(root)
        self.segment = segment

    def _dir(self, session_date: date) -> Path:
        return self.root / "derivatives" / self.segment / str(session_date.year)

    def _path(self, session_date: date) -> Path:
        return self._dir(session_date) / f"{session_date.isoformat()}.parquet"

    def _conform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Check the columns and the one invariant that matters.

        A contract appearing twice in one session is a parse fault, not a
        market event: the same terms cannot settle at two prices on one day.
        Caught here rather than at read time, where it would present as a chain
        with duplicated strikes and no explanation.
        """
        missing = [c for c in CONTRACT_SCHEMA if c not in frame.columns]
        if missing:
            raise ValueError(f"derivatives session is missing columns {missing}")
        ordered = frame.select(list(CONTRACT_SCHEMA))
        duplicates = ordered.height - ordered["contract_id"].n_unique()
        if duplicates:
            raise ValueError(
                f"{duplicates} duplicate contract_id rows in one session; "
                "the same terms cannot settle at two prices on one day"
            )
        return ordered

    def write_session(self, session_date: date, frame: pl.DataFrame) -> int:
        """Write one session, replacing any existing file for that date."""
        conformed = self._conform(frame)
        directory = self._dir(session_date)
        directory.mkdir(parents=True, exist_ok=True)
        conformed.write_parquet(self._path(session_date), compression="zstd")
        return conformed.height

    def sessions(self) -> list[date]:
        """Every session present, ascending."""
        base = self.root / "derivatives" / self.segment
        if not base.exists():
            return []
        return sorted(
            date.fromisoformat(p.stem)
            for year_dir in base.iterdir()
            if year_dir.is_dir()
            for p in year_dir.glob("*.parquet")
        )

    def latest_session(self) -> date | None:
        """The newest session held, or None."""
        found = self.sessions()
        return found[-1] if found else None

    def chain(
        self,
        underlying: str,
        *,
        as_of: DecisionTime,
        session: date | None = None,
    ) -> pl.DataFrame:
        """Every contract on one underlying, for one session.

        Args:
            underlying: Ticker of the underlying, e.g. RELIANCE or NIFTY.
            as_of: Decision time. Rows published after it are not returned,
                which is the same point-in-time rule the equity panel keeps.
            session: Which session to read. Defaults to the newest held that is
                observable at `as_of`.

        Raises:
            NoDataError: if nothing has been ingested at all.
        """
        cutoff = require_utc(as_of)
        available = [d for d in self.sessions() if d <= cutoff.date()]
        if not available:
            if not self.sessions():
                raise NoDataError(f"{self.segment} derivatives", self.root)
            return pl.DataFrame(schema=CONTRACT_SCHEMA)

        wanted = session or available[-1]
        path = self._path(wanted)
        if not path.exists():
            return pl.DataFrame(schema=CONTRACT_SCHEMA)

        return (
            pl.scan_parquet(path)
            .filter(
                (pl.col("receive_time") <= cutoff) & (pl.col("underlying") == underlying.upper())
            )
            .sort(["expiry", "strike", "right"])
            .collect()
        )

    def underlyings(self, *, as_of: DecisionTime) -> list[str]:
        """Every underlying with contracts in the newest observable session."""
        cutoff = require_utc(as_of)
        available = [d for d in self.sessions() if d <= cutoff.date()]
        if not available:
            return []
        return sorted(
            pl.scan_parquet(self._path(available[-1]))
            .filter(pl.col("receive_time") <= cutoff)
            .select("underlying")
            .unique()
            .collect()["underlying"]
            .to_list()
        )
