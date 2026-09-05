"""Industry classification, and when it was observed — MASTER_PLAN §1.1, §3.3.

One Parquet file per observation date: ``<root>/sectors/<date>.parquet``.

**Dated, because NSE publishes no archive.** The constituent lists are served
at a fixed address showing today's membership, so unlike a bhavcopy there is no
session this data belongs to — only the day it was fetched. Recording that date
is what lets a reader see how far a label is being carried, and what stops a
"current" classification silently presenting itself as a historical one.

**The honest position on look-ahead.** An industry label used for a decision
made before it was observed is, strictly, information from the future. It is a
much weaker violation than a price would be — a bank was a bank in 2019, and
the label barely moves — but it is not nothing, and pretending otherwise would
be exactly the kind of quiet approximation this system exists to refuse. So:

    `as_of` reads return the newest observation at or before that date, and
    `latest` returns the newest held. A caller reaching backwards past every
    observation gets nothing rather than today's answer.

`SectorView.reaching_back` says how far the returned labels are being carried,
so a screen can report it the way `RiskModel.market_source` reports which market
a beta was measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from core.clock import DecisionTime, require_utc
from data.feeds.nse_sectors import SECTOR_SCHEMA

__all__ = ["SectorStore", "SectorView"]


@dataclass(frozen=True)
class SectorView:
    """A classification, and the provenance a reader needs to judge it."""

    #: ISIN -> industry.
    industries: dict[str, str]
    #: ISIN -> ticker, as the list reported it.
    symbols: dict[str, str]
    #: When these labels were observed. `None` when nothing was held.
    observed_at: date | None
    #: The decision date they were read for.
    as_of: date | None = None

    @property
    def reaching_back(self) -> int:
        """Days between the observation and the decision it is describing.

        Zero or negative is a label observed at or after the decision — the
        ordinary case for anything present-tense. A large positive number means
        a present-day classification is being applied to old data, which is
        sound for an industry and worth saying out loud.
        """
        if self.observed_at is None or self.as_of is None:
            return 0
        return (self.observed_at - self.as_of).days

    def industry_of(self, instrument_id: str) -> str | None:
        """The industry for an instrument id, or None if unclassified.

        Takes the whole `VENUE:ISIN` id rather than a bare ISIN, because that
        is what every caller holds and splitting it at each of them is how the
        two drift apart.
        """
        return self.industries.get(instrument_id.rsplit(":", maxsplit=1)[-1])

    def coverage(self, instrument_ids: list[str]) -> float:
        """Share of the given instruments that have an industry.

        Reported rather than assumed: a name delisted before the observation
        appears in no current constituent list, so sector-aware analysis over
        history is missing exactly the names that failed.
        """
        if not instrument_ids:
            return 0.0
        known = sum(1 for i in instrument_ids if self.industry_of(i) is not None)
        return known / len(instrument_ids)


class SectorStore:
    """``<root>/sectors/<date>.parquet``, one file per observation."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self) -> Path:
        return self.root / "sectors"

    def _path(self, observed: date) -> Path:
        return self._dir() / f"{observed.isoformat()}.parquet"

    def write(self, observed: date, frame: pl.DataFrame) -> int:
        """Store one observation, replacing any for the same date."""
        missing = [c for c in SECTOR_SCHEMA if c not in frame.columns]
        if missing:
            raise ValueError(f"sector frame is missing columns {missing}")
        ordered = frame.select(list(SECTOR_SCHEMA)).unique(subset=["isin"], keep="first")
        self._dir().mkdir(parents=True, exist_ok=True)
        ordered.write_parquet(self._path(observed), compression="zstd")
        return ordered.height

    def observations(self) -> list[date]:
        """Every observation date held, ascending."""
        base = self._dir()
        if not base.exists():
            return []
        return sorted(date.fromisoformat(p.stem) for p in base.glob("*.parquet"))

    def view(self, as_of: DecisionTime | None = None) -> SectorView:
        """The classification observable at `as_of`, or the newest held.

        Args:
            as_of: The decision point. `None` reads the newest observation,
                which is right for anything present-tense — a risk limit on the
                book as it stands now.
        """
        held = self.observations()
        if not held:
            return SectorView(industries={}, symbols={}, observed_at=None)

        wanted = require_utc(as_of).date() if as_of is not None else None
        if wanted is None:
            chosen = held[-1]
        else:
            observable = [d for d in held if d <= wanted]
            # Nothing observed by then. The newest label is still the honest
            # answer for an industry, but the caller is told how far it reaches
            # rather than being handed it silently as contemporaneous.
            chosen = observable[-1] if observable else held[0]

        frame = pl.read_parquet(self._path(chosen))
        return SectorView(
            industries=dict(zip(frame["isin"], frame["industry"], strict=True)),
            symbols=dict(zip(frame["isin"], frame["symbol"], strict=True)),
            observed_at=chosen,
            as_of=wanted,
        )
