"""NSE bhavcopy loader — MASTER_PLAN §13.4, §M2.

The daily bhavcopy is NSE's end-of-day file listing every security that traded
that session. That property is what makes it the free path to a
survivorship-bias-free universe: **a stock delisted in 2022 is still present in
every bhavcopy up to its last trading day.** Union the archive across dates and
you have the real historical universe, including the failures — which is
precisely what a 2019 backtest must be able to hold (§M2 gate).

Two format eras are supported:

    legacy  cm01JAN2020bhav.csv     SYMBOL, SERIES, OPEN, HIGH, ...
    UDiFF   BhavCopy_NSE_CM_...csv  TckrSymb, SctySrs, OpnPric, ...

**Timestamps.** `event_time` is the session close, 15:30 IST = 10:00 UTC.
`receive_time` is when the file is actually published, ~18:00 IST. They are
never equal: a strategy cannot act on today's close at today's close (§7.6).

**ISIN is the stable identity**, not the symbol. Symbols get renamed and
recycled; the ISIN persists, which is what makes point-in-time symbol
resolution possible (§1.1).

**Access note.** NSE serves these files for personal use but rate-limits and
requires browser-like headers. Review NSE's terms before automating bulk
downloads (§32). The parser is fully usable offline against files fetched by
any means.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import polars as pl

from core.calendars import IST
from core.clock import UTC
from data.store.bars import BAR_SCHEMA

__all__ = [
    "PUBLICATION_LAG",
    "SESSION_CLOSE_IST",
    "BhavcopyDay",
    "Listing",
    "legacy_url",
    "parse_bhavcopy",
    "udiff_url",
]

SESSION_CLOSE_IST = time(15, 30)

#: Bhavcopy lands ~2.5h after the close. Conservative: too long is merely
#: pessimistic, too short is look-ahead.
PUBLICATION_LAG = timedelta(hours=2, minutes=30)

#: Series worth trading. EQ is the rolling-settlement equity segment; BE is the
#: trade-for-trade segment (illiquid, no intraday netting) and is excluded by
#: default because fills there are not realistic.
DEFAULT_SERIES = ("EQ",)

_LEGACY_COLUMNS = {
    "symbol": "SYMBOL",
    "series": "SERIES",
    "open": "OPEN",
    "high": "HIGH",
    "low": "LOW",
    "close": "CLOSE",
    "volume": "TOTTRDQTY",
    "value": "TOTTRDVAL",
    "trades": "TOTALTRADES",
    "isin": "ISIN",
}

_UDIFF_COLUMNS = {
    "symbol": "TckrSymb",
    "series": "SctySrs",
    "open": "OpnPric",
    "high": "HghPric",
    "low": "LwPric",
    "close": "ClsPric",
    "volume": "TtlTradgVol",
    "value": "TtlTrfVal",
    "trades": "TtlNbOfTxsExctd",
    "isin": "ISIN",
}

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


@dataclass(frozen=True)
class Listing:
    """A security observed trading on a given session.

    The accumulation of these across the archive is the instrument master, and
    the last session a symbol appears is its de-facto delisting date.
    """

    symbol: str
    isin: str
    series: str
    session_date: date


@dataclass
class BhavcopyDay:
    """One parsed session."""

    session_date: date
    bars: pl.DataFrame  # BAR_SCHEMA + symbol, isin
    listings: list[Listing]

    @property
    def symbols(self) -> set[str]:
        return {listing.symbol for listing in self.listings}


class BhavcopyFormatError(ValueError):
    """The file did not match either known layout. Never guessed at (§14.1.5)."""


def _session_timestamps(session_date: date) -> tuple[datetime, datetime]:
    close_ist = datetime.combine(session_date, SESSION_CLOSE_IST, tzinfo=IST)
    event = close_ist.astimezone(UTC)
    return event, event + PUBLICATION_LAG


def _detect_layout(columns: set[str]) -> dict[str, str]:
    if _LEGACY_COLUMNS["symbol"] in columns:
        return _LEGACY_COLUMNS
    if _UDIFF_COLUMNS["symbol"] in columns:
        return _UDIFF_COLUMNS
    raise BhavcopyFormatError(f"unrecognised bhavcopy layout; columns were {sorted(columns)[:12]}")


def _read_csv(payload: bytes) -> pl.DataFrame:
    """Read a bhavcopy, transparently unzipping if needed."""
    if not payload:
        raise BhavcopyFormatError("empty payload — download truncated or file missing")
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise BhavcopyFormatError("zip contains no CSV")
            payload = archive.read(names[0])
    return pl.read_csv(io.BytesIO(payload), infer_schema_length=0)


def parse_bhavcopy(
    payload: bytes,
    session_date: date,
    series: tuple[str, ...] = DEFAULT_SERIES,
) -> BhavcopyDay:
    """Parse one bhavcopy into bars and listings.

    Args:
        payload: Raw CSV or ZIP bytes.
        session_date: The trading session this file covers.
        series: Which security series to keep. Empty tuple keeps all.

    Raises:
        BhavcopyFormatError: on an unrecognised layout or an empty result.
    """
    raw = _read_csv(payload)
    if raw.is_empty():
        raise BhavcopyFormatError(f"bhavcopy for {session_date} is empty")

    layout = _detect_layout(set(raw.columns))
    event_time, receive_time = _session_timestamps(session_date)

    frame = raw.rename({v: k for k, v in layout.items() if v in raw.columns})
    frame = frame.with_columns(
        pl.col("symbol").str.strip_chars(),
        pl.col("series").str.strip_chars(),
        pl.col("isin").str.strip_chars() if "isin" in frame.columns else pl.lit("").alias("isin"),
    )
    if series:
        frame = frame.filter(pl.col("series").is_in(list(series)))
    if frame.is_empty():
        raise BhavcopyFormatError(f"bhavcopy for {session_date} has no rows in series {series}")

    numeric = ["open", "high", "low", "close", "volume"]
    frame = frame.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in numeric]
        + [pl.col("trades").cast(pl.Int64, strict=False).fill_null(0)]
    ).drop_nulls(subset=numeric)

    # A zero or negative print is bad data, not a tradable price. Dropping is
    # correct here and the count surfaces in the quality report.
    frame = frame.filter(pl.col("low") > 0)

    bars = frame.with_columns(
        pl.lit(event_time).cast(BAR_SCHEMA["event_time"]).alias("event_time"),
        pl.lit(receive_time).cast(BAR_SCHEMA["receive_time"]).alias("receive_time"),
    ).select([*BAR_SCHEMA.keys(), "symbol", "isin"])

    listings = [
        Listing(
            symbol=row["symbol"], isin=row["isin"], series=row["series"], session_date=session_date
        )
        for row in frame.select("symbol", "isin", "series").to_dicts()
    ]
    return BhavcopyDay(session_date=session_date, bars=bars, listings=listings)


def legacy_url(session_date: date) -> str:
    """Archive URL for the pre-2024 bhavcopy layout."""
    month = _MONTHS[session_date.month - 1]
    return (
        "https://archives.nseindia.com/content/historical/EQUITIES/"
        f"{session_date.year}/{month}/"
        f"cm{session_date.day:02d}{month}{session_date.year}bhav.csv.zip"
    )


def udiff_url(session_date: date) -> str:
    """Archive URL for the current UDiFF bhavcopy layout."""
    stamp = session_date.strftime("%Y%m%d")
    return (
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{stamp}_F_0000.csv.zip"
    )


def nse_instrument_id(isin: str, symbol: str) -> str:
    """Canonical internal id.

    Prefers ISIN: symbols are renamed and recycled, ISINs are not. Falls back to
    the symbol only when a historical file carries no ISIN column.
    """
    return f"NSE:{isin}" if isin else f"NSE:{symbol.upper()}"
