"""NSE index closes — MASTER_PLAN §6, §13.4.

**This exists because the risk model's market factor was a column of ones.**
`build_risk_model` needs a market return to regress against, and with no index
series in the lake it used a constant — which is not a market, it is a
placeholder that happens to have the right shape. Every beta, every
market-neutral result and every regime split computed against it was
approximate in a way nothing on screen admitted.

The central finding of the research so far is that a signal is 72.7% market
beta. That number deserves a real market.

NSE publishes one file per session covering every index it computes, from 2019
through today — the same span as the equity panel, so a beta can be estimated
over exactly the history a backtest trades.

**Stored as its own series, not as a row in the panel.** An index is not a
security: it cannot be held, it has no ISIN, and putting it in the
cross-section would let a factor rank it alongside the stocks it is built from.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from core.clock import UTC

__all__ = [
    "BENCHMARK",
    "IndexFormatError",
    "nse_index_url",
    "parse_index_close",
]

#: The default benchmark. NIFTY 50 is the index Indian equity beta is quoted
#: against, and the one a market-neutral book is neutral *to*.
BENCHMARK = "Nifty 50"

#: Columns the file must carry. Named so a layout change fails here, loudly,
#: rather than producing a frame of nulls that regresses to nothing.
REQUIRED_COLUMNS = (
    "Index Name",
    "Index Date",
    "Open Index Value",
    "High Index Value",
    "Low Index Value",
    "Closing Index Value",
)


class IndexFormatError(ValueError):
    """The file was not an NSE index close. Never guessed at."""


@dataclass(frozen=True)
class IndexSession:
    """One session of index closes."""

    session_date: date
    rows: pl.DataFrame


def nse_index_url(session_date: date) -> str:
    """Daily close for every index NSE computes.

    Note the date format differs from the bhavcopy: DDMMYYYY here against
    YYYYMMDD there. Getting it wrong returns a 404 rather than the wrong day,
    which is the better failure but still a surprise worth naming.
    """
    return (
        "https://nsearchives.nseindia.com/content/indices/"
        f"ind_close_all_{session_date.strftime('%d%m%Y')}.csv"
    )


def parse_index_close(payload: bytes, session_date: date) -> IndexSession:
    """Parse one index close file.

    Args:
        payload: The CSV as downloaded.
        session_date: The session it describes, used for the timestamps rather
            than read from the file — a mislabelled download should not write
            itself into the wrong session.

    Raises:
        IndexFormatError: if the layout is not the expected one, or the file is
            a web page rather than data.
    """
    head = payload[:512].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html")):
        raise IndexFormatError(
            f"NSE returned a web page rather than index closes for {session_date}"
        )

    try:
        frame = pl.read_csv(io.BytesIO(payload), infer_schema_length=2000)
    except Exception as exc:
        raise IndexFormatError(f"could not read {session_date} as CSV: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise IndexFormatError(f"{session_date} is missing columns {missing}")

    event_time = datetime(
        session_date.year, session_date.month, session_date.day, 10, 0, tzinfo=UTC
    )
    receive_time = datetime(
        session_date.year, session_date.month, session_date.day, 12, 30, tzinfo=UTC
    )

    rows = (
        frame.with_columns(
            pl.lit(event_time).alias("event_time"),
            pl.lit(receive_time).alias("receive_time"),
            pl.col("Index Name").str.strip_chars().alias("index_name"),
            # A holiday prints a row with a dash instead of a number. Cast
            # non-strictly so those become null and are dropped, rather than
            # failing the whole session.
            pl.col("Open Index Value").cast(pl.Float64, strict=False).alias("open"),
            pl.col("High Index Value").cast(pl.Float64, strict=False).alias("high"),
            pl.col("Low Index Value").cast(pl.Float64, strict=False).alias("low"),
            pl.col("Closing Index Value").cast(pl.Float64, strict=False).alias("close"),
        )
        .filter(pl.col("close").is_not_null() & (pl.col("close") > 0))
        .select(
            "event_time",
            "receive_time",
            "index_name",
            "open",
            "high",
            "low",
            "close",
        )
        .sort("index_name")
    )
    return IndexSession(session_date, rows)
