"""NSE derivatives bhavcopy — MASTER_PLAN §13.4.

**A derivative is not a security with extra columns.** An equity is identified
by its ISIN and trades in one series; an option is identified by underlying,
expiry, strike and right, and there are twenty-four thousand of them on a
single NSE session against three thousand equities. They cannot share the panel
store: a cross-section keyed on instrument id would carry every strike of every
expiry as a separate name, and every factor in the library would score them.

So this parses into its own frame with its own identity, and nothing in
`quant/` sees it unless it asks.

**The file already answers most of what an option screen needs.** Strike,
expiry, right, open interest and its change, the underlying's own price, the
settlement price and the board lot are all columns. What is *not* in the file
is implied volatility or any Greek — those are computed, and computing them
requires assumptions (a rate, a dividend yield, a model) that belong where they
can be seen rather than buried in a loader.

**Four instrument types, and the distinction matters.** `STO` and `IDO` are
stock and index options; `STF` and `IDF` are the futures. An index option has
no ISIN because an index is not a security you can hold, which is exactly why
identity here is built from the contract terms rather than from ISIN.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from core.clock import UTC

__all__ = [
    "FO_TYPES",
    "OPTION_TYPES",
    "DerivativesFormatError",
    "contract_id",
    "nse_fo_url",
    "parse_fo_bhavcopy",
]

#: Instrument types in the F&O file.
#:
#:     STO  stock option      IDO  index option
#:     STF  stock future      IDF  index future
FO_TYPES: tuple[str, ...] = ("STO", "IDO", "STF", "IDF")

#: The option types. Futures carry an empty right, which is how they are told
#: apart from options without consulting `FinInstrmTp` twice.
OPTION_TYPES: tuple[str, ...] = ("CE", "PE")

#: Columns the parser requires. Named explicitly so a layout change fails here,
#: loudly, rather than producing a frame with silently missing strikes.
REQUIRED_COLUMNS = (
    "TradDt",
    "FinInstrmTp",
    "TckrSymb",
    "XpryDt",
    "StrkPric",
    "OptnTp",
    "OpnPric",
    "HghPric",
    "LwPric",
    "ClsPric",
    "SttlmPric",
    "UndrlygPric",
    "OpnIntrst",
    "ChngInOpnIntrst",
    "TtlTradgVol",
    "TtlNbOfTxsExctd",
    "NewBrdLotQty",
)


class DerivativesFormatError(ValueError):
    """The file did not match the UDiFF derivatives layout. Never guessed at."""


@dataclass(frozen=True)
class DerivativesSession:
    """One session of derivatives, parsed."""

    session_date: date
    contracts: pl.DataFrame


def nse_fo_url(session_date: date) -> str:
    """Daily F&O bhavcopy, in the current UDiFF layout."""
    stamp = session_date.strftime("%Y%m%d")
    return (
        f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{stamp}_F_0000.csv.zip"
    )


def contract_id(symbol: str, expiry: date, strike: float, right: str) -> str:
    """Canonical id for one contract.

    **Built from the terms, not from an ISIN.** Index options have no ISIN
    because an index is not a holdable security, and two contracts on the same
    underlying differ only by expiry, strike and right — so those four fields
    *are* the identity. The strike is formatted to a fixed precision so that
    2500 and 2500.0 are the same contract rather than two.

    A future has no strike and no right, and reads as `NFO:RELIANCE:20260924:FUT`.
    """
    stamp = expiry.strftime("%Y%m%d")
    if right not in OPTION_TYPES:
        return f"NFO:{symbol}:{stamp}:FUT"
    return f"NFO:{symbol}:{stamp}:{strike:.2f}:{right}"


def _session_timestamps(session_date: date) -> tuple[datetime, datetime]:
    """Event and receive time for a session.

    The same discipline the equity loader keeps (§3.3): a bhavcopy describes a
    session that closed at 15:30 and is published afterwards, so `receive_time`
    is later than `event_time` and a decision made at the close cannot read it.
    """
    event = datetime(session_date.year, session_date.month, session_date.day, 10, 0, tzinfo=UTC)
    receive = datetime(session_date.year, session_date.month, session_date.day, 12, 30, tzinfo=UTC)
    return event, receive


def _unzip(payload: bytes) -> bytes:
    """The archive holds one CSV. Returns it, or the payload if it is not a zip."""
    if payload[:2] != b"PK":
        return payload
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise DerivativesFormatError("archive contains no CSV")
        return archive.read(names[0])


def parse_fo_bhavcopy(
    payload: bytes,
    session_date: date,
    types: tuple[str, ...] = FO_TYPES,
) -> DerivativesSession:
    """Parse one derivatives bhavcopy.

    Args:
        payload: Zipped or bare CSV, as downloaded.
        session_date: The session this file describes, used for the timestamps
            rather than read from the file — a mislabelled download should not
            silently write itself into the wrong session.
        types: Which instrument types to keep.

    Raises:
        DerivativesFormatError: if the layout is not the expected one, or the
            file is a web page rather than data.
    """
    body = _unzip(payload)
    head = body[:512].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html")):
        raise DerivativesFormatError(
            f"NSE returned a web page rather than a derivatives bhavcopy for {session_date}"
        )

    try:
        frame = pl.read_csv(io.BytesIO(body), infer_schema_length=10_000)
    except Exception as exc:
        raise DerivativesFormatError(f"could not read {session_date} as CSV: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DerivativesFormatError(f"{session_date} is missing columns {missing}")

    event_time, receive_time = _session_timestamps(session_date)
    kept = frame.filter(pl.col("FinInstrmTp").is_in(list(types)))
    if kept.is_empty():
        return DerivativesSession(session_date, kept.head(0))

    contracts = (
        kept.with_columns(
            pl.lit(event_time).alias("event_time"),
            pl.lit(receive_time).alias("receive_time"),
            pl.col("TckrSymb").str.strip_chars().alias("underlying"),
            pl.col("FinInstrmTp").alias("instrument_type"),
            pl.col("XpryDt").str.strip_chars().str.to_date("%Y-%m-%d").alias("expiry"),
            pl.col("StrkPric").cast(pl.Float64).alias("strike"),
            # Futures carry no right. Normalised to an empty string rather than
            # null so grouping and comparison do not have to special-case it.
            pl.col("OptnTp").fill_null("").str.strip_chars().alias("right"),
            pl.col("OpnPric").cast(pl.Float64).alias("open"),
            pl.col("HghPric").cast(pl.Float64).alias("high"),
            pl.col("LwPric").cast(pl.Float64).alias("low"),
            pl.col("ClsPric").cast(pl.Float64).alias("close"),
            pl.col("SttlmPric").cast(pl.Float64).alias("settlement"),
            pl.col("UndrlygPric").cast(pl.Float64).alias("underlying_price"),
            pl.col("OpnIntrst").cast(pl.Float64).alias("open_interest"),
            pl.col("ChngInOpnIntrst").cast(pl.Float64).alias("oi_change"),
            pl.col("TtlTradgVol").cast(pl.Float64).alias("volume"),
            pl.col("TtlNbOfTxsExctd").cast(pl.Float64).alias("trades"),
            pl.col("NewBrdLotQty").cast(pl.Float64).alias("lot_size"),
        )
        .with_columns(
            pl.struct(["underlying", "expiry", "strike", "right"])
            .map_elements(
                lambda row: contract_id(
                    row["underlying"], row["expiry"], row["strike"] or 0.0, row["right"]
                ),
                return_dtype=pl.String,
            )
            .alias("contract_id")
        )
        .select(
            "event_time",
            "receive_time",
            "contract_id",
            "underlying",
            "instrument_type",
            "expiry",
            "strike",
            "right",
            "open",
            "high",
            "low",
            "close",
            "settlement",
            "underlying_price",
            "open_interest",
            "oi_change",
            "volume",
            "trades",
            "lot_size",
        )
        .sort("contract_id")
    )
    return DerivativesSession(session_date, contracts)
