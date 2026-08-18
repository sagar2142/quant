"""Unified market data events — MASTER_PLAN §1.3 and PDF §10.

Every feed normalises into these types, so strategies never learn which venue
their data came from. Adding a market means writing an adapter, not touching
anything downstream.

**On Decimal vs float here.** These objects are the canonical record: venues
publish prices as exact decimal strings, and a price that round-trips through
float64 can no longer be compared to a broker's contract note. So `Bar` and
`Tick` carry `Decimal`.

That is *not* the research path. Bulk research reads Parquet into Polars/NumPy
float64 columns and never materialises these objects — millions of Decimal
constructions would be pointlessly slow. The split is deliberate (§14.1.2):

    event objects   exact, low volume, live path, reconcilable
    columnar frames float64, high volume, research path, statistical
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.clock import EventTime, ReceiveTime
from core.instruments import InstrumentId

__all__ = [
    "Bar",
    "CausalityError",
    "EventType",
    "Tick",
    "Timeframe",
]


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.M30: 1800,
            Timeframe.H1: 3600,
            Timeframe.H4: 14400,
            Timeframe.D1: 86400,
        }[self]


class EventType(str, Enum):
    BAR = "BAR"
    TRADE = "TRADE"
    QUOTE = "QUOTE"


class CausalityError(ValueError):
    """An event claims to have been received before it happened.

    Almost always clock drift between the venue and this machine, occasionally
    a bug in a feed adapter. Either way the timestamps cannot be trusted for
    look-ahead enforcement (§3.3), so construction fails rather than silently
    producing an unsound event stream.
    """

    def __init__(self, event_time: EventTime, receive_time: ReceiveTime) -> None:
        super().__init__(
            f"receive_time {receive_time.isoformat()} precedes event_time "
            f"{event_time.isoformat()} — clock drift or adapter bug"
        )


class _MarketEvent(BaseModel):
    """Shared spine of every market data event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    source: str = Field(min_length=1, description="Feed adapter that produced this")
    event_time: EventTime
    receive_time: ReceiveTime
    sequence: int | None = Field(default=None, description="Venue sequence, for gap detection")

    @model_validator(mode="after")
    def _check_causality(self) -> _MarketEvent:
        if self.receive_time < self.event_time:
            raise CausalityError(self.event_time, self.receive_time)
        return self


class Bar(_MarketEvent):
    """An OHLCV bar. `event_time` is the bar's CLOSE, never its open.

    This convention is load-bearing: a bar stamped with its open time can be
    observed before it has finished forming, which is look-ahead. Every adapter
    converts to close-stamped on ingest.
    """

    timeframe: Timeframe
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)
    trades: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_ohlc(self) -> Bar:
        if self.high < self.low:
            raise ValueError(f"high {self.high} below low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open {self.open} outside [{self.low}, {self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close {self.close} outside [{self.low}, {self.high}]")
        return self

    @property
    def typical_price(self) -> Decimal:
        return (self.high + self.low + self.close) / 3

    @property
    def is_doji(self) -> bool:
        """Zero-range bar. Common in illiquid names and a slippage-model trap."""
        return self.high == self.low


class Tick(_MarketEvent):
    """A trade print or a top-of-book quote update."""

    event_type: EventType = EventType.TRADE
    price: Decimal | None = Field(default=None, gt=0)
    quantity: Decimal | None = Field(default=None, ge=0)
    bid: Decimal | None = Field(default=None, gt=0)
    ask: Decimal | None = Field(default=None, gt=0)
    bid_size: Decimal | None = Field(default=None, ge=0)
    ask_size: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_book(self) -> Tick:
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError(f"crossed book: bid {self.bid} above ask {self.ask}")
        return self

    @property
    def spread(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def mid(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2
