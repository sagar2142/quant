"""Instrument master — MASTER_PLAN §1.1.

The abstraction that decides whether a new market is a plugin or a rewrite.
Every market-specific fact lives here so that strategies, the backtester and
the risk engine stay asset-class agnostic.

Two rules carry most of the weight:

1. ``instrument_id`` is internal, stable and never reused. Exchange symbols are
   *not* identity — they get recycled and renamed. NSE reassigned symbols after
   delistings; a company renames and its ticker moves with it.
2. Symbol lookup is always point-in-time. ``resolve("INFOSYSTCH", as_of=2010)``
   and ``resolve("INFY", as_of=2024)`` must reach the same instrument, and
   neither may be answered with today's symbol table (§3.3).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.clock import require_utc

__all__ = [
    "AssetClass",
    "Currency",
    "Exchange",
    "Instrument",
    "InstrumentId",
    "OptionType",
    "SymbolAlias",
    "SymbolResolver",
    "UnknownSymbolError",
]

InstrumentId = NewType("InstrumentId", str)


class AssetClass(str, Enum):
    """Equities and their derivatives. Nothing else (§0.0).

    Deliberately narrow. The system trades stock exchanges, so there is no
    crypto, no FX and no commodity member — an unused enum value is an
    invitation to write code paths nobody tests.
    """

    EQUITY = "EQUITY"
    ETF = "ETF"
    INDEX = "INDEX"
    FUTURE = "FUTURE"
    OPTION = "OPTION"

    @property
    def is_derivative(self) -> bool:
        return self in {AssetClass.FUTURE, AssetClass.OPTION}

    @property
    def is_cash_equity(self) -> bool:
        """Tradable on the cash segment, and settles into a demat account.

        The distinction that decides which cost model applies: cash equities
        attract STT on both legs and DP charges on exit; derivatives do not.
        """
        return self in {AssetClass.EQUITY, AssetClass.ETF}


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    #: Alpaca routes to US venues; the fill venue is not always disclosed, so
    #: it is recorded as the broker rather than pretending to know.
    ARCA = "ARCA"

    @property
    def is_indian(self) -> bool:
        return self in {Exchange.NSE, Exchange.BSE}


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"


class OptionType(str, Enum):
    CALL = "CE"
    PUT = "PE"


class Instrument(BaseModel):
    """A tradable instrument. Immutable once created."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    symbol: str = Field(min_length=1, description="Current venue symbol; NOT identity")
    name: str = ""
    asset_class: AssetClass
    exchange: Exchange
    currency: Currency

    tick_size: Decimal = Field(gt=0, description="Minimum price increment")
    lot_size: int = Field(default=1, ge=1, description="Minimum tradable quantity")
    multiplier: Decimal = Field(
        default=Decimal(1), gt=0, description="Contract size: notional = price * qty * multiplier"
    )

    # Derivatives only.
    expiry: datetime | None = None
    strike: Decimal | None = Field(default=None, gt=0)
    option_type: OptionType | None = None
    underlying_id: InstrumentId | None = None

    # Lifecycle. listed_on/delisted_on are what make survivorship-bias-free
    # universes possible (§M2 gate).
    listed_on: datetime | None = None
    delisted_on: datetime | None = None

    @field_validator("expiry", "listed_on", "delisted_on")
    @classmethod
    def _tz_aware(cls, v: datetime | None) -> datetime | None:
        return None if v is None else require_utc(v)

    @model_validator(mode="after")
    def _check_derivative_fields(self) -> Instrument:
        if self.asset_class.is_derivative and self.expiry is None:
            raise ValueError(f"{self.asset_class} requires an expiry")
        if self.asset_class is AssetClass.OPTION:
            if self.strike is None or self.option_type is None:
                raise ValueError("OPTION requires both strike and option_type")
        elif self.strike is not None or self.option_type is not None:
            raise ValueError(f"{self.asset_class} must not carry strike/option_type")
        if (
            self.listed_on is not None
            and self.delisted_on is not None
            and self.delisted_on < self.listed_on
        ):
            raise ValueError("delisted_on precedes listed_on")
        return self

    def is_tradable_on(self, ts: datetime) -> bool:
        """Whether this instrument existed and was tradable at `ts`.

        The guard against survivorship bias: a 2019 backtest must be able to
        hold a company that was delisted in 2022, and must not hold one that
        listed in 2023.
        """
        ts = require_utc(ts)
        if self.listed_on is not None and ts < self.listed_on:
            return False
        if self.delisted_on is not None and ts >= self.delisted_on:
            return False
        if self.expiry is not None and ts >= self.expiry:
            return False
        return True

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Snap a price to the venue's tick grid. Orders off-tick get rejected."""
        return (price / self.tick_size).quantize(Decimal(1)) * self.tick_size

    def notional(self, price: Decimal, quantity: int) -> Decimal:
        """Cash value of a position. Decimal throughout (§14.1.2)."""
        return price * Decimal(quantity) * self.multiplier


class SymbolAlias(BaseModel):
    """A symbol an instrument was known by, over a validity window.

    `valid_to = None` means "still current".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    symbol: str = Field(min_length=1)
    exchange: Exchange
    valid_from: datetime
    valid_to: datetime | None = None

    @field_validator("valid_from", "valid_to")
    @classmethod
    def _tz_aware(cls, v: datetime | None) -> datetime | None:
        return None if v is None else require_utc(v)

    @model_validator(mode="after")
    def _check_window(self) -> SymbolAlias:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self

    def covers(self, ts: datetime) -> bool:
        ts = require_utc(ts)
        if ts < self.valid_from:
            return False
        return self.valid_to is None or ts < self.valid_to


class UnknownSymbolError(LookupError):
    """No instrument carried this symbol at this time.

    Raised rather than returning None (§14.1.5): a silently unresolved symbol
    becomes a silently skipped position.
    """

    def __init__(self, symbol: str, exchange: Exchange, as_of: datetime) -> None:
        super().__init__(
            f"no instrument for symbol {symbol!r} on {exchange.value} as of {as_of.isoformat()}"
        )


class SymbolResolver:
    """Point-in-time symbol → instrument_id lookup.

    Resolution is always asked *as of* a timestamp. There is no method to
    resolve "now" without saying so, because that is how a 2015 backtest ends
    up using 2026 symbol mappings.
    """

    def __init__(self, aliases: list[SymbolAlias]) -> None:
        self._by_key: dict[tuple[str, Exchange], list[SymbolAlias]] = {}
        for alias in aliases:
            self._by_key.setdefault((alias.symbol.upper(), alias.exchange), []).append(alias)

    def resolve(self, symbol: str, exchange: Exchange, as_of: datetime) -> InstrumentId:
        """Return the instrument that carried `symbol` at `as_of`.

        Raises:
            UnknownSymbolError: if no alias window covers `as_of`.
        """
        ts = require_utc(as_of)
        for alias in self._by_key.get((symbol.upper(), exchange), ()):
            if alias.covers(ts):
                return alias.instrument_id
        raise UnknownSymbolError(symbol, exchange, ts)

    def symbols_for(self, instrument_id: InstrumentId, as_of: datetime) -> list[str]:
        """Every symbol this instrument was known by at `as_of`."""
        ts = require_utc(as_of)
        return sorted(
            alias.symbol
            for aliases in self._by_key.values()
            for alias in aliases
            if alias.instrument_id == instrument_id and alias.covers(ts)
        )
