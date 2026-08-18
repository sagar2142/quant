"""Corporate actions — MASTER_PLAN §9, PDF §9.

**Design decision: bars are stored RAW, exactly as the exchange printed them.**

The common shortcut is to store a back-adjusted series: walk every known split
and dividend backwards and rewrite history. It is convenient and it is subtly
wrong, in two ways:

1. *It embeds future information.* A back-adjusted 2015 price reflects a 2019
   split. A backtest deciding in 2015 is reading a number that could not have
   existed in 2015. The distortion is small per event and compounds silently.
2. *History mutates.* Every new corporate action rewrites every prior price, so
   a backtest run today cannot reproduce one run last year — which breaks the
   M3 reproducibility gate outright.

So: raw prices in the lake, corporate actions as separate dated events, and the
backtester applies them to *positions* as they occur — a split multiplies your
share count on the ex-date, a dividend credits cash. That is what actually
happens to a real portfolio, and it carries no look-ahead.

`back_adjust()` exists for charting and for comparing against vendor series. It
is explicitly labelled as containing future information and must never feed a
backtest.

**Announcement date versus ex-date.** Both are kept. You learn about an action
on its announcement date; it takes effect on its ex-date. A strategy that
positions for a split must only know about it from the announcement onward
(§3.3).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.clock import require_utc
from core.instruments import InstrumentId

__all__ = [
    "ActionType",
    "CorporateAction",
    "CorporateActionBook",
    "back_adjust",
]


class ActionType(str, Enum):
    SPLIT = "SPLIT"
    BONUS = "BONUS"
    DIVIDEND = "DIVIDEND"
    RIGHTS = "RIGHTS"
    MERGER = "MERGER"
    DELISTING = "DELISTING"

    @property
    def changes_share_count(self) -> bool:
        return self in {ActionType.SPLIT, ActionType.BONUS}


class CorporateAction(BaseModel):
    """One dated corporate action.

    `ratio` is expressed as **new shares per existing share**:

        2-for-1 split      ratio = 2      (100 shares become 200, price halves)
        1:1 bonus issue    ratio = 2      (one free share per share held)
        3:1 bonus issue    ratio = 4      (three free shares per share held)
        reverse 1-for-10   ratio = 0.1

    `cash_per_share` carries dividends, in the instrument's currency.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: InstrumentId
    action_type: ActionType
    ex_date: datetime
    announcement_date: datetime | None = None

    ratio: Decimal = Field(default=Decimal(1), gt=0)
    cash_per_share: Decimal = Field(default=Decimal(0), ge=0)
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> CorporateAction:
        object.__setattr__(self, "ex_date", require_utc(self.ex_date))
        if self.announcement_date is not None:
            announced = require_utc(self.announcement_date)
            object.__setattr__(self, "announcement_date", announced)
            if announced > self.ex_date:
                raise ValueError(
                    f"announcement {announced.date()} is after ex-date "
                    f"{self.ex_date.date()}; an action cannot take effect "
                    "before it is known"
                )
        if self.action_type.changes_share_count and self.ratio == 1:
            raise ValueError(f"{self.action_type} with ratio 1 has no effect")
        if self.action_type is ActionType.DIVIDEND and self.cash_per_share == 0:
            raise ValueError("DIVIDEND requires a non-zero cash_per_share")
        return self

    @property
    def quantity_multiplier(self) -> Decimal:
        """Factor applied to a held position on the ex-date."""
        return self.ratio if self.action_type.changes_share_count else Decimal(1)

    @property
    def price_multiplier(self) -> Decimal:
        """Factor by which the quoted price mechanically changes on the ex-date.

        Exactly the reciprocal of the quantity change, so position value is
        preserved across the event.
        """
        return Decimal(1) / self.ratio if self.action_type.changes_share_count else Decimal(1)

    def known_at(self, ts: datetime) -> bool:
        """Whether this action was public at `ts`.

        Falls back to the ex-date when no announcement date is recorded — the
        conservative choice, since assuming earlier knowledge would grant
        look-ahead.
        """
        reference = self.announcement_date or self.ex_date
        return require_utc(ts) >= reference


class CorporateActionBook:
    """Actions for a set of instruments, queryable point-in-time."""

    def __init__(self, actions: list[CorporateAction]) -> None:
        self._by_instrument: dict[InstrumentId, list[CorporateAction]] = {}
        for action in sorted(actions, key=lambda a: a.ex_date):
            self._by_instrument.setdefault(action.instrument_id, []).append(action)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_instrument.values())

    @property
    def instruments(self) -> frozenset[InstrumentId]:
        """Instruments with at least one recorded action.

        Coverage is not the same as membership: a name absent here either had no
        actions or was never fetched, and only the caller knows which.
        """
        return frozenset(self._by_instrument)

    def for_instrument(self, instrument_id: InstrumentId) -> list[CorporateAction]:
        return list(self._by_instrument.get(instrument_id, ()))

    def known_at(self, instrument_id: InstrumentId, as_of: datetime) -> list[CorporateAction]:
        """Actions whose existence was public at `as_of`, whether or not effective."""
        return [a for a in self.for_instrument(instrument_id) if a.known_at(as_of)]

    def effective_between(
        self, instrument_id: InstrumentId, start: datetime, end: datetime
    ) -> list[CorporateAction]:
        """Actions with an ex-date in (start, end].

        Half-open at the start so that stepping a backtest bar by bar applies
        each action exactly once.
        """
        lo, hi = require_utc(start), require_utc(end)
        return [a for a in self.for_instrument(instrument_id) if lo < a.ex_date <= hi]

    def cumulative_quantity_factor(
        self, instrument_id: InstrumentId, start: datetime, end: datetime
    ) -> Decimal:
        """Combined share-count multiplier over (start, end]."""
        factor = Decimal(1)
        for action in self.effective_between(instrument_id, start, end):
            factor *= action.quantity_multiplier
        return factor


def back_adjust(
    bars: pl.DataFrame,
    actions: list[CorporateAction],
) -> pl.DataFrame:
    """Return a back-adjusted price series.

    **Contains future information by construction — never feed this to a
    backtest.** Prices before each ex-date are scaled so the split-induced jump
    disappears, which means a 2015 row reflects a 2019 event.

    Legitimate uses: charting, and reconciling against a vendor's adjusted
    series to confirm the corporate-action book is complete.
    """
    if bars.is_empty() or not actions:
        return bars

    price_columns = ["open", "high", "low", "close"]
    adjusted = bars
    for action in sorted(actions, key=lambda a: a.ex_date, reverse=True):
        factor = float(action.price_multiplier)
        if factor == 1.0:
            continue
        before = pl.col("event_time") < pl.lit(action.ex_date)
        adjusted = adjusted.with_columns(
            [
                pl.when(before)
                .then(pl.col(column) * factor)
                .otherwise(pl.col(column))
                .alias(column)
                for column in price_columns
            ]
            # Volume moves the other way: more shares outstanding, more traded.
            + [
                pl.when(before)
                .then(pl.col("volume") / factor)
                .otherwise(pl.col("volume"))
                .alias("volume")
            ]
        )
    return adjusted
