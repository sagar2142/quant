"""Backtest inputs and outputs — MASTER_PLAN §14.

Separated from the event loop so that `engine.py` contains only the replay
logic. These are the things a caller supplies and receives; the loop is the
thing that turns one into the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import cast

import polars as pl

from core.instruments import Instrument, InstrumentId
from data.corpactions.actions import CorporateActionBook
from engine.accounting import Portfolio
from engine.backtest.fills import FillModel
from engine.costs.model import CostModel

__all__ = [
    "MIN_ORDER_VALUE",
    "BacktestConfig",
    "BacktestResult",
    "MarketModel",
    "RunState",
    "validate_history",
]

#: Below this, an order is not worth its fixed costs; DP charges alone would
#: dominate. Prevents the backtester generating dust trades no operator would
#: ever place (§7.1).
MIN_ORDER_VALUE = Decimal(500)


@dataclass(frozen=True)
class MarketModel:
    """How a market behaves: what trading costs, how orders fill, what exists.

    Grouped rather than passed as four separate collaborators because they are
    one decision — "simulate NSE delivery" or "simulate Binance spot" — and
    changing one without the others is almost always a mistake.
    """

    cost_model: CostModel
    fill_model: FillModel
    instruments: dict[InstrumentId, Instrument]
    actions: CorporateActionBook = field(default_factory=lambda: CorporateActionBook([]))


@dataclass(frozen=True)
class BacktestConfig:
    """Capital, leverage and turnover controls."""

    initial_cash: Decimal = Decimal(1_000_000)
    #: Cash may fall this far below zero. Zero means no leverage at all.
    margin_allowance: Decimal = Decimal(0)
    #: Skip rebalancing a name whose target differs from its holding by less
    #: than this fraction of NAV (§7.1).
    rebalance_threshold: Decimal = Decimal("0.005")
    min_order_value: Decimal = MIN_ORDER_VALUE
    allow_partial_fills: bool = True
    #: Cash held back when sizing buys, covering fees and the gap between the
    #: decision price and the fill price.
    cost_headroom: Decimal = Decimal("0.005")


@dataclass
class BacktestResult:
    """Everything a backtest produced. Serialised onto the experiment row."""

    equity_curve: pl.DataFrame
    trades: pl.DataFrame
    final_portfolio: Portfolio
    config: BacktestConfig
    strategy_fingerprint: dict[str, object]
    bars_processed: int = 0
    orders_generated: int = 0
    orders_filled: int = 0
    #: Something went wrong: an illegal state, an accounting refusal. Should be
    #: zero in a healthy run, which is why it is a golden-number assertion.
    orders_rejected: int = 0
    #: The instrument did not trade on the execution session — delisted, halted,
    #: or absent from that day's cross-section. The market was not there; this
    #: says nothing about our logic.
    orders_no_market: int = 0
    #: The account ran out of cash before this order. Expected, not exceptional:
    #: a strategy targeting 100% gross has no buffer, so the final name in a
    #: rebalance is one adverse tick from being unaffordable. Counting it as a
    #: rejection would make a fully-invested book look permanently broken.
    orders_unfunded: int = 0
    #: Bars where a fill was wanted but liquidity refused it.
    liquidity_failures: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        if self.equity_curve.is_empty():
            return 0.0
        first = float(self.equity_curve["equity"][0])
        last = float(self.equity_curve["equity"][-1])
        return last / first - 1 if first else 0.0

    @property
    def fill_rate(self) -> float:
        if self.orders_generated == 0:
            return 0.0
        return self.orders_filled / self.orders_generated

    @property
    def max_drawdown(self) -> float:
        """Peak-to-trough decline. Always <= 0."""
        if self.equity_curve.is_empty():
            return 0.0
        equity = self.equity_curve["equity"]
        peak = equity.cum_max()
        # Polars scalars are loosely typed; the column is Float64 by schema.
        return cast("float", ((equity - peak) / peak).min() or 0.0)


@dataclass
class RunState:
    """Mutable bookkeeping for one run. Internal to the engine."""

    portfolio: Portfolio
    result: BacktestResult
    trades: list[dict[str, object]] = field(default_factory=list)
    equity: list[dict[str, object]] = field(default_factory=list)


#: Columns the engine replay depends on. Everything else rides along untouched.
REQUIRED_HISTORY_COLUMNS = frozenset(
    {"event_time", "receive_time", "instrument_id", "open", "high", "low", "close", "volume"}
)


def validate_history(history: pl.DataFrame) -> None:
    """Refuse a frame the engine cannot honestly replay.

    Raises:
        ValueError: on missing columns or an empty frame. Loud and early — a
            missing `receive_time` discovered mid-run would mean part of the
            curve was built without point-in-time filtering.
    """
    missing = REQUIRED_HISTORY_COLUMNS - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")
    if history.is_empty():
        raise ValueError("cannot backtest an empty history")
