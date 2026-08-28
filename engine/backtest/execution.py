"""Order execution inside the backtest loop — MASTER_PLAN §14, §7.

Split from `engine` because filling an order and driving the bar loop are
different subjects, and the module carried both past the point where either
could be read without the other in view.

**These take the engine rather than living on it.** They need its fill model,
cost model and instrument master and nothing else; making that dependency an
argument says so, and keeps the loop itself readable as a sequence of steps.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl

from core.instruments import InstrumentId
from core.orders import Side
from engine.accounting import Fill, Portfolio
from engine.backtest.context import RunState
from engine.backtest.fills import ExecutionBar, NoLiquidityError
from engine.costs.model import TradeContext

if TYPE_CHECKING:
    from engine.backtest.engine import BacktestEngine

__all__ = ["affordable_quantity", "execute_order"]


def _to_decimal(value: float) -> Decimal:
    """float64 price -> Decimal, without inheriting binary representation error."""
    return Decimal(str(value))


def execute_order(  # noqa: PLR0913, PLR0917 - one order, and all it needs to fill
    engine: BacktestEngine,
    state: RunState,
    instrument_id: InstrumentId,
    quantity: Decimal,
    execution_slice: pl.DataFrame,
    execution_ts: datetime,
) -> bool:
    """Fill one order into the execution bar. Returns whether it filled."""
    portfolio, result = state.portfolio, state.result
    rows = execution_slice.filter(pl.col("instrument_id") == instrument_id)
    if rows.is_empty():
        # The instrument did not trade this session. Counted separately: a
        # delisting is not a defect in our order logic.
        result.orders_no_market += 1
        return False

    row = rows.row(0, named=True)
    instrument = engine.instruments[instrument_id]
    bar = ExecutionBar(
        instrument=instrument,
        open=_to_decimal(row["open"]),
        high=_to_decimal(row["high"]),
        low=_to_decimal(row["low"]),
        close=_to_decimal(row["close"]),
        volume=_to_decimal(row["volume"]),
    )
    side = Side.BUY if quantity > 0 else Side.SELL
    wanted = abs(quantity)

    if side is Side.BUY:
        wanted = affordable_quantity(
            engine, portfolio, bar, engine.fill_model.reference_price(bar), wanted
        )
        if wanted <= 0:
            result.orders_unfunded += 1
            return False

    try:
        simulated = engine.fill_model.simulate(
            bar,
            side,
            wanted,
            allow_partial=engine.config.allow_partial_fills,
        )
    except NoLiquidityError:
        # The bar could not absorb the order — zero volume, zero range, or
        # past the participation cap. Counted once, under liquidity. It is
        # not a rejection: nothing in our logic went wrong, the market was
        # simply not deep enough.
        result.liquidity_failures += 1
        return False

    quantity = simulated.quantity
    if side is Side.BUY:
        # Final trim against the *realised* fill price. The earlier check
        # used the fill model's reference price, and `simulate` then moved
        # it against us by the slippage. Without this the account overdraws
        # by exactly the slippage on the last order of a fully-invested
        # rebalance — which presents as a rejection rather than a bug.
        quantity = affordable_quantity(engine, portfolio, bar, simulated.price, quantity)
        if quantity <= 0:
            result.orders_unfunded += 1
            return False

    costs = engine.cost_model.cost(
        TradeContext(
            instrument=instrument,
            side=side,
            quantity=quantity,
            price=simulated.price,
            adv_value=bar.volume * bar.typical,
        )
    )
    fill = Fill(
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        price=simulated.price,
        costs=costs,
        event_time=execution_ts,
        multiplier=instrument.multiplier,
    )

    try:
        realised = portfolio.apply_fill(fill)
    except Exception:  # noqa: BLE001 — insufficient cash is a rejection, not a crash
        result.orders_rejected += 1
        return False

    state.trades.append(
        {
            "event_time": execution_ts,
            "instrument_id": instrument_id,
            "side": side.value,
            "quantity": float(quantity),
            "price": float(simulated.price),
            "costs": float(costs.total),
            "realised_pnl": float(realised),
        }
    )
    return True


def affordable_quantity(
    engine: BacktestEngine,
    portfolio: Portfolio,
    bar: ExecutionBar,
    price: Decimal,
    wanted: Decimal,
) -> Decimal:
    """Largest buy the account can fund at `price`.

    Delegates the arithmetic to the planner (§14.2). The important detail is
    that `cost_of` builds the *same* `TradeContext` the charge will use —
    including `adv_value`, which enables the square-root impact term. An
    estimate that omits impact under-charges by exactly the impact, and the
    order then overdraws by that amount.
    """
    instrument = bar.instrument
    adv_value = bar.volume * bar.typical

    def cost_of(quantity: Decimal, at_price: Decimal) -> Decimal:
        return engine.cost_model.cost(
            TradeContext(
                instrument=instrument,
                side=Side.BUY,
                quantity=quantity,
                price=at_price,
                adv_value=adv_value,
            )
        ).total

    return engine.planner.affordable(portfolio, instrument, price, wanted, cost_of)
