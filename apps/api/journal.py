"""Trade journal — MASTER_PLAN §9, §M7.

**The cost model was never checked against anything.** A backtest charges an
estimate at every fill; nothing compared that estimate to a real one, so the
only test the model ever faced was whether the equity curve it produced looked
believable — which it always will, because the model produced it.

This reads what actually happened: orders, their fills, and the marks recorded
beside them. `trading.journal` does the arithmetic; this does the SQL, and only
the SQL, so the decomposition can be tested without a Postgres.

**An empty journal is a state, not an error.** Before the first order there is
nothing to measure, and the endpoint says so rather than returning zeros that
would read as perfect execution.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from apps.api.auth import ReadAccess
from trading.journal import Execution, JournalSummary, summarise

__all__ = ["build_journal_router", "load_executions"]

#: Rows a request will read. A journal is for inspecting recent execution, not
#: for exporting the book — an unbounded query here is one bad day away from a
#: response the browser cannot hold.
DEFAULT_LIMIT = 200

#: One order, its fills aggregated. The average fill is quantity-weighted
#: rather than a plain mean: a 900-share fill at 100 and a 100-share fill at
#: 110 average to 101, and calling that 105 misstates the cost by a factor of
#: four.
ORDERS_SQL = """
    SELECT
        o.order_id::text            AS order_id,
        o.instrument_id             AS instrument_id,
        COALESCE(i.symbol, o.instrument_id) AS symbol,
        o.side::text                AS side,
        o.strategy_id               AS strategy_id,
        o.mode::text                AS mode,
        o.decision_time             AS decision_time,
        o.quantity                  AS quantity_ordered,
        o.limit_price               AS limit_price,
        COALESCE(SUM(f.quantity), 0)                        AS quantity_filled,
        SUM(f.price * f.quantity) / NULLIF(SUM(f.quantity), 0) AS average_fill,
        COALESCE(SUM(
            f.commission + f.taxes + f.exchange_fees + f.other_fees
        ), 0)                                               AS fees,
        MIN(f.intended_price)       AS intended_price,
        MIN(f.event_time)           AS first_fill_time,
        MAX(f.event_time)           AS last_fill_time
    FROM orders o
    LEFT JOIN fills f       ON f.order_id = o.order_id
    LEFT JOIN instruments i ON i.instrument_id = o.instrument_id
    -- Casts are not decoration: without them Postgres cannot infer a type for
    -- a parameter that only ever appears beside NULL, and the query fails with
    -- "could not determine data type of parameter $1".
    WHERE (%(strategy)s::text IS NULL OR o.strategy_id = %(strategy)s::text)
      AND (%(mode)s::text IS NULL OR o.mode::text = %(mode)s::text)
    GROUP BY o.order_id, i.symbol
    ORDER BY o.decision_time DESC
    LIMIT %(limit)s
"""


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _row_to_execution(row: dict[str, Any]) -> Execution:
    """One row, with a limit price standing in for a missing intent.

    A limit order's price *is* what it aimed at, so using it when no
    `intended_price` was recorded recovers execution cost for the orders where
    it is unambiguous. A market order has no such stand-in and stays
    unmeasured, which is the truth about it.
    """
    intended = _decimal(row.get("intended_price")) or _decimal(row.get("limit_price"))
    return Execution(
        order_id=str(row["order_id"]),
        instrument_id=str(row["instrument_id"]),
        symbol=str(row["symbol"]),
        side=str(row["side"]),
        strategy_id=str(row["strategy_id"]),
        mode=str(row["mode"]),
        decision_time=row["decision_time"],
        quantity_ordered=_decimal(row["quantity_ordered"]) or Decimal(0),
        quantity_filled=_decimal(row["quantity_filled"]) or Decimal(0),
        average_fill=_decimal(row["average_fill"]),
        fees=_decimal(row["fees"]) or Decimal(0),
        # No decision mark is recorded today, so shortfall against the decision
        # price is not yet measurable and `slippage` falls back to the intent.
        # Named here rather than silently defaulted: the gap is a schema
        # question, not an arithmetic one.
        decision_price=_decimal(row.get("decision_price")),
        intended_price=intended,
        first_fill_time=row.get("first_fill_time"),
        last_fill_time=row.get("last_fill_time"),
    )


def load_executions(
    strategy: str | None = None,
    mode: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[Execution], str]:
    """Read orders and their fills.

    Returns:
        The executions and a note. The note is non-empty when nothing could be
        read — an unreachable database and an empty book are different states
        and must not both render as "no trades".
    """
    from ops.db import optional_connection  # noqa: PLC0415 - optional at runtime

    with optional_connection() as conn:
        if conn is None:
            return [], "The trade database is unreachable, so nothing can be measured."
        with conn.cursor() as cursor:
            cursor.execute(ORDERS_SQL, {"strategy": strategy, "mode": mode, "limit": limit})
            columns = [c.name for c in cursor.description or []]
            rows = [dict(zip(columns, record, strict=True)) for record in cursor.fetchall()]

    if not rows:
        return [], "No orders yet. Execution quality is measurable from the first one."
    return [_row_to_execution(row) for row in rows], ""


class ExecutionRow(BaseModel):
    order_id: str
    symbol: str
    side: str
    strategy_id: str
    mode: str
    decision_time: datetime
    quantity_ordered: Decimal
    quantity_filled: Decimal
    fill_rate: float
    average_fill: Decimal | None
    intended_price: Decimal | None
    fees: Decimal
    #: null means unmeasured, never zero. An order with no reference price has
    #: an unknown cost, which is a different finding from a costless one.
    delay_cost: Decimal | None
    execution_cost: Decimal | None
    slippage: Decimal | None
    slippage_bps: float | None
    implementation_shortfall: Decimal | None


class JournalResponse(BaseModel):
    """Execution quality, decomposed."""

    orders: int
    filled: int
    #: Filled orders with enough marks to measure. Reported so a total computed
    #: from part of the book cannot be read as the whole book's.
    measured: int
    coverage: float
    notional: Decimal
    fees: Decimal
    delay_cost: Decimal | None
    execution_cost: Decimal | None
    slippage: Decimal | None
    implementation_shortfall: Decimal | None
    shortfall_bps: float | None
    rows: list[ExecutionRow]
    worst: list[str]
    note: str = ""


def _to_row(execution: Execution) -> ExecutionRow:
    return ExecutionRow(
        order_id=execution.order_id,
        symbol=execution.symbol,
        side=execution.side,
        strategy_id=execution.strategy_id,
        mode=execution.mode,
        decision_time=execution.decision_time,
        quantity_ordered=execution.quantity_ordered,
        quantity_filled=execution.quantity_filled,
        fill_rate=execution.fill_rate,
        average_fill=execution.average_fill,
        intended_price=execution.intended_price,
        fees=execution.fees,
        delay_cost=execution.delay_cost,
        execution_cost=execution.execution_cost,
        slippage=execution.slippage,
        slippage_bps=execution.slippage_bps,
        implementation_shortfall=execution.implementation_shortfall,
    )


def _response(executions: list[Execution], summary: JournalSummary, note: str) -> JournalResponse:
    return JournalResponse(
        orders=summary.orders,
        filled=summary.filled,
        measured=summary.measured,
        coverage=summary.coverage,
        notional=summary.notional,
        fees=summary.fees,
        delay_cost=summary.delay_cost,
        execution_cost=summary.execution_cost,
        slippage=summary.slippage,
        implementation_shortfall=summary.implementation_shortfall,
        shortfall_bps=summary.shortfall_bps,
        rows=[_to_row(e) for e in executions],
        worst=[e.order_id for e in summary.worst],
        note=note,
    )


def build_journal_router() -> APIRouter:
    router = APIRouter(prefix="/journal", tags=["journal"])

    @router.get("", response_model=JournalResponse, dependencies=[ReadAccess])
    def journal(
        strategy: str | None = None,
        mode: str | None = Query(default=None, pattern="^(PAPER|LIVE)$"),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=1000),
    ) -> JournalResponse:
        """Orders, fills, and what the difference cost.

        The number to put beside a backtest's modelled cost. Until something
        measures the real one, the model is only ever checked against the
        equity curve it produced.
        """
        executions, note = load_executions(strategy, mode, limit)
        return _response(executions, summarise(executions), note)

    return router
