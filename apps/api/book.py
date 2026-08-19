"""Risk limits and the paper book — MASTER_PLAN §8, §12.7.

**Why the console showed empty screens.** It polled `/vitals` and nothing else,
so Positions, Blotter and Risk had no source and rendered as blank. Risk in
particular said "No limits configured", which was not merely unhelpful — it was
false. `RiskEngine` carries a full set of defaults and enforces them on every
order; nothing exposed them.

An operations screen that reads "no limits" when limits are active is the worst
kind of wrong: it invites you to believe nothing is protecting you, or to go
looking for a configuration screen that does not exist.

**The book comes from the paper state file**, the same one `apps.cli.paper`
writes each cycle. There is no second source of truth and no in-memory
position store to drift out of sync with it — if the file says you hold 133
shares, that is what the console shows, because that is what reconciliation
will compare against tomorrow.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import polars as pl
from fastapi import APIRouter
from pydantic import BaseModel

from core.instruments import InstrumentId
from trading.paper.state import PaperStateStore, StateCorruptError
from trading.risk.limits import RiskLimits

__all__ = ["build_book_router"]

#: Where `apps.cli.paper --state-dir` writes by default.
DEFAULT_STATE_DIR = Path("paper")


class LimitRow(BaseModel):
    name: str
    threshold: float
    unit: str
    detail: str


class PositionRow(BaseModel):
    instrument_id: str
    symbol: str
    quantity: float
    average_price: float
    last_price: float
    market_value: float
    unrealised_pnl: float
    weight_pct: float


class BookResponse(BaseModel):
    """The paper account as of its last recorded cycle."""

    positions: list[PositionRow]
    cash: float
    equity: float
    realised_pnl: float
    unrealised_pnl: float
    fees_paid: float
    cycles: int
    last_session: str | None
    halted: bool
    halt_reason: str
    #: True when no paper state exists yet. Distinguished from an empty book,
    #: because "not started" and "started and flat" call for different actions.
    absent: bool


def _limit_rows(limits: RiskLimits) -> list[LimitRow]:
    """Every enforced limit, in the order the engine checks them (§8)."""
    return [
        LimitRow(
            name="order_notional",
            threshold=float(limits.max_order_notional),
            unit="INR",
            detail="single order size",
        ),
        LimitRow(
            name="position_size",
            threshold=float(limits.max_position_pct),
            unit="pct_nav",
            detail="resulting position as a fraction of NAV",
        ),
        LimitRow(
            name="price_band",
            threshold=float(limits.price_band_pct),
            unit="pct",
            detail="fat-finger guard: distance from last traded price",
        ),
        LimitRow(
            name="order_rate",
            threshold=float(limits.max_orders_per_minute),
            unit="count",
            detail="runaway-loop guard",
        ),
        LimitRow(
            name="open_orders",
            threshold=float(limits.max_open_orders),
            unit="count",
            detail="live orders resting at the venue",
        ),
        LimitRow(
            name="gross_exposure",
            threshold=float(limits.max_gross_exposure_pct),
            unit="pct_nav",
            detail="long plus short",
        ),
        LimitRow(
            name="net_exposure",
            threshold=float(limits.max_net_exposure_pct),
            unit="pct_nav",
            detail="directional tilt",
        ),
        LimitRow(
            name="cluster_concentration",
            threshold=float(limits.max_cluster_pct),
            unit="pct_nav",
            detail="correlated names count as one bet",
        ),
        LimitRow(
            name="daily_loss",
            threshold=float(limits.daily_loss_limit_pct),
            unit="pct",
            detail="session P&L halt threshold",
        ),
        LimitRow(
            name="liquidity",
            threshold=float(limits.max_adv_participation),
            unit="pct_adv",
            detail="order as a fraction of average daily value",
        ),
    ]


def _empty_book() -> BookResponse:
    return BookResponse(
        positions=[],
        cash=0.0,
        equity=0.0,
        realised_pnl=0.0,
        unrealised_pnl=0.0,
        fees_paid=0.0,
        cycles=0,
        last_session=None,
        halted=False,
        halt_reason="",
        absent=True,
    )


def build_book_router(marks_source: object | None = None) -> APIRouter:
    """Risk limits and the paper book.

    Args:
        marks_source: Injected panel for tests. Defaults to the live lake.
    """
    router = APIRouter(tags=["book"])

    @router.get("/risk/limits")
    def risk_limits() -> list[LimitRow]:
        """What the engine actually enforces.

        Read from `RiskLimits()` rather than a config file, because that is the
        object the engine is constructed with. A screen showing limits the
        engine does not use would be worse than showing none.
        """
        return _limit_rows(RiskLimits())

    @router.get("/book", response_model=BookResponse)
    def book() -> BookResponse:
        """The paper account, marked to the latest panel close."""
        store = PaperStateStore(DEFAULT_STATE_DIR)
        if not store.exists():
            return _empty_book()

        try:
            state = store.restore()
        except StateCorruptError:
            # A corrupt file is reported as absent rather than crashing the
            # console. The CLI is where that gets diagnosed; a monitoring
            # surface that 500s on bad state tells you less than one that says
            # "nothing here".
            return _empty_book()

        marks = _latest_marks(marks_source)
        rows: list[PositionRow] = []
        position_value = Decimal(0)

        for instrument_id, position in sorted(state.portfolio.positions.items()):
            if position.is_flat:
                continue
            last = marks.get(instrument_id, position.average_price)
            value = position.market_value(last)
            position_value += value
            rows.append(
                PositionRow(
                    instrument_id=str(instrument_id),
                    symbol=str(instrument_id).split(":")[-1],
                    quantity=float(position.quantity),
                    average_price=float(position.average_price),
                    last_price=float(last),
                    market_value=float(value),
                    unrealised_pnl=float(position.unrealised_pnl(last)),
                    weight_pct=0.0,
                )
            )

        equity = state.portfolio.cash + position_value
        if equity > 0:
            rows = [
                row.model_copy(update={"weight_pct": row.market_value / float(equity)})
                for row in rows
            ]

        return BookResponse(
            positions=rows,
            cash=float(state.portfolio.cash),
            equity=float(equity),
            realised_pnl=float(state.portfolio.realised_pnl),
            unrealised_pnl=sum(r.unrealised_pnl for r in rows),
            fees_paid=float(state.portfolio.fees_paid),
            cycles=state.cycles,
            last_session=state.last_session.isoformat() if state.last_session else None,
            halted=state.halted,
            halt_reason=state.halt_reason,
            absent=False,
        )

    @router.get("/equity")
    def equity_curve() -> list[dict[str, str]]:
        """One row per completed cycle. This is the M9 six-week clock (§M9)."""
        return PaperStateStore(DEFAULT_STATE_DIR).equity_history()

    return router


def _latest_marks(source: object | None) -> dict[InstrumentId, Decimal]:
    """Most recent close per instrument, for marking the book.

    Falls back to an empty map when the panel cannot be read: positions then
    show at cost, which is visibly wrong rather than invisibly stale.
    """
    from apps.api.analytics import _panel  # noqa: PLC0415 - shares the cached panel

    try:
        history = source if isinstance(source, pl.DataFrame) else _panel()
    except Exception:  # noqa: BLE001 - a monitoring surface must not 500 on data
        return {}

    latest = history.sort("event_time").group_by("instrument_id").agg(pl.col("close").last())
    return {
        InstrumentId(i): Decimal(str(c))
        for i, c in zip(latest["instrument_id"], latest["close"], strict=True)
    }
