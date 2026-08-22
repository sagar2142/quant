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
from functools import lru_cache
from pathlib import Path

import polars as pl
from fastapi import APIRouter
from pydantic import BaseModel

from apps.api.auth import ReadAccess
from apps.api.limits import LimitRow, limit_rows
from apps.api.snapshot import book_snapshot
from core.instruments import InstrumentId
from trading.paper.state import PaperStateStore, StateCorruptError
from trading.risk.limits import RiskLimits

__all__ = ["build_book_router"]

#: Where `apps.cli.paper --state-dir` writes by default.
DEFAULT_STATE_DIR = Path("paper")

#: Blotter rows served by default. The log grows without bound over a six-week
#: run and the screen shows the recent end of it.
FILL_PAGE = 200


class ReconciliationResponse(BaseModel):
    """Whether the broker and the book were compared, and what came of it.

    `checked` is separate from `halted` because they answer different
    questions. A book that has never run a cycle is not reconciled-and-clean;
    it is unreconciled, and only one of those is reassuring.
    """

    checked: bool
    halted: bool
    halt_reason: str
    cycles: int


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

    @router.get("/risk/limits", dependencies=[ReadAccess])
    def risk_limits() -> list[LimitRow]:
        """What the engine enforces, and where the book currently sits.

        Read from `RiskLimits()` rather than a config file, because that is the
        object the engine is constructed with. A screen showing limits the
        engine does not use would be worse than showing none.

        Observations come from the paper state, and are null for the limits
        that are decided per order rather than held by a portfolio.
        """
        return limit_rows(
            RiskLimits(), book_snapshot(DEFAULT_STATE_DIR, _latest_marks(marks_source))
        )

    @router.get("/book", response_model=BookResponse, dependencies=[ReadAccess])
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
        symbols = _symbol_map(marks_source)
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
                    symbol=symbols.get(instrument_id, str(instrument_id)),
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

    _register_logs(router, marks_source)

    return router


def _symbol_map(source: object | None) -> dict[InstrumentId, str]:
    """Current venue symbol per instrument, for display only.

    An `instrument_id` is `NSE:INE340A01012`, so taking the last colon-segment
    yielded the ISIN and the Positions screen listed twelve-character ISINs
    where an operator expects BIRLACORPN. Resolved from the panel instead.

    Display only, and deliberately so: §3.3 is explicit that a symbol is not
    identity. The `instrument_id` remains the key everywhere it matters, and a
    name whose ticker cannot be resolved falls back to showing the id rather
    than an empty cell.
    """
    return _resolve(source)[1]


def _latest_marks(source: object | None) -> dict[InstrumentId, Decimal]:
    """Most recent close per instrument, for marking the book.

    Falls back to an empty map when the panel cannot be read: positions then
    show at cost, which is visibly wrong rather than invisibly stale.
    """
    return _resolve(source)[0]


def _tail(history: pl.DataFrame) -> tuple[dict[InstrumentId, Decimal], dict[InstrumentId, str]]:
    """Last close and last symbol per instrument, in a single pass.

    Both maps come from the same sort because they answer the same question of
    the same rows. Computing them separately sorted 3.3M rows twice for one
    screen.
    """
    latest = (
        history.sort("event_time")
        .group_by("instrument_id")
        .agg(pl.col("close").last(), pl.col("symbol").last())
    )
    marks = {
        InstrumentId(i): Decimal(str(c))
        for i, c in zip(latest["instrument_id"], latest["close"], strict=True)
    }
    symbols = {
        InstrumentId(i): str(s)
        for i, s in zip(latest["instrument_id"], latest["symbol"], strict=True)
    }
    return marks, symbols


@lru_cache(maxsize=1)
def _cached_tail() -> tuple[dict[InstrumentId, Decimal], dict[InstrumentId, str]]:
    """`_tail` over the shared panel, computed once per process.

    **The console polls once a second**, and three endpoints now need these
    maps. Sorting 3.3M rows per request meant three full sorts per second for a
    result that cannot change until the lake is re-ingested and the process
    restarts — the same reasoning that caches `_panel` itself.
    """
    from apps.api.analytics import _panel  # noqa: PLC0415 - shares the cached panel

    return _tail(_panel())


def _resolve(
    source: object | None,
) -> tuple[dict[InstrumentId, Decimal], dict[InstrumentId, str]]:
    """Marks and symbols, from an injected panel or the cached one.

    An injected frame is a test fixture and is never cached — caching it would
    leak one test's panel into the next.
    """
    try:
        if isinstance(source, pl.DataFrame):
            return _tail(source)
        return _cached_tail()
    except Exception:  # noqa: BLE001 - a monitoring surface must not 500 on data
        return {}, {}


def _register_logs(router: APIRouter, marks_source: object | None) -> None:
    """The append-only paper logs, plus the reconciliation verdict.

    Split out of `build_book_router` to keep it under the complexity lint,
    the same way `analytics` registers its groups.
    """

    @router.get("/equity", dependencies=[ReadAccess])
    def equity_curve() -> list[dict[str, str]]:
        """One row per completed cycle. This is the M9 six-week clock (§M9)."""
        return PaperStateStore(DEFAULT_STATE_DIR).equity_history()

    @router.get("/fills", dependencies=[ReadAccess])
    def fills(limit: int = FILL_PAGE) -> list[dict[str, str]]:
        """Applied fills, most recent last — the Blotter's source.

        The ticker is resolved here rather than read from the log, because the
        log stores `instrument_id` and a symbol is not identity (§3.3). A name
        renamed since it traded shows the ticker every other screen shows.
        """
        symbols = _symbol_map(marks_source)
        rows = PaperStateStore(DEFAULT_STATE_DIR).fill_history(limit=max(0, limit))
        return [
            {**row, "symbol": symbols.get(InstrumentId(row["instrument_id"]), row["instrument_id"])}
            for row in rows
        ]

    @router.get("/reconciliation", dependencies=[ReadAccess])
    def reconciliation() -> ReconciliationResponse:
        """Whether the last cycle's reconciliation found anything.

        The console previously rendered "Broker and internal records agree."
        unconditionally, from a `breaks` array nothing ever populated. That is
        an affirmative safety claim made about a check that had not run, which
        is worse than showing nothing: it is the screen an operator would look
        at to decide a break had cleared.
        """
        snapshot = book_snapshot(DEFAULT_STATE_DIR)
        return ReconciliationResponse(
            checked=snapshot.present and snapshot.cycles > 0,
            halted=snapshot.halted,
            halt_reason=snapshot.halt_reason,
            cycles=snapshot.cycles,
        )
