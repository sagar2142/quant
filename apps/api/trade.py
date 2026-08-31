"""Order entry — MASTER_PLAN §8, §12.7, §21.

**Why this file did not exist.** Every piece of the order path was already
built: `Order` and its state machine, `RiskEngine.check`, `KiteBroker.submit`,
the kill switch, the notional budget. None of it was reachable from outside the
process, so the console could describe a market in detail and not act on one.

**Preview and submit are separate endpoints with different requirements.** A
preview is arithmetic — notional, costs, which limits the order comes near —
and needs no credentials, no live environment and no permission, so it works
today and is where the thinking happens. A submit sends real money to a real
exchange and passes every guard the system has.

**The gates are reported, not hidden.** `GET /trade/status` answers "could I
place an order right now, and if not, what exactly is missing" — environment,
live permission, credentials, kill switch. A console that greys a button out
without saying why teaches you to distrust it; one that says "KITE_ACCESS_TOKEN
is empty" tells you what to do next.

**Nothing here invents a fallback.** No simulated fills, no pretend order ids,
no demo mode that looks like trading and is not. If the chain is incomplete the
request fails with the reason (§14.1.5) — this system had simulated positions
once and they were removed, because a book you did not trade is worse than no
book at all.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Protocol

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apps.api.analytics import latest_quote
from apps.api.auth import ReadAccess, WriteAccess
from core.clock import utc_now
from core.config import settings
from core.instruments import InstrumentId
from trading.risk.engine import RiskEngine
from trading.risk.limits import PortfolioState, ProposedOrder

if TYPE_CHECKING:  # pragma: no cover - types only; the runtime import is local
    from trading.execution.broker import BrokerFill, BrokerOrder, BrokerPosition

__all__ = ["MANUAL_STRATEGY", "build_trade_router", "live_gates"]

#: Strategy id recorded against a hand-entered order.
#:
#: Not blank, and not the name of a researched strategy. A discretionary order
#: labelled as though a tested strategy produced it would corrupt the only
#: record that says which ideas were actually traded.
MANUAL_STRATEGY = "manual"

#: Indicative NSE delivery costs for the ticket's estimate, as fractions of
#: notional. Selling costs more: STT falls on the sell leg, as does stamp duty.
#:
#: The authoritative model is `engine.costs.india.NseEquityCostModel`, which
#: the backtester uses and which prices every component properly including the
#: per-scrip DP charge. This is a display estimate of the same order of
#: magnitude and is labelled as one — a ticket quoting a number that disagreed
#: with the backtester would make one of the two a liar.
BUY_COST_RATE = Decimal("0.0012")
SELL_COST_RATE = Decimal("0.0022")


class OrderRequest(BaseModel):
    """One order, as the ticket sends it."""

    symbol: str = Field(min_length=1, max_length=32)
    side: str = Field(pattern="^(BUY|SELL)$")
    quantity: str = Field(description="Decimal string. Never a float (§14.1.2).")
    order_type: str = Field(default="MARKET", pattern="^(MARKET|LIMIT)$")
    limit_price: str | None = None
    #: Last traded price the ticket showed. Risk checks against it, so an order
    #: priced off a stale screen is caught rather than sent.
    reference_price: str
    #: Capital this order is sized against.
    #:
    #: Supplied by the caller rather than read from a book, because there is no
    #: book: paper trading was removed deliberately, and every percentage limit
    #: in `RiskLimits` is a fraction of equity. Defaulting it to a made-up
    #: number would make each check pass or fail for a reason unconnected to
    #: the account being traded.
    equity: str


class GateStatus(BaseModel):
    """One precondition for live trading, and whether it is met."""

    name: str
    ready: bool
    detail: str


class TradeStatusResponse(BaseModel):
    can_trade: bool
    mode: str
    gates: list[GateStatus]


class PreviewResponse(BaseModel):
    symbol: str
    instrument_id: str
    last_close: str
    side: str
    quantity: str
    notional: str
    estimated_costs: str
    cash_impact: str
    allowed: bool
    breaches: list[str]
    checks: list[dict[str, object]]


class WorkingOrder(BaseModel):
    broker_order_id: str
    instrument_id: str
    side: str
    quantity: str
    filled_quantity: str
    pending_quantity: str
    price: str | None
    status: str
    placed_at: str


class HeldPosition(BaseModel):
    instrument_id: str
    quantity: str
    average_price: str


class ExecutedFill(BaseModel):
    broker_fill_id: str
    broker_order_id: str
    instrument_id: str
    side: str
    quantity: str
    price: str


class SubmitResponse(BaseModel):
    order_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: str
    submitted_at: datetime


def _decimal(raw: str, field: str) -> Decimal:
    """Parse a Decimal, or say which field was wrong.

    Money never arrives as a float (§14.1.2), and a quantity that silently
    became zero because it failed to parse is an order nobody meant to place.
    """
    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{field} is not a number: {raw!r}") from exc
    if not value.is_finite():
        raise HTTPException(status_code=422, detail=f"{field} must be finite, got {raw!r}")
    return value


def _credential_gate() -> GateStatus:
    """Whether Kite credentials load and the token is actually set.

    Probed by loading them, because a key that is present but empty is not a
    key — a distinction that matters at 08:59 when the daily Kite login has not
    been done and every field still looks configured.
    """
    try:
        from core.secrets import load_broker_credentials  # noqa: PLC0415 - optional path

        credentials = load_broker_credentials("kite")
    except Exception as exc:  # noqa: BLE001 - any failure here means "not ready"
        return GateStatus(
            name="broker_credentials", ready=False, detail=f"{type(exc).__name__}: {exc}"
        )

    ready = credentials.access_token.is_set
    return GateStatus(
        name="broker_credentials",
        ready=ready,
        detail=(
            "Kite key and access token loaded"
            if ready
            else "KITE_ACCESS_TOKEN is empty — complete the daily Kite login"
        ),
    )


def live_gates(risk: RiskEngine) -> list[GateStatus]:
    """Every precondition between a click and the exchange, in failure order."""
    return [
        GateStatus(
            name="environment",
            ready=settings.env.is_live,
            detail=f"env={settings.env.value}; live paths require the live environment",
        ),
        GateStatus(
            name="live_enabled",
            ready=bool(settings.live_enabled),
            detail="NEUTRON_LIVE_ENABLED must be set; the environment alone is not enough (§21)",
        ),
        _credential_gate(),
        GateStatus(
            name="kill_switch",
            ready=not risk.is_killed,
            detail="engaged — release it before trading" if risk.is_killed else "clear",
        ),
    ]


def estimated_costs(notional: Decimal, side: str) -> Decimal:
    """Indicative cost of one leg. See `BUY_COST_RATE`."""
    rate = BUY_COST_RATE if side == "BUY" else SELL_COST_RATE
    return (notional * rate).quantize(Decimal("0.01"))


def _require_armed(risk: RiskEngine) -> None:
    """Refuse unless every gate is open, naming the ones that are not."""
    blocked = [g for g in live_gates(risk) if not g.ready]
    if blocked:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "live trading is not armed",
                "blocked_by": [{"gate": g.name, "detail": g.detail} for g in blocked],
            },
        )
    risk.raise_if_killed()


def _preview(request: OrderRequest, risk: RiskEngine) -> PreviewResponse:
    """Risk verdict and cost for an order, sending nothing."""
    quantity = _decimal(request.quantity, "quantity")
    reference = _decimal(request.reference_price, "reference_price")
    equity = _decimal(request.equity, "equity")
    if quantity <= 0:
        raise HTTPException(status_code=422, detail="quantity must be positive")
    if reference <= 0:
        raise HTTPException(status_code=422, detail="reference_price must be positive")
    if equity <= 0:
        raise HTTPException(status_code=422, detail="equity must be positive")

    quote = latest_quote(request.symbol)
    if quote is None:
        raise HTTPException(status_code=404, detail=f"{request.symbol.upper()} is not in the panel")
    instrument_id = InstrumentId(str(quote["instrument_id"]))
    last_close = Decimal(str(quote["last_close"]))

    # No open positions and no drawdown, because there is no book to read them
    # from. That makes this a check of the order *in isolation* — size, price
    # band, liquidity — and not of portfolio concentration, which needs
    # holdings this system does not have until a broker is connected. Said
    # plainly rather than assumed: a concentration check that always passes
    # because it sees an empty book is worse than no check at all.
    #
    # The last close and ADV are supplied, so the fat-finger band and the
    # participation limit are evaluated against real numbers rather than
    # blocking for want of an input.
    state = PortfolioState(
        equity=equity,
        cash=equity,
        peak_equity=equity,
        day_start_equity=equity,
        last_prices={instrument_id: last_close},
        adv={instrument_id: Decimal(str(quote["adv"]))},
    )
    proposed = ProposedOrder(
        strategy_id=MANUAL_STRATEGY,
        instrument_id=instrument_id,
        quantity=quantity,
        price=reference,
        multiplier=Decimal(1) if request.side == "BUY" else Decimal(-1),
    )
    verdict = risk.check(proposed, state)
    notional = proposed.notional
    costs = estimated_costs(notional, request.side)
    impact = notional + costs if request.side == "BUY" else costs - notional

    return PreviewResponse(
        symbol=request.symbol.upper(),
        instrument_id=str(instrument_id),
        last_close=f"{last_close:.2f}",
        side=request.side,
        quantity=str(quantity),
        notional=f"{notional:.2f}",
        estimated_costs=f"{costs:.2f}",
        cash_impact=f"{impact:.2f}",
        allowed=verdict.allowed,
        breaches=list(verdict.reasons),
        checks=[{"name": c.name, "passed": c.passed, "detail": c.message} for c in verdict.checks],
    )


def _submit(request: OrderRequest, risk: RiskEngine) -> SubmitResponse:
    """Place a live order. Reached only with every gate open."""
    _require_armed(risk)

    # Imported here, not at module scope: `KiteBroker.__post_init__` calls
    # `require_live_permission`, which raises in every non-live environment.
    # A module-scope import would make this file unimportable in development.
    from core.orders import OrderType, Side  # noqa: PLC0415
    from core.secrets import load_broker_credentials  # noqa: PLC0415
    from trading.execution.broker import BrokerError  # noqa: PLC0415
    from trading.execution.kite import KiteBroker  # noqa: PLC0415
    from trading.execution.orders import Order, TradingMode  # noqa: PLC0415

    quantity = _decimal(request.quantity, "quantity")
    reference = _decimal(request.reference_price, "reference_price")
    quote = latest_quote(request.symbol)
    if quote is None:
        raise HTTPException(status_code=404, detail=f"{request.symbol.upper()} is not in the panel")
    order = Order(
        strategy_id=MANUAL_STRATEGY,
        # ISIN-keyed, never the ticker: a symbol can wear more than one ISIN
        # over its life, and an order against the wrong one is an order in the
        # wrong security (§1.1).
        instrument_id=InstrumentId(str(quote["instrument_id"])),
        side=Side.BUY if request.side == "BUY" else Side.SELL,
        quantity=quantity,
        order_type=OrderType.LIMIT if request.order_type == "LIMIT" else OrderType.MARKET,
        mode=TradingMode.LIVE,
        decision_time=utc_now(),
        limit_price=_decimal(request.limit_price, "limit_price") if request.limit_price else None,
    )
    broker = KiteBroker(credentials=load_broker_credentials("kite"), instruments={})
    try:
        broker_order_id = broker.submit(order, reference)
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=f"broker rejected the order: {exc}") from exc

    return SubmitResponse(
        order_id=str(order.order_id),
        broker_order_id=broker_order_id,
        symbol=request.symbol.upper(),
        side=request.side,
        quantity=str(quantity),
        submitted_at=order.decision_time,
    )


class LiveBroker(Protocol):
    """The read surface this module needs from a broker.

    A protocol rather than the concrete class because `KiteBroker` cannot be
    imported at module scope: its constructor calls `require_live_permission`,
    which raises in every environment that is not live. Typing against the
    shape keeps the checker honest without making this file unimportable in
    development.
    """

    def orders(self) -> list[BrokerOrder]: ...

    def positions(self) -> list[BrokerPosition]: ...

    def fills_since(self, marker: str | None) -> list[BrokerFill]: ...


def _broker() -> LiveBroker:
    """A live broker, or an HTTP error explaining why there is not one.

    Constructed per request rather than held: `KiteBroker` carries a session
    notional budget that is deliberately exhausted rather than reset, and a
    long-lived instance behind a web process would spend that budget across
    unrelated sessions.
    """
    from core.secrets import load_broker_credentials  # noqa: PLC0415
    from trading.execution.kite import KiteBroker  # noqa: PLC0415

    return KiteBroker(credentials=load_broker_credentials("kite"), instruments={})


def _working_orders(risk: RiskEngine) -> list[WorkingOrder]:
    """Today's order book, read from the venue.

    Empty is a valid answer only when the broker actually said so. Without
    credentials this refuses rather than returning `[]`, because an empty list
    and "I could not ask" look identical on a screen and only one of them means
    you have nothing resting.
    """
    _require_armed(risk)
    from trading.execution.broker import BrokerError  # noqa: PLC0415

    try:
        found = _broker().orders()
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [
        WorkingOrder(
            broker_order_id=o.broker_order_id,
            instrument_id=str(o.instrument_id),
            side=o.side.value,
            quantity=str(o.quantity),
            filled_quantity=str(o.filled_quantity),
            pending_quantity=str(o.pending_quantity),
            price=None if o.price is None else f"{o.price:.2f}",
            status=o.status,
            placed_at=o.placed_at,
        )
        for o in found
    ]


def _held_positions(risk: RiskEngine) -> list[HeldPosition]:
    """What the broker believes is held — the reconciliation baseline (§9)."""
    _require_armed(risk)
    from trading.execution.broker import BrokerError  # noqa: PLC0415

    try:
        held = _broker().positions()
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [
        HeldPosition(
            instrument_id=str(p.instrument_id),
            quantity=str(p.quantity),
            average_price=f"{p.average_price:.2f}",
        )
        for p in held
    ]


def _executed_fills(risk: RiskEngine) -> list[ExecutedFill]:
    """Today's executions, as the venue reports them."""
    _require_armed(risk)
    from trading.execution.broker import BrokerError  # noqa: PLC0415

    try:
        executed = _broker().fills_since(None)
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [
        ExecutedFill(
            broker_fill_id=f.broker_fill_id,
            broker_order_id=f.broker_order_id,
            instrument_id=str(f.instrument_id),
            side=f.side.value,
            quantity=str(f.quantity),
            price=f"{f.price:.2f}",
        )
        for f in executed
    ]


def build_trade_router(risk: RiskEngine) -> APIRouter:
    """Order entry, risk preview and broker status.

    Args:
        risk: The same engine the rest of the system uses, injected rather than
            constructed here — a second engine would carry a second kill
            switch, and engaging one would not stop the other.
    """
    router = APIRouter(prefix="/trade", tags=["trade"])

    @router.get("/status", response_model=TradeStatusResponse, dependencies=[ReadAccess])
    def status() -> TradeStatusResponse:
        """Whether an order could be placed right now, and what is missing."""
        gates = live_gates(risk)
        armed = all(g.ready for g in gates)
        return TradeStatusResponse(
            can_trade=armed, mode="LIVE" if armed else "BLOCKED", gates=gates
        )

    @router.post("/preview", response_model=PreviewResponse, dependencies=[ReadAccess])
    def preview(request: OrderRequest) -> PreviewResponse:
        """Cost and risk verdict for an order, without sending anything."""
        return _preview(request, risk)

    @router.post("/orders", response_model=SubmitResponse, dependencies=[WriteAccess])
    def submit(request: OrderRequest) -> SubmitResponse:
        """Place a live order. Every gate must be open."""
        return _submit(request, risk)

    @router.get("/orders", response_model=list[WorkingOrder], dependencies=[ReadAccess])
    def orders() -> list[WorkingOrder]:
        """Today's order book, read from the venue."""
        return _working_orders(risk)

    @router.get("/positions", response_model=list[HeldPosition], dependencies=[ReadAccess])
    def positions() -> list[HeldPosition]:
        """What the broker believes is held — the reconciliation baseline (§9)."""
        return _held_positions(risk)

    @router.get("/fills", response_model=list[ExecutedFill], dependencies=[ReadAccess])
    def fills() -> list[ExecutedFill]:
        """Today's executions, as the venue reports them."""
        return _executed_fills(risk)

    @router.delete("/orders/{broker_order_id}", dependencies=[WriteAccess])
    def cancel(broker_order_id: str) -> dict[str, str]:
        """Cancel a working order at the venue."""
        _require_armed(risk)

        from core.secrets import load_broker_credentials  # noqa: PLC0415
        from trading.execution.broker import BrokerError  # noqa: PLC0415
        from trading.execution.kite import KiteBroker  # noqa: PLC0415

        broker = KiteBroker(credentials=load_broker_credentials("kite"), instruments={})
        try:
            broker.cancel(broker_order_id)
        except BrokerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"cancelled": broker_order_id}

    return router
