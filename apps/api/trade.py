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

from apps.api.accounts import SessionCookie
from apps.api.analytics import latest_quote
from apps.api.auth import ReadAccess, WriteAccess
from core.clock import as_decision_time, utc_now
from core.config import settings
from core.instruments import InstrumentId, OptionType
from data.store.bars import NoDataError
from data.store.derivatives import DerivativesStore
from trading.risk.engine import RiskEngine
from trading.risk.limits import PortfolioState, ProposedOrder

if TYPE_CHECKING:  # pragma: no cover - types only; the runtime import is local
    from core.secrets import BrokerCredentials
    from data.store.sectors import SectorView
    from trading.execution.broker import BrokerFill, BrokerOrder, BrokerPosition

__all__ = [
    "INDUSTRY_PREFIX",
    "MANUAL_STRATEGY",
    "UNCLASSIFIED",
    "build_trade_router",
    "live_gates",
    "sector_exposure",
]

#: Group for a name NSE does not classify. Named rather than blank, because an
#: empty cluster is what made the concentration limit unfailable — every
#: unclassified order landed in the same nameless bucket as every other and the
#: check compared against nothing.
UNCLASSIFIED = "industry:unclassified"

#: Namespace for an industry-derived group, mirroring `CORRELATION_PREFIX`. The
#: two groupings answer different questions and must not be mistaken for each
#: other in a breach message.
INDUSTRY_PREFIX = "industry:"

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

#: The chain speaks CE and PE; the instrument model speaks CALL and PUT.
OPTION_RIGHTS = {"CE": OptionType.CALL, "PE": OptionType.PUT}


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
    #: Which market this order is in. An equity order is priced on notional and
    #: an option order on premium, and they are not the same number: 10 lots of
    #: a 20-rupee option on a 500-share lot is 100,000 of premium against
    #: 6,435,000 of underlying exposure. Charging one as though it were the
    #: other is not a rounding error.
    kind: str = Field(default="EQUITY", pattern="^(EQUITY|OPTION)$")
    #: Option contract, required when `kind` is OPTION. The strike is a string
    #: for the same reason every other price here is (§14.1.2).
    expiry: str | None = None
    strike: str | None = None
    right: str | None = Field(default=None, pattern="^(CE|PE)$")


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
    kind: str
    #: Contracts, for an option. One lot is `lot_size` units, and NSE trades
    #: options in whole lots only.
    lot_size: str | None = None
    #: What the underlying exposure actually is, which for an option is the
    #: premium times nothing like the notional. Stated because an option that
    #: costs 100,000 in premium can carry 6,435,000 of delta-one exposure, and
    #: a ticket showing only the premium invites sizing against the wrong
    #: number.
    underlying_exposure: str | None = None
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


def account_credentials(session: str | None, broker: str) -> BrokerCredentials | None:
    """The signed-in account's stored broker keys, or None.

    **This is what makes "connect your broker" mean anything.** The console
    encrypts a key and secret into the account document; without a read path
    they were write-only, and the credential gate below reported "not
    connected" to an operator who had just connected. Everything still ran off
    environment variables, which is a different place from the one the screen
    writes to.

    The environment wins where it is set: a machine deliberately pinned to one
    set of credentials should not be overridden by whoever happens to be signed
    in, and `_environment_pins` is the existing expression of that rule.
    """
    from apps.api.accounts import current_account  # noqa: PLC0415 - optional path
    from apps.api.broker_keys import stored_credentials  # noqa: PLC0415
    from core.secrets import BrokerCredentials, SecretValue  # noqa: PLC0415

    account = current_account(session)
    if account is None:
        return None
    try:
        stored = stored_credentials(account.settings, broker)
    except Exception:  # noqa: BLE001 - an unreadable key is not a usable one
        return None
    if stored is None:
        return None
    return BrokerCredentials(
        broker=broker,
        api_key=SecretValue(stored["api_key"]),
        api_secret=SecretValue(stored["api_secret"]),
        access_token=SecretValue(stored["access_token"]),
    )


def _credential_gate(session: str | None = None) -> GateStatus:
    """Whether broker credentials load and a usable secret is actually set.

    Probed by loading them, because a key that is present but empty is not a
    key — a distinction that matters at 08:59 when the daily login has not been
    done and every field still looks configured.

    Checks the signed-in account's stored keys first, then the environment.
    """
    broker = settings.broker.capitalize()
    from_account = account_credentials(session, settings.broker.lower())
    if from_account is not None:
        ready = from_account.access_token.is_set or from_account.api_secret.is_set
        return GateStatus(
            name="Broker credentials",
            ready=ready,
            detail=f"{broker} connected" if ready else f"{broker} API key and secret required",
        )

    try:
        from core.secrets import load_broker_credentials  # noqa: PLC0415 - optional path

        credentials = load_broker_credentials(settings.broker.lower())
    except Exception:  # noqa: BLE001 - any failure here means "not ready"
        # Deliberately not the exception text. A `PermissionError` repr on an
        # operations screen is a stack trace wearing a status label: it names
        # the internal guard that fired rather than the thing to do about it,
        # and every gate above already reports that thing.
        return GateStatus(
            name="Broker credentials", ready=False, detail=f"{broker} account not connected"
        )

    # Either a minted token, or the key and secret to mint one with.
    ready = credentials.access_token.is_set or credentials.api_secret.is_set
    return GateStatus(
        name="Broker credentials",
        ready=ready,
        detail=f"{broker} connected" if ready else f"{broker} API key and secret required",
    )


def sector_exposure(
    positions: dict[InstrumentId, Decimal],
) -> tuple[dict[str, Decimal], SectorView]:
    """Signed notional per industry, and the classification used.

    **This is what makes the concentration limit mean anything on a manual
    order.** `ProposedOrder.cluster` defaulted to the empty string, so
    `_cluster_check` compared the order against a group that was always empty
    and passed every time — a limit that cannot fail is not a limit. Correlation
    clustering is the better answer where there is history to cluster over
    (`quant.analytics.clusters` argues why, and it is right), but a hand-entered
    order arrives without one and an industry is a real group rather than no
    group at all.

    Present-tense, so the classification carries no look-ahead: the book is
    valued as it stands and the labels were observed at or before now.
    """
    from data.store.sectors import SectorStore  # noqa: PLC0415 - optional at import

    view = SectorStore(settings.lake).view()
    groups: dict[str, Decimal] = {}
    for instrument_id, notional in positions.items():
        found = view.industry_of(str(instrument_id))
        industry = f"{INDUSTRY_PREFIX}{found}" if found else UNCLASSIFIED
        groups[industry] = groups.get(industry, Decimal(0)) + notional
    return groups, view


def _cluster_for(instrument_id: InstrumentId) -> str:
    """The group a hand-entered order belongs to."""
    from data.store.sectors import SectorStore  # noqa: PLC0415

    found = SectorStore(settings.lake).view().industry_of(str(instrument_id))
    return f"{INDUSTRY_PREFIX}{found}" if found else UNCLASSIFIED


def live_gates(risk: RiskEngine, session: str | None = None) -> list[GateStatus]:
    """Every precondition between a click and the exchange, in failure order."""
    return [
        GateStatus(
            name="Environment",
            ready=settings.env.is_live,
            detail="Live" if settings.env.is_live else f"{settings.env.value.capitalize()} mode",
        ),
        GateStatus(
            name="Live trading",
            ready=bool(settings.live_enabled),
            detail="Enabled" if settings.live_enabled else "Disabled",
        ),
        _credential_gate(session),
        GateStatus(
            name="Kill switch",
            ready=not risk.is_killed,
            detail="Engaged — release before trading" if risk.is_killed else "Clear",
        ),
    ]


def estimated_costs(notional: Decimal, side: str) -> Decimal:
    """Indicative cost of one leg. See `BUY_COST_RATE`."""
    rate = BUY_COST_RATE if side == "BUY" else SELL_COST_RATE
    return (notional * rate).quantize(Decimal("0.01"))


def option_quote(request: OrderRequest) -> dict[str, object]:
    """The contract this ticket names, priced from the last session.

    Raises:
        HTTPException: 422 when the contract terms are incomplete, 404 when no
            such contract is listed. Both say which, because "no such option"
            and "you did not say which strike" need different corrections.
    """
    from datetime import date as _date  # noqa: PLC0415 - local to the option path

    if not (request.expiry and request.strike and request.right):
        raise HTTPException(
            status_code=422,
            detail="an option order needs expiry, strike and right (CE or PE)",
        )
    try:
        expiry = _date.fromisoformat(request.expiry)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"expiry is not a date: {request.expiry!r}"
        ) from exc

    strike = _decimal(request.strike, "strike")
    store = DerivativesStore(settings.lake)
    try:
        chain = store.chain(request.symbol, as_of=as_decision_time(utc_now()))
    except NoDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    match = chain.filter(
        (chain["expiry"] == expiry)
        & (chain["strike"] == float(strike))
        & (chain["right"] == request.right)
    )
    if match.is_empty():
        raise HTTPException(
            status_code=404,
            detail=(
                f"no {request.symbol.upper()} {expiry} {strike} {request.right} contract is listed"
            ),
        )
    row = match.to_dicts()[0]
    return {
        "contract_id": str(row["contract_id"]),
        "premium": float(row["close"] or 0.0),
        "underlying_price": float(row["underlying_price"] or 0.0),
        "lot_size": float(row["lot_size"] or 0.0),
        "expiry": expiry,
        "strike": strike,
        "right": str(row["right"]),
    }


def _option_preview(request: OrderRequest, risk: RiskEngine) -> PreviewResponse:
    """Risk verdict and cost for one option order.

    **Sized in lots, and priced on premium.** NSE options trade in whole lots,
    so a quantity that is not a multiple of the lot is not an order the
    exchange will take — refused here rather than rounded, because rounding a
    size is deciding a position on the operator's behalf.

    The risk engine is handed the *premium* outlay, which is what a bought
    option can lose. That is deliberately not the underlying exposure, and both
    are returned so the difference is on the screen rather than in the reader's
    head.
    """
    quote = option_quote(request)
    lot = Decimal(str(quote["lot_size"]))
    premium = Decimal(str(quote["premium"]))
    quantity = _decimal(request.quantity, "quantity")
    equity = _decimal(request.equity, "equity")

    if quantity <= 0:
        raise HTTPException(status_code=422, detail="quantity must be positive")
    if equity <= 0:
        raise HTTPException(status_code=422, detail="equity must be positive")
    if lot <= 0:
        raise HTTPException(status_code=422, detail="the contract has no lot size")
    if quantity % lot != 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"NSE options trade in whole lots: {quantity} is not a multiple of {lot}. "
                f"Nearest lots: {int(quantity // lot)} ({int(quantity // lot) * lot} units) "
                f"or {int(quantity // lot) + 1} ({(int(quantity // lot) + 1) * lot} units)."
            ),
        )
    if premium <= 0:
        raise HTTPException(
            status_code=422,
            detail="the contract has no closing premium — it did not trade in the last session",
        )

    instrument_id = InstrumentId(str(quote["contract_id"]))
    outlay = premium * quantity
    exposure = Decimal(str(quote["underlying_price"])) * quantity

    state = PortfolioState(
        equity=equity,
        cash=equity,
        peak_equity=equity,
        day_start_equity=equity,
        last_prices={instrument_id: premium},
    )
    proposed = ProposedOrder(
        strategy_id=MANUAL_STRATEGY,
        instrument_id=instrument_id,
        quantity=quantity,
        price=premium,
        multiplier=Decimal(1) if request.side == "BUY" else Decimal(-1),
        cluster=_cluster_for(instrument_id),
    )
    verdict = risk.check(proposed, state)
    costs = option_costs(instrument_id, request, quantity, premium, lot)
    impact = outlay + costs if request.side == "BUY" else costs - outlay

    return PreviewResponse(
        symbol=request.symbol.upper(),
        instrument_id=str(instrument_id),
        last_close=f"{premium:.2f}",
        kind="OPTION",
        lot_size=f"{lot:.0f}",
        underlying_exposure=f"{exposure:.2f}",
        side=request.side,
        quantity=str(quantity),
        notional=f"{outlay:.2f}",
        estimated_costs=f"{costs:.2f}",
        cash_impact=f"{impact:.2f}",
        allowed=verdict.allowed,
        breaches=list(verdict.reasons),
        checks=[{"name": c.name, "passed": c.passed, "detail": c.message} for c in verdict.checks],
    )


def option_costs(
    instrument_id: InstrumentId,
    request: OrderRequest,
    quantity: Decimal,
    premium: Decimal,
    lot: Decimal,
) -> Decimal:
    """Charges on one option leg, from the real cost model.

    `NseOptionsCostModel` has existed since the cost work and nothing reached
    it: STT on premium for the sell leg, exchange and SEBI fees, stamp on the
    buy, GST on the fee components but not on STT. Using the flat estimate the
    equity ticket uses would have been wrong in both directions at once, since
    options are charged on premium rather than on notional.
    """
    from core.instruments import AssetClass, Currency, Exchange, Instrument  # noqa: PLC0415
    from core.orders import Side  # noqa: PLC0415
    from engine.costs.india import NseOptionsCostModel  # noqa: PLC0415
    from engine.costs.model import TradeContext  # noqa: PLC0415

    contract = Instrument(
        instrument_id=instrument_id,
        symbol=request.symbol.upper(),
        asset_class=AssetClass.OPTION,
        exchange=Exchange.NSE,
        currency=Currency.INR,
        tick_size=Decimal("0.05"),
        lot_size=int(lot),
        expiry=datetime.fromisoformat(f"{request.expiry}T00:00:00+00:00"),
        strike=_decimal(request.strike or "0", "strike"),
        option_type=OPTION_RIGHTS[request.right or "CE"],
    )
    breakdown = NseOptionsCostModel().cost(
        TradeContext(
            instrument=contract,
            side=Side.BUY if request.side == "BUY" else Side.SELL,
            quantity=quantity,
            price=premium,
        )
    )
    return Decimal(breakdown.total)


def _require_armed(risk: RiskEngine, session: str | None = None) -> None:
    """Refuse unless every gate is open, naming the ones that are not."""
    blocked = [g for g in live_gates(risk, session) if not g.ready]
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
    if request.kind == "OPTION":
        return _option_preview(request, risk)

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
        cluster=_cluster_for(instrument_id),
    )
    verdict = risk.check(proposed, state)
    notional = proposed.notional
    costs = estimated_costs(notional, request.side)
    impact = notional + costs if request.side == "BUY" else costs - notional

    return PreviewResponse(
        symbol=request.symbol.upper(),
        instrument_id=str(instrument_id),
        last_close=f"{last_close:.2f}",
        kind="EQUITY",
        side=request.side,
        quantity=str(quantity),
        notional=f"{notional:.2f}",
        estimated_costs=f"{costs:.2f}",
        cash_impact=f"{impact:.2f}",
        allowed=verdict.allowed,
        breaches=list(verdict.reasons),
        checks=[{"name": c.name, "passed": c.passed, "detail": c.message} for c in verdict.checks],
    )


def _submit(request: OrderRequest, risk: RiskEngine, session: str | None = None) -> SubmitResponse:
    """Place a live order. Reached only with every gate open."""
    _require_armed(risk, session)

    # Imported here, not at module scope: `KiteBroker.__post_init__` calls
    # `require_live_permission`, which raises in every non-live environment.
    # A module-scope import would make this file unimportable in development.
    from core.orders import OrderType, Side  # noqa: PLC0415
    from trading.execution.broker import BrokerError  # noqa: PLC0415
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
    broker = _broker(session)
    try:
        broker_order_id = broker.submit(order, reference)  # type: ignore[attr-defined]
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


def _broker(session: str | None = None) -> LiveBroker:
    """A live broker, or an HTTP error explaining why there is not one.

    Constructed per request rather than held: `KiteBroker` carries a session
    notional budget that is deliberately exhausted rather than reset, and a
    long-lived instance behind a web process would spend that budget across
    unrelated sessions.
    """
    from core.secrets import load_broker_credentials  # noqa: PLC0415

    name = settings.broker.lower()
    # The account's own keys first, then the environment — the same order the
    # gate reports, so what the screen says is connected is what actually
    # places the order.
    credentials = account_credentials(session, name) or load_broker_credentials(name)
    if name == "groww":
        from trading.execution.groww import GrowwBroker  # noqa: PLC0415

        return GrowwBroker(credentials=credentials, instruments={})

    from trading.execution.kite import KiteBroker  # noqa: PLC0415

    return KiteBroker(credentials=credentials, instruments={})


def _working_orders(risk: RiskEngine, session: str | None = None) -> list[WorkingOrder]:
    """Today's order book, read from the venue.

    Empty is a valid answer only when the broker actually said so. Without
    credentials this refuses rather than returning `[]`, because an empty list
    and "I could not ask" look identical on a screen and only one of them means
    you have nothing resting.
    """
    _require_armed(risk)
    from trading.execution.broker import BrokerError  # noqa: PLC0415

    try:
        found = _broker(session).orders()
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


def _held_positions(risk: RiskEngine, session: str | None = None) -> list[HeldPosition]:
    """What the broker believes is held — the reconciliation baseline (§9)."""
    _require_armed(risk)
    from trading.execution.broker import BrokerError  # noqa: PLC0415

    try:
        held = _broker(session).positions()
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


def _executed_fills(risk: RiskEngine, session: str | None = None) -> list[ExecutedFill]:
    """Today's executions, as the venue reports them."""
    _require_armed(risk)
    from trading.execution.broker import BrokerError  # noqa: PLC0415

    try:
        executed = _broker(session).fills_since(None)
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
    def status(session: SessionCookie = None) -> TradeStatusResponse:
        """Whether an order could be placed right now, and what is missing."""
        gates = live_gates(risk, session)
        armed = all(g.ready for g in gates)
        return TradeStatusResponse(
            can_trade=armed, mode="LIVE" if armed else "BLOCKED", gates=gates
        )

    @router.post("/preview", response_model=PreviewResponse, dependencies=[ReadAccess])
    def preview(request: OrderRequest) -> PreviewResponse:
        """Cost and risk verdict for an order, without sending anything."""
        return _preview(request, risk)

    @router.post("/orders", response_model=SubmitResponse, dependencies=[WriteAccess])
    def submit(request: OrderRequest, session: SessionCookie = None) -> SubmitResponse:
        """Place a live order. Every gate must be open."""
        return _submit(request, risk, session)

    @router.get("/orders", response_model=list[WorkingOrder], dependencies=[ReadAccess])
    def orders(session: SessionCookie = None) -> list[WorkingOrder]:
        """Today's order book, read from the venue."""
        return _working_orders(risk, session)

    @router.get("/positions", response_model=list[HeldPosition], dependencies=[ReadAccess])
    def positions(session: SessionCookie = None) -> list[HeldPosition]:
        """What the broker believes is held — the reconciliation baseline (§9)."""
        return _held_positions(risk, session)

    @router.get("/fills", response_model=list[ExecutedFill], dependencies=[ReadAccess])
    def fills(session: SessionCookie = None) -> list[ExecutedFill]:
        """Today's executions, as the venue reports them."""
        return _executed_fills(risk, session)

    @router.delete("/orders/{broker_order_id}", dependencies=[WriteAccess])
    def cancel(broker_order_id: str, session: SessionCookie = None) -> dict[str, str]:
        """Cancel a working order at the venue."""
        _require_armed(risk, session)

        from trading.execution.broker import BrokerError  # noqa: PLC0415

        broker = _broker(session)
        try:
            broker.cancel(broker_order_id)  # type: ignore[attr-defined]
        except BrokerError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"cancelled": broker_order_id}

    return router
