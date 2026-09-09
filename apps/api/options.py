"""Option chain endpoints — MASTER_PLAN §13.4.

**The chain is assembled, not stored.** The bhavcopy holds one row per
contract: a price, an open interest, a lot size. A chain is calls and puts laid
side by side at each strike, with implied volatility and Greeks that exist
nowhere in the file and are computed here under assumptions the caller can see
and change.

**The rate is a parameter with a stated default, not a constant.** Every
implied volatility on the screen depends on it, and a surface computed under a
rate nobody chose is a surface nobody can defend.

**Rows where the maths does not work are returned with nulls, not dropped.** A
far strike that has not traded prints a stale price no volatility explains, and
its open interest is still information — often the most interesting information
on the screen. Dropping the row would hide a large position because its IV
could not be inverted.
"""

from __future__ import annotations

import math
from datetime import date

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.auth import ReadAccess
from core.clock import as_decision_time, utc_now
from core.config import settings
from data.store.bars import NoDataError
from data.store.derivatives import DerivativesStore
from quant.options.pricing import CALL, PUT, greeks, implied_volatility, time_to_expiry
from quant.options.strategy import (
    STRUCTURES,
    Leg,
    LegKind,
    Position,
    PositionError,
    analyse,
)

__all__ = ["DEFAULT_RATE", "analyse_strategy", "build_options_router"]

#: Annualised risk-free rate used when the caller does not supply one.
#:
#: Roughly the 91-day Indian T-bill. It is a default rather than a truth: the
#: whole implied-volatility surface moves with it, so it is exposed as a query
#: parameter and named here where it can be argued with.
DEFAULT_RATE = 0.065

#: Strikes either side of the money returned by default. A full chain on a
#: liquid name is several hundred rows, most of them untraded; the interesting
#: ones are near the underlying.
DEFAULT_STRIKE_WINDOW = 15


class ContractLeg(BaseModel):
    """One side of a strike."""

    contract_id: str
    close: float
    settlement: float
    open_interest: float
    oi_change: float
    volume: float
    trades: float
    implied_vol: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None


class ChainRow(BaseModel):
    strike: float
    call: ContractLeg | None
    put: ContractLeg | None


class ChainResponse(BaseModel):
    underlying: str
    session: date
    expiry: date
    days_to_expiry: int
    #: The cash close NSE stamped on these contracts.
    underlying_price: float
    #: What every price on this chain was actually computed against, and where
    #: it came from. Reported rather than assumed: a reader comparing an
    #: implied volatility here against another screen needs to know which
    #: underlying produced it.
    pricing_price: float
    pricing_source: str
    lot_size: float
    rate: float
    rows: list[ChainRow]
    #: Strike whose call and put are closest in value — the market's own
    #: forward, and a better centre for a chain than the spot when the two
    #: disagree.
    atm_strike: float | None


def _store() -> DerivativesStore:
    return DerivativesStore(settings.lake)


def _f(value: object) -> float:
    """A polars cell as a float, treating null and unparseable as zero.

    Zero rather than None because these are prices and open interests on a
    contract that exists: a strike nobody traded has an open interest of zero,
    which is a fact. The quantities that genuinely may not exist — implied
    volatility and the Greeks — stay `None` and are never coerced here.
    """
    if value is None:
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


#: Paired strikes needed before the chain is allowed to price itself.
#:
#: Three is enough for a median to survive one stale quote and few enough that
#: a thin chain still gets a forward. Below it the cash close is the honest
#: fallback — a forward inferred from two strikes is a guess with a decimal
#: point.
MIN_PARITY_PAIRS = 3

#: How far the *middle half* of the estimates may disagree, as a fraction of
#: the median, before the chain is treated as inconsistent.
#:
#: **Measured across the middle half, not end to end.** The first version of
#: this checked min against max, which is not a robust statistic and threw away
#: the reason for taking a median in the first place. On a real RELIANCE chain
#: it rejected a perfectly good forward: thirty-one strikes spanned 1.60% end to
#: end and 0.19% across the middle half, and every outlier was a strike that had
#: not traded — 1180 implied 1324.50 on zero call volume, 1430 implied 1316.79
#: on zero put volume. One untraded far strike is enough to veto the whole
#: chain when the test is min against max.
MAX_PARITY_SPREAD = 0.005


def implied_underlying(
    by_strike: dict[float, dict[str, dict[str, object]]], years: float, rate: float
) -> float | None:
    """The underlying price the option quotes are consistent with, or None.

    **The cash close is the wrong number to price these with.** NSE stamps
    `UndrlygPric` from the cash market, but Indian stock options trade against
    the futures, which carry. On RELIANCE that gap was ₹6.50 — cash at 1302.50
    against 1309 — and pricing off the smaller number pushed call implied
    volatility up and put implied volatility down by about five points each,
    inventing a skew that is not in the market. Every Greek inherited the same
    bias.

    Put-call parity gives the right number without needing the futures leg at
    all: `C - P = S - K·e^(-rT)`, so each paired strike is one estimate of S,
    and a real chain agrees across all of them. The median is taken rather than
    the mean because one untraded far strike prints a stale quote, and a mean
    would let it move the whole surface.

    Returns None when the chain cannot vouch for itself — too few pairs, or
    estimates that disagree by more than `MAX_PARITY_SPREAD`. The caller then
    falls back to the cash close and says so, because a made-up forward would
    be worse than a known-biased one nobody was told about.
    """
    if years <= 0:
        return None
    discount = math.exp(-rate * years)

    # Both legs must have traded. A strike that printed no volume carries an
    # older quote against today's other leg, and parity across those two
    # describes nothing. On a real RELIANCE chain this is the single largest
    # source of disagreement — 1180 implied 1324.50 on zero call volume, 1430
    # implied 1316.79 on zero put volume — and excluding it costs only the
    # strikes nobody wanted.
    estimates = sorted(
        (_f(legs[CALL]["close"]) - _f(legs[PUT]["close"])) + strike * discount
        for strike, legs in by_strike.items()
        if CALL in legs
        and PUT in legs
        and _f(legs[CALL]["close"]) > 0
        and _f(legs[PUT]["close"]) > 0
        and _f(legs[CALL]["volume"]) > 0
        and _f(legs[PUT]["volume"]) > 0
    )
    if len(estimates) < MIN_PARITY_PAIRS:
        return None

    middle = estimates[len(estimates) // 2]
    if middle <= 0:
        return None
    # Across the middle half, not end to end. See MAX_PARITY_SPREAD.
    low = estimates[len(estimates) // 4]
    high = estimates[(3 * len(estimates)) // 4]
    if (high - low) / middle > MAX_PARITY_SPREAD:
        return None
    return middle


def _leg(row: dict[str, object], spot: float, years: float, rate: float) -> ContractLeg:
    """One contract, priced.

    IV and Greeks are None together: a Greek computed from a volatility that
    could not be inverted would be a number derived from nothing.
    """
    close = _f(row["close"])
    strike = _f(row["strike"])
    right = str(row["right"])
    vol = implied_volatility(close, spot, strike, years, rate, right)
    sensitivities = greeks(spot, strike, years, rate, vol, right) if vol else None
    return ContractLeg(
        contract_id=str(row["contract_id"]),
        close=close,
        settlement=_f(row["settlement"]),
        open_interest=_f(row["open_interest"]),
        oi_change=_f(row["oi_change"]),
        volume=_f(row["volume"]),
        trades=_f(row["trades"]),
        implied_vol=vol,
        delta=sensitivities.delta if sensitivities else None,
        gamma=sensitivities.gamma if sensitivities else None,
        vega=sensitivities.vega if sensitivities else None,
        theta=sensitivities.theta if sensitivities else None,
    )


def assemble_chain(
    underlying: str,
    expiry: date | None,
    rate: float,
    window: int,
) -> ChainResponse:
    """Calls and puts side by side at each strike, with IV and Greeks.

    Module-level rather than nested in the router so the assembly can be
    tested without standing up an application, which is the only way to
    check the pricing against a real chain rather than a fixture.
    """
    as_of = as_decision_time(utc_now())
    try:
        held = _store().chain(underlying, as_of=as_of)
    except NoDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if held.is_empty():
        raise HTTPException(status_code=404, detail=f"{underlying.upper()} has no listed contracts")

    options = held.filter(held["right"].is_in([CALL, PUT]))
    if options.is_empty():
        raise HTTPException(
            status_code=404, detail=f"{underlying.upper()} lists futures but no options"
        )

    available = sorted(set(options["expiry"].to_list()))
    wanted = expiry or available[0]
    if wanted not in available:
        raise HTTPException(
            status_code=404,
            detail=f"{underlying.upper()} has no {wanted} expiry; listed: "
            + ", ".join(d.isoformat() for d in available[:8]),
        )

    slice_ = options.filter(options["expiry"] == wanted)
    session = slice_["event_time"][0].date()
    spot = _f(slice_["underlying_price"].max())
    lot = _f(slice_["lot_size"].max())
    days = (wanted - session).days
    years = time_to_expiry(days)

    by_strike: dict[float, dict[str, dict[str, object]]] = {}
    for row in slice_.to_dicts():
        strike = _f(row["strike"])
        by_strike.setdefault(strike, {})[str(row["right"])] = row

    # What the options are actually trading against, which is not the cash
    # close NSE stamps on them. See `implied_underlying`.
    parity = implied_underlying(by_strike, years, rate)
    pricing_spot = parity if parity is not None else spot

    strikes = sorted(by_strike)
    # Centred on the money. A full chain is mostly strikes nobody trades,
    # and returning all of them buries the ones that matter.
    if spot > 0 and len(strikes) > window * 2:
        nearest = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        strikes = strikes[max(0, nearest - window) : nearest + window + 1]

    rows: list[ChainRow] = []
    atm_strike: float | None = None
    smallest_gap = float("inf")
    for strike in strikes:
        legs = by_strike[strike]
        call_row = legs.get(CALL)
        put_row = legs.get(PUT)
        call = _leg(call_row, pricing_spot, years, rate) if call_row else None
        put = _leg(put_row, pricing_spot, years, rate) if put_row else None
        if call and put:
            gap = abs(call.close - put.close)
            if gap < smallest_gap:
                smallest_gap = gap
                atm_strike = strike
        rows.append(ChainRow(strike=strike, call=call, put=put))

    return ChainResponse(
        underlying=underlying.upper(),
        session=session,
        expiry=wanted,
        days_to_expiry=days,
        underlying_price=spot,
        pricing_price=pricing_spot,
        pricing_source="parity" if parity is not None else "cash",
        lot_size=lot,
        rate=rate,
        rows=rows,
        atm_strike=atm_strike,
    )


class StrategyLeg(BaseModel):
    """One leg of a proposed structure, as the console describes it."""

    kind: str = Field(pattern="^(CE|PE|EQ)$")
    #: Signed lots. Negative is short.
    quantity: int = Field(ge=-100, le=100)
    price: float = Field(ge=0)
    strike: float = Field(default=0.0, ge=0)
    implied_vol: float | None = Field(default=None, ge=0, le=5)


class StrategyRequest(BaseModel):
    underlying: str
    spot: float = Field(gt=0)
    lot_size: float = Field(gt=0, le=100_000)
    days_to_expiry: float = Field(ge=0, le=1000)
    rate: float = Field(default=DEFAULT_RATE, ge=0, le=1)
    legs: list[StrategyLeg] = Field(min_length=1, max_length=8)


class StrategyPoint(BaseModel):
    price: float
    profit: float


class StrategyResponse(BaseModel):
    """What a multi-leg position is, in the terms a trader decides on."""

    underlying: str
    spot: float
    days_to_expiry: float
    net_premium: float
    payoff: list[StrategyPoint]
    #: Marked at today's volatilities. Empty when a leg has none — a curve
    #: built from a mixture of market and assumed vols is unfalsifiable.
    value: list[StrategyPoint]
    breakevens: list[float]
    #: null means unbounded. A short call's loss has no maximum, and a number
    #: taken from the edge of the plotted range would be a confident fiction.
    max_profit: float | None
    max_loss: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    rho: float | None
    note: str


def analyse_strategy(request: StrategyRequest) -> StrategyResponse:
    """Payoff, present value, bounds and aggregate Greeks for one structure."""
    try:
        position = Position(
            underlying=request.underlying.upper(),
            legs=tuple(
                Leg(
                    kind=LegKind(leg.kind),
                    quantity=leg.quantity,
                    price=leg.price,
                    strike=leg.strike,
                    implied_vol=leg.implied_vol,
                )
                for leg in request.legs
            ),
            spot=request.spot,
            lot_size=request.lot_size,
            days_to_expiry=request.days_to_expiry,
            rate=request.rate,
        )
    except PositionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = analyse(position)
    sensitivities = result.greeks
    return StrategyResponse(
        underlying=result.underlying,
        spot=result.spot,
        days_to_expiry=result.days_to_expiry,
        net_premium=result.net_premium,
        payoff=[StrategyPoint(price=p, profit=v) for p, v in result.payoff],
        value=[StrategyPoint(price=p, profit=v) for p, v in result.value],
        breakevens=result.breakevens,
        max_profit=result.max_profit,
        max_loss=result.max_loss,
        delta=sensitivities.delta if sensitivities else None,
        gamma=sensitivities.gamma if sensitivities else None,
        vega=sensitivities.vega if sensitivities else None,
        theta=sensitivities.theta if sensitivities else None,
        rho=sensitivities.rho if sensitivities else None,
        note=result.note,
    )


def build_options_router() -> APIRouter:
    """Underlyings, expiries, the chain, and multi-leg structures."""
    router = APIRouter(prefix="/options", tags=["options"])

    @router.get("/structures", dependencies=[ReadAccess])
    def structures() -> list[dict[str, object]]:
        """The named structures a builder offers, with how many strikes each
        needs. A closed set: a butterfly built from two strikes is a vertical
        spread wearing the wrong name, and it would plot perfectly well."""
        return [
            {
                "name": name,
                "label": label,
                "strikes": max(index for _, _, index in template) + 1,
                "legs": [
                    {"kind": kind.value, "quantity": quantity, "strike_index": index}
                    for kind, quantity, index in template
                ],
            }
            for name, (label, template) in STRUCTURES.items()
        ]

    @router.post("/strategy", response_model=StrategyResponse, dependencies=[ReadAccess])
    def strategy(request: StrategyRequest) -> StrategyResponse:
        """Analyse a multi-leg position.

        A chain prices one contract at a time and nobody trades one contract at
        a time. The offset between legs is the whole point of a structure, and
        it is exactly what a strike-by-strike view cannot show.
        """
        return analyse_strategy(request)

    @router.get("/underlyings", dependencies=[ReadAccess])
    def underlyings(q: str = Query("", max_length=32)) -> list[str]:
        """Every underlying with listed contracts, optionally filtered.

        This is what makes options searchable: the equity panel does not know
        which of its names have a derivatives market, and 216 of roughly three
        thousand do.
        """
        try:
            names = _store().underlyings(as_of=as_decision_time(utc_now()))
        except NoDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        prefix = q.strip().upper()
        return [n for n in names if not prefix or n.startswith(prefix)]

    @router.get("/{underlying}/expiries", dependencies=[ReadAccess])
    def expiries(underlying: str) -> list[dict[str, object]]:
        """Expiries listed on one underlying, nearest first."""
        as_of = as_decision_time(utc_now())
        try:
            chain = _store().chain(underlying, as_of=as_of)
        except NoDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if chain.is_empty():
            raise HTTPException(
                status_code=404, detail=f"{underlying.upper()} has no listed contracts"
            )

        session = chain["event_time"][0].date()
        out: list[dict[str, object]] = []
        for expiry in sorted(set(chain["expiry"].to_list())):
            leg = chain.filter(chain["expiry"] == expiry)
            out.append(
                {
                    "expiry": expiry.isoformat(),
                    "days": (expiry - session).days,
                    "contracts": leg.height,
                    "open_interest": _f(leg["open_interest"].sum()),
                }
            )
        return out

    @router.get("/{underlying}/chain", response_model=ChainResponse, dependencies=[ReadAccess])
    def chain(
        underlying: str,
        # No `Query(...)` wrapper: FastAPI reads an optional scalar as a query
        # parameter from the annotation, and the wrapper adds nothing but a
        # call in a default argument.
        expiry: date | None = None,
        rate: float = Query(DEFAULT_RATE, ge=0, le=1),
        window: int = Query(DEFAULT_STRIKE_WINDOW, ge=1, le=200),
    ) -> ChainResponse:
        """Calls and puts side by side, with IV and Greeks."""
        return assemble_chain(underlying, expiry, rate, window)

    return router
