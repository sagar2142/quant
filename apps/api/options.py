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

from datetime import date

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from apps.api.auth import ReadAccess
from core.clock import as_decision_time, utc_now
from core.config import settings
from data.store.bars import NoDataError
from data.store.derivatives import DerivativesStore
from quant.options.pricing import CALL, PUT, greeks, implied_volatility, time_to_expiry

__all__ = ["DEFAULT_RATE", "build_options_router"]

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
    underlying_price: float
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
        call = _leg(call_row, spot, years, rate) if call_row else None
        put = _leg(put_row, spot, years, rate) if put_row else None
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
        lot_size=lot,
        rate=rate,
        rows=rows,
        atm_strike=atm_strike,
    )


def build_options_router() -> APIRouter:
    """Underlyings, expiries and the chain."""
    router = APIRouter(prefix="/options", tags=["options"])

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
