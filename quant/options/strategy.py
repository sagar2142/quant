"""Multi-leg option positions — MASTER_PLAN §M6.

**A chain prices one contract at a time; nobody trades one contract at a time.**
A spread's risk is not the sum of two rows read separately — the whole point of
the structure is that the legs offset, and the offset is exactly what a
strike-by-strike view cannot show. This computes what a combination is actually
worth, what it is exposed to, and where it stops making money.

Three things it will not do:

**It will not invent a bound that does not exist.** A short call loses without
limit. Reporting the worst point of whatever price range happened to be plotted
would produce a finite, confident, wrong number — and it is precisely the
position where that number matters. Unbounded is returned as `None`.

**Nor will it invent one that does.** Only the upside is unbounded, because a
price cannot fall below zero: a long share, a covered call and a short put all
have a finite worst case, and the last of those is the textbook one (the strike
less the premium). Calling them unlimited-risk is not caution, it is noise — it
teaches the reader to skip the warning on the position that has earned it.

**It will not conflate expiry payoff with present value.** The hockey-stick is
what the position pays if held to expiry; it is not what it is worth now, and a
position can be deeply underwater on the first while comfortably ahead on the
second. Both curves are produced, separately.

**It will not aggregate Greeks across underlyings.** Adding the delta of a
RELIANCE call to the delta of an INFY put yields a number with no meaning. The
caller supplies one underlying's legs, and a mixed basket is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from quant.options.pricing import CALL, PUT, Greeks, black_scholes, greeks, time_to_expiry

__all__ = [
    "STRUCTURES",
    "Leg",
    "LegKind",
    "Position",
    "PositionAnalysis",
    "PositionError",
    "analyse",
    "breakevens",
    "expiry_payoff",
    "position_greeks",
    "structure_legs",
    "value_at",
]

#: Points in a payoff curve. Enough that a kink at a strike is visible, few
#: enough that the browser is not asked to draw a thousand-point path.
CURVE_POINTS = 121

#: How far either side of spot a payoff curve is drawn, as a fraction. Wide
#: enough to contain the strikes of an ordinary structure; the analysis reports
#: unboundedness rather than relying on this range to describe the tails.
CURVE_SPAN = 0.35


class PositionError(ValueError):
    """The legs do not describe a position that can be analysed."""


class LegKind(str, Enum):
    CALL = "CE"
    PUT = "PE"
    #: A future or the underlying itself. Delta one, no strike, no volatility —
    #: included because a covered call and a collar are not option-only
    #: structures and pretending otherwise makes them unrepresentable.
    UNDERLYING = "EQ"


@dataclass(frozen=True)
class Leg:
    """One leg, in lots.

    `quantity` is signed: negative is short. Lots rather than units because
    that is how the contract trades — an NSE option is 1 lot of `lot_size`,
    and quoting a position in units invites an order for a quantity the
    exchange will not accept.
    """

    kind: LegKind
    quantity: int
    #: Premium (or price, for the underlying) per unit, as paid or received.
    price: float
    strike: float = 0.0
    #: Implied volatility for present-value marking. `None` means this leg
    #: cannot be marked, and the whole position's present value is withheld
    #: rather than computed from a mixture of real and assumed volatilities.
    implied_vol: float | None = None

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise PositionError("a leg with zero quantity is not a leg")
        if self.kind is not LegKind.UNDERLYING and self.strike <= 0:
            raise PositionError(f"{self.kind.value} leg needs a positive strike")


@dataclass(frozen=True)
class Position:
    """A structure on one underlying."""

    underlying: str
    legs: tuple[Leg, ...]
    spot: float
    lot_size: float
    days_to_expiry: float
    rate: float = 0.065

    def __post_init__(self) -> None:
        if not self.legs:
            raise PositionError("a position needs at least one leg")
        if self.spot <= 0:
            raise PositionError("spot must be positive")
        if self.lot_size <= 0:
            raise PositionError("lot size must be positive")

    @property
    def net_premium(self) -> float:
        """Cash paid (positive) or received (negative) to open, in rupees.

        Signed from the buyer's side throughout: a debit spread costs money and
        a credit spread pays, and the sign is the difference between a maximum
        loss you have already funded and one you have not.
        """
        return sum(leg.quantity * leg.price for leg in self.legs) * self.lot_size


def _leg_payoff(leg: Leg, spot: float) -> float:
    """One leg's value per unit at expiry."""
    if leg.kind is LegKind.UNDERLYING:
        return spot
    if leg.kind is LegKind.CALL:
        return max(spot - leg.strike, 0.0)
    return max(leg.strike - spot, 0.0)


def expiry_payoff(position: Position, spot: float) -> float:
    """Profit or loss in rupees if the underlying settles at `spot`.

    Net of what was paid to open, so zero is break-even rather than "the
    options expired worthless".
    """
    gross = sum(leg.quantity * _leg_payoff(leg, spot) for leg in position.legs)
    return gross * position.lot_size - position.net_premium


def value_at(position: Position, spot: float, days: float | None = None) -> float | None:
    """Mark-to-model profit or loss before expiry, or None if unmarkable.

    Args:
        position: The structure.
        spot: Underlying price to mark at.
        days: Days remaining. Defaults to the position's own.

    Returns:
        None when any option leg has no implied volatility. Deliberately all
        or nothing: a position marked with half its legs at market vol and half
        at a guess is a number whose error nobody can bound.
    """
    remaining = position.days_to_expiry if days is None else days
    years = time_to_expiry(remaining)

    total = 0.0
    for leg in position.legs:
        if leg.kind is LegKind.UNDERLYING:
            total += leg.quantity * spot
            continue
        vol = leg.implied_vol
        if vol is None or vol <= 0:
            return None
        right = CALL if leg.kind is LegKind.CALL else PUT
        priced = black_scholes(spot, leg.strike, years, position.rate, vol, right)
        if priced is None:
            return None
        total += leg.quantity * priced
    return total * position.lot_size - position.net_premium


def position_greeks(position: Position, spot: float | None = None) -> Greeks | None:
    """Aggregate sensitivities, per lot-scaled position.

    Returns:
        None if any option leg lacks a volatility, for the same reason
        `value_at` does: a partial aggregate is worse than none, because it
        looks complete.
    """
    price = position.spot if spot is None else spot
    years = time_to_expiry(position.days_to_expiry)

    delta = gamma = vega = theta = rho = 0.0
    for leg in position.legs:
        size = leg.quantity * position.lot_size
        if leg.kind is LegKind.UNDERLYING:
            # Delta one, and no second-order exposure at all. Stated rather
            # than skipped: a covered call's delta is the sum of a short call's
            # and the stock's, and omitting the stock inverts the sign.
            delta += size
            continue
        vol = leg.implied_vol
        if vol is None or vol <= 0:
            return None
        right = CALL if leg.kind is LegKind.CALL else PUT
        one = greeks(price, leg.strike, years, position.rate, vol, right)
        if one is None:
            # Expired, or a degenerate input. Withheld for the same reason a
            # missing volatility is: an aggregate short one leg looks complete.
            return None
        delta += size * one.delta
        gamma += size * one.gamma
        vega += size * one.vega
        theta += size * one.theta
        rho += size * one.rho

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def breakevens(curve: list[tuple[float, float]]) -> list[float]:
    """Prices where the expiry payoff crosses zero.

    Found by sign change and linear interpolation between adjacent points. A
    kink exactly on a strike can sit between samples, so these are accurate to
    the curve's resolution and not to the paisa — which is the honest precision
    for a number computed from a plotted line.
    """
    crossings: list[float] = []
    for (x0, y0), (x1, y1) in pairwise(curve):
        if y0 == 0.0:
            crossings.append(x0)
        elif (y0 < 0) != (y1 < 0):
            crossings.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))
    return crossings


def _upside_slope(position: Position) -> float:
    """Payoff gradient per unit above every strike.

    Above every strike, puts are worthless and calls pay one-for-one, so this
    is the exact asymptotic slope — which is what decides whether a bound
    exists at all, rather than reading the worst point of a plotted range and
    calling it the worst case.

    **Only the upside can be unbounded.** A price cannot go below zero, so the
    downside always terminates at spot 0 and every position has a finite worst
    case on the way down. Treating the down tail as running to negative
    infinity reported "unbounded loss" for a long share, a covered call and a
    short put — the last of which is the textbook finite number (the strike
    less the premium) and the first two of which are among the most
    conservative positions there are. A builder that calls a covered call
    unlimited-risk teaches its reader to ignore the warning on the one
    position that has earned it.
    """
    return float(
        sum(leg.quantity for leg in position.legs if leg.kind in (LegKind.UNDERLYING, LegKind.CALL))
    )


@dataclass(frozen=True)
class PositionAnalysis:
    """What a structure is, in the terms a trader decides on."""

    underlying: str
    spot: float
    days_to_expiry: float
    net_premium: float
    #: (price, profit) at expiry.
    payoff: list[tuple[float, float]]
    #: (price, profit) marked now, or empty when a leg has no volatility.
    value: list[tuple[float, float]]
    breakevens: list[float]
    #: None means unbounded. A short call's loss has no maximum, and a number
    #: taken from the edge of a chart would be a confident fiction.
    max_profit: float | None
    max_loss: float | None
    greeks: Greeks | None
    #: Why a bound is missing, in words, when one is.
    note: str = ""


def analyse(
    position: Position,
    span: float = CURVE_SPAN,
    points: int = CURVE_POINTS,
) -> PositionAnalysis:
    """Payoff, present value, break-evens, bounds and Greeks.

    Args:
        position: The structure.
        span: Fraction of spot to plot either side.
        points: Samples in each curve.
    """
    strikes = [leg.strike for leg in position.legs if leg.kind is not LegKind.UNDERLYING]
    low = min([position.spot * (1 - span), *strikes]) * 0.9
    high = max([position.spot * (1 + span), *strikes]) * 1.1
    step = (high - low) / max(1, points - 1)

    # The strikes are in the grid, not merely near it. An expiry payoff is
    # piecewise linear with a kink at every strike, so its maximum and minimum
    # are *always* attained at a strike or at the tails — a uniform grid that
    # steps over 2500 by four rupees reports a straddle's worst case as 5%
    # better than it is, and does so with no sign that anything was missed.
    grid = sorted({low + step * i for i in range(points)} | set(strikes))
    payoff = [(price, expiry_payoff(position, price)) for price in grid]

    marked = [value_at(position, price) for price in grid]
    value = (
        [(price, v) for price, v in zip(grid, marked, strict=True) if v is not None]
        if all(v is not None for v in marked)
        else []
    )

    # The worthless case, whether or not it is plotted. A payoff's downside
    # extremum sits at spot 0, which is usually far outside the display range —
    # reading the bound off the drawn curve would report a short put's worst
    # case as whatever the left edge of the chart happened to be.
    profits = [p for _, p in payoff] + [expiry_payoff(position, 0.0)]

    # The upside slope decides both bounds, not the plotted range. A positive
    # slope means profit grows without limit, a negative one means loss does,
    # and zero means the payoff is flat out there — so the extreme of the
    # sampled interior really is the extreme.
    up = _upside_slope(position)
    unbounded_profit = up > 0
    unbounded_loss = up < 0

    notes: list[str] = []
    if unbounded_loss:
        notes.append("loss is unbounded — a naked short call has no worst case")
    if not value:
        notes.append("present value withheld: a leg has no implied volatility")

    return PositionAnalysis(
        underlying=position.underlying,
        spot=position.spot,
        days_to_expiry=position.days_to_expiry,
        net_premium=position.net_premium,
        payoff=payoff,
        value=value,
        breakevens=breakevens(payoff),
        max_profit=None if unbounded_profit else max(profits),
        max_loss=None if unbounded_loss else min(profits),
        greeks=position_greeks(position),
        note="; ".join(notes),
    )


#: The structures a builder offers, as (leg kind, quantity, strike index).
#: `strike index` selects from the strikes the caller supplies, ascending, so
#: one description serves every width.
STRUCTURES: dict[str, tuple[str, tuple[tuple[LegKind, int, int], ...]]] = {
    "long_call": ("Long call", ((LegKind.CALL, 1, 0),)),
    "long_put": ("Long put", ((LegKind.PUT, 1, 0),)),
    "short_call": ("Short call", ((LegKind.CALL, -1, 0),)),
    "short_put": ("Short put", ((LegKind.PUT, -1, 0),)),
    "bull_call_spread": ("Bull call spread", ((LegKind.CALL, 1, 0), (LegKind.CALL, -1, 1))),
    "bear_put_spread": ("Bear put spread", ((LegKind.PUT, 1, 1), (LegKind.PUT, -1, 0))),
    "long_straddle": ("Long straddle", ((LegKind.CALL, 1, 0), (LegKind.PUT, 1, 0))),
    "short_straddle": ("Short straddle", ((LegKind.CALL, -1, 0), (LegKind.PUT, -1, 0))),
    "long_strangle": ("Long strangle", ((LegKind.PUT, 1, 0), (LegKind.CALL, 1, 1))),
    "short_strangle": ("Short strangle", ((LegKind.PUT, -1, 0), (LegKind.CALL, -1, 1))),
    "iron_condor": (
        "Iron condor",
        (
            (LegKind.PUT, 1, 0),
            (LegKind.PUT, -1, 1),
            (LegKind.CALL, -1, 2),
            (LegKind.CALL, 1, 3),
        ),
    ),
    "call_butterfly": (
        "Call butterfly",
        ((LegKind.CALL, 1, 0), (LegKind.CALL, -2, 1), (LegKind.CALL, 1, 2)),
    ),
    "covered_call": ("Covered call", ((LegKind.UNDERLYING, 1, 0), (LegKind.CALL, -1, 0))),
    "collar": (
        "Collar",
        ((LegKind.UNDERLYING, 1, 0), (LegKind.PUT, 1, 0), (LegKind.CALL, -1, 1)),
    ),
}


def structure_legs(name: str, strikes: list[float]) -> tuple[tuple[LegKind, int, float], ...]:
    """Expand a named structure against a list of strikes, ascending.

    Raises:
        PositionError: if the name is unknown, or too few strikes were given.
            Refused rather than padded — a butterfly built from two strikes is
            a vertical spread wearing the wrong name, and it would price and
            plot perfectly well while being a different position entirely.
    """
    if name not in STRUCTURES:
        raise PositionError(f"unknown structure {name!r}; expected one of {sorted(STRUCTURES)}")
    _label, template = STRUCTURES[name]
    needed = max(index for _, _, index in template) + 1
    if len(strikes) < needed:
        raise PositionError(f"{name} needs {needed} strike(s), got {len(strikes)}")

    ordered = sorted(strikes)
    return tuple((kind, quantity, ordered[index]) for kind, quantity, index in template)
