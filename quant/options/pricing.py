"""Option pricing and implied volatility — MASTER_PLAN §6.

**Black-Scholes, because NSE options are European.** Stock options on the NSE
have been European-style since 2011 and index options always were, so there is
no early-exercise premium to model and the closed form is the right one rather
than a convenient approximation. If this system ever trades an American option,
this file is wrong for it and must say so rather than quietly mispricing.

**Implied volatility is inverted, not read.** The bhavcopy carries no IV and no
Greek — it carries a price. Everything here is derived from that price under
assumptions that are arguments rather than constants buried in a formula: the
rate, the dividend yield, the day count. A surface computed under a rate nobody
chose is a surface nobody can defend.

**The inversion can fail, and says so.** A price below intrinsic value, a
contract that has not traded, an expiry in the past — none of these have an
implied volatility, and returning a plausible number for them would be worse
than returning nothing. Every function here answers `None` where the answer
does not exist (§14.1.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CALL",
    "DAYS_PER_YEAR",
    "PUT",
    "Greeks",
    "black_scholes",
    "greeks",
    "implied_volatility",
    "intrinsic_value",
    "no_arbitrage_floor",
    "time_to_expiry",
]

CALL = "CE"
PUT = "PE"

#: Calendar days per year for the day count.
#:
#: Calendar rather than trading days: the discounting is calendar-time, and an
#: option decays over a weekend. Using 252 here would make a Friday option look
#: cheaper than it is by pricing three days of theta as one.
DAYS_PER_YEAR = 365.0

#: Bounds for the volatility search, and when to stop.
#:
#: 1000% is not a realistic volatility; it is a bound wide enough that hitting
#: it means the price is not invertible rather than that the search was too
#: narrow, which is a distinction worth being able to make.
MIN_VOL = 1e-4
MAX_VOL = 10.0
TOLERANCE = 1e-6
MAX_ITERATIONS = 100


@dataclass(frozen=True)
class Greeks:
    """First and second-order sensitivities.

    Per unit of underlying and per contract-year, in the conventional units:
    delta per 1 of spot, gamma per 1 squared, vega per 1.00 of volatility (not
    per point), theta per calendar day, rho per 1.00 of rate.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, via erf. No SciPy needed for one function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def time_to_expiry(days: float) -> float:
    """Years to expiry from calendar days. Never negative."""
    return max(days, 0.0) / DAYS_PER_YEAR


def intrinsic_value(spot: float, strike: float, right: str) -> float:
    """What the option is worth if exercised now."""
    return max(spot - strike, 0.0) if right == CALL else max(strike - spot, 0.0)


def no_arbitrage_floor(  # noqa: PLR0913, PLR0917 - a contract is its terms
    spot: float,
    strike: float,
    years: float,
    rate: float,
    right: str,
    dividend_yield: float = 0.0,
) -> float:
    """Lowest price a European option can trade at without arbitrage.

    **Not the intrinsic value, and the difference is not academic.** A European
    put deep in the money is worth *less* than `K - S`, because exercise pays
    the strike at expiry rather than today: its floor is `Ke^-rT - Se^-qT`. A
    125-strike put on a 100 spot at 6.5% for three months is worth 23.62 with
    an intrinsic of 25.00, and treating that gap as impossible marks a
    perfectly ordinary quote as un-invertible.

    That is exactly what the first version of `implied_volatility` did: it
    floored at intrinsic and returned None for every deep in-the-money put on
    the chain, which on a screen reads as "these strikes have no volatility"
    rather than as a bug.
    """
    discounted_strike = strike * math.exp(-rate * years)
    discounted_spot = spot * math.exp(-dividend_yield * years)
    if right == CALL:
        return max(discounted_spot - discounted_strike, 0.0)
    return max(discounted_strike - discounted_spot, 0.0)


def _d1_d2(  # noqa: PLR0913, PLR0917 - the Black-Scholes terms, named as they are written
    spot: float, strike: float, years: float, rate: float, vol: float, yield_: float
) -> tuple[float, float]:
    variance = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - yield_ + 0.5 * vol * vol) * years) / variance
    return d1, d1 - variance


def black_scholes(  # noqa: PLR0913, PLR0917 - a contract is its terms
    spot: float,
    strike: float,
    years: float,
    rate: float,
    vol: float,
    right: str,
    dividend_yield: float = 0.0,
) -> float | None:
    """Theoretical value of a European option.

    Returns:
        The price, or None when the inputs do not describe a live option — a
        non-positive spot or strike, a non-positive volatility, or an expiry
        that has passed. At expiry the value is the intrinsic value and is
        returned as such rather than as a limit of a formula that divides by
        zero.
    """
    if spot <= 0 or strike <= 0 or vol <= 0:
        return None
    if years <= 0:
        return intrinsic_value(spot, strike, right)

    d1, d2 = _d1_d2(spot, strike, years, rate, vol, dividend_yield)
    discounted_strike = strike * math.exp(-rate * years)
    discounted_spot = spot * math.exp(-dividend_yield * years)
    if right == CALL:
        return discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
    return discounted_strike * _norm_cdf(-d2) - discounted_spot * _norm_cdf(-d1)


def implied_volatility(  # noqa: PLR0913, PLR0917 - a contract is its terms
    price: float,
    spot: float,
    strike: float,
    years: float,
    rate: float,
    right: str,
    dividend_yield: float = 0.0,
) -> float | None:
    """Volatility that reproduces `price`, or None if there is not one.

    Bisection rather than Newton-Raphson. Newton is faster and fails badly at
    exactly the strikes an option screen cares about: vega goes to zero deep in
    and deep out of the money, and dividing by it sends the iteration
    somewhere arbitrary. Bisection cannot diverge, and a hundred iterations of
    it is microseconds.

    Returns None when the price is not invertible — below the no-arbitrage
    floor, at or past expiry, or outside the volatility bracket. The floor is
    the *discounted* one, not intrinsic value: a deep in-the-money European put
    trades below intrinsic legitimately, and rejecting those would blank the
    implied volatility of half the strikes on a real chain.
    Those are real conditions in a real chain: an untraded far strike prints a
    stale price that no volatility explains.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or years <= 0:
        return None
    if price < no_arbitrage_floor(spot, strike, years, rate, right, dividend_yield) - TOLERANCE:
        return None

    low, high = MIN_VOL, MAX_VOL
    low_price = black_scholes(spot, strike, years, rate, low, right, dividend_yield)
    high_price = black_scholes(spot, strike, years, rate, high, right, dividend_yield)
    if low_price is None or high_price is None:
        return None
    # The price must lie inside the bracket, or no volatility in it explains
    # the quote and reporting one would be inventing a number.
    if not (low_price - TOLERANCE <= price <= high_price + TOLERANCE):
        return None

    for _ in range(MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        value = black_scholes(spot, strike, years, rate, mid, right, dividend_yield)
        if value is None:
            return None
        if abs(value - price) < TOLERANCE:
            return mid
        if value < price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def greeks(  # noqa: PLR0913, PLR0917 - a contract is its terms
    spot: float,
    strike: float,
    years: float,
    rate: float,
    vol: float,
    right: str,
    dividend_yield: float = 0.0,
) -> Greeks | None:
    """Sensitivities under Black-Scholes, or None if they do not exist.

    Theta is per calendar day rather than per year, because that is the number
    a position is actually carried against.
    """
    if spot <= 0 or strike <= 0 or vol <= 0 or years <= 0:
        return None

    d1, d2 = _d1_d2(spot, strike, years, rate, vol, dividend_yield)
    carry = math.exp(-dividend_yield * years)
    discount = math.exp(-rate * years)
    root = math.sqrt(years)

    gamma = carry * _norm_pdf(d1) / (spot * vol * root)
    vega = spot * carry * _norm_pdf(d1) * root

    if right == CALL:
        delta = carry * _norm_cdf(d1)
        theta = (
            -spot * carry * _norm_pdf(d1) * vol / (2 * root)
            - rate * strike * discount * _norm_cdf(d2)
            + dividend_yield * spot * carry * _norm_cdf(d1)
        )
        rho = strike * years * discount * _norm_cdf(d2)
    else:
        delta = -carry * _norm_cdf(-d1)
        theta = (
            -spot * carry * _norm_pdf(d1) * vol / (2 * root)
            + rate * strike * discount * _norm_cdf(-d2)
            - dividend_yield * spot * carry * _norm_cdf(-d1)
        )
        rho = -strike * years * discount * _norm_cdf(-d2)

    return Greeks(
        delta=delta,
        gamma=gamma,
        # Per 1.00 of volatility, so a move from 20% to 21% is vega/100.
        vega=vega,
        theta=theta / DAYS_PER_YEAR,
        rho=rho,
    )
