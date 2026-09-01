"""Option pricing — MASTER_PLAN §6.

The tests worth having here are the identities. A pricing function can be
wrong in ways that look entirely plausible on a screen — a sign error in a put,
a day count that halves theta — and the only cheap defence is that the maths
must satisfy relationships nobody chose: put-call parity, delta parity, and
inversion returning what it was given.
"""

from __future__ import annotations

import math

import pytest

from quant.options.pricing import (
    CALL,
    DAYS_PER_YEAR,
    PUT,
    black_scholes,
    greeks,
    implied_volatility,
    intrinsic_value,
    no_arbitrage_floor,
    time_to_expiry,
)

SPOT = 100.0
STRIKE = 100.0
YEARS = 0.25
RATE = 0.065
VOL = 0.30


class TestIdentities:
    """Relationships the formula cannot violate if it is right."""

    def test_put_call_parity(self):
        """C - P = S - Ke^-rT. A sign error anywhere breaks this."""
        call = black_scholes(SPOT, STRIKE, YEARS, RATE, VOL, CALL)
        put = black_scholes(SPOT, STRIKE, YEARS, RATE, VOL, PUT)
        assert call is not None and put is not None
        forward = SPOT - STRIKE * math.exp(-RATE * YEARS)
        assert call - put == pytest.approx(forward, abs=1e-9)

    def test_delta_parity(self):
        """Call delta minus put delta is one, with no dividend."""
        call = greeks(SPOT, STRIKE, YEARS, RATE, VOL, CALL)
        put = greeks(SPOT, STRIKE, YEARS, RATE, VOL, PUT)
        assert call is not None and put is not None
        assert call.delta - put.delta == pytest.approx(1.0, abs=1e-9)

    def test_gamma_and_vega_do_not_depend_on_the_right(self):
        """Both are second-order in the same variable, so a call and a put on
        the same terms share them."""
        call = greeks(SPOT, STRIKE, YEARS, RATE, VOL, CALL)
        put = greeks(SPOT, STRIKE, YEARS, RATE, VOL, PUT)
        assert call is not None and put is not None
        assert call.gamma == pytest.approx(put.gamma)
        assert call.vega == pytest.approx(put.vega)

    def test_a_call_is_worth_more_when_volatility_rises(self):
        cheap = black_scholes(SPOT, STRIKE, YEARS, RATE, 0.20, CALL)
        dear = black_scholes(SPOT, STRIKE, YEARS, RATE, 0.40, CALL)
        assert cheap is not None and dear is not None
        assert dear > cheap

    def test_value_never_falls_below_intrinsic(self):
        deep = black_scholes(150.0, 100.0, YEARS, RATE, VOL, CALL)
        assert deep is not None
        assert deep >= intrinsic_value(150.0, 100.0, CALL) - 1e-9


class TestImpliedVolatility:
    def test_it_recovers_the_volatility_it_was_priced_with(self):
        price = black_scholes(SPOT, STRIKE, YEARS, RATE, VOL, CALL)
        assert price is not None
        assert implied_volatility(price, SPOT, STRIKE, YEARS, RATE, CALL) == pytest.approx(
            VOL, abs=1e-5
        )

    @pytest.mark.parametrize("right", [CALL, PUT])
    @pytest.mark.parametrize("moneyness", [0.8, 0.95, 1.0, 1.05, 1.25])
    def test_it_round_trips_across_the_chain(self, right, moneyness):
        """Deep strikes are where a Newton solver fails: vega goes to zero and
        the iteration divides by it. Bisection has to hold everywhere."""
        strike = SPOT * moneyness
        price = black_scholes(SPOT, strike, YEARS, RATE, VOL, right)
        assert price is not None
        recovered = implied_volatility(price, SPOT, strike, YEARS, RATE, right)
        assert recovered is not None
        assert recovered == pytest.approx(VOL, abs=1e-4)

    def test_a_deep_itm_put_below_intrinsic_still_inverts(self):
        """The bug this test was written for.

        A European put deep in the money is worth *less* than `K - S`, because
        exercise pays the strike at expiry rather than today. A 125 strike on a
        100 spot is worth 23.62 against an intrinsic of 25.00. Flooring the
        inversion at intrinsic rejected every such quote, which on a chain
        reads as "these strikes have no volatility" rather than as a bug.
        """
        strike = 125.0
        price = black_scholes(SPOT, strike, YEARS, RATE, VOL, PUT)
        assert price is not None
        assert price < intrinsic_value(SPOT, strike, PUT), "the premise of this test"
        assert price >= no_arbitrage_floor(SPOT, strike, YEARS, RATE, PUT)

        recovered = implied_volatility(price, SPOT, strike, YEARS, RATE, PUT)
        assert recovered is not None
        assert recovered == pytest.approx(VOL, abs=1e-4)

    def test_a_price_below_the_no_arbitrage_floor_has_none(self):
        """Below the *discounted* floor there really is no volatility."""
        strike = 125.0
        floor = no_arbitrage_floor(SPOT, strike, YEARS, RATE, PUT)
        assert implied_volatility(floor - 1.0, SPOT, strike, YEARS, RATE, PUT) is None

    def test_a_price_below_intrinsic_has_no_implied_volatility(self):
        """Not a low one. No volatility produces it, and returning a number
        would put a fabricated point on the surface."""
        assert implied_volatility(0.5, 150.0, 100.0, YEARS, RATE, CALL) is None

    def test_an_expired_option_has_no_implied_volatility(self):
        assert implied_volatility(5.0, SPOT, STRIKE, 0.0, RATE, CALL) is None

    def test_an_untraded_contract_has_no_implied_volatility(self):
        assert implied_volatility(0.0, SPOT, STRIKE, YEARS, RATE, CALL) is None


class TestRefusals:
    """Where the answer does not exist, say so (§14.1.5)."""

    def test_greeks_past_expiry(self):
        assert greeks(SPOT, STRIKE, 0.0, RATE, VOL, CALL) is None

    def test_greeks_at_zero_volatility(self):
        assert greeks(SPOT, STRIKE, YEARS, RATE, 0.0, CALL) is None

    def test_price_at_expiry_is_intrinsic_not_an_error(self):
        """A limit that exists is returned, not refused."""
        assert black_scholes(120.0, 100.0, 0.0, RATE, VOL, CALL) == 20.0
        assert black_scholes(80.0, 100.0, 0.0, RATE, VOL, CALL) == 0.0

    def test_a_negative_spot_is_refused(self):
        assert black_scholes(-1.0, STRIKE, YEARS, RATE, VOL, CALL) is None


class TestDayCount:
    def test_the_year_is_calendar_days(self):
        """Not 252. An option decays over a weekend, and a trading-day count
        would price three days of theta as one every Friday."""
        assert DAYS_PER_YEAR == 365.0
        assert time_to_expiry(365) == pytest.approx(1.0)

    def test_time_to_expiry_is_never_negative(self):
        assert time_to_expiry(-10) == 0.0

    def test_theta_is_per_day_not_per_year(self):
        """A quarter-year ATM option loses single-digit paise a day at these
        levels, not a year's worth."""
        sensitivities = greeks(SPOT, STRIKE, YEARS, RATE, VOL, CALL)
        assert sensitivities is not None
        assert -1.0 < sensitivities.theta < 0.0
