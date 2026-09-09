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


class TestImpliedUnderlying:
    """Which underlying the chain is priced against — MASTER_PLAN §9.

    NSE stamps the *cash* close on option rows, but Indian stock options trade
    against the futures. Pricing off the smaller number pushes call implied
    volatility up and put implied volatility down and invents a skew that is
    not in the market. Put-call parity recovers the right number from the
    quotes themselves.
    """

    def chain(self, spot: float = 1309.0, strikes=(1280, 1300, 1320, 1340), **over):
        """A synthetic chain that satisfies parity exactly at `spot`."""
        import math

        from apps.api.options import _f  # noqa: F401 - import guard only

        rate, years = 0.065, 0.0712
        rows: dict[float, dict[str, dict[str, object]]] = {}
        for k in strikes:
            # C - P = S - K*e^(-rT), split around a plausible call value.
            call = max(1.0, spot - k * math.exp(-rate * years)) + 20.0
            put = call - (spot - k * math.exp(-rate * years))
            rows[float(k)] = {
                CALL: {"close": call, "volume": 1000.0},
                PUT: {"close": put, "volume": 1000.0},
            }
        for k, patch in over.items():
            rows[float(k)] = patch
        return rows, years, rate

    def test_it_recovers_the_underlying_parity_implies(self) -> None:
        from apps.api.options import implied_underlying

        rows, years, rate = self.chain(spot=1309.0)
        found = implied_underlying(rows, years, rate)
        assert found is not None
        assert found == pytest.approx(1309.0, abs=0.05)

    def test_an_untraded_strike_does_not_move_it(self) -> None:
        """The failure this guards: one stale far strike vetoing a good chain.

        On a real RELIANCE chain, 1180 implied 1324.50 on zero call volume and
        1430 implied 1316.79 on zero put volume, while every traded strike
        agreed within a rupee.
        """
        from apps.api.options import implied_underlying

        rows, years, rate = self.chain(spot=1309.0)
        rows[1180.0] = {
            CALL: {"close": 140.0, "volume": 0.0},  # stale, never traded
            PUT: {"close": 1.0, "volume": 500.0},
        }
        found = implied_underlying(rows, years, rate)
        assert found is not None
        assert found == pytest.approx(1309.0, abs=0.05)

    def test_too_few_traded_pairs_is_refused(self) -> None:
        """A forward from two strikes is a guess with a decimal point."""
        from apps.api.options import implied_underlying

        rows, years, rate = self.chain(spot=1309.0, strikes=(1300, 1320))
        assert implied_underlying(rows, years, rate) is None

    def test_an_inconsistent_chain_is_refused(self) -> None:
        """Quotes that no single underlying explains fall back to the cash
        close, which is biased but known — better than an invented forward."""
        from apps.api.options import implied_underlying

        rows, years, rate = self.chain(spot=1309.0)
        for i, k in enumerate(sorted(rows)):
            rows[k][CALL]["close"] = float(rows[k][CALL]["close"]) + i * 40.0
        assert implied_underlying(rows, years, rate) is None

    def test_expired_contracts_have_no_forward(self) -> None:
        from apps.api.options import implied_underlying

        rows, _years, rate = self.chain()
        assert implied_underlying(rows, 0.0, rate) is None

    def test_calls_and_puts_agree_once_priced_off_it(self) -> None:
        """The point of the whole exercise. Priced against the parity
        underlying, a call and a put on the same strike must return the same
        implied volatility — it is the same contract seen from two sides."""
        from apps.api.options import implied_underlying

        rows, years, rate = self.chain(spot=1309.0)
        spot = implied_underlying(rows, years, rate)
        assert spot is not None
        for strike, legs in rows.items():
            call_vol = implied_volatility(
                float(legs[CALL]["close"]), spot, strike, years, rate, CALL
            )
            put_vol = implied_volatility(float(legs[PUT]["close"]), spot, strike, years, rate, PUT)
            if call_vol is None or put_vol is None:
                continue
            assert call_vol == pytest.approx(put_vol, abs=1e-4)
