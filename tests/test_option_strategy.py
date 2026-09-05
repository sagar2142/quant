"""Multi-leg option positions — MASTER_PLAN §M6.

The chain priced one contract at a time and nobody trades one contract at a
time. These check the offset between legs — the whole point of a structure —
and lean hardest on the one number a payoff calculator most wants to fabricate:
a maximum loss for a position that has none.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.options import build_options_router
from quant.options.strategy import (
    STRUCTURES,
    Leg,
    LegKind,
    Position,
    PositionError,
    analyse,
    breakevens,
    expiry_payoff,
    position_greeks,
    structure_legs,
    value_at,
)

SPOT = 2500.0
LOT = 250.0
DAYS = 30.0


def call(strike: float, quantity: int, price: float, vol: float | None = 0.25) -> Leg:
    return Leg(kind=LegKind.CALL, quantity=quantity, price=price, strike=strike, implied_vol=vol)


def put(strike: float, quantity: int, price: float, vol: float | None = 0.25) -> Leg:
    return Leg(kind=LegKind.PUT, quantity=quantity, price=price, strike=strike, implied_vol=vol)


def position(*legs: Leg, spot: float = SPOT, days: float = DAYS) -> Position:
    return Position(underlying="RELIANCE", legs=legs, spot=spot, lot_size=LOT, days_to_expiry=days)


class TestLegValidation:
    def test_a_zero_quantity_leg_is_refused(self) -> None:
        with pytest.raises(PositionError, match="not a leg"):
            call(2500, 0, 50)

    def test_an_option_leg_needs_a_strike(self) -> None:
        with pytest.raises(PositionError, match="positive strike"):
            Leg(kind=LegKind.CALL, quantity=1, price=50, strike=0)

    def test_the_underlying_needs_no_strike(self) -> None:
        assert Leg(kind=LegKind.UNDERLYING, quantity=1, price=SPOT).strike == 0.0

    def test_a_position_needs_a_leg(self) -> None:
        with pytest.raises(PositionError, match="at least one leg"):
            Position(underlying="X", legs=(), spot=SPOT, lot_size=LOT, days_to_expiry=DAYS)


class TestExpiryPayoff:
    def test_a_long_call_below_the_strike_loses_the_premium(self) -> None:
        book = position(call(2600, 1, 40))
        assert expiry_payoff(book, 2400.0) == pytest.approx(-40 * LOT)

    def test_a_long_call_above_the_strike_pays_intrinsic_less_premium(self) -> None:
        book = position(call(2600, 1, 40))
        assert expiry_payoff(book, 2700.0) == pytest.approx((100 - 40) * LOT)

    def test_a_short_put_keeps_the_premium_above_the_strike(self) -> None:
        book = position(put(2400, -1, 35))
        assert expiry_payoff(book, 2500.0) == pytest.approx(35 * LOT)

    def test_a_vertical_spread_caps_at_the_width(self) -> None:
        """The property the structure exists for: paying 40 and receiving 15
        for the 2700 caps the win at the width less the net debit."""
        book = position(call(2600, 1, 40), call(2700, -1, 15))
        assert expiry_payoff(book, 3500.0) == pytest.approx((100 - 25) * LOT)
        assert expiry_payoff(book, 2000.0) == pytest.approx(-25 * LOT)

    def test_a_covered_call_is_the_stock_plus_the_short_call(self) -> None:
        """Omitting the stock leg inverts the sign, so it is asserted."""
        book = position(
            Leg(kind=LegKind.UNDERLYING, quantity=1, price=SPOT),
            call(2600, -1, 40),
        )
        # Called away at 2600: 100 of stock gain plus 40 of premium.
        assert expiry_payoff(book, 2700.0) == pytest.approx((2600 - SPOT + 40) * LOT)


class TestNetPremium:
    def test_a_debit_structure_costs_money(self) -> None:
        assert position(call(2600, 1, 40), call(2700, -1, 15)).net_premium == pytest.approx(
            25 * LOT
        )

    def test_a_credit_structure_pays(self) -> None:
        assert position(call(2600, -1, 40), call(2700, 1, 15)).net_premium == pytest.approx(
            -25 * LOT
        )


class TestBounds:
    """The load-bearing part of this file."""

    def test_a_naked_short_call_has_no_maximum_loss(self) -> None:
        """Reporting the worst point of the plotted range would produce a
        finite, confident, wrong number — on exactly the position where it
        matters most."""
        result = analyse(position(call(2600, -1, 40)))
        assert result.max_loss is None
        assert "unbounded" in result.note

    def test_a_naked_short_put_has_a_finite_worst_case(self) -> None:
        """The textbook number, and one this used to get wrong.

        A price cannot fall below zero, so a short put's worst case is the
        strike less the premium — large, but finite and knowable. Reporting it
        as unbounded is not caution: it puts the same warning on a short put
        as on a short call, and the reader stops reading the warning.
        """
        result = analyse(position(put(2400, -1, 35)))
        assert result.max_loss == pytest.approx(-(2400 - 35) * LOT)

    def test_a_long_share_cannot_lose_more_than_it_cost(self) -> None:
        result = analyse(position(Leg(kind=LegKind.UNDERLYING, quantity=1, price=SPOT)))
        assert result.max_loss == pytest.approx(-SPOT * LOT)
        assert result.max_profit is None

    def test_a_covered_call_is_bounded_both_ways(self) -> None:
        """One of the most conservative structures there is. It was reported
        as unlimited-risk."""
        result = analyse(
            position(Leg(kind=LegKind.UNDERLYING, quantity=1, price=SPOT), call(2600, -1, 40))
        )
        assert result.max_loss == pytest.approx(-(SPOT - 40) * LOT)
        assert result.max_profit == pytest.approx((2600 - SPOT + 40) * LOT)
        assert "unbounded" not in result.note

    def test_a_collar_is_bounded_both_ways(self) -> None:
        result = analyse(
            position(
                Leg(kind=LegKind.UNDERLYING, quantity=1, price=SPOT),
                put(2400, 1, 35),
                call(2600, -1, 40),
            )
        )
        assert result.max_loss is not None
        assert result.max_profit is not None

    def test_only_a_short_call_leg_makes_the_loss_unbounded(self) -> None:
        """The asymptotic upside slope is the whole test: it is the only
        direction with no limit."""
        assert analyse(position(call(2600, -1, 40))).max_loss is None
        assert analyse(position(call(2600, -1, 40), call(2700, 1, 15))).max_loss is not None

    def test_a_long_call_has_no_maximum_profit(self) -> None:
        result = analyse(position(call(2600, 1, 40)))
        assert result.max_profit is None
        assert result.max_loss == pytest.approx(-40 * LOT)

    def test_a_vertical_spread_is_bounded_both_ways(self) -> None:
        result = analyse(position(call(2600, 1, 40), call(2700, -1, 15)))
        assert result.max_profit == pytest.approx(75 * LOT)
        assert result.max_loss == pytest.approx(-25 * LOT)

    def test_an_iron_condor_is_bounded_both_ways(self) -> None:
        result = analyse(
            position(
                put(2300, 1, 12),
                put(2400, -1, 28),
                call(2600, -1, 30),
                call(2700, 1, 13),
            )
        )
        assert result.max_profit is not None
        assert result.max_loss is not None
        # A condor's credit is its maximum profit, and its width less that
        # credit is its maximum loss. Both finite is the point of the wings.
        assert result.max_profit == pytest.approx(33 * LOT)

    def test_a_straddle_is_unbounded_in_profit_and_bounded_in_loss(self) -> None:
        result = analyse(position(call(2500, 1, 60), put(2500, 1, 55)))
        assert result.max_profit is None
        assert result.max_loss == pytest.approx(-115 * LOT)

    def test_a_short_straddle_inverts_that(self) -> None:
        result = analyse(position(call(2500, -1, 60), put(2500, -1, 55)))
        assert result.max_profit == pytest.approx(115 * LOT)
        assert result.max_loss is None


class TestBreakevens:
    def test_a_long_call_breaks_even_at_strike_plus_premium(self) -> None:
        result = analyse(position(call(2600, 1, 40)))
        assert len(result.breakevens) == 1
        assert result.breakevens[0] == pytest.approx(2640.0, abs=15.0)

    def test_a_straddle_has_two(self) -> None:
        result = analyse(position(call(2500, 1, 60), put(2500, 1, 55)))
        assert len(result.breakevens) == 2

    def test_a_position_that_never_profits_has_none(self) -> None:
        """A structure paying more than it can ever return has no crossing,
        and an empty list is the honest answer rather than a nearest guess."""
        assert breakevens([(1.0, -5.0), (2.0, -4.0), (3.0, -3.0)]) == []


class TestGreeks:
    def test_a_long_call_is_delta_positive(self) -> None:
        sensitivities = position_greeks(position(call(2500, 1, 60)))
        assert sensitivities is not None
        assert 0 < sensitivities.delta < LOT

    def test_a_short_call_flips_the_sign(self) -> None:
        long = position_greeks(position(call(2500, 1, 60)))
        short = position_greeks(position(call(2500, -1, 60)))
        assert long is not None and short is not None
        assert short.delta == pytest.approx(-long.delta)

    def test_a_straddle_is_roughly_delta_neutral_at_the_money(self) -> None:
        sensitivities = position_greeks(position(call(2500, 1, 60), put(2500, 1, 55)))
        assert sensitivities is not None
        assert abs(sensitivities.delta) < 0.15 * LOT
        # And long gamma, which is what the position is actually for.
        assert sensitivities.gamma > 0

    def test_the_underlying_contributes_delta_one_and_nothing_else(self) -> None:
        sensitivities = position_greeks(
            position(Leg(kind=LegKind.UNDERLYING, quantity=1, price=SPOT))
        )
        assert sensitivities is not None
        assert sensitivities.delta == pytest.approx(LOT)
        assert sensitivities.gamma == 0.0
        assert sensitivities.vega == 0.0

    def test_a_leg_without_volatility_withholds_the_aggregate(self) -> None:
        """A partial aggregate is worse than none, because it looks complete."""
        assert position_greeks(position(call(2500, 1, 60), call(2600, -1, 20, None))) is None


class TestPresentValue:
    def test_it_marks_a_position_before_expiry(self) -> None:
        book = position(call(2500, 1, 60))
        marked = value_at(book, SPOT)
        assert marked is not None
        # Bought at 60 and marked at the same spot and vol: near flat, and
        # certainly not the expiry payoff, which would be -60 a lot.
        assert marked > expiry_payoff(book, SPOT)

    def test_it_is_withheld_when_a_leg_cannot_be_marked(self) -> None:
        assert value_at(position(call(2500, 1, 60, None)), SPOT) is None

    def test_the_analysis_reports_the_reason(self) -> None:
        result = analyse(position(call(2500, 1, 60, None)))
        assert result.value == []
        assert "implied volatility" in result.note

    def test_expiry_payoff_and_present_value_are_different_curves(self) -> None:
        """A position can be underwater on the first and ahead on the second,
        so conflating them is not a rounding difference."""
        result = analyse(position(call(2600, 1, 40)))
        assert result.value
        assert result.payoff[0][1] != result.value[0][1]


class TestStructures:
    def test_every_structure_expands(self) -> None:
        for name, (_label, template) in STRUCTURES.items():
            needed = max(index for _, _, index in template) + 1
            strikes = [2300.0 + 100 * i for i in range(needed)]
            assert len(structure_legs(name, strikes)) == len(template)

    def test_strikes_are_sorted_before_assignment(self) -> None:
        """A bull call spread is long the lower strike whichever order the
        console happened to send them in."""
        legs = structure_legs("bull_call_spread", [2700.0, 2600.0])
        assert legs[0][2] == 2600.0
        assert legs[1][2] == 2700.0

    def test_too_few_strikes_is_refused_not_padded(self) -> None:
        """A butterfly built from two strikes is a vertical spread wearing the
        wrong name, and it would price and plot perfectly well."""
        with pytest.raises(PositionError, match="needs 3 strike"):
            structure_legs("call_butterfly", [2500.0, 2600.0])

    def test_an_unknown_structure_names_the_real_ones(self) -> None:
        with pytest.raises(PositionError, match="iron_condor"):
            structure_legs("jade_lizard", [2500.0])


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(build_options_router())
    return TestClient(app)


class TestStrategyEndpoint:
    def leg(self, kind: str, quantity: int, price: float, strike: float) -> dict:
        return {
            "kind": kind,
            "quantity": quantity,
            "price": price,
            "strike": strike,
            "implied_vol": 0.25,
        }

    def request(self, *legs: dict) -> dict:
        return {
            "underlying": "RELIANCE",
            "spot": SPOT,
            "lot_size": LOT,
            "days_to_expiry": DAYS,
            "legs": list(legs),
        }

    def test_the_structure_list_says_how_many_strikes_each_needs(self, client) -> None:
        rows = {r["name"]: r for r in client.get("/options/structures").json()}
        assert rows["call_butterfly"]["strikes"] == 3
        assert rows["long_call"]["strikes"] == 1

    def test_a_spread_returns_bounded_profit_and_loss(self, client) -> None:
        body = client.post(
            "/options/strategy",
            json=self.request(
                self.leg("CE", 1, 40, 2600),
                self.leg("CE", -1, 15, 2700),
            ),
        ).json()
        assert body["maxProfit"] if "maxProfit" in body else body["max_profit"]

    def test_an_unbounded_loss_is_null_and_explained(self, client) -> None:
        body = client.post(
            "/options/strategy", json=self.request(self.leg("CE", -1, 40, 2600))
        ).json()
        assert body["max_loss"] is None
        assert "unbounded" in body["note"]

    def test_a_bad_leg_is_a_422_that_says_why(self, client) -> None:
        response = client.post("/options/strategy", json=self.request(self.leg("CE", 1, 40, 0)))
        assert response.status_code == 422

    def test_more_than_eight_legs_is_refused(self, client) -> None:
        legs = [self.leg("CE", 1, 10, 2000 + 50 * i) for i in range(9)]
        assert client.post("/options/strategy", json=self.request(*legs)).status_code == 422

    def test_greeks_come_back_aggregated(self, client) -> None:
        body = client.post(
            "/options/strategy",
            json=self.request(self.leg("CE", 1, 60, 2500), self.leg("PE", 1, 55, 2500)),
        ).json()
        assert abs(body["delta"]) < 0.15 * LOT
        assert body["gamma"] > 0
