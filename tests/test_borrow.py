"""What a short costs to hold — MASTER_PLAN §7.3.

The point of these is not that the arithmetic works. It is that the charge
accrues with *time held* rather than with trading, that an unknown borrow rate
costs something rather than nothing, and that adding this model cannot change a
long-only result — because if it could, every backtest in the repo would need
re-running.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.instruments import InstrumentId
from engine.costs.borrow import (
    DEFAULT_ANNUAL_RATE,
    HARD_TO_BORROW_RATE,
    SESSIONS_PER_YEAR,
    BorrowModel,
)

A = InstrumentId("NSE:INE001A01001")
B = InstrumentId("NSE:INE002A01002")


class TestWhoPays:
    def test_only_the_short_side_is_charged(self):
        model = BorrowModel()
        book = {A: Decimal(-100_000), B: Decimal(100_000)}
        one_sided = model.session_charge({A: Decimal(-100_000)}, 21)
        assert model.session_charge(book, 21) == one_sided

    def test_a_long_only_book_is_free(self):
        """Load-bearing: this model exists alongside every long-only backtest
        already run, and those results must be unchanged by its arrival."""
        model = BorrowModel()
        assert model.session_charge({A: Decimal(500_000), B: Decimal(250_000)}, 252) == Decimal(0)

    def test_a_flat_position_pays_nothing(self):
        assert BorrowModel().session_charge({A: Decimal(0)}, 252) == Decimal(0)


class TestItPricesTimeNotTrades:
    def test_the_charge_scales_with_sessions_held(self):
        """A short carried for a month pays twenty-one times, having traded
        once. That is the whole reason it cannot live in `CostModel.cost`."""
        model = BorrowModel()
        position = {A: Decimal(-100_000)}
        month = model.session_charge(position, 21)
        assert float(month) == pytest.approx(
            float(model.session_charge(position, 1)) * 21, rel=1e-3
        )

    def test_a_block_is_charged_once_rather_than_accrued_session_by_session(self):
        """Each call rounds to paisa, so twenty-one one-session charges do not
        equal one twenty-one-session charge — 500.01 against 500.00 on a lakh,
        because half-up rounding of 23.8095 pushes each session up.

        The block is the truthful figure and the drift is bounded at a paisa
        per session, which is why a caller should charge for the holding period
        it actually has rather than accruing daily. Asserting the bound rather
        than a direction: the sign of the drift is an artefact of where the
        rate lands, and pinning it would pin the wrong thing.
        """
        model = BorrowModel()
        position = {A: Decimal(-100_000)}
        drift = abs(model.session_charge(position, 21) - model.session_charge(position, 1) * 21)
        assert drift <= Decimal("0.01") * 21

    def test_a_full_year_costs_the_annual_rate(self):
        model = BorrowModel()
        charge = model.session_charge({A: Decimal(-100_000)}, SESSIONS_PER_YEAR)
        assert float(charge) == pytest.approx(float(Decimal(100_000) * DEFAULT_ANNUAL_RATE))

    def test_zero_or_negative_sessions_cost_nothing(self):
        model = BorrowModel()
        assert model.session_charge({A: Decimal(-100_000)}, 0) == Decimal(0)
        assert model.session_charge({A: Decimal(-100_000)}, -5) == Decimal(0)


class TestRates:
    def test_an_unknown_name_is_charged_not_exempted(self):
        """An unquoted borrow cost is not a zero borrow cost — the default has
        to bite, or the model flatters exactly the names it knows least about.
        """
        assert BorrowModel().rate_for(A) == DEFAULT_ANNUAL_RATE
        assert BorrowModel().session_charge({A: Decimal(-100_000)}, 21) > 0

    def test_a_known_rate_overrides_the_default(self):
        model = BorrowModel(rates={A: HARD_TO_BORROW_RATE})
        assert model.rate_for(A) == HARD_TO_BORROW_RATE
        assert model.rate_for(B) == DEFAULT_ANNUAL_RATE

    def test_a_scarce_name_costs_multiples_of_an_ordinary_one(self):
        model = BorrowModel(rates={A: HARD_TO_BORROW_RATE})
        scarce = model.session_charge({A: Decimal(-100_000)}, 252)
        ordinary = model.session_charge({B: Decimal(-100_000)}, 252)
        assert scarce > ordinary * 3

    def test_the_default_is_pessimistic_on_purpose(self):
        """Documented as a deliberate choice, so it is pinned as one. Six
        percent is the expensive end of ordinary, not a market mid."""
        assert Decimal("0.06") == DEFAULT_ANNUAL_RATE
        assert HARD_TO_BORROW_RATE > DEFAULT_ANNUAL_RATE * 3


class TestBorrowability:
    def test_an_empty_set_models_no_restriction(self):
        """The permissive default, asserted rather than assumed: making it
        restrictive would silently empty every existing long-short test."""
        assert BorrowModel().can_short(A)

    def test_a_real_borrow_list_excludes_everything_else(self):
        model = BorrowModel(borrowable=frozenset({B}))
        assert model.can_short(B)
        assert not model.can_short(A)


class TestMoney:
    def test_the_charge_is_quantised_to_paisa(self):
        """Decimal for money (§14.1.2) — a cost carried at full precision
        drifts against a ledger that rounds."""
        charge = BorrowModel().session_charge({A: Decimal(-33_333)}, 7)
        assert charge == charge.quantize(Decimal("0.01"))

    def test_the_cost_is_positive_though_the_position_is_negative(self):
        """Sign discipline: positions are signed, costs are not. A borrow
        charge that came back negative would *add* to equity."""
        assert BorrowModel().session_charge({A: Decimal(-100_000)}, 21) > 0
