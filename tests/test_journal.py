"""Execution quality — MASTER_PLAN §9, §M7.

The cost model was never checked against anything but the equity curve it
produced. These cover the arithmetic that checks it, and lean on the two ways
a journal flatters itself: treating an unmeasured cost as zero, and calling an
adverse sale a gain because the difference was not signed.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from core.clock import UTC
from trading.journal import Execution, summarise

T0 = datetime(2026, 6, 1, 5, 0, tzinfo=UTC)


def execution(  # noqa: PLR0913, PLR0917 - a builder for a nine-field record
    side: str = "BUY",
    ordered: str = "100",
    filled: str = "100",
    fill: str | None = "101",
    decision: str | None = "100",
    intended: str | None = "100.5",
    fees: str = "0",
    order_id: str = "o1",
) -> Execution:
    return Execution(
        order_id=order_id,
        instrument_id="NSE:INE002A01018",
        symbol="RELIANCE",
        side=side,
        strategy_id="s1",
        mode="PAPER",
        decision_time=T0,
        quantity_ordered=Decimal(ordered),
        quantity_filled=Decimal(filled),
        average_fill=None if fill is None else Decimal(fill),
        fees=Decimal(fees),
        decision_price=None if decision is None else Decimal(decision),
        intended_price=None if intended is None else Decimal(intended),
    )


class TestSign:
    def test_a_buy_above_the_decision_price_is_a_cost(self) -> None:
        assert execution("BUY", fill="101").slippage == Decimal(100)

    def test_a_sell_below_the_decision_price_is_also_a_cost(self) -> None:
        """The mistake a signless difference makes: paying 1 over on a buy and
        receiving 1 under on a sell are the same loss, and an unsigned
        subtraction calls one of them a gain."""
        assert execution("SELL", fill="99").slippage == Decimal(100)

    def test_a_sell_above_the_decision_price_is_a_gain(self) -> None:
        assert execution("SELL", fill="101").slippage == Decimal(-100)

    def test_case_does_not_matter(self) -> None:
        assert execution("buy", fill="101").slippage == Decimal(100)


class TestDecomposition:
    def test_delay_and_execution_sum_to_slippage(self) -> None:
        """The whole point of splitting them: the parts must still be the
        whole, or the decomposition is a second, disagreeing measurement."""
        one = execution(decision="100", intended="100.5", fill="101")
        assert one.delay_cost is not None and one.execution_cost is not None
        assert one.delay_cost + one.execution_cost == one.slippage

    def test_delay_cost_is_the_move_before_sending(self) -> None:
        assert execution(decision="100", intended="100.5").delay_cost == Decimal(50)

    def test_execution_cost_is_the_move_after_sending(self) -> None:
        assert execution(intended="100.5", fill="101").execution_cost == Decimal(50)

    def test_shortfall_adds_the_fees(self) -> None:
        one = execution(fill="101", fees="37.5")
        assert one.implementation_shortfall == Decimal("137.5")


class TestUnmeasured:
    def test_a_missing_decision_price_leaves_delay_unmeasured(self) -> None:
        """Not zero. A zero would average into the totals and flatter them."""
        assert execution(decision=None).delay_cost is None

    def test_a_missing_intent_leaves_execution_cost_unmeasured(self) -> None:
        assert execution(intended=None).execution_cost is None

    def test_slippage_falls_back_to_the_intent(self) -> None:
        """A smaller claim than shortfall, and made only when the decision
        mark is genuinely absent."""
        assert execution(decision=None, intended="100.5", fill="101").slippage == Decimal(50)

    def test_an_unfilled_order_has_no_slippage(self) -> None:
        assert execution(filled="0", fill=None).slippage is None

    def test_a_totally_unmeasurable_set_totals_to_none_not_zero(self) -> None:
        """ "No order could be measured" and "the orders cost nothing" are
        opposite findings that a zero renders identically."""
        summary = summarise([execution(decision=None, intended=None, fill="101")])
        assert summary.implementation_shortfall is None
        assert summary.delay_cost is None

    def test_coverage_reports_how_much_was_measurable(self) -> None:
        summary = summarise(
            [
                execution(order_id="a"),
                execution(order_id="b", decision=None, intended=None),
            ]
        )
        assert summary.filled == 2
        assert summary.measured == 1
        assert summary.coverage == pytest.approx(0.5)

    def test_the_format_names_the_gap(self) -> None:
        summary = summarise(
            [execution(order_id="a"), execution(order_id="b", decision=None, intended=None)]
        )
        assert "unmeasured, not zero" in summary.format()


class TestPartialFills:
    def test_costs_scale_with_what_was_filled_not_what_was_ordered(self) -> None:
        """Charging the unfilled part would invent a cost that was never
        incurred, and on a badly filled order that is most of it."""
        assert execution(ordered="100", filled="40", fill="101").slippage == Decimal(40)

    def test_the_unfilled_quantity_is_reported(self) -> None:
        one = execution(ordered="100", filled="40")
        assert one.unfilled_quantity == Decimal(60)
        assert one.fill_rate == pytest.approx(0.4)

    def test_opportunity_cost_needs_a_later_mark(self) -> None:
        one = execution(ordered="100", filled="40", decision="100")
        assert one.opportunity_cost(None) is None
        assert one.opportunity_cost(Decimal(105)) == Decimal(300)

    def test_a_fully_filled_order_has_no_opportunity_cost(self) -> None:
        assert execution(ordered="100", filled="100").opportunity_cost(Decimal(105)) is None


class TestBasisPoints:
    def test_slippage_in_bps_is_against_the_reference_notional(self) -> None:
        # 1 rupee on a 100-rupee decision price is 100bps.
        assert execution(decision="100", fill="101").slippage_bps == pytest.approx(100.0)

    def test_a_zero_reference_yields_none_rather_than_a_division(self) -> None:
        assert execution(decision=None, intended=None).slippage_bps is None


class TestSummary:
    def test_an_empty_journal_is_a_state_not_an_error(self) -> None:
        summary = summarise([])
        assert summary.orders == 0
        assert summary.coverage == 0.0
        assert summary.implementation_shortfall is None

    def test_the_worst_orders_are_named_not_just_counted(self) -> None:
        """A shortfall concentrated in two orders and one spread across two
        hundred call for different fixes."""
        summary = summarise(
            [
                execution(order_id="cheap", fill="100.1"),
                execution(order_id="awful", fill="110"),
                execution(order_id="fine", fill="100.2"),
            ],
            worst_n=1,
        )
        assert [e.order_id for e in summary.worst] == ["awful"]

    def test_shortfall_bps_is_volume_weighted(self) -> None:
        """A large order filled badly must not be averaged away by a small one
        filled well — the equal-weighted number is the one that looks fine
        while the size that matters is being filled badly."""
        summary = summarise(
            [
                execution(order_id="big", ordered="1000", filled="1000", fill="101"),
                execution(order_id="small", ordered="10", filled="10", fill="100"),
            ]
        )
        assert summary.shortfall_bps is not None
        # The big order is 99% of notional and cost ~99bps; a plain mean of
        # the two orders' bps would report about half that.
        assert summary.shortfall_bps > 90

    def test_notional_uses_the_fill_price(self) -> None:
        summary = summarise([execution(ordered="100", filled="100", fill="101")])
        assert summary.notional == Decimal(10_100)

    def test_unfilled_orders_are_counted_but_not_priced(self) -> None:
        summary = summarise([execution(order_id="x", filled="0", fill=None)])
        assert summary.orders == 1
        assert summary.filled == 0
        assert summary.notional == Decimal(0)


class TestZeroReference:
    def test_a_zero_decision_price_is_not_treated_as_missing(self) -> None:
        """`or` would fall through to the intent on a zero decision price and
        measure a different quantity than the one it claims to. `is None`
        declines to measure instead."""
        zero = execution(decision="0", intended="100.5", fill="101")
        # Zero is a nonsense reference, so bps must not silently come back as
        # the intent-based number.
        assert zero.slippage == Decimal(10100)
        assert zero.slippage_bps is None
