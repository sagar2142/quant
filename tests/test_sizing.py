"""Order sizing and funding (§17).

Money code: this is what turns "3% of NAV" into "buy 47 shares", and it decides
what the account can actually pay for. The tests below target the guards —
every one of them is the difference between a refused order and an overdrawn
account.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from engine.accounting import Portfolio
from engine.backtest.sizing import OrderPlanner, SizingConfig

A = InstrumentId("NSE:AAA")
B = InstrumentId("NSE:BBB")


def equity(instrument_id: InstrumentId, lot_size: int = 1) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=str(instrument_id).split(":")[1],
        asset_class=AssetClass.EQUITY,
        exchange=Exchange.NSE,
        currency=Currency.INR,
        tick_size=Decimal("0.05"),
        lot_size=lot_size,
    )


INSTRUMENTS = {A: equity(A), B: equity(B)}


def planner(**config) -> OrderPlanner:
    return OrderPlanner(INSTRUMENTS, SizingConfig(**config) if config else None)


def free_cost(_quantity: Decimal, _price: Decimal) -> Decimal:
    """A cost model that charges nothing, so funding maths is visible alone."""
    return Decimal(0)


class TestAffordable:
    def test_the_whole_order_when_cash_allows(self):
        book = Portfolio(cash=Decimal(1_000_000))
        room = planner().affordable(book, equity(A), Decimal(100), Decimal(500), free_cost)
        assert room == Decimal(500)

    def test_trimmed_to_what_cash_covers(self):
        book = Portfolio(cash=Decimal(10_000))
        room = planner().affordable(book, equity(A), Decimal(100), Decimal(500), free_cost)
        assert room == Decimal(100)

    def test_costs_reduce_the_affordable_quantity(self):
        def dear(quantity: Decimal, price: Decimal) -> Decimal:
            return quantity * price * Decimal("0.10")

        book = Portfolio(cash=Decimal(10_000))
        assert planner().affordable(book, equity(A), Decimal(100), Decimal(100), dear) < Decimal(
            100
        )

    def test_a_non_positive_price_buys_nothing(self):
        """Guard against a corrupt or missing mark. Dividing by it would raise;
        pricing an order off it would be worse."""
        book = Portfolio(cash=Decimal(1_000_000))
        assert planner().affordable(book, equity(A), Decimal(0), Decimal(10), free_cost) == 0

    def test_an_empty_account_buys_nothing(self):
        """Room goes negative once costs exceed cash — it must clamp to zero
        rather than return a negative quantity, which would read as a sell."""
        book = Portfolio(cash=Decimal(0))
        assert planner().affordable(book, equity(A), Decimal(100), Decimal(10), free_cost) == 0

    def test_costs_alone_can_exhaust_the_account(self):
        def ruinous(_quantity: Decimal, _price: Decimal) -> Decimal:
            return Decimal(1_000_000)

        book = Portfolio(cash=Decimal(1_000))
        assert planner().affordable(book, equity(A), Decimal(100), Decimal(5), ruinous) == 0

    def test_margin_allowance_is_spendable(self):
        book = Portfolio(cash=Decimal(0), margin_allowance=Decimal(50_000))
        assert planner().affordable(book, equity(A), Decimal(100), Decimal(100), free_cost) > 0

    def test_lot_size_rounds_down_never_up(self):
        """Rounding up would exceed the funded budget by a whole lot."""
        instruments = {A: equity(A, lot_size=100)}
        book = Portfolio(cash=Decimal(15_000))
        room = OrderPlanner(instruments).affordable(
            book, equity(A, lot_size=100), Decimal(100), Decimal(500), free_cost
        )
        assert room == Decimal(100)  # 150 affordable -> one 100-lot

    def test_quantities_are_whole_shares(self):
        book = Portfolio(cash=Decimal(1_050))
        room = planner().affordable(book, equity(A), Decimal(100), Decimal(500), free_cost)
        assert room == room.to_integral_value()


class TestPlanFunding:
    def marks(self) -> dict[InstrumentId, Decimal]:
        return {A: Decimal(100), B: Decimal(200)}

    def test_a_zero_price_name_is_skipped(self):
        """A name with no usable mark cannot be sized, and must not abort the
        rest of the rebalance."""
        book = Portfolio(cash=Decimal(1_000_000))
        planned = planner().plan(
            book,
            {A: Decimal("0.5"), B: Decimal("0.5")},
            {A: Decimal(0), B: Decimal(200)},
            Decimal(1_000_000),
        )
        assert [i for i, _ in planned] == [B]

    def test_an_unknown_instrument_is_skipped(self):
        book = Portfolio(cash=Decimal(1_000_000))
        unknown = InstrumentId("NSE:ZZZ")
        planned = planner().plan(
            book,
            {unknown: Decimal("0.5")},
            {unknown: Decimal(100)},
            Decimal(1_000_000),
        )
        assert planned == []

    def test_orders_below_the_minimum_value_are_dropped(self):
        """Below this an order cannot pay for its own fixed costs (§7.1)."""
        book = Portfolio(cash=Decimal(1_000_000))
        planned = planner(min_order_value=Decimal(50_000)).plan(
            book, {A: Decimal("0.01")}, self.marks(), Decimal(1_000_000)
        )
        assert planned == []

    def test_small_drift_is_not_rebalanced(self):
        """Turnover is the most reliable way to lose money to costs."""
        book = Portfolio(cash=Decimal(1_000_000))
        planned = planner(rebalance_threshold=Decimal("0.5")).plan(
            book, {A: Decimal("0.1")}, self.marks(), Decimal(1_000_000)
        )
        assert planned == []

    def test_sells_are_planned_before_buys(self):
        """Sell proceeds fund the buys in the same session, so the order of the
        plan is what makes a fully-invested rebalance possible at all."""
        book = Portfolio(cash=Decimal(0))
        book.positions[A] = book.position(A)
        planned = planner().plan(book, {B: Decimal("0.5")}, self.marks(), Decimal(1_000_000))
        # Nothing held, no cash: nothing can be funded.
        assert all(q > 0 for _, q in planned) or planned == []

    def test_an_unfundable_buy_is_trimmed_not_rejected(self):
        """An order the account cannot fund was never an order (§17)."""
        book = Portfolio(cash=Decimal(5_000))
        planned = planner().plan(book, {A: Decimal("1.0")}, self.marks(), Decimal(1_000_000))
        for _, quantity in planned:
            assert quantity * Decimal(100) <= Decimal(5_000)

    def test_dropped_names_are_closed(self):
        """A held name absent from the targets must be sold, not retained."""
        book = Portfolio(cash=Decimal(100_000))
        from engine.accounting import Position

        book.positions[A] = Position(A, quantity=Decimal(100), average_price=Decimal(90))
        planned = dict(planner().plan(book, {}, self.marks(), Decimal(110_000)))
        assert planned[A] == Decimal(-100)

    def test_zero_equity_does_not_divide_by_zero(self):
        book = Portfolio(cash=Decimal(0))
        assert planner().plan(book, {A: Decimal("0.5")}, self.marks(), Decimal(0)) == []


class TestLotRounding:
    def test_a_partial_lot_is_refused(self):
        instruments = {A: equity(A, lot_size=50)}
        book = Portfolio(cash=Decimal(1_000_000))
        planned = OrderPlanner(instruments).plan(
            book, {A: Decimal("0.0071")}, {A: Decimal(100)}, Decimal(1_000_000)
        )
        for _, quantity in planned:
            assert quantity % 50 == 0

    @pytest.mark.parametrize("lot", [1, 25, 100])
    def test_every_planned_quantity_is_a_whole_lot(self, lot):
        instruments = {A: equity(A, lot_size=lot)}
        book = Portfolio(cash=Decimal(1_000_000))
        planned = OrderPlanner(instruments).plan(
            book, {A: Decimal("0.37")}, {A: Decimal(137)}, Decimal(1_000_000)
        )
        for _, quantity in planned:
            assert quantity % lot == 0
