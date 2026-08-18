"""Position and cash accounting (§14.1.2, §17).

Money module: §14.5 requires 100% coverage. The decisive cases are
`TestPositionFlip` (long straight through to short) and `TestShorts`, because
both are where sign errors hide without ever crashing.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from core.clock import UTC
from core.instruments import InstrumentId
from core.orders import Side
from engine.accounting import Fill, InsufficientCashError, Portfolio, Position
from engine.costs.model import CostBreakdown

IID = InstrumentId("NSE:RELIANCE")
OTHER = InstrumentId("NSE:TCS")
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)

FREE = CostBreakdown()
FEE = CostBreakdown(brokerage=Decimal("20.00"))


def fill(side: Side, qty: str, price: str, costs: CostBreakdown = FREE, **kw) -> Fill:
    return Fill(
        instrument_id=kw.pop("instrument_id", IID),
        side=side,
        quantity=Decimal(qty),
        price=Decimal(price),
        costs=costs,
        event_time=T0,
        **kw,
    )


class TestFill:
    def test_buy_consumes_cash(self):
        assert fill(Side.BUY, "100", "1000").cash_delta == Decimal("-100000.00")

    def test_sell_releases_cash(self):
        assert fill(Side.SELL, "100", "1000").cash_delta == Decimal("100000.00")

    def test_costs_always_consume(self):
        assert fill(Side.BUY, "100", "1000", FEE).cash_delta == Decimal("-100020.00")
        assert fill(Side.SELL, "100", "1000", FEE).cash_delta == Decimal("99980.00")

    def test_multiplier_scales_value(self):
        f = fill(Side.BUY, "1", "24000", multiplier=Decimal(50))
        assert f.gross_value == Decimal(1_200_000)

    def test_signed_quantity(self):
        assert fill(Side.BUY, "10", "100").signed_quantity == Decimal(10)
        assert fill(Side.SELL, "10", "100").signed_quantity == Decimal(-10)

    def test_zero_quantity_rejected(self):
        with pytest.raises(ValueError, match="quantity"):
            fill(Side.BUY, "0", "100")

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="price"):
            fill(Side.BUY, "10", "-1")

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="naive"):
            Fill(
                instrument_id=IID,
                side=Side.BUY,
                quantity=Decimal(1),
                price=Decimal(100),
                costs=FREE,
                event_time=datetime(2024, 1, 1),
            )


class TestOpeningAndAdding:
    def test_open_long(self):
        pos, realised = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        assert pos.quantity == 100
        assert pos.average_price == 1000
        assert realised == 0
        assert pos.is_long

    def test_average_price_is_weighted(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        pos, realised = pos.apply(fill(Side.BUY, "100", "1200"))
        assert pos.quantity == 200
        assert pos.average_price == 1100
        assert realised == 0  # adding never realises

    def test_uneven_weighting(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        pos, _ = pos.apply(fill(Side.BUY, "300", "1200"))
        assert pos.average_price == Decimal(1150)

    def test_fees_accumulate(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000", FEE))
        pos, _ = pos.apply(fill(Side.BUY, "100", "1000", FEE))
        assert pos.fees_paid == Decimal("40.00")


class TestReducingAndClosing:
    def test_partial_close_realises_proportionally(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        pos, realised = pos.apply(fill(Side.SELL, "40", "1100"))
        assert realised == Decimal("4000.00")  # 40 x 100 gain
        assert pos.quantity == 60
        # Average price is untouched: the remaining 60 were bought at 1000.
        assert pos.average_price == 1000

    def test_full_close_flattens(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        pos, realised = pos.apply(fill(Side.SELL, "100", "1100"))
        assert pos.is_flat
        assert pos.average_price == 0
        assert realised == Decimal("10000.00")

    def test_loss_realised_correctly(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        _, realised = pos.apply(fill(Side.SELL, "100", "900"))
        assert realised == Decimal("-10000.00")

    def test_realised_accumulates_on_position(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        pos, _ = pos.apply(fill(Side.SELL, "50", "1100"))
        pos, _ = pos.apply(fill(Side.SELL, "50", "1200"))
        assert pos.realised_pnl == Decimal("15000.00")


class TestShorts:
    def test_open_short(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        assert pos.quantity == -100
        assert pos.average_price == 1000
        assert pos.is_short

    def test_short_profits_when_price_falls(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        _, realised = pos.apply(fill(Side.BUY, "100", "900"))
        assert realised == Decimal("10000.00")

    def test_short_loses_when_price_rises(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        _, realised = pos.apply(fill(Side.BUY, "100", "1100"))
        assert realised == Decimal("-10000.00")

    def test_short_unrealised_sign(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        assert pos.unrealised_pnl(Decimal(900)) == Decimal("10000.00")
        assert pos.unrealised_pnl(Decimal(1100)) == Decimal("-10000.00")

    def test_adding_to_short_averages(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        pos, realised = pos.apply(fill(Side.SELL, "100", "1200"))
        assert pos.quantity == -200
        assert pos.average_price == 1100
        assert realised == 0

    def test_short_market_value_is_negative(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        assert pos.market_value(Decimal(1000)) == Decimal("-100000.00")


class TestPositionFlip:
    """Long straight through to short in one fill.

    Naive netting leaves a phantom average price mixing long and short entries,
    and the error persists for the life of the position.
    """

    def test_flip_long_to_short(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        pos, realised = pos.apply(fill(Side.SELL, "150", "1100"))
        # The full 100-share long is realised...
        assert realised == Decimal("10000.00")
        # ...and the remaining 50 open a short at the fill price, not a blend.
        assert pos.quantity == -50
        assert pos.average_price == 1100

    def test_flip_short_to_long(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        pos, realised = pos.apply(fill(Side.BUY, "150", "900"))
        assert realised == Decimal("10000.00")
        assert pos.quantity == 50
        assert pos.average_price == 900

    def test_flip_then_close_is_consistent(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        pos, _ = pos.apply(fill(Side.SELL, "150", "1100"))
        pos, realised = pos.apply(fill(Side.BUY, "50", "1000"))
        # Short 50 from 1100, covered at 1000 = +5,000.
        assert realised == Decimal("5000.00")
        assert pos.is_flat
        assert pos.realised_pnl == Decimal("15000.00")


class TestCorporateActions:
    def test_split_preserves_value(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        before = pos.market_value(Decimal(1000))
        split = pos.apply_split(Decimal(2))
        assert split.quantity == 200
        assert split.average_price == 500
        # Price halves alongside, so value is unchanged.
        assert split.market_value(Decimal(500)) == before

    def test_reverse_split(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "100", "100"))
        reverse = pos.apply_split(Decimal("0.1"))
        assert reverse.quantity == 10
        assert reverse.average_price == 1000

    def test_split_on_short(self):
        pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        assert pos.apply_split(Decimal(2)).quantity == -200

    def test_invalid_ratio_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Position(IID).apply_split(Decimal(0))


class TestPortfolioCash:
    def test_buy_reduces_cash(self):
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000", FEE))
        assert p.cash == Decimal("899980.00")

    def test_sell_increases_cash(self):
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000"))
        p.apply_fill(fill(Side.SELL, "100", "1100"))
        assert p.cash == Decimal("1010000.00")

    def test_overdraft_rejected(self):
        # An unnoticed overdraft is free leverage, and free leverage flatters
        # every metric downstream (§14.1.5).
        p = Portfolio(cash=Decimal(1000))
        with pytest.raises(InsufficientCashError):
            p.apply_fill(fill(Side.BUY, "100", "1000"))

    def test_margin_allowance_permits_overdraft(self):
        p = Portfolio(cash=Decimal(1000), margin_allowance=Decimal(200_000))
        p.apply_fill(fill(Side.BUY, "100", "1000"))
        assert p.cash == Decimal("-99000.00")

    def test_failed_fill_leaves_state_untouched(self):
        p = Portfolio(cash=Decimal(1000))
        with pytest.raises(InsufficientCashError):
            p.apply_fill(fill(Side.BUY, "100", "1000"))
        assert p.cash == Decimal(1000)
        assert p.position(IID).is_flat

    def test_fees_tracked(self):
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000", FEE))
        assert p.fees_paid == Decimal("20.00")


class TestPortfolioValuation:
    def portfolio(self) -> Portfolio:
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000"))
        p.apply_fill(fill(Side.SELL, "50", "2000", instrument_id=OTHER))
        return p

    def test_equity_combines_cash_and_marks(self):
        p = self.portfolio()
        prices = {IID: Decimal(1100), OTHER: Decimal(1900)}
        # cash 1,000,000 - 100,000 + 100,000 = 1,000,000
        # long  100 x 1100 =  110,000
        # short -50 x 1900 =  -95,000
        assert p.equity(prices) == Decimal("1015000.00")

    def test_missing_mark_raises(self):
        # A stale or zero mark would silently misstate equity.
        p = self.portfolio()
        with pytest.raises(KeyError, match="no mark"):
            p.equity({IID: Decimal(1100)})

    def test_gross_versus_net_exposure(self):
        p = self.portfolio()
        prices = {IID: Decimal(1000), OTHER: Decimal(2000)}
        assert p.gross_exposure(prices) == Decimal("200000.00")  # 100k + 100k
        assert p.net_exposure(prices) == Decimal("0.00")  # market neutral

    def test_unrealised_pnl(self):
        p = self.portfolio()
        assert p.unrealised_pnl({IID: Decimal(1100), OTHER: Decimal(2000)}) == Decimal("10000.00")

    def test_flat_positions_excluded(self):
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000"))
        p.apply_fill(fill(Side.SELL, "100", "1000"))
        assert p.open_positions() == {}
        assert p.equity({}) == p.cash

    def test_position_of_unknown_instrument_is_flat(self):
        assert Portfolio(cash=Decimal(0)).position(IID).is_flat


class TestPortfolioCorporateActions:
    def test_split_applied_to_holding(self):
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000"))
        p.apply_split(IID, Decimal(2))
        assert p.position(IID).quantity == 200

    def test_split_on_flat_position_is_noop(self):
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_split(IID, Decimal(2))
        assert p.position(IID).is_flat

    def test_dividend_credits_long(self):
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000"))
        assert p.apply_dividend(IID, Decimal(10)) == Decimal("1000.00")
        assert p.cash == Decimal("901000.00")

    def test_dividend_debits_short(self):
        # A short position pays the dividend.
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.SELL, "100", "1000"))
        assert p.apply_dividend(IID, Decimal(10)) == Decimal("-1000.00")

    def test_dividend_on_flat_is_zero(self):
        p = Portfolio(cash=Decimal(1000))
        assert p.apply_dividend(IID, Decimal(10)) == Decimal(0)

    def test_funding_payment_reduces_cash(self):
        p = Portfolio(cash=Decimal(100_000))
        p.apply_funding(Decimal(50))
        assert p.cash == Decimal("99950.00")
        assert p.fees_paid == Decimal("50.00")

    def test_funding_received_increases_cash(self):
        p = Portfolio(cash=Decimal(100_000))
        p.apply_funding(Decimal(-50))
        assert p.cash == Decimal("100050.00")


class TestPositionEdgeCases:
    def test_cost_basis_signed(self):
        long_pos, _ = Position(IID).apply(fill(Side.BUY, "100", "1000"))
        short_pos, _ = Position(IID).apply(fill(Side.SELL, "100", "1000"))
        assert long_pos.cost_basis == Decimal(100_000)
        assert short_pos.cost_basis == Decimal(-100_000)

    def test_cost_basis_uses_multiplier(self):
        pos, _ = Position(IID).apply(fill(Side.BUY, "1", "24000", multiplier=Decimal(50)))
        assert pos.cost_basis == Decimal(1_200_000)

    def test_flat_position_has_no_unrealised_pnl(self):
        assert Position(IID).unrealised_pnl(Decimal(1000)) == Decimal(0)

    def test_unmarked_position_skipped_in_aggregates(self):
        # A held position with no mark contributes nothing to these summaries
        # rather than raising — only `equity` insists on a complete mark set,
        # because that is the number a drawdown is measured on.
        p = Portfolio(cash=Decimal(1_000_000))
        p.apply_fill(fill(Side.BUY, "100", "1000"))
        p.apply_fill(fill(Side.BUY, "10", "2000", instrument_id=OTHER))
        prices = {IID: Decimal(1100)}
        assert p.unrealised_pnl(prices) == Decimal("10000.00")
        assert p.gross_exposure(prices) == Decimal("110000.00")


class TestAssetClassGuards:
    """A model must refuse products it cannot price, never guess (§14.1.5)."""

    def test_equity_model_rejects_future(self):
        from core.instruments import AssetClass, Currency, Exchange, Instrument
        from engine.costs.india import NseEquityCostModel
        from engine.costs.model import TradeContext

        future = Instrument(
            instrument_id=InstrumentId("NSE:FUT"),
            symbol="FUT",
            asset_class=AssetClass.FUTURE,
            exchange=Exchange.NSE,
            currency=Currency.INR,
            tick_size=Decimal("0.05"),
            expiry=datetime(2024, 12, 26, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="cannot price"):
            NseEquityCostModel().cost(
                TradeContext(
                    instrument=future,
                    side=Side.BUY,
                    quantity=Decimal(1),
                    price=Decimal(100),
                )
            )

    def test_us_model_rejects_derivatives(self):
        from core.instruments import AssetClass, Currency, Exchange, Instrument
        from engine.costs.model import TradeContext
        from engine.costs.us_equity import UsEquityCostModel

        future = Instrument(
            instrument_id=InstrumentId("US:ESZ4"),
            symbol="ESZ4",
            asset_class=AssetClass.FUTURE,
            exchange=Exchange.NASDAQ,
            currency=Currency.USD,
            tick_size=Decimal("0.25"),
            expiry=datetime(2024, 12, 20, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="cannot price"):
            UsEquityCostModel().cost(
                TradeContext(
                    instrument=future,
                    side=Side.BUY,
                    quantity=Decimal(1),
                    price=Decimal(5000),
                )
            )


class TestExactness:
    """Float would drift here; Decimal does not (§14.1.2)."""

    def test_no_accumulation_error_over_many_trades(self):
        p = Portfolio(cash=Decimal(1_000_000))
        for _ in range(1000):
            p.apply_fill(fill(Side.BUY, "1", "0.1"))
            p.apply_fill(fill(Side.SELL, "1", "0.1"))
        # 0.1 is not representable in binary floating point; 1000 round trips
        # would visibly drift. Decimal returns to the exact starting cash.
        assert p.cash == Decimal("1000000.00")
        assert p.position(IID).is_flat
