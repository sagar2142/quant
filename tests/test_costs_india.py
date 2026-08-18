"""Indian cost models (§7.1, §7.2), hand-verified.

§14.5 requires every cost model to be asserted against a hand-computed example.
The worked round trip below is the reference; if a statutory rate changes, this
test fails first and loudly, which is the intended behaviour.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from core.clock import UTC
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId, OptionType
from core.orders import Side
from engine.costs.india import (
    DP_CHARGE_PER_SCRIP,
    NseEquityCostModel,
    NseFuturesCostModel,
    NseOptionsCostModel,
)
from engine.costs.model import ScaledCostModel, TradeContext
from engine.costs.slippage import SlippageModel

# No-op slippage isolates the statutory fees, which are what the hand
# calculation covers. Slippage gets its own tests.
NO_SLIPPAGE = SlippageModel(spread_k=Decimal(0), impact_lambda=Decimal(0))

EQUITY = Instrument(
    instrument_id=InstrumentId("NSE:INE002A01018"),
    symbol="RELIANCE",
    asset_class=AssetClass.EQUITY,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.05"),
)

FUTURE = Instrument(
    instrument_id=InstrumentId("NSE:NIFTY24DECFUT"),
    symbol="NIFTY24DECFUT",
    asset_class=AssetClass.FUTURE,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.05"),
    multiplier=Decimal(50),
    expiry=datetime(2024, 12, 26, tzinfo=UTC),
)

OPTION = Instrument(
    instrument_id=InstrumentId("NSE:NIFTY24DEC24000CE"),
    symbol="NIFTY24DEC24000CE",
    asset_class=AssetClass.OPTION,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.05"),
    multiplier=Decimal(50),
    expiry=datetime(2024, 12, 26, tzinfo=UTC),
    strike=Decimal(24000),
    option_type=OptionType.CALL,
)


def ctx(side: Side, **kw) -> TradeContext:
    defaults = dict(instrument=EQUITY, side=side, quantity=Decimal(100), price=Decimal(1000))
    return TradeContext(**{**defaults, **kw})


class TestDeliveryHandCalculation:
    """₹1,00,000 delivery trade, Zerodha schedule. Worked by hand:

    BUY                                  SELL
    brokerage      0.00                  brokerage      0.00
    STT   0.1%   100.00                  STT   0.1%   100.00
    exch 0.00297%  2.97                  exch 0.00297%  2.97
    SEBI ₹10/cr    0.10                  SEBI ₹10/cr    0.10
    stamp 0.015%  15.00                  stamp          0.00
    GST 18%x3.07   0.55                  GST 18%x3.07   0.55
    DP             0.00                  DP            15.34
    ─────────────────────                ─────────────────────
                 118.62                               118.96
    """

    model = NseEquityCostModel(slippage=NO_SLIPPAGE)

    def test_buy_itemised(self):
        c = self.model.cost(ctx(Side.BUY))
        assert c.brokerage == Decimal("0.00")
        assert c.stt == Decimal("100.00")
        assert c.exchange_fee == Decimal("2.97")
        assert c.regulatory_fee == Decimal("0.10")
        assert c.stamp_duty == Decimal("15.00")
        assert c.gst == Decimal("0.55")
        assert c.depository_fee == Decimal("0.00")

    def test_buy_total(self):
        assert self.model.cost(ctx(Side.BUY)).explicit == Decimal("118.62")

    def test_sell_itemised(self):
        c = self.model.cost(ctx(Side.SELL))
        assert c.stt == Decimal("100.00")  # both legs, unlike intraday
        assert c.stamp_duty == Decimal("0.00")  # buy side only
        assert c.depository_fee == DP_CHARGE_PER_SCRIP

    def test_sell_total(self):
        assert self.model.cost(ctx(Side.SELL)).explicit == Decimal("118.96")

    def test_round_trip_matches_plan(self):
        buy = self.model.cost(ctx(Side.BUY)).explicit
        sell = self.model.cost(ctx(Side.SELL)).explicit
        # 237.58 on 1,00,000 = 0.2376%. Excluding DP it is 0.222%, which is the
        # ~0.22% the plan uses for break-even arithmetic (§7.1).
        assert buy + sell == Decimal("237.58")
        assert (buy + sell - DP_CHARGE_PER_SCRIP) / Decimal(100_000) == pytest.approx(
            Decimal("0.002222"), abs=Decimal("0.000002")
        )

    def test_break_even_arithmetic(self):
        """The number that constrains the whole strategy space (§7.1)."""
        buy = self.model.cost(ctx(Side.BUY)).explicit
        sell = self.model.cost(ctx(Side.SELL)).explicit
        round_trip_pct = (buy + sell) / Decimal(100_000)
        # Weekly rebalancing needs >10% gross annual just to break even.
        assert round_trip_pct * 52 > Decimal("0.10")
        # Monthly is workable.
        assert round_trip_pct * 12 < Decimal("0.03")


class TestIntraday:
    model = NseEquityCostModel(brokerage_per_order=Decimal(20), slippage=NO_SLIPPAGE)

    def test_no_stt_on_intraday_buy(self):
        # Intraday STT is sell-side only, unlike delivery.
        assert self.model.cost(ctx(Side.BUY, intraday=True)).stt == Decimal("0.00")

    def test_reduced_stt_on_intraday_sell(self):
        # 0.025% of 1,00,000
        assert self.model.cost(ctx(Side.SELL, intraday=True)).stt == Decimal("25.00")

    def test_no_dp_charge_intraday(self):
        # Nothing leaves the demat account.
        assert self.model.cost(ctx(Side.SELL, intraday=True)).depository_fee == Decimal("0.00")

    def test_flat_brokerage_applied(self):
        assert self.model.cost(ctx(Side.BUY, intraday=True)).brokerage == Decimal("20.00")

    def test_intraday_cheaper_than_delivery(self):
        delivery = NseEquityCostModel(slippage=NO_SLIPPAGE)
        intra = self.model.cost(ctx(Side.SELL, intraday=True)).explicit
        deliv = delivery.cost(ctx(Side.SELL)).explicit
        assert intra < deliv


class TestSmallAccountPenalty:
    """DP charges are flat per scrip, so they bite hardest on small positions."""

    model = NseEquityCostModel(slippage=NO_SLIPPAGE)

    def test_dp_dominates_small_exit(self):
        small = TradeContext(
            instrument=EQUITY, side=Side.SELL, quantity=Decimal(1), price=Decimal(1000)
        )
        cost = self.model.cost(small)
        # ₹15.34 on ₹1,000 is 153bp before anything else.
        assert cost.depository_fee > cost.stt
        assert cost.bps_of(small.notional) > Decimal(150)

    def test_dp_negligible_on_large_exit(self):
        large = TradeContext(
            instrument=EQUITY, side=Side.SELL, quantity=Decimal(10_000), price=Decimal(1000)
        )
        assert self.model.cost(large).bps_of(large.notional) < Decimal(25)

    def test_dp_can_be_disabled(self):
        model = NseEquityCostModel(slippage=NO_SLIPPAGE, charge_dp_fees=False)
        assert model.cost(ctx(Side.SELL)).depository_fee == Decimal("0.00")


class TestFutures:
    model = NseFuturesCostModel(slippage=NO_SLIPPAGE)

    def _ctx(self, side: Side) -> TradeContext:
        return TradeContext(
            instrument=FUTURE, side=side, quantity=Decimal(1), price=Decimal(24_000)
        )

    def test_notional_uses_multiplier(self):
        assert self._ctx(Side.BUY).notional == Decimal(1_200_000)

    def test_stt_sell_side_only(self):
        assert self.model.cost(self._ctx(Side.BUY)).stt == Decimal("0.00")
        assert self.model.cost(self._ctx(Side.SELL)).stt == Decimal("240.00")

    def test_no_dp_charge(self):
        assert self.model.cost(self._ctx(Side.SELL)).depository_fee == Decimal("0.00")

    def test_rejects_wrong_asset_class(self):
        with pytest.raises(ValueError, match="cannot price"):
            self.model.cost(ctx(Side.BUY))


class TestOptions:
    model = NseOptionsCostModel(slippage=NO_SLIPPAGE)

    def _ctx(self, side: Side) -> TradeContext:
        # Premium ₹100 x lot 50 = ₹5,000 premium value.
        return TradeContext(instrument=OPTION, side=side, quantity=Decimal(1), price=Decimal(100))

    def test_charges_are_on_premium_not_notional(self):
        cost = self.model.cost(self._ctx(Side.SELL))
        # 0.1% of ₹5,000 premium, not of the ₹12L underlying notional.
        assert cost.stt == Decimal("5.00")

    def test_no_stt_on_buy(self):
        assert self.model.cost(self._ctx(Side.BUY)).stt == Decimal("0.00")

    def test_exercise_stt_on_intrinsic_value(self):
        """The classic account-killer (§7.2)."""
        # An option whose premium was ₹5,000 but which expires ₹50,000 ITM.
        exercise = NseOptionsCostModel.exercise_stt(Decimal(50_000))
        assert exercise == Decimal("62.50")
        # Far exceeds the STT that squaring off would have cost.
        assert exercise > self.model.cost(self._ctx(Side.SELL)).stt

    def test_rejects_wrong_asset_class(self):
        with pytest.raises(ValueError, match="cannot price"):
            self.model.cost(ctx(Side.BUY))


class TestCostScaling:
    """Gauntlet test 7 requires survival at 3x modelled costs (§5.4)."""

    def test_triple_costs_triples_every_component(self):
        base = NseEquityCostModel(slippage=NO_SLIPPAGE)
        scaled = ScaledCostModel(base, Decimal(3))
        b, s = base.cost(ctx(Side.BUY)), scaled.cost(ctx(Side.BUY))
        assert s.stt == b.stt * 3
        assert s.total == b.total * 3

    def test_scaled_name_records_multiplier(self):
        scaled = ScaledCostModel(NseEquityCostModel(), Decimal(3))
        assert "x3" in scaled.name

    def test_negative_multiplier_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            ScaledCostModel(NseEquityCostModel(), Decimal(-1))


class TestContextValidation:
    def test_zero_quantity_rejected(self):
        with pytest.raises(ValueError, match="quantity"):
            ctx(Side.BUY, quantity=Decimal(0))

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="price"):
            ctx(Side.BUY, price=Decimal(-1))


class TestBreakdownArithmetic:
    model = NseEquityCostModel(slippage=NO_SLIPPAGE)

    def test_breakdowns_add(self):
        combined = self.model.cost(ctx(Side.BUY)) + self.model.cost(ctx(Side.SELL))
        assert combined.stt == Decimal("200.00")
        assert combined.total == Decimal("237.58")

    def test_bps_of_zero_notional_is_zero(self):
        assert self.model.cost(ctx(Side.BUY)).bps_of(Decimal(0)) == Decimal(0)

    def test_format_lists_nonzero_components(self):
        text = self.model.cost(ctx(Side.BUY)).format()
        assert "STT" in text
        assert "TOTAL" in text
        assert "depository" not in text  # zero on a buy, so omitted
