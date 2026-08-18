"""US equity cost model (§7.3) and the shared slippage model (§7.5).

Numbers here are computed by hand from the published schedules, not from the
implementation. A test that asserts what the code already does cannot catch the
code being wrong.

**Why this matters beyond the US book.** The plan's claim that NSE is ~50x more
expensive than US equities decides which strategies are viable in each market
(§7.1). If either cost model drifts, that comparison silently stops being true
and a strategy gets validated against a schedule it will never trade against.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from core.clock import UTC
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from core.orders import Side
from engine.costs.model import TradeContext
from engine.costs.slippage import SlippageModel
from engine.costs.us_equity import (
    SEC_FEE_RATE,
    TAF_CAP,
    TAF_PER_SHARE,
    UsEquityCostModel,
)


def us_stock(symbol: str = "AAPL") -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(f"NASDAQ:{symbol}"),
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange=Exchange.NASDAQ,
        currency=Currency.USD,
        tick_size=Decimal("0.01"),
    )


def index_future() -> Instrument:
    return Instrument(
        instrument_id=InstrumentId("NASDAQ:ESZ5"),
        symbol="ESZ5",
        asset_class=AssetClass.FUTURE,
        exchange=Exchange.NASDAQ,
        currency=Currency.USD,
        tick_size=Decimal("0.25"),
        # A derivative without an expiry is refused at construction.
        expiry=datetime(2025, 12, 19, tzinfo=UTC),
        multiplier=Decimal(50),
    )


def ctx(
    side: Side,
    quantity: Decimal = Decimal(100),
    price: Decimal = Decimal(200),
    instrument: Instrument | None = None,
    **kwargs,
) -> TradeContext:
    return TradeContext(
        instrument=instrument or us_stock(),
        side=side,
        quantity=quantity,
        price=price,
        **kwargs,
    )


#: No slippage, so the fee arithmetic is visible on its own.
FEES_ONLY = UsEquityCostModel(slippage=SlippageModel(spread_k=Decimal(0)))


class TestSellSideOnly:
    """The asymmetry that makes a US round trip unlike an Indian one."""

    def test_a_buy_carries_no_regulatory_fee(self):
        cost = FEES_ONLY.cost(ctx(Side.BUY))
        assert cost.regulatory_fee == Decimal(0)
        assert cost.brokerage == Decimal(0)
        assert cost.total == Decimal(0)

    def test_a_sell_pays_sec_fee_and_taf(self):
        """$20,000 sale of 100 shares, by hand:

        SEC  = 20000 * 0.0000278 = 0.556
        TAF  = 100 * 0.000166    = 0.0166
        sum  = 0.5726  ->  0.58 rounded UP to the cent
        """
        cost = FEES_ONLY.cost(ctx(Side.SELL))
        assert cost.regulatory_fee == Decimal("0.58")

    def test_round_trip_is_entirely_on_the_exit(self):
        assert FEES_ONLY.cost(ctx(Side.BUY)).total == Decimal(0)
        assert FEES_ONLY.cost(ctx(Side.SELL)).total > 0


class TestRounding:
    def test_regulatory_fees_round_up_not_down(self):
        """Venues round up to the cent. Matching that avoids a systematic
        one-cent-per-trade reconciliation drift — small per trade, and a
        guaranteed break every single day."""
        cost = FEES_ONLY.cost(ctx(Side.SELL, quantity=Decimal(1), price=Decimal(1)))
        assert cost.regulatory_fee == Decimal("0.01")

    def test_the_fee_is_never_silently_zero_on_a_sale(self):
        """Rounding down would make small sales free, which they are not."""
        cost = FEES_ONLY.cost(ctx(Side.SELL, quantity=Decimal(1), price=Decimal("0.01")))
        assert cost.regulatory_fee > 0


class TestTafCap:
    def test_taf_is_capped_per_trade(self):
        """The cap binds above 50,000 shares (8.30 / 0.000166)."""
        shares = Decimal(1_000_000)
        cost = FEES_ONLY.cost(ctx(Side.SELL, quantity=shares, price=Decimal("0.10")))
        notional = shares * Decimal("0.10")
        uncapped = notional * SEC_FEE_RATE + shares * TAF_PER_SHARE
        assert cost.regulatory_fee < uncapped

    def test_below_the_cap_taf_scales_with_shares(self):
        cheap = FEES_ONLY.cost(ctx(Side.SELL, quantity=Decimal(1000), price=Decimal("0.01")))
        dearer = FEES_ONLY.cost(ctx(Side.SELL, quantity=Decimal(10_000), price=Decimal("0.01")))
        assert dearer.regulatory_fee > cheap.regulatory_fee

    def test_cap_threshold_matches_the_published_rate(self):
        assert pytest.approx(50_000, rel=0.01) == TAF_CAP / TAF_PER_SHARE


class TestCommission:
    def test_commission_free_is_the_default(self):
        assert UsEquityCostModel().commission_per_order == Decimal(0)

    def test_a_charging_broker_is_supported(self):
        model = UsEquityCostModel(
            commission_per_order=Decimal("0.65"),
            slippage=SlippageModel(spread_k=Decimal(0)),
        )
        assert model.cost(ctx(Side.BUY)).brokerage == Decimal("0.65")


class TestAssetClassGuard:
    def test_a_future_is_refused(self):
        """Futures carry exchange and clearing fees this model does not know.

        Refused rather than approximated: a silently wrong cost on a
        derivatives backtest is indistinguishable from a right one.
        """
        with pytest.raises(ValueError, match="cannot price"):
            FEES_ONLY.cost(ctx(Side.BUY, instrument=index_future()))

    def test_an_etf_is_accepted(self):
        etf = Instrument(
            instrument_id=InstrumentId("ARCA:SPY"),
            symbol="SPY",
            asset_class=AssetClass.ETF,
            exchange=Exchange.ARCA,
            currency=Currency.USD,
            tick_size=Decimal("0.01"),
        )
        assert FEES_ONLY.cost(ctx(Side.SELL, instrument=etf)).total > 0


class TestRoundTripBps:
    def test_matches_hand_arithmetic(self):
        """$20,000 / 100 shares:  (0.556 + 0.0166) / 20000 * 10000 = 0.2863 bps"""
        bps = UsEquityCostModel.round_trip_bps(Decimal(20_000), Decimal(100))
        assert abs(bps - Decimal("0.2863")) < Decimal("0.0001")

    def test_zero_notional_is_zero_not_a_division_error(self):
        assert UsEquityCostModel.round_trip_bps(Decimal(0), Decimal(100)) == Decimal(0)

    def test_us_is_far_cheaper_than_nse(self):
        """The §7.3 claim, asserted rather than assumed.

        NSE delivery is ~22 bps one way; US explicit fees are well under 1 bp.
        If this ever fails, the market-viability comparison in the plan has
        stopped being true and every cross-market conclusion needs revisiting.
        """
        assert UsEquityCostModel.round_trip_bps(Decimal(1_000_000), Decimal(5_000)) < Decimal(1)


class TestSlippageModel:
    """Shared by both cost models, so its edges are money in either market."""

    def test_explicit_spread_beats_the_default(self):
        """A quoted spread is a measurement; the default is a guess."""
        model = SlippageModel()
        quoted = model.cost(ctx(Side.BUY, spread=Decimal("0.02")))
        # 100 shares * half of $0.02 = $1.00, times spread_k.
        assert quoted == model.spread_k * Decimal(1)

    def test_impact_is_zero_when_adv_is_unknown(self):
        """Zero rather than a guess: an invented ADV produces an invented cost
        indistinguishable from a measured one."""
        model = SlippageModel()
        no_adv = model.cost(ctx(Side.BUY, spread=Decimal("0.02")))
        with_adv = model.cost(ctx(Side.BUY, spread=Decimal("0.02"), adv_value=Decimal(1_000_000)))
        assert with_adv > no_adv

    @pytest.mark.parametrize("adv", [Decimal(0), Decimal(-1)])
    def test_nonpositive_adv_contributes_no_impact(self, adv):
        model = SlippageModel()
        baseline = model.cost(ctx(Side.BUY, spread=Decimal("0.02")))
        assert model.cost(ctx(Side.BUY, spread=Decimal("0.02"), adv_value=adv)) == baseline

    def test_explicit_volatility_beats_the_default(self):
        model = SlippageModel()
        calm = model.cost(ctx(Side.BUY, adv_value=Decimal(1_000_000), volatility=Decimal("0.005")))
        wild = model.cost(ctx(Side.BUY, adv_value=Decimal(1_000_000), volatility=Decimal("0.05")))
        assert wild > calm

    def test_impact_grows_superlinearly_with_size(self):
        """Square-root impact on participation, applied to notional: the total
        impact cost rises faster than the trade does."""
        model = SlippageModel()
        small = model.cost(ctx(Side.BUY, quantity=Decimal(100), adv_value=Decimal(10_000_000)))
        large = model.cost(ctx(Side.BUY, quantity=Decimal(10_000), adv_value=Decimal(10_000_000)))
        assert large > small * 100

    def test_scaled_multiplies_both_coefficients(self):
        """Gauntlet check 7 runs the whole model at 3x (§5.4)."""
        base = SlippageModel()
        tripled = base.scaled(Decimal(3))
        assert tripled.spread_k == base.spread_k * 3
        assert tripled.impact_lambda == base.impact_lambda * 3
        assert tripled.default_spread_bps == base.default_spread_bps
        assert tripled.default_volatility == base.default_volatility

    def test_scaling_raises_the_cost(self):
        context = ctx(Side.BUY, spread=Decimal("0.02"), adv_value=Decimal(1_000_000))
        assert SlippageModel().scaled(Decimal(3)).cost(context) > SlippageModel().cost(context)

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_slippage_is_always_a_cost_never_a_credit(self, side):
        assert SlippageModel().cost(ctx(side, adv_value=Decimal(1_000_000))) > 0
