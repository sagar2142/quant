"""Indian market cost models — MASTER_PLAN §7.1, §7.2.

Every rate below is a *statutory or exchange-published* number, not a modelling
choice, and each carries its source (§14.11). Rates change: STT was revised in
2024, exchange transaction charges in 2023-24. Verify against a recent contract
note before trusting any backtest that depends on them, and re-verify before
going live.

**Why this file constrains the whole project.** NSE delivery round trip is
~0.22% before slippage. Annualised break-even gross return by rebalance
frequency:

    weekly    ~11.4%       ← rules out most weekly strategies
    monthly    ~2.6%       ← workable
    quarterly  ~0.9%

That is the arithmetic behind the plan's push toward monthly, cross-sectional,
low-turnover designs (§6). It is not a preference.

**DP charges are flat per scrip per sell day**, so they hit small accounts
hardest: ₹15.34 on a ₹25,000 position is 6bp on its own, on top of everything
else. A 20-position portfolio at ₹5L capital pays it twenty times per exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from core.instruments import AssetClass
from core.orders import Side
from engine.costs.model import CostBreakdown, TradeContext, quantize_money
from engine.costs.slippage import SlippageModel

__all__ = ["NseEquityCostModel", "NseFuturesCostModel", "NseOptionsCostModel"]

# ── statutory rates ─────────────────────────────────────────────────────────
# Securities Transaction Tax, Finance Act 2004 as amended (2024 revision).
STT_DELIVERY = Decimal("0.001")  # 0.1%, BOTH legs
STT_INTRADAY_SELL = Decimal("0.00025")  # 0.025%, sell side only
STT_FUTURES_SELL = Decimal("0.0002")  # 0.02%, sell side only
STT_OPTIONS_SELL = Decimal("0.001")  # 0.1% of PREMIUM, sell side only
#: On exercised in-the-money options, charged on INTRINSIC value, not premium.
#: The classic retail account-killer — an option worth ₹5 of premium can attract
#: STT on ₹50,000 of intrinsic value if allowed to expire ITM.
STT_OPTIONS_EXERCISE = Decimal("0.00125")

# NSE exchange transaction charges (per NSE circular, current schedule).
NSE_TXN_EQUITY = Decimal("0.0000297")  # 0.00297%
NSE_TXN_FUTURES = Decimal("0.0000173")  # 0.00173%
NSE_TXN_OPTIONS = Decimal("0.0003503")  # 0.03503% of premium

# SEBI turnover fee: ₹10 per crore.
SEBI_TURNOVER_FEE = Decimal("0.000001")

# Stamp duty, Indian Stamp Act as amended 2020. Buy side only.
STAMP_DELIVERY = Decimal("0.00015")  # 0.015%
STAMP_INTRADAY = Decimal("0.00003")  # 0.003%
STAMP_FUTURES = Decimal("0.00002")  # 0.002%
STAMP_OPTIONS = Decimal("0.00003")  # 0.003%

# GST on (brokerage + exchange + SEBI). Not on STT or stamp duty.
GST_RATE = Decimal("0.18")

# Discount-broker defaults (Zerodha schedule).
BROKERAGE_DELIVERY = Decimal("0")
BROKERAGE_FLAT = Decimal("20")  # per executed order, intraday and F&O
#: Per scrip, per day, on the sell side. CDSL charge plus broker markup.
DP_CHARGE_PER_SCRIP = Decimal("15.34")


@dataclass(frozen=True)
class _IndiaBase:
    """Shared GST and regulatory arithmetic."""

    slippage: SlippageModel = field(default_factory=SlippageModel)

    @staticmethod
    def _gst(brokerage: Decimal, exchange: Decimal, sebi: Decimal) -> Decimal:
        return (brokerage + exchange + sebi) * GST_RATE

    @staticmethod
    def _sebi(notional: Decimal) -> Decimal:
        return notional * SEBI_TURNOVER_FEE


@dataclass(frozen=True)
class NseEquityCostModel(_IndiaBase):
    """NSE cash equity, delivery (CNC) or intraday (MIS).

    Args:
        brokerage_per_order: 0 for delivery on a discount broker; ₹20 flat for
            intraday.
        charge_dp_fees: Whether to apply DP charges on exits. Set False only
            when modelling a broker that does not levy them.
    """

    brokerage_per_order: Decimal = BROKERAGE_DELIVERY
    charge_dp_fees: bool = True
    name: str = "NSE_EQUITY_DELIVERY"

    def cost(self, ctx: TradeContext) -> CostBreakdown:
        if ctx.instrument.asset_class not in {AssetClass.EQUITY, AssetClass.ETF}:
            raise ValueError(
                f"{self.name} cannot price {ctx.instrument.asset_class}; "
                "use the futures or options model"
            )

        notional = ctx.notional
        is_sell = ctx.side is Side.SELL

        if ctx.intraday:
            stt = notional * STT_INTRADAY_SELL if is_sell else Decimal(0)
            stamp = Decimal(0) if is_sell else notional * STAMP_INTRADAY
        else:
            # Delivery STT applies to BOTH legs — the dominant cost, and the
            # reason intraday-frequency delivery strategies cannot work.
            stt = notional * STT_DELIVERY
            stamp = Decimal(0) if is_sell else notional * STAMP_DELIVERY

        brokerage = self.brokerage_per_order
        exchange = notional * NSE_TXN_EQUITY
        sebi = self._sebi(notional)
        gst = self._gst(brokerage, exchange, sebi)

        # Flat, per scrip, per sell day — punishing on small positions.
        dp = (
            DP_CHARGE_PER_SCRIP
            if (is_sell and not ctx.intraday and self.charge_dp_fees)
            else Decimal(0)
        )

        return CostBreakdown(
            brokerage=quantize_money(brokerage),
            stt=quantize_money(stt),
            exchange_fee=quantize_money(exchange),
            regulatory_fee=quantize_money(sebi),
            stamp_duty=quantize_money(stamp),
            gst=quantize_money(gst),
            depository_fee=quantize_money(dp),
            slippage=quantize_money(self.slippage.cost(ctx)),
        )


@dataclass(frozen=True)
class NseFuturesCostModel(_IndiaBase):
    """NSE equity and index futures."""

    brokerage_per_order: Decimal = BROKERAGE_FLAT
    name: str = "NSE_FUTURES"

    def cost(self, ctx: TradeContext) -> CostBreakdown:
        if ctx.instrument.asset_class is not AssetClass.FUTURE:
            raise ValueError(f"{self.name} cannot price {ctx.instrument.asset_class}")

        notional = ctx.notional
        is_sell = ctx.side is Side.SELL

        stt = notional * STT_FUTURES_SELL if is_sell else Decimal(0)
        stamp = Decimal(0) if is_sell else notional * STAMP_FUTURES
        brokerage = self.brokerage_per_order
        exchange = notional * NSE_TXN_FUTURES
        sebi = self._sebi(notional)

        return CostBreakdown(
            brokerage=quantize_money(brokerage),
            stt=quantize_money(stt),
            exchange_fee=quantize_money(exchange),
            regulatory_fee=quantize_money(sebi),
            stamp_duty=quantize_money(stamp),
            gst=quantize_money(self._gst(brokerage, exchange, sebi)),
            slippage=quantize_money(self.slippage.cost(ctx)),
        )


@dataclass(frozen=True)
class NseOptionsCostModel(_IndiaBase):
    """NSE index and stock options.

    Charges are levied on **premium**, not on notional — except the exercise
    STT, which is charged on intrinsic value and has wiped out retail accounts
    that let ITM options expire rather than squaring off.
    """

    brokerage_per_order: Decimal = BROKERAGE_FLAT
    name: str = "NSE_OPTIONS"

    def cost(self, ctx: TradeContext) -> CostBreakdown:
        if ctx.instrument.asset_class is not AssetClass.OPTION:
            raise ValueError(f"{self.name} cannot price {ctx.instrument.asset_class}")

        premium = ctx.notional  # price is the premium for an option
        is_sell = ctx.side is Side.SELL

        stt = premium * STT_OPTIONS_SELL if is_sell else Decimal(0)
        stamp = Decimal(0) if is_sell else premium * STAMP_OPTIONS
        brokerage = self.brokerage_per_order
        exchange = premium * NSE_TXN_OPTIONS
        sebi = self._sebi(premium)

        return CostBreakdown(
            brokerage=quantize_money(brokerage),
            stt=quantize_money(stt),
            exchange_fee=quantize_money(exchange),
            regulatory_fee=quantize_money(sebi),
            stamp_duty=quantize_money(stamp),
            gst=quantize_money(self._gst(brokerage, exchange, sebi)),
            slippage=quantize_money(self.slippage.cost(ctx)),
        )

    @staticmethod
    def exercise_stt(intrinsic_value: Decimal) -> Decimal:
        """STT on an option allowed to expire in the money.

        Charged on intrinsic value, so a position whose premium was trivial can
        attract a tax proportional to the full moneyness. Any options strategy
        must model expiry explicitly rather than assuming a square-off.
        """
        return quantize_money(intrinsic_value * STT_OPTIONS_EXERCISE)
