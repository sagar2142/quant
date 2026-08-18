"""US equity cost model — MASTER_PLAN §7.3.

Roughly **50x cheaper than NSE delivery**, and that is not a detail — it decides
which strategies are viable in each market:

    break-even gross annual return, by rebalance frequency
                        NSE delivery      US large cap
    weekly                   ~11.4%            ~0.2%
    monthly                   ~2.6%            ~0.05%

A strategy validated only against US costs and then pointed at NSE will be
destroyed by the difference. The reverse — validated on NSE, run on US — is
merely conservative. **Always re-check a strategy against the cost model of the
market it will actually trade** (§7.1).

**The regulatory fees are sell-side only**, which makes a round trip asymmetric
in a way the Indian model is not: buying is genuinely free at most US retail
brokers, and the entire explicit cost sits on the exit.

**Rates change annually.** The SEC fee in particular is revised each fiscal
year, sometimes substantially. Verify against a recent trade confirmation before
trusting a backtest that depends on it (§14.11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_UP, Decimal

from core.orders import Side
from engine.costs.model import CostBreakdown, TradeContext, quantize_money
from engine.costs.slippage import SlippageModel

__all__ = ["SEC_FEE_RATE", "TAF_CAP", "TAF_PER_SHARE", "UsEquityCostModel"]

#: SEC Section 31 fee, on the *value* of a sale. Revised annually — this is the
#: FY2025 rate. Verify against a recent confirmation.
SEC_FEE_RATE = Decimal("0.0000278")

#: FINRA Trading Activity Fee, per share sold, capped per trade.
TAF_PER_SHARE = Decimal("0.000166")
TAF_CAP = Decimal("8.30")

#: Commission-free is the retail norm (Alpaca, Schwab, Fidelity). Set a value
#: here for a broker that charges.
DEFAULT_COMMISSION = Decimal(0)


@dataclass(frozen=True)
class UsEquityCostModel:
    """US cash equities, retail commission-free schedule.

    Args:
        commission_per_order: Flat per-order commission. Zero at Alpaca.
        slippage: Spread and impact. On US large caps the spread term dominates
            the explicit fees by an order of magnitude, which is why modelling
            it properly matters more here than the fee schedule does.
    """

    commission_per_order: Decimal = DEFAULT_COMMISSION
    slippage: SlippageModel = field(default_factory=SlippageModel)
    name: str = "US_EQUITY"

    def cost(self, ctx: TradeContext) -> CostBreakdown:
        if not ctx.instrument.asset_class.is_cash_equity:
            raise ValueError(
                f"{self.name} cannot price {ctx.instrument.asset_class}; "
                "it covers US cash equities and ETFs only"
            )

        is_sell = ctx.side is Side.SELL
        notional = ctx.notional

        # Both regulatory fees are sell-side only, so a buy carries no explicit
        # cost at all beyond any commission.
        sec_fee = notional * SEC_FEE_RATE if is_sell else Decimal(0)
        taf = min(ctx.quantity * TAF_PER_SHARE, TAF_CAP) if is_sell else Decimal(0)

        return CostBreakdown(
            brokerage=quantize_money(self.commission_per_order),
            # Regulatory fees are rounded UP to the cent by the venue, never
            # down — matching that avoids a systematic one-cent-per-trade
            # reconciliation drift.
            regulatory_fee=(sec_fee + taf).quantize(Decimal("0.01"), rounding=ROUND_UP),
            slippage=quantize_money(self.slippage.cost(ctx)),
        )

    @staticmethod
    def round_trip_bps(notional: Decimal, shares: Decimal) -> Decimal:
        """Explicit round-trip cost in basis points, excluding spread.

        Useful for the break-even arithmetic in §7.1: compare it against the
        expected gross edge per trade before believing a strategy is viable.
        """
        if notional <= 0:
            return Decimal(0)
        sec_fee = notional * SEC_FEE_RATE
        taf = min(shares * TAF_PER_SHARE, TAF_CAP)
        return ((sec_fee + taf) / notional) * Decimal(10_000)
