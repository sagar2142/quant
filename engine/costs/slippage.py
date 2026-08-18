"""Slippage and market impact — MASTER_PLAN §7.5.

    slippage = k * half_spread + lambda * sigma * sqrt(order_value / ADV)

The square root matters. Impact is *concave* in size, not linear — the
empirically supported form across equity markets. A linear model badly
overstates the cost of large orders and understates small ones.

At personal-account size the impact term is small and the spread term dominates.
It is built correctly now anyway, because a model that is only right at current
size stops being right exactly when it matters.

**Calibration is the point.** `LAMBDA` is a literature default, not a
measurement. §9 requires logging intended-versus-actual fill prices from live
trading and refitting these constants from your own fills. Until then, every
number produced here is an assumption wearing a decimal point.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from engine.costs.model import TradeContext

__all__ = ["SPREAD_CROSSING_K", "SQRT_IMPACT_LAMBDA", "SlippageModel"]

#: Marketable orders cross the full spread; passive ones would use ~0.
SPREAD_CROSSING_K = Decimal("1.0")

#: Literature default for the square-root impact coefficient. Replace with a
#: value fitted to your own fills as soon as live data exists (§9).
SQRT_IMPACT_LAMBDA = Decimal("0.7")

#: Used only when a venue quotes no spread. 5bp is a reasonable liquid-equity
#: stand-in and is deliberately pessimistic for large caps.
DEFAULT_SPREAD_BPS = Decimal("5")

#: Fallback daily volatility when none is supplied.
DEFAULT_VOLATILITY = Decimal("0.02")


@dataclass(frozen=True)
class SlippageModel:
    """Square-root market impact plus spread crossing."""

    spread_k: Decimal = SPREAD_CROSSING_K
    impact_lambda: Decimal = SQRT_IMPACT_LAMBDA
    default_spread_bps: Decimal = DEFAULT_SPREAD_BPS
    default_volatility: Decimal = DEFAULT_VOLATILITY

    def cost(self, ctx: TradeContext) -> Decimal:
        """Expected slippage in currency units. Always a positive cost."""
        return self._spread_cost(ctx) + self._impact_cost(ctx)

    def _spread_cost(self, ctx: TradeContext) -> Decimal:
        """Half-spread paid on crossing, in currency units."""
        if ctx.spread is not None:
            half_spread = ctx.spread / 2
        else:
            half_spread = ctx.price * self.default_spread_bps / Decimal(20_000)
        return self.spread_k * half_spread * ctx.quantity * ctx.instrument.multiplier

    def _impact_cost(self, ctx: TradeContext) -> Decimal:
        """Square-root impact. Zero when ADV is unknown or non-positive.

        Returning zero rather than guessing is deliberate: an invented ADV
        produces an invented cost that is indistinguishable from a measured one.
        The absence shows up in the cost-sensitivity sweep instead.
        """
        if ctx.adv_value is None or ctx.adv_value <= 0:
            return Decimal(0)

        volatility = ctx.volatility if ctx.volatility is not None else self.default_volatility
        participation = ctx.notional / ctx.adv_value
        # Decimal.sqrt keeps the whole calculation exact-typed; §14.1.2 bans
        # float in this module and a round-trip through float would reintroduce
        # representation error into a money figure.
        return self.impact_lambda * volatility * participation.sqrt() * ctx.notional

    def scaled(self, multiplier: Decimal) -> SlippageModel:
        """A copy with both coefficients scaled, for sensitivity sweeps."""
        return SlippageModel(
            spread_k=self.spread_k * multiplier,
            impact_lambda=self.impact_lambda * multiplier,
            default_spread_bps=self.default_spread_bps,
            default_volatility=self.default_volatility,
        )
