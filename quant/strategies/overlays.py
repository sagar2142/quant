"""Strategy overlays — MASTER_PLAN §6, §8.

An overlay wraps a strategy and rescales its weights without touching its
opinions. The relative sizes stay exactly as the inner strategy expressed them;
only the overall magnitude changes.

**Volatility targeting is the one the plan calls nearly free alpha.** It
requires no new edge: scale position size inversely to recent realised
volatility so the portfolio runs at a constant risk level. Most strategies
size by conviction and let risk float, which means they take their largest
risk exactly when markets are most dangerous. Inverting that improves
risk-adjusted returns on almost anything.

It is not literally free. Rescaling on every bar is turnover, and turnover is
the most reliable way to lose money to costs (§7.1). Hence `rebalance_band`:
leave the scale alone until it has drifted enough to be worth paying for.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from quant.strategies.base import MarketView, Strategy, StrategySpec, TargetWeights

__all__ = ["VolatilityTarget", "vol_scale"]

#: Below this many observations a volatility estimate is noise.
MIN_VOL_OBSERVATIONS = 20


def vol_scale(
    realised_vol: float,
    target_vol: float,
    max_leverage: float,
) -> Decimal:
    """Scaling factor that maps realised volatility onto the target.

    Returns 0 when volatility is unmeasurable, rather than 1: an unknown risk
    level is not the same as an acceptable one, and defaulting to full size on
    missing data is how a strategy takes its largest position in the least
    understood conditions (§14.1.5).
    """
    if realised_vol <= 0 or not np.isfinite(realised_vol):
        return Decimal(0)
    scale = min(target_vol / realised_vol, max_leverage)
    return Decimal(str(round(scale, 4)))


class VolatilityTarget(Strategy):
    """Wraps a strategy and holds portfolio volatility roughly constant.

    Args:
        inner: The strategy whose views are being rescaled.
        target_vol: Annualised volatility to aim for, e.g. 0.15 for 15%.
        lookback: Bars used to estimate realised volatility.
        max_leverage: Cap on the scale factor. Without it, a quiet market
            produces enormous positions right before volatility returns — the
            precise mechanism behind several well-known blow-ups.
        rebalance_band: Leave the scale untouched until it has drifted this far
            proportionally. Pure turnover control.
        periods_per_year: Annualisation factor. 252 for equities, 365 crypto.
    """

    def __init__(
        self,
        inner: Strategy,
        *,
        target_vol: float = 0.15,
        lookback: int = 60,
        max_leverage: float = 2.0,
        rebalance_band: float = 0.10,
        periods_per_year: int = 252,
    ) -> None:
        if target_vol <= 0:
            raise ValueError(f"target_vol must be positive, got {target_vol}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {max_leverage}")
        if lookback < MIN_VOL_OBSERVATIONS:
            raise ValueError(
                f"lookback {lookback} is below {MIN_VOL_OBSERVATIONS}; "
                "a volatility estimate that short is noise"
            )

        super().__init__(
            StrategySpec(
                name=f"voltarget({inner.name})",
                universe=inner.spec.universe,
                timeframe=inner.spec.timeframe,
                parameters={
                    **inner.spec.parameters,
                    "target_vol": target_vol,
                    "vol_lookback": lookback,
                    "max_leverage": max_leverage,
                    "rebalance_band": rebalance_band,
                },
                # Need whichever history is longer.
                lookback=max(inner.spec.lookback, lookback + 1),
                max_position=inner.spec.max_position,
                max_gross=Decimal(str(max_leverage)),
            )
        )
        self.inner = inner
        self.target_vol = target_vol
        self.vol_lookback = lookback
        self.max_leverage = max_leverage
        self.rebalance_band = rebalance_band
        self.periods_per_year = periods_per_year
        #: Last applied scale. Purely a turnover optimisation — it never changes
        #: *what* is held, only whether a small resize is worth its cost.
        self._last_scale: Decimal | None = None

    def realised_volatility(self, view: MarketView) -> float:
        """Annualised volatility of the equal-weighted universe.

        Uses the universe rather than the strategy's own past returns because a
        strategy has no access to its own P&L by design (§13) — and because
        market volatility is the thing actually being targeted.
        """
        closes = view.closes()
        if closes.is_empty() or closes.height < self.vol_lookback + 1:
            return 0.0

        columns = [c for c in closes.columns if c != "event_time"]
        if not columns:
            return 0.0

        matrix = closes.select(columns).tail(self.vol_lookback + 1).to_numpy()
        matrix = np.asarray(matrix, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = matrix[1:] / matrix[:-1] - 1.0

        # Equal-weighted portfolio return per bar, ignoring names with gaps.
        portfolio = np.nanmean(np.where(np.isfinite(returns), returns, np.nan), axis=1)
        portfolio = portfolio[np.isfinite(portfolio)]
        if portfolio.size < MIN_VOL_OBSERVATIONS:
            return 0.0
        return float(np.std(portfolio, ddof=1) * np.sqrt(self.periods_per_year))

    def generate(self, view: MarketView) -> TargetWeights:
        raw = self.inner(view)
        if not raw.weights:
            return raw

        scale = vol_scale(self.realised_volatility(view), self.target_vol, self.max_leverage)
        scale = self._apply_band(scale)
        if scale == 0:
            return TargetWeights(view.as_of, {})

        return TargetWeights(view.as_of, {k: w * scale for k, w in raw.weights.items()})

    def _apply_band(self, scale: Decimal) -> Decimal:
        """Hold the previous scale unless it has drifted outside the band."""
        previous = self._last_scale
        if previous is None or previous == 0:
            self._last_scale = scale
            return scale

        drift = abs(scale - previous) / previous
        if drift < Decimal(str(self.rebalance_band)):
            return previous
        self._last_scale = scale
        return scale
