"""Mean-reversion strategy family — MASTER_PLAN §6 rows 4 and 5.

The plan is candid that these are *learning* strategies, not funding
candidates: single-asset z-score reversion and classical pairs trading were
both heavily arbitraged years ago, and what edge remains is thin and
cost-sensitive. They earn their place by teaching the failure modes that
matter — non-stationarity, spurious correlation, regime break — on instruments
where you can see them clearly.

**Why mean reversion is dangerous in a way trend-following is not.** A trend
strategy's losing trades close themselves: the trend reverses and the stop
triggers. A mean-reversion strategy adds to a losing position by construction,
because a bigger deviation is a stronger signal. When the relationship breaks
rather than reverting, the position is largest at exactly the wrong moment.
Hence `max_zscore`: beyond some deviation, assume the relationship is broken
rather than merely stretched.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import numpy.typing as npt

from core.instruments import InstrumentId
from quant.strategies.base import MarketView, Strategy, StrategySpec, TargetWeights

__all__ = ["PairsTrading", "ZScoreReversion", "half_life"]

#: Minimum observations before a z-score means anything.
MIN_ZSCORE_OBSERVATIONS = 20


def half_life(spread: npt.NDArray[np.float64]) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life, in bars.

    Fits ``d(spread) = lambda * spread + c`` and reports ``-ln(2)/lambda``: how
    long a deviation takes to decay by half.

    This is the number that decides whether a pair is tradable at all. A
    half-life of 3 days is a strategy; 200 days is a buy-and-hold position with
    extra steps, and its costs will exceed its reversion. Returns `inf` when
    the series is not mean-reverting.
    """
    series = np.asarray(spread, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]
    if series.size < MIN_ZSCORE_OBSERVATIONS:
        return float("inf")

    lagged = series[:-1]
    delta = np.diff(series)
    centred = lagged - lagged.mean()
    denominator = float(np.sum(centred**2))
    if denominator == 0:
        return float("inf")

    lam = float(np.sum(centred * (delta - delta.mean())) / denominator)
    # lambda >= 0 means deviations grow rather than decay: not mean-reverting.
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


class ZScoreReversion(Strategy):
    """Fade deviations of a single asset from its own rolling mean.

    Enters when |z| exceeds `entry_z`, exits inside `exit_z`, and abandons the
    position entirely beyond `max_zscore` on the assumption that the level has
    genuinely moved rather than temporarily deviated.
    """

    def __init__(
        self,
        *,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_zscore: float = 4.0,
        gross: Decimal = Decimal(1),
    ) -> None:
        if exit_z >= entry_z:
            raise ValueError(f"exit_z ({exit_z}) must be inside entry_z ({entry_z})")
        if max_zscore <= entry_z:
            raise ValueError(f"max_zscore ({max_zscore}) must exceed entry_z ({entry_z})")
        if lookback < MIN_ZSCORE_OBSERVATIONS:
            raise ValueError(f"lookback {lookback} is too short for a z-score")

        super().__init__(
            StrategySpec(
                name="zscore_reversion",
                universe="fixed",
                timeframe="1d",
                parameters={
                    "lookback": lookback,
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "max_zscore": max_zscore,
                },
                lookback=lookback + 1,
                max_position=Decimal("0.25"),
                max_gross=gross,
            )
        )
        self.window = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_zscore = max_zscore
        self.gross = gross

    def zscore(self, view: MarketView, instrument_id: InstrumentId) -> float | None:
        series = view.series(instrument_id)
        if series.height < self.window:
            return None
        closes = np.asarray(
            [float(c) for c in series["close"].tail(self.window).to_list()],
            dtype=np.float64,
        )
        std = float(np.std(closes, ddof=1))
        if std == 0:
            return None
        return float((closes[-1] - closes.mean()) / std)

    def generate(self, view: MarketView) -> TargetWeights:
        signals: dict[InstrumentId, Decimal] = {}
        for name in view.universe:
            z = self.zscore(view, name)
            if z is None:
                continue
            if abs(z) > self.max_zscore:
                # Too far to be a deviation. Assume the level moved and stay out.
                continue
            if abs(z) >= self.entry_z:
                # Fade: short when high, long when low.
                signals[name] = Decimal(-1) if z > 0 else Decimal(1)

        if not signals:
            return TargetWeights(view.as_of, {})
        weight = self.gross / Decimal(len(signals))
        return TargetWeights(view.as_of, {k: v * weight for k, v in signals.items()})


class PairsTrading(Strategy):
    """Trade the spread between two instruments.

    Uses a rolling OLS hedge ratio rather than a fixed one, because the
    relationship between two names drifts and a stale ratio silently turns a
    market-neutral position into a directional bet.

    **Correlation is not cointegration.** Two assets that both trend upward are
    correlated and will look like a wonderful pair right up until they diverge
    permanently. The half-life filter below is the cheap defence: a spread that
    does not decay is not a spread, it is a drift.
    """

    def __init__(
        self,
        left: InstrumentId,
        right: InstrumentId,
        *,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_half_life: float = 30.0,
        gross: Decimal = Decimal(1),
    ) -> None:
        if exit_z >= entry_z:
            raise ValueError(f"exit_z ({exit_z}) must be inside entry_z ({entry_z})")

        super().__init__(
            StrategySpec(
                name="pairs_trading",
                universe="pair",
                timeframe="1d",
                parameters={
                    "left": str(left),
                    "right": str(right),
                    "lookback": lookback,
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "max_half_life": max_half_life,
                },
                lookback=lookback + 1,
                max_position=Decimal("0.5"),
                max_gross=gross,
            )
        )
        self.left = left
        self.right = right
        self.window = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_half_life = max_half_life
        self.gross = gross

    def _prices(
        self, view: MarketView
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        left = view.series(self.left)
        right = view.series(self.right)
        if left.height < self.window or right.height < self.window:
            return None
        size = min(left.height, right.height, self.window)
        a = np.asarray([float(c) for c in left["close"].tail(size).to_list()], dtype=np.float64)
        b = np.asarray([float(c) for c in right["close"].tail(size).to_list()], dtype=np.float64)
        if a.size != b.size or a.size < MIN_ZSCORE_OBSERVATIONS:
            return None
        return a, b

    def hedge_ratio(self, left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
        """Rolling OLS slope of left on right."""
        variance = float(np.var(right, ddof=1))
        if variance == 0:
            return 0.0
        return float(np.cov(left, right, ddof=1)[0, 1] / variance)

    def generate(self, view: MarketView) -> TargetWeights:
        prices = self._prices(view)
        if prices is None:
            return TargetWeights(view.as_of, {})
        left, right = prices

        beta = self.hedge_ratio(left, right)
        if beta == 0:
            return TargetWeights(view.as_of, {})

        spread = left - beta * right
        # A spread that does not decay is a drift, not a tradable relationship.
        if half_life(spread) > self.max_half_life:
            return TargetWeights(view.as_of, {})

        std = float(np.std(spread, ddof=1))
        if std == 0:
            return TargetWeights(view.as_of, {})
        z = float((spread[-1] - spread.mean()) / std)

        if abs(z) < self.entry_z:
            return TargetWeights(view.as_of, {})

        # Spread high: short the rich leg, long the cheap one.
        direction = Decimal(-1) if z > 0 else Decimal(1)
        leg = self.gross / 2
        hedge = Decimal(str(round(abs(beta), 4)))
        # Normalise so gross exposure stays at the requested level regardless
        # of how large the hedge ratio happens to be.
        scale = leg / (Decimal(1) + hedge) if hedge > 0 else leg

        return TargetWeights(
            view.as_of,
            {
                self.left: direction * scale,
                self.right: -direction * scale * hedge,
            },
        )
