"""Cointegration — MASTER_PLAN §254, strategy family 5 (pairs).

**Correlation is not cointegration, and the difference is the whole strategy.**
Two random walks with a shared drift correlate beautifully and their spread
wanders off forever. Cointegration is the stronger claim: a *linear combination
of them is stationary*, so the spread has a level to revert to. Only the second
supports a pairs trade.

The plan flags "spurious correlation" as the risk of strategy family 5. This
module is what turns that flag into a gate.

**Engle-Granger, deliberately.** Two legs, one hedge ratio, a regression and a
unit-root test on the residual. Johansen handles baskets of three or more and
is the right tool there — but a solo book trading two names does not need a
vector error-correction model, and the plan's own advice is to learn pairs
rather than fund them.

**The hedge ratio is estimated on the same data the test runs on**, which is
in-sample by construction. That is standard for Engle-Granger and it is also
why a cointegration result must still clear the gauntlet's walk-forward check
before anyone believes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant.math.timeseries.stationarity import (
    MIN_OBSERVATIONS,
    SIGNIFICANCE,
    Stationarity,
    assess_stationarity,
)

__all__ = [
    "CointegrationReport",
    "engle_granger",
    "hedge_ratio",
    "spread_series",
]


@dataclass(frozen=True)
class CointegrationReport:
    """Whether two series share a stationary linear combination."""

    cointegrated: bool
    hedge_ratio: float
    intercept: float
    #: Stationarity verdict of the residual spread.
    spread_verdict: Stationarity
    adf_pvalue: float
    half_life_bars: float
    correlation: float
    observations: int

    @property
    def tradable(self) -> bool:
        """Cointegrated *and* reverting fast enough to pay for the round trip.

        A half-life longer than the sample cannot be verified within it, and a
        spread that takes a year to close is a directional position wearing a
        pairs-trade label — its costs will exceed its reversion (§7.1).
        """
        return (
            self.cointegrated
            and np.isfinite(self.half_life_bars)
            and 0 < self.half_life_bars < self.observations / 4
        )

    def format(self) -> str:
        state = "COINTEGRATED" if self.cointegrated else "NOT COINTEGRATED"
        half_life = f"{self.half_life_bars:.1f}" if np.isfinite(self.half_life_bars) else "inf"
        return (
            f"  {state:<18} beta={self.hedge_ratio:+.4f}  "
            f"ADF p={self.adf_pvalue:.4f}  half-life={half_life} bars  "
            f"corr={self.correlation:+.3f}  n={self.observations}"
        )


def _aligned(
    left: npt.ArrayLike, right: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Both series as finite float arrays of equal length.

    Rows where either side is missing are dropped from *both*, so the pair
    stays aligned bar for bar. Dropping independently would silently compare
    different dates — a spread computed across misaligned days is noise that
    looks like a signal.
    """
    a: npt.NDArray[np.float64] = np.asarray(left, dtype=np.float64).ravel()
    b: npt.NDArray[np.float64] = np.asarray(right, dtype=np.float64).ravel()
    size = min(a.size, b.size)
    a, b = a[:size], b[:size]
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def hedge_ratio(dependent: npt.ArrayLike, independent: npt.ArrayLike) -> tuple[float, float]:
    """Ordinary least squares slope and intercept: how many units of
    `independent` hedge one unit of `dependent`.

    Returns:
        (beta, intercept). (0.0, 0.0) when the regression is degenerate, which
        makes the resulting spread the dependent series itself and lets the
        stationarity test reach the obvious conclusion.
    """
    y, x = _aligned(dependent, independent)
    if y.size < MIN_OBSERVATIONS or float(np.std(x)) == 0.0:
        return 0.0, 0.0
    beta, intercept = np.polyfit(x, y, 1)
    return float(beta), float(intercept)


def spread_series(
    dependent: npt.ArrayLike, independent: npt.ArrayLike, beta: float, intercept: float = 0.0
) -> npt.NDArray[np.float64]:
    """The residual that a pairs trade actually trades."""
    y, x = _aligned(dependent, independent)
    return y - (beta * x + intercept)


def engle_granger(
    dependent: npt.ArrayLike,
    independent: npt.ArrayLike,
    significance: float = SIGNIFICANCE,
) -> CointegrationReport:
    """Two-step Engle-Granger test.

    1. Regress one series on the other to get the hedge ratio.
    2. Test the residual spread for stationarity.

    A stationary residual means the pair is cointegrated: the spread has a
    level, and deviations from it are the trade.

    Returns:
        A full report rather than a boolean, so a rejection can be explained.
        Correlation is reported alongside precisely because a high correlation
        with a failed test is the spurious-pair case worth seeing.
    """
    from quant.strategies.mean_reversion import half_life  # noqa: PLC0415 - avoids a cycle

    y, x = _aligned(dependent, independent)
    if y.size < MIN_OBSERVATIONS:
        return CointegrationReport(
            cointegrated=False,
            hedge_ratio=0.0,
            intercept=0.0,
            spread_verdict=Stationarity.INCONCLUSIVE,
            adf_pvalue=1.0,
            half_life_bars=float("inf"),
            correlation=0.0,
            observations=int(y.size),
        )

    beta, intercept = hedge_ratio(y, x)
    spread = spread_series(y, x, beta, intercept)
    report = assess_stationarity(spread, significance)
    correlation = (
        float(np.corrcoef(y, x)[0, 1]) if float(np.std(y)) > 0 and float(np.std(x)) > 0 else 0.0
    )

    return CointegrationReport(
        # The residual must be outright stationary. INCONCLUSIVE is not a pair.
        cointegrated=report.verdict.supports_mean_reversion,
        hedge_ratio=beta,
        intercept=intercept,
        spread_verdict=report.verdict,
        adf_pvalue=report.adf_pvalue,
        half_life_bars=half_life(spread),
        correlation=correlation,
        observations=int(y.size),
    )
