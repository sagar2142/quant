"""Stationarity and memory — MASTER_PLAN §253, strategy families 4 and 5.

**Why a mean-reversion strategy needs this before it needs anything else.** A
z-score strategy assumes the level it fades toward exists. On a non-stationary
series that assumption is false: the "mean" is wherever the random walk
happened to have been, and fading deviations from it is a machine for buying
things on their way down. The plan flags exactly this as the risk of strategy
family 4, and the flag is worthless without a test attached.

**Two tests, opposite null hypotheses, on purpose.** ADF's null is *has a unit
root* (non-stationary); KPSS's null is *is stationary*. Each is weak alone —
failing to reject a null is not evidence for it — so agreement is what carries
information:

    ADF rejects + KPSS does not reject  ->  stationary, tradable
    ADF does not + KPSS rejects        ->  unit root, do not fade it
    both reject / neither rejects      ->  inconclusive, and saying so is the
                                           honest answer rather than picking
                                           whichever supports the trade

The Hurst exponent is the third opinion and the most intuitive: below 0.5 the
series reverts, at 0.5 it is a random walk, above 0.5 it trends.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

__all__ = [
    "MIN_OBSERVATIONS",
    "Stationarity",
    "StationarityReport",
    "adf_test",
    "assess_stationarity",
    "hurst_exponent",
    "kpss_test",
]

#: Below this, every test here reports noise with confidence. Roughly a
#: trading quarter — enough for the lag structure to mean something.
MIN_OBSERVATIONS = 60

#: Conventional 5% significance. Named rather than inlined so a change is a
#: visible decision (§14.3).
SIGNIFICANCE = 0.05

#: Hurst bands. The 0.5 midpoint is a random walk; these are the margins either
#: side within which the estimate cannot distinguish itself from one.
HURST_REVERTING = 0.45
HURST_TRENDING = 0.55

#: Two points define a line; the Hurst regression needs at least that many
#: window sizes before its slope means anything.
MIN_REGRESSION_POINTS = 2


class Stationarity(str, Enum):
    """What the two tests, read together, actually support."""

    STATIONARY = "STATIONARY"
    UNIT_ROOT = "UNIT_ROOT"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def supports_mean_reversion(self) -> bool:
        """Only an outright STATIONARY verdict does.

        INCONCLUSIVE is deliberately not tradable. The whole point of running
        two tests with opposite nulls is to detect disagreement; treating
        disagreement as permission would discard the information.
        """
        return self is Stationarity.STATIONARY


@dataclass(frozen=True)
class StationarityReport:
    """The full picture, so a rejection can be explained rather than asserted."""

    verdict: Stationarity
    adf_pvalue: float
    kpss_pvalue: float
    hurst: float
    observations: int

    @property
    def tradable_as_mean_reversion(self) -> bool:
        return self.verdict.supports_mean_reversion and self.hurst < HURST_TRENDING

    def format(self) -> str:
        return (
            f"  {self.verdict.value:<14} ADF p={self.adf_pvalue:.4f}  "
            f"KPSS p={self.kpss_pvalue:.4f}  Hurst={self.hurst:.3f}  "
            f"n={self.observations}"
        )


def _clean(series: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values: npt.NDArray[np.float64] = np.asarray(series, dtype=np.float64).ravel()
    finite: npt.NDArray[np.float64] = values[np.isfinite(values)]
    return finite


def adf_test(series: npt.ArrayLike) -> float:
    """Augmented Dickey-Fuller p-value. Null: the series has a unit root.

    A small p-value rejects the unit root and therefore *supports* stationarity.

    Returns:
        The p-value, or 1.0 (the most conservative answer — cannot reject a
        unit root) when the series is too short or degenerate. Never raises:
        a strategy gate must fail closed, and 1.0 means "do not trade this".
    """
    values = _clean(series)
    if values.size < MIN_OBSERVATIONS or float(np.std(values)) == 0.0:
        return 1.0

    from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415 - heavy import

    try:
        pvalue: float = float(adfuller(values, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        return 1.0
    return pvalue


def kpss_test(series: npt.ArrayLike) -> float:
    """KPSS p-value. Null: the series *is* stationary around a level.

    A small p-value rejects stationarity. Note the inversion relative to ADF —
    mixing them up flips every verdict, which is why they are wrapped here
    rather than called directly at the point of use.

    Returns:
        The p-value, or 0.0 (rejects stationarity) when the series is unusable.
        Conservative in the same direction as `adf_test`.
    """
    values = _clean(series)
    if values.size < MIN_OBSERVATIONS or float(np.std(values)) == 0.0:
        return 0.0

    import warnings  # noqa: PLC0415

    from statsmodels.tsa.stattools import kpss  # noqa: PLC0415

    try:
        with warnings.catch_warnings():
            # statsmodels warns when the statistic falls outside its p-value
            # lookup table. That is not an error: it means the result is more
            # extreme than the table covers, and the clipped p-value is still
            # the right decision.
            warnings.simplefilter("ignore")
            return float(kpss(values, regression="c", nlags="auto")[1])
    except (ValueError, OverflowError):
        return 0.0


def hurst_exponent(series: npt.ArrayLike) -> float:
    """Hurst exponent by the variance of lagged differences.

        < 0.5   mean-reverting
        = 0.5   random walk
        > 0.5   trending

    ``Var(X_t - X_{t-k})`` scales as ``k^(2H)``, so a regression of the log
    variance on the log lag has slope ``2H``.

    **This estimator takes a level series** — a price, or a pair's spread — and
    that is deliberate. Classical rescaled-range analysis expects *increments*;
    feeding it levels returns ~1.0 for every random walk, which reads as a
    strong trend and would wave through exactly the series a mean-reversion
    strategy must refuse. The lagged-difference form does the differencing
    itself, so the caller cannot get it wrong.

    Returns:
        The exponent, or 0.5 (a random walk — no edge either way) when the
        series is too short or too degenerate to support the regression.
    """
    values = _clean(series)
    if values.size < MIN_OBSERVATIONS:
        return 0.5

    # Lags up to a tenth of the sample: beyond that too few differences remain
    # for the variance at that lag to be estimated stably.
    max_lag = max(MIN_REGRESSION_POINTS + 1, min(values.size // 10, 40))
    lags = range(2, max_lag)

    log_lags: list[float] = []
    log_variance: list[float] = []
    for lag in lags:
        differences = values[lag:] - values[:-lag]
        variance = float(np.var(differences))
        if variance > 0:
            log_lags.append(float(np.log(lag)))
            log_variance.append(float(np.log(variance)))

    if len(log_lags) < MIN_REGRESSION_POINTS:
        return 0.5
    slope = float(np.polyfit(log_lags, log_variance, 1)[0])
    # Clipped: the estimator is noisy on short samples and an exponent outside
    # [0, 1] is an artefact, not a discovery.
    return float(np.clip(slope / 2.0, 0.0, 1.0))


def assess_stationarity(
    series: npt.ArrayLike, significance: float = SIGNIFICANCE
) -> StationarityReport:
    """Run both tests plus Hurst and reconcile them into one verdict.

    This is the function a strategy should call. It exists so that the
    ADF/KPSS null-hypothesis inversion is resolved in exactly one place.
    """
    values = _clean(series)
    adf_p = adf_test(values)
    kpss_p = kpss_test(values)

    adf_says_stationary = adf_p < significance
    kpss_says_stationary = kpss_p >= significance

    if adf_says_stationary and kpss_says_stationary:
        verdict = Stationarity.STATIONARY
    elif not adf_says_stationary and not kpss_says_stationary:
        verdict = Stationarity.UNIT_ROOT
    else:
        verdict = Stationarity.INCONCLUSIVE

    return StationarityReport(
        verdict=verdict,
        adf_pvalue=adf_p,
        kpss_pvalue=kpss_p,
        hurst=hurst_exponent(values),
        observations=int(values.size),
    )
