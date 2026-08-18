"""Volatility estimation — MASTER_PLAN §256, strategy family 6.

Volatility targeting is the one overlay the plan says to **apply to
everything**, and it is only as good as the volatility number underneath it.

**Three estimators, increasing in cost and responsiveness:**

    realised    equal-weighted standard deviation. Honest, and slow to react —
                a shock stays in the window at full weight until it falls out,
                so the estimate drops on a date determined by the window length
                rather than by the market.
    ewma        exponentially weighted. Reacts immediately, decays smoothly,
                one parameter. RiskMetrics used lambda=0.94 on daily data.
    garch       GARCH(1,1) by MLE. Models mean reversion *in volatility
                itself*, so it forecasts rather than merely describes.

**Default to EWMA.** GARCH is better when its assumptions hold and worse when
they do not, it needs a few hundred observations to fit stably, and the
difference rarely justifies the fragility for a daily-rebalanced book. The plan
budgets vol targeting as a risk control, not as a research project.

**Forecasts are annualised in the same units as everything else** (§ metrics):
252 sessions for equities. A vol number on a different annualisation silently
scales every position it touches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "MAX_VOL_SCALE",
    "MIN_GARCH_OBSERVATIONS",
    "RISKMETRICS_LAMBDA",
    "VolatilityForecast",
    "ewma_volatility",
    "forecast_volatility",
    "garch_volatility",
    "realised_volatility",
]

#: RiskMetrics' daily decay. Roughly a 33-day effective half-life.
RISKMETRICS_LAMBDA = 0.94

#: GARCH needs this many points before its parameters mean anything. Below it
#: the fit is unstable and the forecast is worse than a plain EWMA.
MIN_GARCH_OBSERVATIONS = 250

#: Any estimator needs at least this much to say anything.
MIN_OBSERVATIONS = 20

TRADING_DAYS = 252

#: Leverage justified purely by a low recent volatility reading is how a calm
#: market becomes a large loss. This cap is the line between vol targeting and
#: vol chasing.
MAX_VOL_SCALE = 3.0


@dataclass(frozen=True)
class VolatilityForecast:
    """A volatility estimate and how it was produced.

    The method is carried alongside the number because a GARCH forecast that
    silently fell back to EWMA — which happens whenever the fit fails — must
    not be reported as a GARCH forecast.
    """

    daily: float
    annualised: float
    method: str
    observations: int

    def scale_to(self, target_annual_vol: float) -> float:
        """Position multiplier that maps this forecast onto a vol target.

        Capped at `MAX_VOL_SCALE`.
        """
        if self.annualised <= 0:
            return 0.0
        scale: float = min(target_annual_vol / self.annualised, MAX_VOL_SCALE)
        return scale


def _clean(returns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values: npt.NDArray[np.float64] = np.asarray(returns, dtype=np.float64).ravel()
    finite: npt.NDArray[np.float64] = values[np.isfinite(values)]
    return finite


def realised_volatility(
    returns: npt.ArrayLike, window: int | None = None, periods_per_year: int = TRADING_DAYS
) -> VolatilityForecast:
    """Equal-weighted standard deviation over the trailing window."""
    values = _clean(returns)
    if window is not None:
        values = values[-window:]
    if values.size < MIN_OBSERVATIONS:
        return VolatilityForecast(0.0, 0.0, "insufficient", int(values.size))

    daily = float(np.std(values, ddof=1))
    return VolatilityForecast(
        daily=daily,
        annualised=daily * float(np.sqrt(periods_per_year)),
        method="realised",
        observations=int(values.size),
    )


def ewma_volatility(
    returns: npt.ArrayLike,
    decay: float = RISKMETRICS_LAMBDA,
    periods_per_year: int = TRADING_DAYS,
) -> VolatilityForecast:
    """Exponentially weighted volatility.

    ``sigma_t^2 = decay * sigma_{t-1}^2 + (1 - decay) * r_t^2``

    Iterative rather than a weighted sum, because that is the recursion the
    estimator is defined by and it makes the seeding explicit: the first
    variance is the sample variance of the opening window, so the series does
    not start from zero and spend its first fifty bars climbing out.
    """
    values = _clean(returns)
    if values.size < MIN_OBSERVATIONS:
        return VolatilityForecast(0.0, 0.0, "insufficient", int(values.size))
    if not 0 < decay < 1:
        raise ValueError(f"decay must be in (0, 1), got {decay}")

    variance = float(np.var(values[:MIN_OBSERVATIONS], ddof=1))
    for value in values[MIN_OBSERVATIONS:].tolist():
        variance = decay * variance + (1 - decay) * float(value) ** 2

    daily = float(np.sqrt(max(variance, 0.0)))
    return VolatilityForecast(
        daily=daily,
        annualised=daily * float(np.sqrt(periods_per_year)),
        method="ewma",
        observations=int(values.size),
    )


def garch_volatility(
    returns: npt.ArrayLike, periods_per_year: int = TRADING_DAYS
) -> VolatilityForecast:
    """One-step-ahead GARCH(1,1) forecast.

    **Falls back to EWMA rather than raising**, and says so in `method`. A
    volatility model that fails is a normal event — non-convergence on a short
    or badly behaved sample — and a risk control that disappears when its
    preferred estimator fails is not a risk control.

    Returns are scaled to percent for the fit: `arch` is documented to
    optimise poorly on raw decimal returns, and the scaling is undone before
    the number is reported.
    """
    values = _clean(returns)
    if values.size < MIN_GARCH_OBSERVATIONS:
        return ewma_volatility(values, periods_per_year=periods_per_year)

    import warnings  # noqa: PLC0415

    try:
        from arch import arch_model  # noqa: PLC0415 - heavy optional import
    except ImportError:  # pragma: no cover - arch is a declared dependency
        return ewma_volatility(values, periods_per_year=periods_per_year)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = arch_model(values * 100, vol="GARCH", p=1, q=1, mean="Zero").fit(
                disp="off", show_warning=False
            )
            forecast = fitted.forecast(horizon=1, reindex=False)
            # Via numpy rather than .iloc: pandas types a scalar lookup as a
            # union that includes dates, which float() cannot accept.
            variance = float(np.asarray(forecast.variance, dtype=np.float64)[-1, 0])
            daily = float(np.sqrt(variance)) / 100
    except (ValueError, RuntimeError, np.linalg.LinAlgError, IndexError):
        return ewma_volatility(values, periods_per_year=periods_per_year)

    if not np.isfinite(daily) or daily <= 0:
        return ewma_volatility(values, periods_per_year=periods_per_year)

    return VolatilityForecast(
        daily=daily,
        annualised=daily * float(np.sqrt(periods_per_year)),
        method="garch",
        observations=int(values.size),
    )


def forecast_volatility(
    returns: npt.ArrayLike, method: str = "ewma", periods_per_year: int = TRADING_DAYS
) -> VolatilityForecast:
    """Dispatch to a named estimator.

    Raises:
        ValueError: on an unknown name. Silently defaulting would let a typo
            in a config change every position size in the book.
    """
    estimators: dict[str, Callable[..., VolatilityForecast]] = {
        "realised": realised_volatility,
        "ewma": ewma_volatility,
        "garch": garch_volatility,
    }
    if method not in estimators:
        raise ValueError(f"unknown volatility method {method!r}; use one of {sorted(estimators)}")
    return estimators[method](returns, periods_per_year=periods_per_year)
