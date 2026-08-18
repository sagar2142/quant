"""Complete quantitative profile of one security — MASTER_PLAN §2, §5, §253.

**What this is for.** Every statistic the system can compute already existed,
scattered across `quant/math`. Nothing composed them into an answer to the
question a quant actually asks: *what is this instrument, statistically, and
what can I do with it?* This module is that composition.

The profile answers, in order:

    what it did      returns over standard horizons, CAGR
    what it costs    volatility, drawdown, downside deviation
    how it pays      Sharpe, Sortino, Calmar
    how it fails     skew, kurtosis, tail ratio, VaR, CVaR
    what it is       stationary or trending, Hurst, autocorrelation
    what to expect   EWMA and GARCH volatility, forecast versus realised

**Distribution statistics are not decoration.** A 2.0 Sharpe with skew -3 and
kurtosis 15 is a strategy that makes small gains until it does not. The
moments are what separate that from a 2.0 Sharpe on a symmetric distribution,
and reporting the ratio without them is how a return series gets mistaken for
an edge.

**Everything is computed from OHLCV.** No fundamentals, no vendor analytics.
That is a real limit and it is stated rather than hidden: this describes price
behaviour, not a company.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant.math.metrics.performance import (
    TRADING_DAYS,
    cagr,
    calmar_ratio,
    hit_rate,
    kurtosis,
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    volatility,
)
from quant.math.timeseries.stationarity import (
    StationarityReport,
    assess_stationarity,
)
from quant.math.timeseries.volatility import (
    VolatilityForecast,
    ewma_volatility,
    realised_volatility,
)

__all__ = [
    "HORIZONS",
    "SecurityProfile",
    "profile_security",
]

#: Standard lookbacks, in sessions. A quant reads a return series at several
#: horizons at once because a name can be up on the year and broken on the
#: quarter, and one number hides that.
HORIZONS: dict[str, int] = {
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
    "3y": 756,
}

#: Percentile for tail risk. 5% is the convention and it is shallow — CVaR is
#: reported alongside because VaR says nothing about how bad the bad days are.
VAR_PERCENTILE = 5.0

#: Sessions of history below which the profile is noise. About a quarter.
MIN_HISTORY = 60

#: Lags reported for autocorrelation. Lag 1 dominates the mean-reversion
#: question; the others show whether the effect persists or is a one-bar
#: artefact of stale prices.
AUTOCORR_LAGS = (1, 2, 5, 10)

#: EWMA volatility this far from its own long-run realised level counts as a
#: regime, not noise. A name is never simply "volatile" — it is volatile
#: relative to itself.
VOL_ELEVATED = 1.3
VOL_COMPRESSED = 0.7

#: §2.1 smell test. On a single name this usually means a missed corporate
#: action rather than a discovery.
IMPLAUSIBLE_SHARPE = 2.5

#: Negative skew past this, with excess kurtosis past that, is the "small
#: gains until ruin" distribution.
FAT_TAIL_SKEW = -0.5
FAT_TAIL_KURTOSIS = 5.0


@dataclass(frozen=True)
class SecurityProfile:
    """Everything the system knows about one instrument's price behaviour."""

    symbol: str
    observations: int
    first_close: float
    last_close: float

    # what it did
    horizon_returns: dict[str, float | None]
    cagr: float
    high_52w: float
    low_52w: float
    off_high: float

    # what it costs
    annual_volatility: float
    max_drawdown: float
    current_drawdown: float

    # how it pays
    sharpe: float
    sortino: float
    calmar: float
    hit_rate: float

    # how it fails
    skewness: float
    kurtosis: float
    var_5: float
    cvar_5: float
    tail_ratio: float

    # what it is
    stationarity: StationarityReport
    autocorrelation: dict[int, float]

    # what to expect
    realised_vol: VolatilityForecast
    ewma_vol: VolatilityForecast

    # liquidity
    adv_value: float | None

    @property
    def vol_regime(self) -> str:
        """Where current volatility sits against its own history.

        The comparison that matters for sizing: a name is not "volatile", it is
        volatile *relative to itself*, and a position sized on the long-run
        average is wrong in both directions at the extremes.
        """
        if self.realised_vol.annualised <= 0:
            return "unknown"
        ratio = self.ewma_vol.annualised / self.realised_vol.annualised
        if ratio > VOL_ELEVATED:
            return "elevated"
        if ratio < VOL_COMPRESSED:
            return "compressed"
        return "normal"

    @property
    def is_implausible(self) -> bool:
        """Sharpe above the §2.1 smell test on a single name.

        Not impossible — but on price data alone it usually means a corporate
        action was missed, so it is surfaced rather than celebrated.
        """
        return self.sharpe > IMPLAUSIBLE_SHARPE

    @property
    def fat_left_tail(self) -> bool:
        """Negative skew with excess kurtosis: small gains, occasional ruin."""
        return self.skewness < FAT_TAIL_SKEW and self.kurtosis > FAT_TAIL_KURTOSIS


def _clean(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array: npt.NDArray[np.float64] = np.asarray(values, dtype=np.float64).ravel()
    finite: npt.NDArray[np.float64] = array[np.isfinite(array)]
    return finite


def horizon_return(closes: npt.NDArray[np.float64], sessions: int) -> float | None:
    """Simple return over the trailing window, or None when history is short.

    None rather than a partial-window number: a "1y return" computed over four
    months is not a small inaccuracy, it is a different statistic under the
    same label.
    """
    if closes.size <= sessions or closes[-(sessions + 1)] <= 0:
        return None
    return float(closes[-1] / closes[-(sessions + 1)] - 1.0)


def autocorrelation(returns: npt.NDArray[np.float64], lag: int) -> float:
    """Correlation of the series with itself `lag` bars back.

    Negative at lag 1 is the mean-reversion signature; positive is momentum.
    Both are usually small and both are usually gone after costs, which is why
    this is reported rather than traded on directly.
    """
    if returns.size <= lag + 1:
        return 0.0
    a, b = returns[lag:], returns[:-lag]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def value_at_risk(returns: npt.NDArray[np.float64], percentile: float = VAR_PERCENTILE) -> float:
    """Historical VaR: the loss exceeded on `percentile`% of days."""
    if returns.size == 0:
        return 0.0
    return float(np.percentile(returns, percentile))


def conditional_var(returns: npt.NDArray[np.float64], percentile: float = VAR_PERCENTILE) -> float:
    """Expected loss *given* the VaR threshold was breached.

    The number VaR should always be quoted with. VaR says how often; CVaR says
    how bad, and a strategy dies of the second.
    """
    if returns.size == 0:
        return 0.0
    threshold = value_at_risk(returns, percentile)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if tail.size else threshold


def tail_ratio(returns: npt.NDArray[np.float64]) -> float:
    """Right tail over left tail, at the 5th/95th percentiles.

    Above 1.0 the good days outrun the bad ones. Below 1.0 the distribution is
    paying you in pennies and charging in notes.
    """
    if returns.size == 0:
        return 1.0
    left = abs(float(np.percentile(returns, 5)))
    right = abs(float(np.percentile(returns, 95)))
    return right / left if left > 0 else 1.0


def current_drawdown(closes: npt.NDArray[np.float64]) -> float:
    """Distance below the running peak, right now."""
    if closes.size == 0:
        return 0.0
    peak = float(np.maximum.accumulate(closes)[-1])
    return float(closes[-1] / peak - 1.0) if peak > 0 else 0.0


def profile_security(
    symbol: str,
    closes: npt.ArrayLike,
    volumes: npt.ArrayLike | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> SecurityProfile:
    """Compute the full profile from a close series.

    Args:
        symbol: For display only.
        closes: Ascending close prices, raw (not back-adjusted).
        volumes: Optional, for average daily traded value.

    Raises:
        ValueError: when history is too short for the statistics to mean
            anything. Refused rather than returned with zeros, because a
            profile full of zeros reads as a calm instrument.
    """
    price = _clean(closes)
    if price.size < MIN_HISTORY:
        raise ValueError(
            f"{symbol}: {price.size} observations is below the {MIN_HISTORY} "
            "needed for these statistics to mean anything"
        )

    returns = returns_from_equity(price)
    window_52w = price[-min(HORIZONS["1y"], price.size) :]

    adv = None
    if volumes is not None:
        traded = _clean(volumes)
        size = min(traded.size, price.size)
        if size:
            adv = float(np.mean(traded[-size:] * price[-size:]))

    return SecurityProfile(
        symbol=symbol,
        observations=int(price.size),
        first_close=float(price[0]),
        last_close=float(price[-1]),
        horizon_returns={name: horizon_return(price, n) for name, n in HORIZONS.items()},
        cagr=cagr(returns, periods_per_year=periods_per_year),
        high_52w=float(window_52w.max()),
        low_52w=float(window_52w.min()),
        off_high=float(price[-1] / window_52w.max() - 1.0),
        annual_volatility=volatility(returns, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(returns),
        current_drawdown=current_drawdown(price),
        sharpe=sharpe_ratio(returns, periods_per_year=periods_per_year),
        sortino=sortino_ratio(returns, periods_per_year=periods_per_year),
        calmar=calmar_ratio(returns, periods_per_year=periods_per_year),
        hit_rate=hit_rate(returns),
        skewness=skewness(returns),
        kurtosis=kurtosis(returns),
        var_5=value_at_risk(returns),
        cvar_5=conditional_var(returns),
        tail_ratio=tail_ratio(returns),
        stationarity=assess_stationarity(price),
        autocorrelation={lag: autocorrelation(returns, lag) for lag in AUTOCORR_LAGS},
        realised_vol=realised_volatility(returns, periods_per_year=periods_per_year),
        ewma_vol=ewma_volatility(returns, periods_per_year=periods_per_year),
        adv_value=adv,
    )
