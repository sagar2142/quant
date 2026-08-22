"""Performance metrics — MASTER_PLAN Part 4, §35.

**float64 throughout, deliberately.** These are statistics, not money (§14.1.2).
The ledger is exact; the statistics computed over it are not, and pretending
otherwise would be slow for no benefit.

Every function takes a *return series*, not an equity curve, because returns
compose across periods while equity levels do not. `returns_from_equity`
converts.

**On the Sharpe ratio.** It assumes returns are roughly IID and roughly
symmetric, and trading strategies routinely violate both. A strategy selling
options has a wonderful Sharpe right up until it does not. That is why §5.4
requires the Deflated Sharpe Ratio, which corrects for skew, kurtosis and the
number of trials, rather than the raw figure reported here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "TRADING_DAYS",
    "PerformanceStats",
    "cagr",
    "calmar_ratio",
    "max_drawdown",
    "returns_from_equity",
    "sharpe_ratio",
    "sortino_ratio",
    "summarise",
]

#: NSE trades ~250 sessions a year; crypto trades 365. Annualisation is only as
#: honest as this constant, so it is always passed explicitly where it matters.
TRADING_DAYS = 252

#: Minimum observations before each moment is meaningful rather than noise.
MIN_OBS_VARIANCE = 2
MIN_OBS_SKEW = 3
MIN_OBS_KURTOSIS = 4

#: §2.1 smell test. Any daily-frequency strategy above this, after realistic
#: costs, is a bug or a leak until proven otherwise.
IMPLAUSIBLE_SHARPE = 2.5

#: Kurtosis of a normal distribution, the non-excess convention DSR expects.
NORMAL_KURTOSIS = 3.0

FloatArray = npt.NDArray[np.float64]


def _clean(returns: npt.ArrayLike) -> FloatArray:
    """Coerce to a finite float64 array.

    Non-finite values are dropped rather than zero-filled: a NaN return is a
    missing observation, and treating it as a flat day silently understates
    volatility (§14.1.5 in spirit — do not invent data).
    """
    array = np.asarray(returns, dtype=np.float64).ravel()
    return np.asarray(array[np.isfinite(array)], dtype=np.float64)


def returns_from_equity(equity: npt.ArrayLike) -> FloatArray:
    """Simple period returns from an equity curve."""
    levels = np.asarray(equity, dtype=np.float64).ravel()
    if levels.size < MIN_OBS_VARIANCE:
        return np.array([], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = levels[1:] / levels[:-1] - 1.0
    return np.asarray(rets[np.isfinite(rets)], dtype=np.float64)


#: A dispersion below this fraction of the series' own scale is numerical
#: noise, not risk. `np.std` of a constant array is ~1e-19 rather than exactly
#: zero, so an `== 0.0` guard never fires and the ratio divides by the noise.
VARIANCE_FLOOR = 1e-12


def _has_variance(dispersion: float, series: FloatArray) -> bool:
    """Whether a dispersion is real rather than floating-point residue.

    Scaled against the series itself: an absolute floor would call a genuine
    but tiny return stream constant. A constant series of 0.001/day produced a
    Sharpe of 7.3e16 against a docstring promising 0.0 — and every consumer of
    that number treats a large Sharpe as good news.
    """
    if not np.isfinite(dispersion) or dispersion <= 0.0:
        return False
    scale = float(np.max(np.abs(series))) if series.size else 0.0
    return dispersion > VARIANCE_FLOOR * max(scale, 1.0)


def sharpe_ratio(
    returns: npt.ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    annualise: bool = True,
) -> float:
    """Mean excess return over volatility.

    Returns 0.0 for a constant series: a strategy with no variance has no
    measurable risk-adjusted return, and dividing by zero to produce `inf`
    would flatter it enormously.
    """
    rets = _clean(returns)
    if rets.size < MIN_OBS_VARIANCE:
        return 0.0
    excess = rets - risk_free / periods_per_year
    # ddof=1: this is a sample, not the population.
    volatility = float(np.std(excess, ddof=1))
    if not _has_variance(volatility, excess):
        return 0.0
    ratio = float(np.mean(excess)) / volatility
    return ratio * np.sqrt(periods_per_year) if annualise else ratio


def sortino_ratio(
    returns: npt.ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Sharpe, but penalising only downside deviation.

    The right measure when a strategy's upside is deliberately asymmetric —
    trend-following, for example, whose big winning months would inflate the
    denominator of a plain Sharpe and understate its quality.
    """
    rets = _clean(returns)
    if rets.size < MIN_OBS_VARIANCE:
        return 0.0
    excess = rets - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0  # no losing periods in sample: undefined, not infinite
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    if not _has_variance(downside_dev, downside):
        return 0.0
    return float(float(np.mean(excess)) / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(returns: npt.ArrayLike) -> float:
    """Worst peak-to-trough decline, as a negative fraction.

    The number that decides whether a strategy is survivable in practice: a
    2.0 Sharpe with a 60% drawdown gets abandoned at the bottom by every human
    who has ever run one.
    """
    rets = _clean(returns)
    if rets.size == 0:
        return 0.0
    # The curve starts at 1.0 *before* the first return. Without that leading
    # point the first bar is its own peak, so a decline beginning at inception
    # is invisible: [-0.5, +0.5] reported 0.00% against a true -50%. The error
    # only ever runs one way — it understates — and it bites hardest on the
    # short windows the gauntlet is built from, where a fold that opens with a
    # loss looks more survivable than it was.
    equity = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    peak = np.maximum.accumulate(equity)
    drawdowns = np.asarray(equity / peak - 1.0, dtype=np.float64)
    return float(drawdowns.min())


def cagr(returns: npt.ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate.

    Returns -1.0 (total loss) if the equity curve reaches zero or below, rather
    than raising: a strategy that blew up has a defined outcome.
    """
    rets = _clean(returns)
    if rets.size == 0:
        return 0.0
    total = float(np.prod(1.0 + rets))
    if total <= 0.0:
        return -1.0
    years = rets.size / periods_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def calmar_ratio(returns: npt.ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by the absolute max drawdown."""
    drawdown = max_drawdown(returns)
    if drawdown == 0.0:
        return 0.0
    return cagr(returns, periods_per_year) / abs(drawdown)


def volatility(returns: npt.ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns."""
    rets = _clean(returns)
    if rets.size < MIN_OBS_VARIANCE:
        return 0.0
    return float(np.std(rets, ddof=1) * np.sqrt(periods_per_year))


def hit_rate(returns: npt.ArrayLike) -> float:
    """Fraction of periods with a positive return.

    Deliberately not a quality measure. Trend-following wins 35-45% of the time
    and is profitable; a strategy winning 95% of the time is usually selling
    tail risk (§2.1).
    """
    rets = _clean(returns)
    if rets.size == 0:
        return 0.0
    return float(np.mean(rets > 0))


def skewness(returns: npt.ArrayLike) -> float:
    """Third standardised moment. Feeds the Deflated Sharpe Ratio."""
    rets = _clean(returns)
    if rets.size < MIN_OBS_SKEW:
        return 0.0
    centred = rets - np.mean(rets)
    std = float(np.std(rets, ddof=0))
    if std == 0.0:
        return 0.0
    return float(np.mean(centred**3) / std**3)


def kurtosis(returns: npt.ArrayLike, excess: bool = False) -> float:
    """Fourth standardised moment.

    Returns *non-excess* kurtosis by default (3.0 for a normal distribution),
    which is the convention the Deflated Sharpe Ratio formula expects.
    """
    rets = _clean(returns)
    if rets.size < MIN_OBS_KURTOSIS:
        return 0.0 if excess else NORMAL_KURTOSIS
    centred = rets - np.mean(rets)
    std = float(np.std(rets, ddof=0))
    if std == 0.0:
        return 0.0 if excess else NORMAL_KURTOSIS
    value = float(np.mean(centred**4) / std**4)
    return value - NORMAL_KURTOSIS if excess else value


@dataclass(frozen=True)
class PerformanceStats:
    """A full performance summary. Persisted onto `backtest_metrics`."""

    n_periods: int
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    skewness: float
    kurtosis: float

    def format(self) -> str:
        rows = [
            ("periods", f"{self.n_periods}"),
            ("total return", f"{self.total_return:>10.2%}"),
            ("CAGR", f"{self.cagr:>10.2%}"),
            ("volatility", f"{self.volatility:>10.2%}"),
            ("Sharpe", f"{self.sharpe:>10.2f}"),
            ("Sortino", f"{self.sortino:>10.2f}"),
            ("max drawdown", f"{self.max_drawdown:>10.2%}"),
            ("Calmar", f"{self.calmar:>10.2f}"),
            ("hit rate", f"{self.hit_rate:>10.2%}"),
            ("skewness", f"{self.skewness:>10.2f}"),
            ("kurtosis", f"{self.kurtosis:>10.2f}"),
        ]
        return "\n".join(f"  {name:<14}{value}" for name, value in rows)

    @property
    def is_implausible(self) -> bool:
        """Smell test from §2.1.

        Any daily-frequency strategy showing Sharpe > 2.5 after realistic costs
        is a bug or a leak until proven otherwise. Flagging it is not the same
        as rejecting it — but it should never pass unremarked.
        """
        return self.sharpe > IMPLAUSIBLE_SHARPE


def summarise(
    returns: npt.ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceStats:
    """Every metric in one pass."""
    rets = _clean(returns)
    total = float(np.prod(1.0 + rets) - 1.0) if rets.size else 0.0
    return PerformanceStats(
        n_periods=int(rets.size),
        total_return=total,
        cagr=cagr(rets, periods_per_year),
        volatility=volatility(rets, periods_per_year),
        sharpe=sharpe_ratio(rets, risk_free, periods_per_year),
        sortino=sortino_ratio(rets, risk_free, periods_per_year),
        max_drawdown=max_drawdown(rets),
        calmar=calmar_ratio(rets, periods_per_year),
        hit_rate=hit_rate(rets),
        skewness=skewness(rets),
        kurtosis=kurtosis(rets),
    )
