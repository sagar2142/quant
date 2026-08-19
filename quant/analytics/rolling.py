"""Rolling statistics — MASTER_PLAN §2.1, §5.4.

**Every other number in this system is full-sample, and that hides everything
that matters.** A 0.53 Sharpe computed over seven years is indistinguishable
from +2.0 for three years followed by -1.0 for four. The first is a strategy
that stopped working; the second is steady mediocrity; the full-sample figure
reports them identically.

Rolling windows are how a single number becomes a claim you can check. They are
also the cheapest possible version of the walk-forward check the gauntlet runs
(§5.4 test 5) — not a substitute for it, but enough to kill an idea before it
costs forty-eight backtests.

**Trailing windows only.** Each value labels the bar it ends on and uses
nothing after it. Nothing here feeds a trading decision, but a look-ahead habit
in analysis code eventually becomes one in decision code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "DEFAULT_WINDOW",
    "RollingSeries",
    "rolling_beta",
    "rolling_sharpe",
    "rolling_stats",
    "rolling_volatility",
]

#: Six months. Long enough for a Sharpe to mean something, short enough to
#: show a regime change while it is happening rather than a year later.
DEFAULT_WINDOW = 126

TRADING_DAYS = 252

#: Below this a rolling statistic is noise dressed as a time series.
MIN_WINDOW = 20


@dataclass(frozen=True)
class RollingSeries:
    """One rolling statistic, aligned to the input series.

    Leading positions are NaN until the window fills — never a partial-window
    value, because a 20-bar "six-month Sharpe" is a different statistic wearing
    the same label.
    """

    name: str
    window: int
    values: npt.NDArray[np.float64]

    @property
    def last(self) -> float:
        finite = self.values[np.isfinite(self.values)]
        return float(finite[-1]) if finite.size else float("nan")

    @property
    def worst(self) -> float:
        finite = self.values[np.isfinite(self.values)]
        return float(finite.min()) if finite.size else float("nan")

    @property
    def best(self) -> float:
        finite = self.values[np.isfinite(self.values)]
        return float(finite.max()) if finite.size else float("nan")

    @property
    def fraction_positive(self) -> float:
        """How often the statistic was above zero.

        For a rolling Sharpe this is the honest answer to "did this work most
        of the time", which a full-sample average cannot give.
        """
        finite = self.values[np.isfinite(self.values)]
        return float(np.mean(finite > 0)) if finite.size else 0.0

    def format(self) -> str:
        return (
            f"  {self.name:<22} last {self.last:>8.2f}   "
            f"worst {self.worst:>8.2f}   best {self.best:>8.2f}   "
            f"positive {self.fraction_positive:>6.1%}"
        )


def _windows(values: npt.NDArray[np.float64], window: int) -> tuple[npt.NDArray[np.float64], int]:
    """Sliding windows, plus the count of leading NaNs needed to realign."""
    if window < MIN_WINDOW:
        raise ValueError(f"window {window} is below the {MIN_WINDOW} minimum")
    if values.size < window:
        return np.empty((0, window)), values.size
    return sliding_window_view(values, window), window - 1


def _aligned(reduced: npt.NDArray[np.float64], size: int, pad: int) -> npt.NDArray[np.float64]:
    out = np.full(size, np.nan, dtype=np.float64)
    if reduced.size:
        out[pad:] = reduced
    return out


def rolling_volatility(
    returns: npt.ArrayLike, window: int = DEFAULT_WINDOW, periods_per_year: int = TRADING_DAYS
) -> RollingSeries:
    """Annualised volatility over a trailing window."""
    values = np.asarray(returns, dtype=np.float64).ravel()
    frames, pad = _windows(values, window)
    reduced = frames.std(axis=1, ddof=1) * np.sqrt(periods_per_year) if frames.size else np.empty(0)
    return RollingSeries(f"volatility({window})", window, _aligned(reduced, values.size, pad))


def rolling_sharpe(
    returns: npt.ArrayLike, window: int = DEFAULT_WINDOW, periods_per_year: int = TRADING_DAYS
) -> RollingSeries:
    """Annualised Sharpe over a trailing window.

    The most useful of these by a distance: it is the number that reveals a
    strategy whose edge decayed, which a full-sample Sharpe averages away.
    """
    values = np.asarray(returns, dtype=np.float64).ravel()
    frames, pad = _windows(values, window)
    if not frames.size:
        return RollingSeries(f"sharpe({window})", window, _aligned(np.empty(0), values.size, pad))

    means = frames.mean(axis=1)
    deviations = frames.std(axis=1, ddof=1)
    # Zero-variance windows would divide by zero. A flat window has no Sharpe
    # rather than an infinite one.
    with np.errstate(divide="ignore", invalid="ignore"):
        reduced = np.where(deviations > 0, means / deviations * np.sqrt(periods_per_year), np.nan)
    return RollingSeries(f"sharpe({window})", window, _aligned(reduced, values.size, pad))


def rolling_beta(
    returns: npt.ArrayLike, market: npt.ArrayLike, window: int = DEFAULT_WINDOW
) -> RollingSeries:
    """Beta to a market proxy over a trailing window.

    A drifting beta is the quiet way a market-neutral book stops being neutral,
    and a single full-sample beta cannot show it.
    """
    a: npt.NDArray[np.float64] = np.asarray(returns, dtype=np.float64).ravel()
    b: npt.NDArray[np.float64] = np.asarray(market, dtype=np.float64).ravel()
    size = min(a.size, b.size)
    # Tail-aligned: both series end on the same bar, so trimming from the
    # front is what pairs them. Trimming the end would pair each return
    # with a different session's market move.
    a = a[a.size - size :]
    b = b[b.size - size :]

    frames_a, pad = _windows(a, window)
    frames_b, _ = _windows(b, window)
    if not frames_a.size or not frames_b.size:
        return RollingSeries(f"beta({window})", window, _aligned(np.empty(0), size, pad))

    centred_a = frames_a - frames_a.mean(axis=1, keepdims=True)
    centred_b = frames_b - frames_b.mean(axis=1, keepdims=True)
    covariance = (centred_a * centred_b).sum(axis=1)
    variance = (centred_b**2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        reduced = np.where(variance > 0, covariance / variance, np.nan)
    return RollingSeries(f"beta({window})", window, _aligned(reduced, size, pad))


def rolling_stats(
    returns: npt.ArrayLike,
    market: npt.ArrayLike | None = None,
    window: int = DEFAULT_WINDOW,
    periods_per_year: int = TRADING_DAYS,
) -> list[RollingSeries]:
    """Sharpe, volatility and — when a market proxy is supplied — beta."""
    series = [
        rolling_sharpe(returns, window, periods_per_year),
        rolling_volatility(returns, window, periods_per_year),
    ]
    if market is not None:
        series.append(rolling_beta(returns, market, window))
    return series
