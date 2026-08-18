"""Resampling tests — MASTER_PLAN §5.4 tests 10 and 11.

Distribution-free ways to ask "could this have happened by chance?", which is
the only honest question to ask of a backtest.

**Every function takes an explicit seed** (§14.1.1). An unseeded resampling test
gives a different answer each run, which means it can be re-rolled until it
agrees with you — the exact failure mode the gauntlet exists to prevent.

**Monte Carlo trade shuffling** answers a different question from the bootstrap.
Shuffling the *order* of trades preserves the return distribution exactly while
destroying the sequence, which isolates how much of a drawdown was sequence
luck. A strategy whose 5th-percentile shuffled drawdown is twice its realised
one got lucky in the ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant.math.metrics.performance import max_drawdown, sharpe_ratio

__all__ = [
    "BootstrapResult",
    "ShuffleResult",
    "block_bootstrap",
    "bootstrap_sharpe",
    "monte_carlo_drawdown",
    "permutation_test",
]

FloatArray = npt.NDArray[np.float64]

#: Below this many observations a resampled statistic is noise, not evidence.
MIN_OBS_RESAMPLE = 4
MIN_TRADES_SHUFFLE = 2


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely one side of zero."""
        return self.lower > 0.0 or self.upper < 0.0

    def format(self) -> str:
        return (
            f"  {self.point_estimate:.3f} "
            f"[{self.lower:.3f}, {self.upper:.3f}] at {self.confidence:.0%} "
            f"({self.n_resamples} resamples)"
        )


def bootstrap_sharpe(
    returns: npt.ArrayLike,
    seed: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    periods_per_year: int = 252,
) -> BootstrapResult:
    """Confidence interval for the Sharpe ratio by IID bootstrap.

    Makes no normality assumption, which matters because trading returns are
    reliably non-normal. Note it *does* assume independence — for autocorrelated
    returns use `block_bootstrap` instead.
    """
    rets = np.asarray(returns, dtype=np.float64).ravel()
    rets = rets[np.isfinite(rets)]
    if rets.size < MIN_OBS_RESAMPLE:
        return BootstrapResult(0.0, 0.0, 0.0, confidence, 0)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        draw = rng.choice(rets, size=rets.size, replace=True)
        samples[i] = sharpe_ratio(draw, periods_per_year=periods_per_year)

    tail = (1.0 - confidence) / 2.0
    return BootstrapResult(
        point_estimate=sharpe_ratio(rets, periods_per_year=periods_per_year),
        lower=float(np.quantile(samples, tail)),
        upper=float(np.quantile(samples, 1.0 - tail)),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def block_bootstrap(
    returns: npt.ArrayLike,
    seed: int,
    block_size: int = 20,
    n_resamples: int = 2000,
    periods_per_year: int = 252,
) -> BootstrapResult:
    """Bootstrap preserving short-range autocorrelation.

    Resamples contiguous blocks rather than individual observations, so
    momentum and volatility clustering survive the resampling. The IID
    bootstrap destroys both and will overstate confidence for any trending
    strategy.
    """
    rets = np.asarray(returns, dtype=np.float64).ravel()
    rets = rets[np.isfinite(rets)]
    if rets.size < block_size * 2:
        return BootstrapResult(0.0, 0.0, 0.0, 0.95, 0)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(rets.size / block_size))
    max_start = rets.size - block_size

    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        draw = np.concatenate([rets[s : s + block_size] for s in starts])[: rets.size]
        samples[i] = sharpe_ratio(draw, periods_per_year=periods_per_year)

    return BootstrapResult(
        point_estimate=sharpe_ratio(rets, periods_per_year=periods_per_year),
        lower=float(np.quantile(samples, 0.025)),
        upper=float(np.quantile(samples, 0.975)),
        confidence=0.95,
        n_resamples=n_resamples,
    )


def permutation_test(
    strategy_returns: npt.ArrayLike,
    benchmark_returns: npt.ArrayLike,
    seed: int,
    n_permutations: int = 2000,
    periods_per_year: int = 252,
) -> float:
    """One-sided p-value that the strategy beats the benchmark.

    Pools both series and repeatedly re-splits them at random. If the real
    difference in Sharpe sits comfortably inside that null distribution, the
    strategy has not distinguished itself from the benchmark.
    """
    strategy = np.asarray(strategy_returns, dtype=np.float64).ravel()
    benchmark = np.asarray(benchmark_returns, dtype=np.float64).ravel()
    strategy = strategy[np.isfinite(strategy)]
    benchmark = benchmark[np.isfinite(benchmark)]
    if strategy.size < MIN_OBS_RESAMPLE or benchmark.size < MIN_OBS_RESAMPLE:
        return 1.0

    observed = sharpe_ratio(strategy, periods_per_year=periods_per_year) - sharpe_ratio(
        benchmark, periods_per_year=periods_per_year
    )

    pooled = np.concatenate([strategy, benchmark])
    rng = np.random.default_rng(seed)
    split = strategy.size

    at_least_as_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        difference = sharpe_ratio(
            shuffled[:split], periods_per_year=periods_per_year
        ) - sharpe_ratio(shuffled[split:], periods_per_year=periods_per_year)
        if difference >= observed:
            at_least_as_extreme += 1

    # +1 in both terms: the observed arrangement is itself one valid draw, and
    # omitting it can produce an impossible p-value of exactly zero.
    return (at_least_as_extreme + 1) / (n_permutations + 1)


@dataclass(frozen=True)
class ShuffleResult:
    realised_drawdown: float
    median_drawdown: float
    percentile_5: float
    n_shuffles: int

    @property
    def sequence_luck_ratio(self) -> float:
        """How much worse the unlucky case is than what actually happened.

        Above ~2.0 means the realised drawdown owes a lot to favourable
        ordering, and the risk limits should be set against the shuffled tail
        rather than against history.
        """
        if self.realised_drawdown == 0.0:
            return 0.0
        return self.percentile_5 / self.realised_drawdown

    def format(self) -> str:
        return (
            f"  realised DD {self.realised_drawdown:.2%}, "
            f"median {self.median_drawdown:.2%}, "
            f"5th pct {self.percentile_5:.2%} "
            f"(luck ratio {self.sequence_luck_ratio:.2f})"
        )


def monte_carlo_drawdown(
    trade_returns: npt.ArrayLike,
    seed: int,
    n_shuffles: int = 2000,
) -> ShuffleResult:
    """Drawdown distribution under random trade ordering (§5.4 test 11).

    Preserves the exact set of trade outcomes and destroys only their sequence.
    Whatever drawdown you actually experienced was one draw from this
    distribution; the 5th percentile is a far better basis for a risk limit
    than the single path history happened to take.
    """
    trades = np.asarray(trade_returns, dtype=np.float64).ravel()
    trades = trades[np.isfinite(trades)]
    if trades.size < MIN_TRADES_SHUFFLE:
        return ShuffleResult(0.0, 0.0, 0.0, 0)

    rng = np.random.default_rng(seed)
    drawdowns = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        drawdowns[i] = max_drawdown(rng.permutation(trades))

    return ShuffleResult(
        realised_drawdown=max_drawdown(trades),
        median_drawdown=float(np.median(drawdowns)),
        percentile_5=float(np.quantile(drawdowns, 0.05)),
        n_shuffles=n_shuffles,
    )
