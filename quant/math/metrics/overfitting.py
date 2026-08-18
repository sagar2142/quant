"""Overfitting statistics — MASTER_PLAN §5.2, §5.4.

The two numbers that make the trial counter mean something.

**Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014). If you test 100
variants at p<0.05, about five pass by chance. The Sharpe you finally report is
the *maximum* of 100 draws, not a single draw, and comparing it to a
single-draw threshold is simply the wrong test. DSR asks the right question:

    given that I tried N times, and given these returns' skew and kurtosis,
    what is the probability the true Sharpe exceeds zero?

Two corrections are folded in, and both matter:

1. **Selection.** The benchmark is raised from zero to the expected maximum
   Sharpe under the null that every strategy is worthless.
2. **Non-normality.** Negative skew and fat tails inflate a naive Sharpe.
   Option-selling strategies look superb under a Gaussian assumption and are
   the classic casualty of ignoring this.

**Probability of Backtest Overfitting** (Bailey et al., 2015), via
Combinatorially Symmetric Cross-Validation. Splits the sample into S blocks,
takes every balanced in-sample/out-of-sample partition, and asks how often the
in-sample winner lands in the bottom half out-of-sample. A PBO above 0.5 means
your selection procedure is worse than choosing at random.

Neither statistic can be computed without an honest trial count, which is why
`hypotheses.n_trials` is maintained by a database trigger rather than by
anyone's memory (§5.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

from quant.math.metrics.performance import kurtosis, sharpe_ratio, skewness

__all__ = [
    "EULER_MASCHERONI",
    "DsrResult",
    "PboResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probability_of_backtest_overfitting",
]

EULER_MASCHERONI = 0.5772156649015329

#: §5.4 thresholds.
DSR_PASS = 0.95
PBO_FAIL = 0.5

#: Minimum observations before DSR is meaningful.
MIN_OBS_DSR = 4
#: A choice needs at least two options, and a split needs at least two blocks.
MIN_CONFIGURATIONS = 2
MIN_BLOCKS_PER_SPLIT = 2
#: An out-of-sample rank at or below this is the bottom half.
MEDIAN_RANK = 0.5
#: The sweep matrix is (observations, configurations).
MATRIX_DIMENSIONS = 2

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DsrResult:
    observed_sharpe: float
    expected_max_sharpe: float
    dsr: float
    n_trials: int
    n_observations: int
    skew: float
    kurt: float

    @property
    def passes(self) -> bool:
        return self.dsr > DSR_PASS

    def format(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"  DSR {self.dsr:.4f} [{verdict}]  observed SR {self.observed_sharpe:.3f} "
            f"vs expected-max {self.expected_max_sharpe:.3f} "
            f"(N={self.n_trials}, T={self.n_observations}, "
            f"skew={self.skew:.2f}, kurt={self.kurt:.2f})"
        )


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across `n_trials` worthless strategies.

    The benchmark a real result must clear. Derived from the expected maximum
    of N independent standard normals:

        E[max] ~ sqrt(V) * [ (1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) ]

    with g the Euler-Mascheroni constant. It grows with N, which is exactly the
    point: the more you search, the better a worthless strategy looks.

    Args:
        n_trials: Independent configurations tested.
        sharpe_variance: Variance of the Sharpe estimates across those trials.
    """
    if n_trials <= 1:
        return 0.0
    if sharpe_variance <= 0:
        return 0.0

    n = float(n_trials)
    first = scipy_stats.norm.ppf(1.0 - 1.0 / n)
    second = scipy_stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(
        np.sqrt(sharpe_variance) * ((1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second)
    )


def deflated_sharpe_ratio(
    returns: npt.ArrayLike,
    n_trials: int,
    sharpe_variance: float | None = None,
    periods_per_year: int = 252,
) -> DsrResult:
    """Probability that the true Sharpe is positive, given the search effort.

    Args:
        returns: Per-period returns of the selected strategy.
        n_trials: How many configurations were tried to find it. Passing 1 here
            when the truth is 200 is the single easiest way to fool yourself,
            and is why the count is kept in the database.
        sharpe_variance: Variance of Sharpe estimates across the trials. When
            unknown, the estimator's own sampling variance (1/T) is used, which
            is conservative in the wrong direction — supply the real value when
            a parameter sweep makes it available.

    Returns:
        A `DsrResult` whose `dsr` is a probability in [0, 1].
    """
    rets = np.asarray(returns, dtype=np.float64).ravel()
    rets = rets[np.isfinite(rets)]
    observations = rets.size

    if observations < MIN_OBS_DSR:
        return DsrResult(0.0, 0.0, 0.0, n_trials, observations, 0.0, 3.0)

    # Non-annualised: the formula is defined per observation.
    observed = sharpe_ratio(rets, periods_per_year=periods_per_year, annualise=False)
    skew = skewness(rets)
    kurt = kurtosis(rets)

    variance = sharpe_variance if sharpe_variance is not None else 1.0 / observations
    benchmark = expected_max_sharpe(n_trials, variance)

    # Denominator: the standard error of the Sharpe estimator, corrected for
    # non-normality. Negative skew and excess kurtosis both inflate it, which
    # is how the correction penalises option-selling return profiles.
    denominator = np.sqrt(max(1e-12, 1.0 - skew * observed + ((kurt - 1.0) / 4.0) * observed**2))
    statistic = (observed - benchmark) * np.sqrt(observations - 1) / denominator
    dsr = float(scipy_stats.norm.cdf(statistic))

    return DsrResult(
        observed_sharpe=observed,
        expected_max_sharpe=benchmark,
        dsr=dsr,
        n_trials=n_trials,
        n_observations=observations,
        skew=skew,
        kurt=kurt,
    )


@dataclass(frozen=True)
class PboResult:
    pbo: float
    n_splits: int
    n_configurations: int
    n_combinations: int

    @property
    def passes(self) -> bool:
        return self.pbo < PBO_FAIL

    def format(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"  PBO {self.pbo:.3f} [{verdict}]  "
            f"{self.n_configurations} configs, {self.n_combinations} CSCV splits"
        )


def probability_of_backtest_overfitting(
    returns_matrix: npt.ArrayLike,
    n_splits: int = 10,
) -> PboResult:
    """PBO via Combinatorially Symmetric Cross-Validation.

    Args:
        returns_matrix: Shape (T, C) — per-period returns for C candidate
            configurations of the *same* strategy family. Give it the whole
            parameter sweep, not just the winner; the sweep is the thing being
            evaluated.
        n_splits: Number of blocks, S. Must be even. C(S, S/2) partitions are
            evaluated, so 10 gives 252 and 16 gives 12,870.

    Returns:
        `PboResult` whose `pbo` is the fraction of partitions where the
        in-sample winner ranked in the bottom half out-of-sample.

    Raises:
        ValueError: if `n_splits` is odd or the matrix is too small to split.
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != MATRIX_DIMENSIONS:
        raise ValueError(f"expected a 2-D (T, C) matrix, got shape {matrix.shape}")
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")

    observations, configurations = matrix.shape
    if configurations < MIN_CONFIGURATIONS:
        raise ValueError("PBO needs at least two configurations to choose between")
    if observations < n_splits * MIN_BLOCKS_PER_SPLIT:
        raise ValueError(
            f"need at least {n_splits * MIN_BLOCKS_PER_SPLIT} observations for {n_splits} splits, got {observations}"
        )

    block_size = observations // n_splits
    blocks = [matrix[i * block_size : (i + 1) * block_size] for i in range(n_splits)]

    logits_below_median = 0
    total = 0
    half = n_splits // 2

    for in_sample_idx in combinations(range(n_splits), half):
        out_idx = [i for i in range(n_splits) if i not in in_sample_idx]

        in_sample = np.vstack([blocks[i] for i in in_sample_idx])
        out_sample = np.vstack([blocks[i] for i in out_idx])

        in_scores = np.array(
            [sharpe_ratio(in_sample[:, c], annualise=False) for c in range(configurations)]
        )
        out_scores = np.array(
            [sharpe_ratio(out_sample[:, c], annualise=False) for c in range(configurations)]
        )

        best = int(np.argmax(in_scores))
        # Rank of the in-sample winner among out-of-sample results.
        rank = float(np.mean(out_scores <= out_scores[best]))
        if rank <= MEDIAN_RANK:
            logits_below_median += 1
        total += 1

    return PboResult(
        pbo=logits_below_median / total if total else 0.0,
        n_splits=n_splits,
        n_configurations=configurations,
        n_combinations=total,
    )
