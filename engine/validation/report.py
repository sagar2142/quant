"""Gauntlet inputs and results — MASTER_PLAN §5.4.

Twelve tests, all of which must pass. This is the component the plan calls the
actual product: **your enemy is not the market, it is yourself.** A decent
backtester will generate a hundred beautiful equity curves in a month and
ninety-five of them are noise. What separates a research process from a
backtest toy is the machinery that kills ideas efficiently.

A healthy rejection rate is 90%+. If most ideas pass, the gauntlet is broken,
not the ideas (§5.5).

Each test returns a `GauntletResult` carrying its statistic, its threshold and
a plain-language reason. Results are persisted to `gauntlet_results` so that
"why was this rejected?" is answerable a year later.

**Tests are ordered cheapest-first and short-circuit by default.** There is no
point running 12,870 CSCV partitions on a strategy that already failed the
shuffle-future check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["GauntletInputs", "GauntletReport", "GauntletResult", "array_or_none", "skipped"]

#: §5.4 thresholds, in one place so a change is visible and reviewable.
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.5
WALK_FORWARD_EFFICIENCY = 0.5
PARAMETER_PLATEAU_RETENTION = 0.6
COST_MULTIPLE = 3.0
UNIVERSE_DROPOUT_PERCENTILE = 0.05
PLACEBO_PERCENTILE = 0.95
REGIME_MIN_POSITIVE = 2

#: Minimum sample sizes before a check's statistic is evidence, not noise.
MIN_NEIGHBOURHOOD_POINTS = 3
MIN_DROPOUT_SAMPLES = 10
MIN_PLACEBO_SAMPLES = 20
MIN_TRADES_FOR_SHUFFLE = 10

#: Float equality tolerance for the shuffle-future comparison. Anything above
#: this is a genuine dependence on future data, not rounding.
LOOK_AHEAD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GauntletResult:
    test: str
    passed: bool
    statistic: float | None = None
    threshold: float | None = None
    reason: str = ""
    skipped: bool = False

    def format(self) -> str:
        if self.skipped:
            return f"  [SKIP] {self.test:<24} {self.reason}"
        mark = "PASS" if self.passed else "FAIL"
        stat = "" if self.statistic is None else f"{self.statistic:>10.4f}"
        thresh = "" if self.threshold is None else f" (limit {self.threshold:g})"
        return f"  [{mark}] {self.test:<24}{stat}{thresh}  {self.reason}"


@dataclass
class GauntletReport:
    results: list[GauntletResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """All twelve must pass. Not most, not the important ones — all."""
        return all(r.passed for r in self.results if not r.skipped)

    @property
    def failures(self) -> list[GauntletResult]:
        return [r for r in self.results if not r.passed and not r.skipped]

    @property
    def first_failure(self) -> str | None:
        return self.failures[0].test if self.failures else None

    def format(self) -> str:
        verdict = "PASSED" if self.passed else f"REJECTED at {self.first_failure}"
        lines = [f"gauntlet: {verdict}"]
        lines.extend(r.format() for r in self.results)
        return "\n".join(lines)


@dataclass
class GauntletInputs:
    """Everything the twelve tests need.

    Optional fields cause their test to be *skipped and reported*, never
    silently passed: an untested claim is not a satisfied one.
    """

    #: Per-period returns of the candidate strategy.
    returns: npt.ArrayLike
    #: Trials that produced this candidate. Feeds DSR (§5.2).
    n_trials: int
    seed: int

    #: Data-quality findings from the ingest that produced these returns.
    critical_data_findings: int = 0

    #: Same backtest re-run with all post-decision data corrupted (test 2).
    shuffled_future_returns: npt.ArrayLike | None = None

    #: (T, C) matrix of the whole parameter sweep (test 4).
    sweep_returns: npt.ArrayLike | None = None

    #: In-sample and out-of-sample returns from walk-forward (test 5).
    in_sample_returns: npt.ArrayLike | None = None
    out_of_sample_returns: npt.ArrayLike | None = None

    #: Sharpe at each point of a parameter neighbourhood (test 6).
    parameter_neighbourhood: npt.ArrayLike | None = None

    #: Returns re-run at COST_MULTIPLE times modelled costs (test 7).
    tripled_cost_returns: npt.ArrayLike | None = None

    #: Sharpe from each of many random universe subsets (test 8).
    universe_dropout_sharpes: npt.ArrayLike | None = None

    #: Returns sliced by market regime (test 9).
    regime_returns: dict[str, npt.ArrayLike] | None = None

    #: Sharpes from random-entry strategies with matched exposure (test 10).
    placebo_sharpes: npt.ArrayLike | None = None

    #: Per-trade returns, for the ordering shuffle (test 11).
    trade_returns: npt.ArrayLike | None = None
    max_drawdown_limit: float = -0.30

    #: Result on the locked test period, touched once ever (test 12).
    locked_test_returns: npt.ArrayLike | None = None
    locked_test_min_sharpe: float = 0.0

    periods_per_year: int = 252


def array_or_none(values: npt.ArrayLike | None) -> npt.NDArray[np.float64] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    return array if array.size else None


def skipped(test: str, missing: str) -> GauntletResult:
    return GauntletResult(
        test=test, passed=False, skipped=True, reason=f"not run: {missing} not supplied"
    )
