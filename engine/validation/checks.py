"""The twelve gauntlet checks — MASTER_PLAN §5.4.

Each check answers one question about whether a backtest result is evidence or
noise, and returns a `GauntletResult` carrying its statistic, its threshold and
a plain-language reason. Results are persisted to `gauntlet_results` so that
"why was this rejected?" is answerable a year later.

Ordering and short-circuiting live in `gauntlet`; this module is only the
checks themselves.
"""

from __future__ import annotations

import numpy as np

from engine.validation.report import (
    COST_MULTIPLE,
    DSR_THRESHOLD,
    LOOK_AHEAD_TOLERANCE,
    MIN_DROPOUT_SAMPLES,
    MIN_NEIGHBOURHOOD_POINTS,
    MIN_PLACEBO_SAMPLES,
    MIN_TRADES_FOR_SHUFFLE,
    PARAMETER_PLATEAU_RETENTION,
    PBO_THRESHOLD,
    PLACEBO_PERCENTILE,
    REGIME_MIN_POSITIVE,
    UNIVERSE_DROPOUT_PERCENTILE,
    WALK_FORWARD_EFFICIENCY,
    GauntletInputs,
    GauntletResult,
    array_or_none,
    skipped,
)
from quant.math.metrics.overfitting import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from quant.math.metrics.performance import max_drawdown, sharpe_ratio
from quant.math.stats.resampling import monte_carlo_drawdown

__all__ = [
    "check_cost_sensitivity",
    "check_data_integrity",
    "check_deflated_sharpe",
    "check_locked_oos",
    "check_look_ahead",
    "check_monte_carlo_drawdown",
    "check_parameter_plateau",
    "check_pbo",
    "check_placebo",
    "check_regimes",
    "check_universe_dropout",
    "check_walk_forward",
]

# ── the twelve ──────────────────────────────────────────────────────────────


def check_data_integrity(inputs: GauntletInputs) -> GauntletResult:
    count = inputs.critical_data_findings
    return GauntletResult(
        test="1_data_integrity",
        passed=count == 0,
        statistic=float(count),
        threshold=0,
        reason="clean" if count == 0 else f"{count} CRITICAL data-quality finding(s)",
    )


def check_look_ahead(inputs: GauntletInputs) -> GauntletResult:
    """Shuffle-future — the cheapest high-value check in the system.

    Corrupt every observation after each decision point. If results barely
    move, the strategy could not have been reading them. If they move a lot,
    it was.
    """
    baseline = array_or_none(inputs.returns)
    corrupted = array_or_none(inputs.shuffled_future_returns)
    if baseline is None or corrupted is None:
        return skipped("2_look_ahead", "shuffled_future_returns")

    size = min(baseline.size, corrupted.size)
    difference = float(np.max(np.abs(baseline[:size] - corrupted[:size])))
    return GauntletResult(
        test="2_look_ahead",
        passed=difference < LOOK_AHEAD_TOLERANCE,
        statistic=difference,
        threshold=LOOK_AHEAD_TOLERANCE,
        reason=(
            "decisions unchanged by future data"
            if difference < LOOK_AHEAD_TOLERANCE
            else "results move when future data changes — the strategy is reading it"
        ),
    )


def check_deflated_sharpe(inputs: GauntletInputs) -> GauntletResult:
    result = deflated_sharpe_ratio(
        inputs.returns, inputs.n_trials, periods_per_year=inputs.periods_per_year
    )
    return GauntletResult(
        test="3_deflated_sharpe",
        passed=result.dsr > DSR_THRESHOLD,
        statistic=result.dsr,
        threshold=DSR_THRESHOLD,
        reason=(
            f"SR {result.observed_sharpe:.3f} vs expected-max "
            f"{result.expected_max_sharpe:.3f} over {result.n_trials} trials"
        ),
    )


def check_pbo(inputs: GauntletInputs) -> GauntletResult:
    if inputs.sweep_returns is None:
        return skipped("4_pbo", "sweep_returns")
    try:
        result = probability_of_backtest_overfitting(inputs.sweep_returns)
    except ValueError as exc:
        return GauntletResult("4_pbo", passed=False, reason=f"could not compute: {exc}")
    return GauntletResult(
        test="4_pbo",
        passed=result.pbo < PBO_THRESHOLD,
        statistic=result.pbo,
        threshold=PBO_THRESHOLD,
        reason=f"{result.n_combinations} CSCV partitions over {result.n_configurations} configs",
    )


def check_walk_forward(inputs: GauntletInputs) -> GauntletResult:
    in_sample = array_or_none(inputs.in_sample_returns)
    out_sample = array_or_none(inputs.out_of_sample_returns)
    if in_sample is None or out_sample is None:
        return skipped("5_walk_forward", "in/out-of-sample returns")

    is_sharpe = sharpe_ratio(in_sample, periods_per_year=inputs.periods_per_year)
    oos_sharpe = sharpe_ratio(out_sample, periods_per_year=inputs.periods_per_year)
    if is_sharpe <= 0:
        return GauntletResult(
            "5_walk_forward",
            passed=False,
            statistic=0.0,
            reason="in-sample Sharpe is not positive; nothing to degrade from",
        )

    efficiency = oos_sharpe / is_sharpe
    return GauntletResult(
        test="5_walk_forward",
        passed=efficiency > WALK_FORWARD_EFFICIENCY,
        statistic=efficiency,
        threshold=WALK_FORWARD_EFFICIENCY,
        reason=f"IS {is_sharpe:.2f} -> OOS {oos_sharpe:.2f}",
    )


def check_parameter_plateau(inputs: GauntletInputs) -> GauntletResult:
    """A mesa, not a needle.

    A performance spike at one parameter value is a fitted artefact of one
    noise realisation. Neighbours must retain most of the performance.
    """
    neighbourhood = array_or_none(inputs.parameter_neighbourhood)
    if neighbourhood is None or neighbourhood.size < MIN_NEIGHBOURHOOD_POINTS:
        return skipped("6_parameter_plateau", "parameter_neighbourhood")

    best = float(np.max(neighbourhood))
    if best <= 0:
        return GauntletResult(
            "6_parameter_plateau",
            passed=False,
            statistic=best,
            reason="no positive Sharpe anywhere in the neighbourhood",
        )
    retention = float(np.median(neighbourhood)) / best
    return GauntletResult(
        test="6_parameter_plateau",
        passed=retention > PARAMETER_PLATEAU_RETENTION,
        statistic=retention,
        threshold=PARAMETER_PLATEAU_RETENTION,
        reason=(
            "plateau"
            if retention > PARAMETER_PLATEAU_RETENTION
            else "needle — performance collapses away from the chosen parameters"
        ),
    )


def check_cost_sensitivity(inputs: GauntletInputs) -> GauntletResult:
    tripled = array_or_none(inputs.tripled_cost_returns)
    if tripled is None:
        return skipped("7_cost_sensitivity", "tripled_cost_returns")
    sharpe = sharpe_ratio(tripled, periods_per_year=inputs.periods_per_year)
    return GauntletResult(
        test="7_cost_sensitivity",
        passed=sharpe > 0,
        statistic=sharpe,
        threshold=0.0,
        reason=f"Sharpe at {COST_MULTIPLE:g}x modelled costs",
    )


def check_universe_dropout(inputs: GauntletInputs) -> GauntletResult:
    sharpes = array_or_none(inputs.universe_dropout_sharpes)
    if sharpes is None or sharpes.size < MIN_DROPOUT_SAMPLES:
        return skipped("8_universe_dropout", "universe_dropout_sharpes")
    percentile = float(np.quantile(sharpes, UNIVERSE_DROPOUT_PERCENTILE))
    return GauntletResult(
        test="8_universe_dropout",
        passed=percentile > 0,
        statistic=percentile,
        threshold=0.0,
        reason=f"5th-percentile Sharpe across {sharpes.size} random universe subsets",
    )


def check_regimes(inputs: GauntletInputs) -> GauntletResult:
    if not inputs.regime_returns:
        return skipped("9_regimes", "regime_returns")

    sharpes = {
        name: sharpe_ratio(rets, periods_per_year=inputs.periods_per_year)
        for name, rets in inputs.regime_returns.items()
    }
    positive = sum(1 for s in sharpes.values() if s > 0)
    detail = ", ".join(f"{k} {v:.2f}" for k, v in sorted(sharpes.items()))
    return GauntletResult(
        test="9_regimes",
        passed=positive >= REGIME_MIN_POSITIVE,
        statistic=float(positive),
        threshold=REGIME_MIN_POSITIVE,
        reason=detail,
    )


def check_placebo(inputs: GauntletInputs) -> GauntletResult:
    """Random entries with matched holding period and exposure.

    If a coin flip trading the same instruments does about as well, the market
    did the work and the signal contributed nothing.
    """
    placebo = array_or_none(inputs.placebo_sharpes)
    if placebo is None or placebo.size < MIN_PLACEBO_SAMPLES:
        return skipped("10_placebo", "placebo_sharpes")

    actual = sharpe_ratio(inputs.returns, periods_per_year=inputs.periods_per_year)
    percentile = float(np.mean(placebo < actual))
    return GauntletResult(
        test="10_placebo",
        passed=percentile >= PLACEBO_PERCENTILE,
        statistic=percentile,
        threshold=PLACEBO_PERCENTILE,
        reason=f"beats {percentile:.1%} of {placebo.size} random-entry strategies",
    )


def check_monte_carlo_drawdown(inputs: GauntletInputs) -> GauntletResult:
    trades = array_or_none(inputs.trade_returns)
    if trades is None or trades.size < MIN_TRADES_FOR_SHUFFLE:
        return skipped("11_mc_drawdown", "trade_returns")

    result = monte_carlo_drawdown(trades, seed=inputs.seed)
    return GauntletResult(
        test="11_mc_drawdown",
        passed=result.percentile_5 > inputs.max_drawdown_limit,
        statistic=result.percentile_5,
        threshold=inputs.max_drawdown_limit,
        reason=(
            f"realised {result.realised_drawdown:.1%}, 5th-pct shuffled {result.percentile_5:.1%}"
        ),
    )


def check_locked_oos(inputs: GauntletInputs) -> GauntletResult:
    """The locked test set. Touched once per strategy, ever (§5.3).

    A failure here is terminal. The strategy may not be tweaked and re-run —
    that would burn the only untouched evidence remaining.
    """
    locked = array_or_none(inputs.locked_test_returns)
    if locked is None:
        return skipped("12_locked_oos", "locked_test_returns")

    sharpe = sharpe_ratio(locked, periods_per_year=inputs.periods_per_year)
    drawdown = max_drawdown(locked)
    passed = sharpe > inputs.locked_test_min_sharpe
    return GauntletResult(
        test="12_locked_oos",
        passed=passed,
        statistic=sharpe,
        threshold=inputs.locked_test_min_sharpe,
        reason=(
            f"max DD {drawdown:.1%}"
            if passed
            else "FAILED ON LOCKED DATA — this strategy is dead, not adjustable"
        ),
    )
