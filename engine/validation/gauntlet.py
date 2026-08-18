"""The validation gauntlet runner — MASTER_PLAN §5.4.

Twelve checks, all of which must pass. This is the component the plan calls the
actual product: **your enemy is not the market, it is yourself.** A decent
backtester will generate a hundred beautiful equity curves in a month and
ninety-five of them are noise. What separates a research process from a
backtest toy is the machinery that kills ideas efficiently.

A healthy rejection rate is 90%+. If most ideas pass, the gauntlet is broken,
not the ideas (§5.5).

The checks themselves live in `checks`; this module only orders and runs them.
"""

from __future__ import annotations

from collections.abc import Callable

from engine.validation.checks import (
    check_cost_sensitivity,
    check_data_integrity,
    check_deflated_sharpe,
    check_locked_oos,
    check_look_ahead,
    check_monte_carlo_drawdown,
    check_parameter_plateau,
    check_pbo,
    check_placebo,
    check_regimes,
    check_universe_dropout,
    check_walk_forward,
)
from engine.validation.report import GauntletInputs, GauntletReport, GauntletResult

__all__ = [
    "ALL_CHECKS",
    "GauntletInputs",
    "GauntletReport",
    "GauntletResult",
    "run_gauntlet",
]

#: Cheapest first, so an obviously broken strategy dies before the expensive
#: combinatorial checks run. There is no point evaluating 12,870 CSCV
#: partitions on a strategy that already failed the shuffle-future test.
ALL_CHECKS: tuple[Callable[[GauntletInputs], GauntletResult], ...] = (
    check_data_integrity,
    check_look_ahead,
    check_cost_sensitivity,
    check_walk_forward,
    check_parameter_plateau,
    check_regimes,
    check_deflated_sharpe,
    check_universe_dropout,
    check_placebo,
    check_monte_carlo_drawdown,
    check_pbo,
    check_locked_oos,
)


def run_gauntlet(inputs: GauntletInputs, short_circuit: bool = True) -> GauntletReport:
    """Run all twelve checks.

    Args:
        short_circuit: Stop at the first hard failure. Set False to collect a
            complete diagnostic picture when debugging a strategy you already
            know is dead.
    """
    report = GauntletReport()
    for check in ALL_CHECKS:
        result = check(inputs)
        report.results.append(result)
        if short_circuit and not result.passed and not result.skipped:
            break
    return report
