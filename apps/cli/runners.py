"""How to run the strategy under test — MASTER_PLAN §5.4.

The gauntlet's twelve checks are one backtest perturbed twelve ways, so the
only thing that differs between validating a momentum strategy and validating a
factor signal is *how to run it once*. That difference lives here, and the
checks themselves stay identical for both — a dropout test that behaved
differently per strategy would be measuring the harness rather than the
strategy.

Split from `validate` because assembling the gauntlet's inputs and deciding
what a strategy *is* are two subjects, and the module was carrying both.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import numpy.typing as npt

from apps.cli.runs import (
    SWEEP_LOOKBACK,
    SWEEP_SKIP,
    SWEEP_TOP_FRACTION,
    Panel,
    SweepTooShortError,
    dropout_runner,
    factor_dropout_runner,
    factor_placebo_runner,
    factor_scores,
    placebo_runner,
    run_factor,
    run_one,
)
from engine.validation.generators import SeededRunner, UniverseRunner
from quant.research.factors import Factor

__all__ = ["Runners", "build_runners", "sweep_configurations"]


def sweep_configurations() -> list[tuple[int, int]]:
    """Every (lookback, skip) the neighbourhood is measured over."""
    return [
        (lookback, skip) for lookback in SWEEP_LOOKBACK for skip in SWEEP_SKIP if skip < lookback
    ]


@dataclass(frozen=True)
class Runners:
    """How to run the strategy under test, in every form the checks need.

    The gauntlet's twelve checks are the same backtest perturbed twelve ways,
    so what varies between a momentum run and a factor run is only *how to run
    it once*. Grouping that here keeps the checks identical for both — a
    dropout test that differed between strategies would be measuring the
    harness rather than the strategy.
    """

    label: str
    #: One run over a panel, at a cost multiple.
    once: Callable[[Panel, Decimal], npt.NDArray[np.float64]]
    #: One run per swept configuration, with its label.
    sweep: Callable[[Panel], list[tuple[str, npt.NDArray[np.float64]]]]
    dropout: Callable[[Panel], UniverseRunner]
    placebo: Callable[[Panel], SeededRunner]


def momentum_runners(args: argparse.Namespace) -> Runners:
    return Runners(
        label=f"momentum({args.lookback}/{args.skip})",
        once=lambda panel, cost: run_one(panel, args.lookback, args.skip, cost),
        sweep=lambda panel: [
            (f"{lookback}/{skip}", run_one(panel, lookback, skip))
            for lookback, skip in sweep_configurations()
        ],
        dropout=lambda panel: dropout_runner(panel, args.lookback, args.skip),
        placebo=lambda panel: placebo_runner(panel, args.lookback),
    )


def factor_runners(panel: Panel, args: argparse.Namespace) -> Runners:
    """The same twelve checks, driven by a precomputed signal.

    The swept parameter is concentration rather than lookback: a factor's
    window is fixed by its definition, and how much of the scored universe to
    hold is the one choice the strategy actually makes. A plateau across it is
    the same evidence a plateau across lookbacks would be.
    """
    factor = Factor(args.factor)
    scores = factor_scores(panel, factor, args.sessions)
    if scores.is_empty():
        raise SweepTooShortError

    held = Decimal(str(args.top_fraction))
    return Runners(
        label=f"{factor.value}(top {held:.0%})",
        once=lambda p, cost: run_factor(p, scores, held, cost),
        sweep=lambda p: [
            (f"top {frac:.0%}", run_factor(p, scores, frac)) for frac in SWEEP_TOP_FRACTION
        ],
        dropout=lambda p: factor_dropout_runner(p, scores, held),
        placebo=lambda p: factor_placebo_runner(p, held, args.lookback),
    )


def build_runners(panel: Panel, args: argparse.Namespace) -> Runners:
    return factor_runners(panel, args) if args.factor else momentum_runners(args)
