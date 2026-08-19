"""Executing a backtest under perturbed conditions — MASTER_PLAN §5.4.

The gauntlet judges a strategy by re-running it: over a reduced universe, with
future data corrupted, at tripled costs, against random entries. Those runs are
the *experiment*; `apps.cli.validate` and `apps.cli.report` are two interfaces
to it, and both need this identically.

Kept in one module because the runners cannot be separated from the primitives
they perturb — a dropout runner is `run_one` with a different universe, and
splitting them only creates an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np
import numpy.typing as npt
import polars as pl

from core.instruments import Instrument, InstrumentId
from engine.backtest import BacktestConfig, BacktestEngine, MarketModel, NextOpenFill
from engine.costs.india import NseEquityCostModel
from engine.costs.model import ScaledCostModel
from engine.validation.generators import SeededRunner, UniverseRunner
from quant.math.metrics.performance import returns_from_equity
from quant.strategies.base import Strategy
from quant.strategies.baselines import CrossSectionalMomentum, RandomEntry

__all__ = [
    "COMPARE_FRACTION",
    "MOMENTUM_TOP_FRACTION",
    "NSE_SESSIONS",
    "SEED",
    "SWEEP_LOOKBACK",
    "SWEEP_SKIP",
    "Panel",
    "SweepTooShortError",
    "build_market",
    "corrupt_future",
    "dropout_runner",
    "placebo_runner",
    "run_one",
    "run_strategy",
]

SEED = 20260816

#: Fraction of the universe the momentum strategy holds. The placebo copies it
#: so the two books are the same size.
MOMENTUM_TOP_FRACTION = Decimal("0.3")

#: Momentum windows swept to build the parameter neighbourhood and PBO matrix.
SWEEP_LOOKBACK = (20, 40, 60, 90, 120)
SWEEP_SKIP = (0, 5, 10)

#: Prices are corrupted from here on, but only the first COMPARE_FRACTION of
#: returns is compared. The gap matters: at the corruption boundary the price
#: triples, producing a return that is an artefact of the test itself.
CORRUPT_FROM = 0.6
COMPARE_FRACTION = 0.45

#: NSE trades ~250 sessions a year.
NSE_SESSIONS = 252


class SweepTooShortError(RuntimeError):
    """The parameter sweep produced nothing usable.

    Raised rather than returned: without a sweep there is no PBO matrix and no
    neighbourhood, so two of the twelve checks cannot run at all. Continuing
    would produce a report claiming twelve checks while silently running ten.
    """

    def __init__(self) -> None:
        super().__init__(
            "parameter sweep produced no returns — the panel is shorter than "
            "the longest swept lookback. Ingest more sessions."
        )


def build_market(
    instruments: dict[InstrumentId, Instrument], cost_multiple: Decimal = Decimal(1)
) -> MarketModel:
    base = NseEquityCostModel()
    costs = base if cost_multiple == 1 else ScaledCostModel(base, cost_multiple)
    return MarketModel(cost_model=costs, fill_model=NextOpenFill(costs), instruments=instruments)


@dataclass(frozen=True)
class Panel:
    """The three things that never vary across a sweep, grouped so they do not
    have to be threaded through every call individually (§14.2)."""

    history: pl.DataFrame
    instruments: dict[InstrumentId, Instrument]
    universe: tuple[InstrumentId, ...]


def run_strategy(
    panel: Panel, strategy: Strategy, cost_multiple: Decimal = Decimal(1)
) -> npt.NDArray[np.float64]:
    """Per-bar return series for one strategy over one panel."""
    engine = BacktestEngine(
        strategy=strategy,
        market=build_market(panel.instruments, cost_multiple),
        config=BacktestConfig(initial_cash=Decimal(1_000_000)),
    )
    result = engine.run(panel.history, universe=panel.universe)
    if result.equity_curve.is_empty():
        return np.array([], dtype=np.float64)
    return returns_from_equity(result.equity_curve["equity"].to_list())


def run_one(
    panel: Panel,
    lookback: int,
    skip: int,
    cost_multiple: Decimal = Decimal(1),
) -> npt.NDArray[np.float64]:
    """Per-bar return series for one momentum configuration."""
    return run_strategy(
        panel,
        CrossSectionalMomentum(
            lookback_bars=lookback, skip_bars=skip, top_fraction=MOMENTUM_TOP_FRACTION
        ),
        cost_multiple,
    )


def dropout_runner(panel: Panel, lookback: int, skip: int) -> UniverseRunner:
    """Runs the same strategy over a reduced universe — test 8.

    Only the universe changes. The panel, the costs and the parameters are held
    fixed so that a difference in the result is attributable to the names that
    were removed and to nothing else.
    """

    def run(universe: tuple[InstrumentId, ...]) -> npt.NDArray[np.float64]:
        return run_one(replace(panel, universe=universe), lookback, skip)

    return run


def placebo_runner(panel: Panel, lookback: int) -> SeededRunner:
    """Runs a random-entry strategy with matched exposure — test 10.

    Matching matters more than the randomness. The placebo holds the same number
    of names, at the same gross, starting on the same bar (`lookback` is copied
    from the real strategy, so neither gets a head start). Any remaining
    difference in performance is the signal's contribution, which is exactly the
    quantity test 10 is trying to measure.
    """
    n_names = max(1, int(len(panel.universe) * float(MOMENTUM_TOP_FRACTION)))

    def run(seed: int) -> npt.NDArray[np.float64]:
        return run_strategy(
            panel,
            RandomEntry(
                seed=seed,
                n_names=n_names,
                hold_bars=1,  # momentum re-decides every bar
                lookback=lookback + 1,
            ),
        )

    return run


def corrupt_future(history: pl.DataFrame, fraction: float = CORRUPT_FROM) -> pl.DataFrame:
    """Scale every price after `fraction` of the sample. Earlier decisions must
    not move."""
    timestamps = history["event_time"].unique().sort().to_list()
    cutoff = timestamps[int(len(timestamps) * fraction)]
    return history.with_columns(
        [
            pl.when(pl.col("event_time") >= cutoff)
            .then(pl.col(c) * 3.0)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("open", "high", "low", "close")
        ]
    )
