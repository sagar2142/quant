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
from quant.research.factors import Factor, FactorSpec, build_factor
from quant.strategies.base import Strategy
from quant.strategies.baselines import CrossSectionalMomentum, RandomEntry
from quant.strategies.signal import SignalStrategy

__all__ = [
    "COMPARE_FRACTION",
    "MOMENTUM_TOP_FRACTION",
    "NSE_SESSIONS",
    "SEED",
    "SWEEP_LOOKBACK",
    "SWEEP_SKIP",
    "SWEEP_TOP_FRACTION",
    "CadenceResult",
    "Panel",
    "SweepTooShortError",
    "build_market",
    "corrupt_future",
    "dropout_runner",
    "factor_dropout_runner",
    "factor_placebo_runner",
    "factor_scores",
    "placebo_runner",
    "run_cadence",
    "run_factor",
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


@dataclass(frozen=True)
class CadenceResult:
    """One rebalance cadence, measured on the things its hypothesis named.

    Fees are carried alongside the returns because the whole claim is about
    them: a cadence question that reported only Sharpe would be judged on half
    of what it predicted.
    """

    returns: npt.NDArray[np.float64]
    fees: Decimal
    orders_filled: int


def run_cadence(panel: Panel, lookback: int, skip: int, every: int) -> CadenceResult:
    """The same momentum book, re-decided every `every` sessions.

    Only the cadence changes. Same lookback, same skip, same universe, same
    costs, same starting cash — so a difference in the outcome is attributable
    to how often the book was re-decided and to nothing else. That is the
    entire content of the two rebalance hypotheses, and running them any other
    way would answer a question neither of them asked.
    """
    engine = BacktestEngine(
        strategy=CrossSectionalMomentum(
            lookback_bars=lookback, skip_bars=skip, top_fraction=MOMENTUM_TOP_FRACTION
        ),
        market=build_market(panel.instruments),
        config=BacktestConfig(initial_cash=Decimal(1_000_000), rebalance_every=every),
    )
    result = engine.run(panel.history, universe=panel.universe)
    if result.equity_curve.is_empty():
        return CadenceResult(np.array([], dtype=np.float64), Decimal(0), 0)
    return CadenceResult(
        returns=returns_from_equity(result.equity_curve["equity"].to_list()),
        fees=result.final_portfolio.fees_paid,
        orders_filled=result.orders_filled,
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


#: Fractions of the scored universe a factor strategy holds. The sweep for a
#: factor, in place of momentum's lookback and skip: concentration is the one
#: parameter a precomputed signal actually has, and a plateau across it is the
#: same evidence a plateau across lookbacks would be.
SWEEP_TOP_FRACTION = (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"), Decimal("0.4"))


def factor_scores(panel: Panel, factor: Factor, sessions: int) -> pl.DataFrame:
    """Signal panel for one factor, scored over the same history the gauntlet
    trades.

    Built once and reused by every runner below. Rebuilding per configuration
    would be slow and, worse, would let a sweep silently score each
    configuration on a slightly different universe.

    The forward-return column `build_factor` attaches is dropped here.
    `SignalStrategy` refuses a panel carrying one and is right to: those are
    the future, and a backtest handed them would be reading the answer. The
    guard caught exactly that when this function first passed them through.
    """
    scored = build_factor(panel.history, FactorSpec(factor, window=sessions), (1,))
    if scored.is_empty():
        # Selected even when empty. Returning the raw frame handed an empty
        # panel that still carried `fwd_1` to `SignalStrategy`, which refused it
        # as a forward leak — so a window too short to score reported itself as
        # look-ahead, which is a different and much more alarming problem.
        return scored.select("event_time", "symbol", "signal") if scored.width else scored
    return scored.select("event_time", "symbol", "signal")


def run_factor(
    panel: Panel,
    scores: pl.DataFrame,
    top_fraction: Decimal,
    cost_multiple: Decimal = Decimal(1),
) -> npt.NDArray[np.float64]:
    """Per-bar returns for one factor configuration."""
    return run_strategy(
        panel,
        SignalStrategy(scores, top_fraction=top_fraction, name="factor"),
        cost_multiple,
    )


def factor_dropout_runner(
    panel: Panel, scores: pl.DataFrame, top_fraction: Decimal
) -> UniverseRunner:
    """The factor equivalent of `dropout_runner` — test 8."""

    def run(universe: tuple[InstrumentId, ...]) -> npt.NDArray[np.float64]:
        return run_factor(replace(panel, universe=universe), scores, top_fraction)

    return run


def factor_placebo_runner(panel: Panel, top_fraction: Decimal, lookback: int) -> SeededRunner:
    """Random entry holding the same number of names — test 10.

    Matched to the factor strategy's concentration rather than momentum's, so
    the comparison isolates the signal rather than the position count.
    """
    n_names = max(1, int(len(panel.universe) * float(top_fraction)))

    def run(seed: int) -> npt.NDArray[np.float64]:
        return run_strategy(
            panel,
            RandomEntry(seed=seed, n_names=n_names, hold_bars=1, lookback=lookback + 1),
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
