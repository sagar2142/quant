"""Run the validation gauntlet against a real backtest — MASTER_PLAN §5.4.

    python -m apps.cli.validate --top 30

Assembles the gauntlet's inputs by actually re-running the backtest under the
conditions each check requires: corrupted future data, tripled costs, a
parameter sweep, split samples. Nothing is asserted that was not computed.

Expect rejection. A 90%+ rejection rate is the system working (§5.5); a strategy
that sails through on the first attempt more likely indicates a broken gauntlet
than a discovered edge.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np
import numpy.typing as npt
import polars as pl

from apps.cli.backtest import build_universe, load_panel, nse_instrument
from core.config import settings
from core.instruments import Instrument, InstrumentId
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from engine.backtest import BacktestConfig, BacktestEngine, MarketModel, NextOpenFill
from engine.costs.india import NseEquityCostModel
from engine.costs.model import ScaledCostModel
from engine.validation import GauntletInputs, run_gauntlet
from engine.validation.generators import (
    DROPOUT_FRACTION,
    SamplingSpec,
    SeededRunner,
    UniverseRunner,
    market_proxy,
    placebo_sharpes,
    regime_slices,
    universe_dropout_sharpes,
)
from engine.validation.report import MIN_DROPOUT_SAMPLES, MIN_PLACEBO_SAMPLES
from quant.math.metrics.performance import returns_from_equity, summarise
from quant.strategies.base import Strategy
from quant.strategies.baselines import CrossSectionalMomentum, RandomEntry

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the validation gauntlet")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--lake", default=None)
    # Each sample is a full backtest, so these are the run's cost knobs. The
    # defaults are the floors the checks accept; raising them buys a percentile
    # you can believe, at linear expense.
    parser.add_argument(
        "--dropout-samples", type=int, default=MIN_DROPOUT_SAMPLES * 3, help="Test 8 subsets"
    )
    parser.add_argument(
        "--placebo-samples", type=int, default=MIN_PLACEBO_SAMPLES, help="Test 10 random runs"
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")

    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}")
        return 1

    universe = build_universe(store, args.top)
    if not universe:
        print("universe is empty — ingest more sessions")
        return 1

    symbols = dict(history.select("instrument_id", "symbol").unique().iter_rows())
    instruments = {InstrumentId(i): nse_instrument(i, symbols.get(i, i)) for i in universe}

    print(
        f"assembling gauntlet inputs for momentum({args.lookback}/{args.skip}) "
        f"over {len(universe)} NSE names..."
    )

    panel = Panel(history=history, instruments=instruments, universe=universe)
    baseline = run_one(panel, args.lookback, args.skip)
    if baseline.size == 0:
        print("no returns produced — the panel is shorter than the lookback")
        return 1

    stats = summarise(baseline, periods_per_year=NSE_SESSIONS)
    print("\nbaseline performance:")
    print(stats.format())
    if stats.is_implausible:
        print("\n  WARNING: Sharpe above the 2.5 smell test (§2.1) — suspect a leak")

    corrupted = run_one(replace(panel, history=corrupt_future(history)), args.lookback, args.skip)
    size = int(min(baseline.size, corrupted.size) * COMPARE_FRACTION)
    shuffled = np.concatenate([corrupted[:size], baseline[size:]])

    sweep: list[npt.NDArray[np.float64]] = []
    neighbourhood: list[float] = []
    for lookback in SWEEP_LOOKBACK:
        for skip in SWEEP_SKIP:
            if skip >= lookback:
                continue
            rets = run_one(panel, lookback, skip)
            if rets.size:
                sweep.append(rets)
                neighbourhood.append(summarise(rets, periods_per_year=NSE_SESSIONS).sharpe)

    if not sweep:
        print("parameter sweep produced nothing — the panel is too short")
        return 1

    width = min(r.size for r in sweep)
    sweep_matrix = np.column_stack([r[:width] for r in sweep])
    split = baseline.size // 2

    print(f"universe dropout: {args.dropout_samples} subsets at {DROPOUT_FRACTION:.0%} removed...")
    dropout = universe_dropout_sharpes(
        dropout_runner(panel, args.lookback, args.skip),
        universe,
        SamplingSpec(seed=SEED, samples=args.dropout_samples, periods_per_year=NSE_SESSIONS),
    )

    print(f"placebo: {args.placebo_samples} random-entry runs...")
    placebo = placebo_sharpes(
        placebo_runner(panel, args.lookback),
        SamplingSpec(seed=SEED, samples=args.placebo_samples, periods_per_year=NSE_SESSIONS),
    )

    market = market_proxy(history, universe)
    regimes = regime_slices(baseline, market["market_return"].to_numpy())
    print(f"regimes found: {', '.join(sorted(regimes)) if regimes else 'none — sample too short'}")

    inputs = GauntletInputs(
        returns=baseline,
        n_trials=len(sweep),  # honest count: this is what the sweep tested
        seed=SEED,
        shuffled_future_returns=shuffled,
        sweep_returns=sweep_matrix,
        in_sample_returns=baseline[:split],
        out_of_sample_returns=baseline[split:],
        parameter_neighbourhood=np.array(neighbourhood),
        tripled_cost_returns=run_one(panel, args.lookback, args.skip, Decimal(3)),
        universe_dropout_sharpes=dropout,
        regime_returns=dict(regimes) if regimes else None,
        placebo_sharpes=placebo,
        trade_returns=baseline[baseline != 0],
        periods_per_year=NSE_SESSIONS,
    )

    print()
    print(run_gauntlet(inputs, short_circuit=False).format())
    # Test 12 stays SKIP by design. The locked test set is touched once per
    # strategy, ever (§5.3) — running it here would burn the only untouched
    # evidence on a routine validation pass.
    return 0


if __name__ == "__main__":
    sys.exit(run())
