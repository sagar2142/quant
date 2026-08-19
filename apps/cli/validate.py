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
from dataclasses import replace
from decimal import Decimal

import numpy as np
import numpy.typing as npt
import polars as pl

from apps.cli.backtest import build_universe, load_panel, nse_instrument
from apps.cli.runs import (
    COMPARE_FRACTION,
    MOMENTUM_TOP_FRACTION,
    NSE_SESSIONS,
    SEED,
    SWEEP_LOOKBACK,
    SWEEP_SKIP,
    Panel,
    SweepTooShortError,
    build_market,
    corrupt_future,
    dropout_runner,
    placebo_runner,
    run_one,
)
from core.config import settings
from core.instruments import InstrumentId
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from engine.backtest import BacktestConfig, BacktestEngine
from engine.validation import GauntletInputs, run_gauntlet
from engine.validation.generators import (
    DROPOUT_FRACTION,
    SamplingSpec,
    market_proxy,
    placebo_sharpes,
    regime_slices,
    universe_dropout_sharpes,
)
from engine.validation.report import MIN_DROPOUT_SAMPLES, MIN_PLACEBO_SAMPLES
from quant.math.metrics.performance import summarise
from quant.strategies.baselines import CrossSectionalMomentum


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the validation gauntlet")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--lake", default=None)
    parser.add_argument(
        "--sessions",
        type=int,
        default=0,
        help=(
            "Trailing sessions to test over. 0 (default) uses the whole panel. "
            "Cost is linear in this and in every sample count below — the "
            "gauntlet re-runs the backtest dozens of times."
        ),
    )
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


def load_market(args: argparse.Namespace) -> tuple[Panel, list[float]] | None:
    """The panel, its universe, and the baseline equity curve.

    Shared by `apps.cli.validate` and `apps.cli.report` so the two can never
    disagree about what was tested. Returns None, having explained why, when
    there is nothing to test.
    """
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")
    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}")
        return None

    universe = build_universe(store, args.top)
    if not universe:
        print("universe is empty — ingest more sessions")
        return None

    if args.sessions:
        # Trailing window. The universe is still built from the whole panel, so
        # membership stays point-in-time correct; only the tested span shrinks.
        recent = history["event_time"].unique().sort().tail(args.sessions)
        history = history.filter(pl.col("event_time").is_in(recent.implode()))

    symbols = dict(history.select("instrument_id", "symbol").unique().iter_rows())
    instruments = {InstrumentId(i): nse_instrument(i, symbols.get(i, i)) for i in universe}
    panel = Panel(history=history, instruments=instruments, universe=universe)

    sessions = history["event_time"].n_unique()
    runs = len(sweep_configurations()) + args.dropout_samples + args.placebo_samples + 3
    print(
        f"{runs} backtests over {sessions:,} sessions x {len(universe)} names. "
        "Reduce with --sessions, --dropout-samples, --placebo-samples."
    )

    engine = BacktestEngine(
        strategy=CrossSectionalMomentum(
            lookback_bars=args.lookback, skip_bars=args.skip, top_fraction=MOMENTUM_TOP_FRACTION
        ),
        market=build_market(instruments),
        config=BacktestConfig(initial_cash=Decimal(1_000_000)),
    )
    result = engine.run(history, universe=universe)
    equity = [float(v) for v in result.equity_curve["equity"].to_list()]
    return panel, equity


def sweep_configurations() -> list[tuple[int, int]]:
    """Every (lookback, skip) the neighbourhood is measured over."""
    return [
        (lookback, skip) for lookback in SWEEP_LOOKBACK for skip in SWEEP_SKIP if skip < lookback
    ]


def assemble_inputs(
    panel: Panel, args: argparse.Namespace, baseline: npt.NDArray[np.float64]
) -> tuple[GauntletInputs, list[float], list[str]]:
    """Re-run the backtest under every condition the twelve checks require.

    Returns the inputs, the parameter-neighbourhood Sharpes, and their labels.
    The last two are returned rather than recomputed because plotting the
    neighbourhood from a second sweep would risk plotting a different sweep
    than the one that was judged.
    """
    history = panel.history
    corrupted = run_one(replace(panel, history=corrupt_future(history)), args.lookback, args.skip)
    size = int(min(baseline.size, corrupted.size) * COMPARE_FRACTION)
    shuffled = np.concatenate([corrupted[:size], baseline[size:]])

    sweep: list[npt.NDArray[np.float64]] = []
    neighbourhood: list[float] = []
    labels: list[str] = []
    for lookback, skip in sweep_configurations():
        rets = run_one(panel, lookback, skip)
        if rets.size:
            sweep.append(rets)
            neighbourhood.append(summarise(rets, periods_per_year=NSE_SESSIONS).sharpe)
            labels.append(f"{lookback}/{skip}")

    if not sweep:
        raise SweepTooShortError

    width = min(r.size for r in sweep)
    sweep_matrix = np.column_stack([r[:width] for r in sweep])
    split = baseline.size // 2

    print(f"universe dropout: {args.dropout_samples} subsets at {DROPOUT_FRACTION:.0%} removed...")
    dropout = universe_dropout_sharpes(
        dropout_runner(panel, args.lookback, args.skip),
        panel.universe,
        SamplingSpec(seed=SEED, samples=args.dropout_samples, periods_per_year=NSE_SESSIONS),
    )

    print(f"placebo: {args.placebo_samples} random-entry runs...")
    placebo = placebo_sharpes(
        placebo_runner(panel, args.lookback),
        SamplingSpec(seed=SEED, samples=args.placebo_samples, periods_per_year=NSE_SESSIONS),
    )

    market = market_proxy(history, panel.universe)
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
    return inputs, neighbourhood, labels


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    market = load_market(args)
    if market is None:
        return 1
    panel, _equity = market

    print(
        f"assembling gauntlet inputs for momentum({args.lookback}/{args.skip}) "
        f"over {len(panel.universe)} NSE names..."
    )

    baseline = run_one(panel, args.lookback, args.skip)
    if baseline.size == 0:
        print("no returns produced — the panel is shorter than the lookback")
        return 1

    stats = summarise(baseline, periods_per_year=NSE_SESSIONS)
    print("\nbaseline performance:")
    print(stats.format())
    if stats.is_implausible:
        print("\n  WARNING: Sharpe above the 2.5 smell test (§2.1) — suspect a leak")

    try:
        inputs, _neighbourhood, _labels = assemble_inputs(panel, args, baseline)
    except SweepTooShortError as exc:
        print(exc)
        return 1

    print()
    print(run_gauntlet(inputs, short_circuit=False).format())
    # Test 12 stays SKIP by design. The locked test set is touched once per
    # strategy, ever (§5.3) — running it here would burn the only untouched
    # evidence on a routine validation pass.
    return 0


if __name__ == "__main__":
    sys.exit(run())
