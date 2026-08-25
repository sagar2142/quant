"""Run the validation gauntlet against a real backtest — MASTER_PLAN §5.4.

    python -m apps.cli.validate --top 30

Assembles the gauntlet's inputs by actually re-running the backtest under the
conditions each check requires: corrupted future data, tripled costs, a
parameter sweep, split samples. Nothing is asserted that was not computed.

Expect rejection. A 90%+ rejection rate is the system working (§5.5); a strategy
that sails through on the first attempt more likely indicates a broken gauntlet
than a discovered edge.

**The trial count comes from the database, not from this process.** The
Deflated Sharpe Ratio divides by how many strategies were tried before this one
looked good, and that history outlives any single run. Counting only the sweep
in front of you understates it every time, and DSR climbs steeply as the count
falls — a 2.0-Sharpe candidate scores 0.951 against 16 trials and 0.659 against
the 500 a few months of searching produce. So the count is read from
`hypotheses.n_trials`, which a trigger maintains, and a run that cannot reach it
reports its DSR but is not permitted to record a pass (§5.2).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import replace
from decimal import Decimal

import numpy as np
import numpy.typing as npt
import polars as pl

from apps.cli.backtest import build_universe, load_panel, nse_instrument
from apps.cli.runners import Runners, build_runners, sweep_configurations
from apps.cli.runs import (
    COMPARE_FRACTION,
    MOMENTUM_TOP_FRACTION,
    NSE_SESSIONS,
    SEED,
    Panel,
    SweepTooShortError,
    build_market,
    corrupt_future,
)
from core.config import settings
from core.instruments import InstrumentId
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from engine.backtest import BacktestConfig, BacktestEngine
from engine.experiments.recording import RunInputs, record_run
from engine.experiments.repository import ExperimentRepository, UnregisteredHypothesisError
from engine.validation import GauntletInputs, run_gauntlet
from engine.validation.generators import (
    DROPOUT_FRACTION,
    SamplingSpec,
    market_proxy,
    placebo_sharpes,
    regime_slices,
    universe_dropout_sharpes,
)
from engine.validation.report import MIN_DROPOUT_SAMPLES, MIN_PLACEBO_SAMPLES, GauntletReport
from ops.db import optional_connection
from quant.math.metrics.performance import summarise
from quant.research.factors import Factor
from quant.strategies.baselines import CrossSectionalMomentum


def resolve_trials(hypothesis: str | None, sweep_size: int) -> tuple[int, bool]:
    """Cumulative trials behind this candidate, and whether that is verified.

    The durable count lives in `hypotheses.n_trials`, incremented by a database
    trigger on every recorded experiment so that no code path can forget it.
    The sweep about to run has not been recorded yet, so it is added on top.

    Returns:
        `(n_trials, verified)`. When the database is unreachable or the
        hypothesis is unknown, the sweep size is returned with `verified=False`
        — a number the report still prints, but one the DSR check refuses to
        pass on. Falling back silently to the sweep size is what made the
        deflation cosmetic in the first place.
    """
    if hypothesis is None:
        print("trials: no --hypothesis given; DSR cannot pass on an unverified count")
        return sweep_size, False

    try:
        identifier = uuid.UUID(hypothesis)
    except ValueError:
        print(f"trials: {hypothesis!r} is not a UUID; DSR cannot pass")
        return sweep_size, False

    with optional_connection() as connection:
        if connection is None:
            print("trials: database unreachable; DSR cannot pass on an unverified count")
            return sweep_size, False
        try:
            prior = ExperimentRepository(connection).trials_for(identifier)
        except UnregisteredHypothesisError:
            print(f"trials: hypothesis {identifier} is not registered; DSR cannot pass")
            return sweep_size, False

    total = prior + sweep_size
    print(f"trials: {prior} recorded + {sweep_size} in this sweep = {total} (verified)")
    return total, True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the validation gauntlet")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument(
        "--factor",
        default=None,
        choices=[f.value for f in Factor],
        help=(
            "Validate a factor signal instead of the momentum strategy. The "
            "swept parameter becomes concentration rather than lookback."
        ),
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.2,
        help="Share of the scored universe a factor strategy holds.",
    )
    parser.add_argument("--lake", default=None)
    parser.add_argument(
        "--hypothesis",
        default=None,
        help=(
            "UUID of the pre-registered hypothesis under test. Its recorded "
            "trial count feeds the Deflated Sharpe Ratio; without it the DSR "
            "check reports its number but cannot pass."
        ),
    )
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


def assemble_inputs(
    panel: Panel,
    args: argparse.Namespace,
    baseline: npt.NDArray[np.float64],
    runners: Runners,
) -> tuple[GauntletInputs, list[float], list[str]]:
    """Re-run the backtest under every condition the twelve checks require.

    Returns the inputs, the parameter-neighbourhood Sharpes, and their labels.
    The last two are returned rather than recomputed because plotting the
    neighbourhood from a second sweep would risk plotting a different sweep
    than the one that was judged.
    """
    history = panel.history
    corrupted = runners.once(replace(panel, history=corrupt_future(history)), Decimal(1))
    size = int(min(baseline.size, corrupted.size) * COMPARE_FRACTION)
    shuffled = np.concatenate([corrupted[:size], baseline[size:]])

    sweep: list[npt.NDArray[np.float64]] = []
    neighbourhood: list[float] = []
    labels: list[str] = []
    for label, rets in runners.sweep(panel):
        if rets.size:
            sweep.append(rets)
            neighbourhood.append(summarise(rets, periods_per_year=NSE_SESSIONS).sharpe)
            labels.append(label)

    if not sweep:
        raise SweepTooShortError

    width = min(r.size for r in sweep)
    sweep_matrix = np.column_stack([r[:width] for r in sweep])
    split = baseline.size // 2

    print(f"universe dropout: {args.dropout_samples} subsets at {DROPOUT_FRACTION:.0%} removed...")
    dropout = universe_dropout_sharpes(
        runners.dropout(panel),
        panel.universe,
        SamplingSpec(seed=SEED, samples=args.dropout_samples, periods_per_year=NSE_SESSIONS),
    )

    print(f"placebo: {args.placebo_samples} random-entry runs...")
    placebo = placebo_sharpes(
        runners.placebo(panel),
        SamplingSpec(seed=SEED, samples=args.placebo_samples, periods_per_year=NSE_SESSIONS),
    )

    market = market_proxy(history, panel.universe)
    regimes = regime_slices(baseline, market["market_return"].to_numpy())
    print(f"regimes found: {', '.join(sorted(regimes)) if regimes else 'none — sample too short'}")

    n_trials, trials_verified = resolve_trials(args.hypothesis, len(sweep))

    inputs = GauntletInputs(
        returns=baseline,
        n_trials=n_trials,
        trials_verified=trials_verified,
        seed=SEED,
        shuffled_future_returns=shuffled,
        sweep_returns=sweep_matrix,
        in_sample_returns=baseline[:split],
        out_of_sample_returns=baseline[split:],
        parameter_neighbourhood=np.array(neighbourhood),
        tripled_cost_returns=runners.once(panel, Decimal(3)),
        universe_dropout_sharpes=dropout,
        regime_returns=dict(regimes) if regimes else None,
        placebo_sharpes=placebo,
        trade_returns=baseline[baseline != 0],
        periods_per_year=NSE_SESSIONS,
    )
    return inputs, neighbourhood, labels


def record_gauntlet_run(args: argparse.Namespace, panel: Panel, report: GauntletReport) -> None:
    """Persist this run so the trial counter it reads next time includes it.

    **Without this the counter can never grow.** `record_run` existed, was
    tested, and had no callers, so `hypotheses.n_trials` stayed wherever it
    happened to be while every gauntlet pass deflated against a number that
    never moved. Reading the counter honestly and never writing to it is only
    half a fix — the count would be verified and permanently wrong.

    Recorded after the verdict, not before: the gauntlet's own results are part
    of the row, and a run that crashed mid-gauntlet should not be counted as a
    trial that produced an answer.
    """
    if args.hypothesis is None:
        return

    with optional_connection() as connection:
        if connection is None:
            print("run not recorded: database unreachable; the trial count will understate")
            return
        try:
            repository = ExperimentRepository(connection)
            hypothesis = repository.hypothesis(uuid.UUID(args.hypothesis))
        except (ValueError, UnregisteredHypothesisError) as exc:
            print(f"run not recorded: {exc}")
            return

        record = record_run(
            connection,
            RunInputs(
                hypothesis=hypothesis,
                strategy_name="xs_momentum",
                parameters={"lookback": args.lookback, "skip": args.skip, "top": args.top},
                universe=[str(i) for i in panel.universe],
                history=panel.history,
                storage_uri=str(args.lake or settings.lake),
                seed=SEED,
                cost_model="NSE_EQUITY_DELIVERY",
            ),
            gauntlet=report,
        )
    print(f"recorded: experiment {record.experiment_id}, trial count now {record.trials}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    market = load_market(args)
    if market is None:
        return 1
    panel, _equity = market

    runners_label = build_runners(panel, args).label
    print(f"assembling gauntlet inputs for {runners_label} over {len(panel.universe)} NSE names...")

    runners = build_runners(panel, args)
    baseline = runners.once(panel, Decimal(1))
    if baseline.size == 0:
        print("no returns produced — the panel is shorter than the lookback")
        return 1

    stats = summarise(baseline, periods_per_year=NSE_SESSIONS)
    print("\nbaseline performance:")
    print(stats.format())
    if stats.is_implausible:
        print("\n  WARNING: Sharpe above the 2.5 smell test (§2.1) — suspect a leak")

    try:
        inputs, _neighbourhood, _labels = assemble_inputs(panel, args, baseline, runners)
    except SweepTooShortError as exc:
        print(exc)
        return 1

    print()
    report = run_gauntlet(inputs, short_circuit=False)
    print(report.format())
    record_gauntlet_run(args, panel, report)
    # Test 12 stays SKIP by design. The locked test set is touched once per
    # strategy, ever (§5.3) — running it here would burn the only untouched
    # evidence on a routine validation pass.
    return 0


if __name__ == "__main__":
    sys.exit(run())
