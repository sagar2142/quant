"""Parallel gauntlet runs — MASTER_PLAN §5.4, §M3.

The gauntlet is sixty-eight independent backtests and ran them one at a time.
Running them at once is only allowed if the answer does not change, and §M3
makes that a gate rather than a preference: two runs over the same data with
the same spec must produce byte-identical output.

So the load-bearing test here is not "is it faster" — it is **"is it the same"**.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from apps.cli.pool import JobSpec, Workload, auto_workers, run_jobs
from core.clock import UTC
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from engine.validation.generators import SamplingSpec, dropout_subsets, placebo_seeds

NAMES = tuple(f"NSE:N{i:02d}" for i in range(8))
SESSIONS = 90


def instrument(instrument_id: str) -> Instrument:
    return Instrument(
        instrument_id=InstrumentId(instrument_id),
        symbol=instrument_id.rsplit(":", maxsplit=1)[-1],
        asset_class=AssetClass.EQUITY,
        exchange=Exchange.NSE,
        currency=Currency.INR,
        tick_size=Decimal("0.01"),
    )


@pytest.fixture(scope="module")
def workload() -> Workload:
    """A small but genuinely tradable panel."""
    rng = np.random.default_rng(20260904)
    rows = []
    for index, name in enumerate(NAMES):
        price = 100.0 + 10 * index
        for day in range(SESSIONS):
            price *= float(1 + rng.normal(0.0005, 0.015))
            event = datetime(2024, 1, 1, 10, 0, tzinfo=UTC) + timedelta(days=day)
            rows.append(
                {
                    "event_time": event,
                    "receive_time": event + timedelta(hours=2, minutes=30),
                    "instrument_id": name,
                    "symbol": name.split(":")[-1],
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 5e6,
                }
            )
    history = pl.DataFrame(rows).sort(["event_time", "instrument_id"])
    return Workload(
        history=history,
        instruments={InstrumentId(n): instrument(n) for n in NAMES},
        universe=tuple(InstrumentId(n) for n in NAMES),
    )


def momentum_jobs(workload: Workload) -> list[JobSpec]:
    return [
        JobSpec(kind="momentum", lookback=lookback, skip=1, universe=workload.universe)
        for lookback in (20, 25, 30, 35, 40, 45)
    ]


class TestWorkerCount:
    def test_one_worker_is_honoured(self) -> None:
        assert auto_workers(jobs=64, requested=1) == 1

    def test_a_small_batch_stays_in_process(self) -> None:
        """Starting an interpreter on Windows costs most of a second. Below a
        handful of jobs the pool is pure overhead."""
        assert auto_workers(jobs=2) == 1

    def test_it_never_asks_for_more_workers_than_jobs(self) -> None:
        """A request for sixty-four workers on five jobs is still five at most,
        and fewer if this machine has fewer cores. The count is bounded by
        every constraint, not only by the one the caller named."""
        assert 1 <= auto_workers(jobs=5, requested=64) <= 5

    def test_auto_leaves_the_machine_usable(self) -> None:
        chosen = auto_workers(jobs=64)
        assert 1 <= chosen <= 4


class TestParallelMatchesSequential:
    """The gate. Everything else here is about speed; this is about truth."""

    def test_a_sweep_is_identical(self, workload: Workload) -> None:
        jobs = momentum_jobs(workload)
        one_at_a_time = run_jobs(jobs, workload, workers=1)
        all_at_once = run_jobs(jobs, workload, workers=3)

        assert len(all_at_once) == len(one_at_a_time) == len(jobs)
        for parallel, sequential in zip(all_at_once, one_at_a_time, strict=True):
            # Exact, not approximate. The same code over the same inputs must
            # produce the same float64s; a tolerance here would hide the very
            # thing the check exists to catch.
            assert np.array_equal(parallel, sequential)

    def test_dropout_subsets_are_identical(self, workload: Workload) -> None:
        spec = SamplingSpec(seed=7, samples=6, periods_per_year=252)
        subsets = dropout_subsets(workload.universe, spec, drop_fraction=0.25)
        jobs = [
            JobSpec(kind="momentum", lookback=20, skip=1, universe=subset) for subset in subsets
        ]
        assert np.array_equal(
            np.concatenate(run_jobs(jobs, workload, workers=1) or [np.array([])]),
            np.concatenate(run_jobs(jobs, workload, workers=3) or [np.array([])]),
        )

    def test_placebo_seeds_are_identical(self, workload: Workload) -> None:
        spec = SamplingSpec(seed=11, samples=6, periods_per_year=252)
        jobs = [
            JobSpec(
                kind="placebo",
                lookback=20,
                skip=1,
                universe=workload.universe,
                seed=seed,
                n_names=3,
            )
            for seed in placebo_seeds(spec)
        ]
        for parallel, sequential in zip(
            run_jobs(jobs, workload, workers=3),
            run_jobs(jobs, workload, workers=1),
            strict=True,
        ):
            assert np.array_equal(parallel, sequential)

    def test_order_follows_submission_not_completion(self, workload: Workload) -> None:
        """Workers finish out of order; results must not. A sweep whose rows
        came back shuffled would still average the same and would rank the
        parameter neighbourhood wrongly."""
        jobs = momentum_jobs(workload)
        results = run_jobs(jobs, workload, workers=3)
        for index, job in enumerate(jobs):
            alone = run_jobs([job], workload, workers=1)[0]
            assert np.array_equal(results[index], alone)


class TestPlanIsSeparableFromExecution:
    """The sample plan is a pure function of the seed, which is what lets the
    runs happen in any order at all."""

    def test_dropout_subsets_reproduce(self) -> None:
        spec = SamplingSpec(seed=3, samples=5, periods_per_year=252)
        universe = tuple(InstrumentId(n) for n in NAMES)
        assert dropout_subsets(universe, spec) == dropout_subsets(universe, spec)

    def test_placebo_seeds_reproduce(self) -> None:
        spec = SamplingSpec(seed=3, samples=5, periods_per_year=252)
        assert placebo_seeds(spec) == placebo_seeds(spec)

    def test_a_different_seed_gives_a_different_plan(self) -> None:
        universe = tuple(InstrumentId(n) for n in NAMES)
        first = dropout_subsets(universe, SamplingSpec(seed=1, samples=5, periods_per_year=252))
        second = dropout_subsets(universe, SamplingSpec(seed=2, samples=5, periods_per_year=252))
        assert first != second

    def test_the_plan_does_not_depend_on_how_many_runs_succeed(self) -> None:
        """Seeds are drawn up front. Drawing them as runs complete would make
        the sequence depend on which subsets happened to be tradable."""
        spec = SamplingSpec(seed=5, samples=8, periods_per_year=252)
        assert len(placebo_seeds(spec)) == 8


class TestEmptyAndDegenerate:
    def test_no_jobs_returns_nothing(self, workload: Workload) -> None:
        assert run_jobs([], workload, workers=3) == []

    def test_an_unknown_kind_is_refused(self, workload: Workload) -> None:
        job = JobSpec(kind="wishful", lookback=20, skip=1, universe=workload.universe)
        with pytest.raises(ValueError, match="unknown job kind"):
            run_jobs([job], workload, workers=1)


class TestSpawnSafety:
    """Workers must not re-run whatever `__main__` happens to be.

    On Windows a worker is a fresh interpreter that re-imports the parent's
    `__main__` before it can unpickle anything. From `python -m
    apps.cli.validate` that is harmless. From the console — `python -m uvicorn
    apps.api.main` — every worker would re-run *uvicorn's* main and try to
    start another API server. From a heredoc it re-runs `<stdin>`, cannot open
    it, and the parent waits forever.
    """

    def test_main_is_hidden_while_the_pool_is_built(self) -> None:
        import __main__
        from apps.cli.pool import _no_main_reimport

        with _no_main_reimport():
            assert not hasattr(__main__, "__file__")
            assert __main__.__spec__ is None

    def test_main_is_restored_afterwards(self) -> None:
        import __main__
        from apps.cli.pool import _no_main_reimport

        before_file = getattr(__main__, "__file__", None)
        before_spec = getattr(__main__, "__spec__", None)
        with _no_main_reimport():
            pass
        assert getattr(__main__, "__file__", None) == before_file
        assert getattr(__main__, "__spec__", None) == before_spec

    def test_it_restores_even_when_the_body_raises(self) -> None:
        """A pool that fails to start must not leave the process without a
        `__main__.__file__` — anything later that introspects it would then
        see a different interpreter than the one it is running in."""
        import __main__
        from apps.cli.pool import _no_main_reimport

        before = getattr(__main__, "__file__", None)
        with pytest.raises(RuntimeError), _no_main_reimport():
            raise RuntimeError("pool failed to start")
        assert getattr(__main__, "__file__", None) == before

    def test_a_simulated_server_main_does_not_reach_the_children(self) -> None:
        """Stand in for `python -m uvicorn`: a `__main__` pointing at a
        third-party entry point that would start a server if re-run."""
        from multiprocessing import spawn

        from apps.cli.pool import _no_main_reimport

        with _no_main_reimport():
            data = spawn.get_preparation_data("probe")
        assert "init_main_from_path" not in data
        assert "init_main_from_name" not in data
