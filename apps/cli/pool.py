"""Run the gauntlet's independent backtests at once — MASTER_PLAN §5.4, §M3.

**The gauntlet is sixty-eight backtests and it ran them one at a time.** The
sweep, the thirty dropout subsets and the twenty placebo seeds share nothing:
each is the same panel with a different universe, seed or parameter pair, and
none reads another's result. On a four-core machine seven threads sat idle for
two minutes.

**Processes, not threads.** The backtest loop is Decimal arithmetic, dict
updates and Python-level branching — it holds the GIL almost throughout, so
threads would interleave rather than overlap. Polars already uses every core
*within* one operation; what is idle is the space between operations.

**Not a GPU.** Worth stating because it is the obvious question. The event loop
is a sequential state machine — bar T+1's cash depends on bar T's fills, so
there is nothing to parallelise across bars — the money is `Decimal` and a GPU
has no decimal type, and the per-bar work is a few dozen names, far below the
point where a kernel launch pays for itself. The parallelism here is across
*runs*, which is where it actually exists.

**Determinism is preserved by construction, not by luck.** The sample plan —
which subsets, which seeds — is drawn in the parent by `dropout_subsets` and
`placebo_seeds` before anything is dispatched, and results come back in
submission order. A worker computes the same function of the same inputs
whatever order it runs in, so the returned list is identical to the sequential
one. `tests/test_pool.py` asserts that rather than assuming it.

**It degrades rather than fails.** One worker, or a machine where a pool cannot
start, runs the same code sequentially in-process.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from core.instruments import Instrument, InstrumentId
from ops.hardware import Hardware, WorkerPlan, detect, plan_workers

__all__ = ["JobSpec", "Workload", "auto_workers", "run_jobs", "worker_plan"]

#: Below this many jobs the pool costs more than it saves. Starting a process
#: on Windows is a fresh interpreter and a re-import of polars, which is most
#: of a second before any work happens.
MIN_JOBS_FOR_POOL = 4

#: Polars reads this at import to size its thread pool.
_POLARS_THREADS = "POLARS_MAX_THREADS"


@lru_cache(maxsize=1)
def _machine() -> Hardware:
    """This machine, measured once.

    Cached because the probe shells out — `nvidia-smi`, `Get-CimInstance` —
    and the hardware does not change while the process runs.
    """
    return detect()


def worker_plan(jobs: int, per_worker_bytes: int = 0, requested: int = 0) -> WorkerPlan:
    """How many processes to use here, and the constraint that decided it.

    **Derived, not declared.** This used to be `min(4, cores - 1, jobs)`, which
    was right for the four-core laptop it was written on and wrong in both
    directions elsewhere: a thirty-two-core workstation ran four backtests at a
    time, and a small cloud box ran four copies of a 131MB panel into swap.
    Neither says anything; both just feel slow.
    """
    return plan_workers(
        _machine(),
        jobs=jobs,
        per_worker_bytes=per_worker_bytes,
        requested=requested,
        minimum_jobs=MIN_JOBS_FOR_POOL,
    )


def auto_workers(jobs: int, requested: int = 0) -> int:
    """How many processes to actually use. See `worker_plan` for the reason."""
    return worker_plan(jobs, requested=requested).workers


@dataclass(frozen=True)
class JobSpec:
    """One backtest, described rather than closed over.

    A closure cannot cross a process boundary on Windows, where a worker is a
    fresh interpreter rather than a fork. So a job says *what* to run and the
    worker rebuilds it, which also makes every dispatched run inspectable —
    printable, loggable, reproducible by hand.
    """

    kind: str
    lookback: int
    skip: int
    #: Universe for this run. Dropout varies it; everything else repeats the
    #: panel's own.
    universe: tuple[InstrumentId, ...]
    #: Placebo only.
    seed: int = 0
    #: Placebo only: how many names the control holds. Carried rather than
    #: derived, because momentum matches its own concentration and a factor
    #: run matches the factor's — deriving it here would put that difference
    #: in two places and let them drift.
    n_names: int = 1
    #: Factor runs only.
    factor_top_fraction: str = "0.2"
    cost_multiple: str = "1"


#: Set once per worker by `_initialise`. A module global because that is the
#: only thing a `ProcessPoolExecutor` initializer can leave behind, and because
#: the alternative — shipping the panel with every job — would send 175MB
#: across the boundary sixty-eight times.
_PANEL: Any = None
_SCORES: pl.DataFrame | None = None


def _initialise(
    history_ipc: bytes,
    instruments: dict[InstrumentId, Instrument],
    universe: tuple[InstrumentId, ...],
    scores_ipc: bytes | None,
) -> None:
    """Rebuild the panel inside a worker, once."""
    global _PANEL, _SCORES  # noqa: PLW0603 - the only channel an initializer has

    from apps.cli.runs import Panel  # noqa: PLC0415 - worker-side import

    _PANEL = Panel(
        history=pl.read_ipc(io.BytesIO(history_ipc)),
        instruments=instruments,
        universe=universe,
    )
    _SCORES = None if scores_ipc is None else pl.read_ipc(io.BytesIO(scores_ipc))


def _execute(job: JobSpec) -> list[float]:
    """Run one job. Returns the per-bar return series as a plain list.

    A list rather than an array: it crosses a process boundary, and pickling a
    numpy array of a few thousand floats buys nothing over the list while
    making the payload's dtype another thing that has to match.
    """
    from dataclasses import replace  # noqa: PLC0415 - worker-side

    from apps.cli.runs import run_factor, run_one, run_strategy  # noqa: PLC0415 - worker-side
    from quant.strategies.baselines import RandomEntry  # noqa: PLC0415

    panel = replace(_PANEL, universe=job.universe)
    cost = Decimal(job.cost_multiple)

    if job.kind == "momentum":
        returns = run_one(panel, job.lookback, job.skip, cost)
    elif job.kind == "factor":
        assert _SCORES is not None, "a factor job needs scores"
        returns = run_factor(panel, _SCORES, Decimal(job.factor_top_fraction), cost)
    elif job.kind == "placebo":
        returns = run_strategy(
            panel,
            RandomEntry(
                seed=job.seed,
                n_names=job.n_names,
                hold_bars=1,
                lookback=job.lookback + 1,
            ),
        )
    else:  # pragma: no cover - JobSpec.kind and this branch are edited together
        raise ValueError(f"unknown job kind {job.kind!r}")

    return [float(v) for v in np.asarray(returns, dtype=np.float64).ravel()]


@dataclass(frozen=True)
class Workload:
    """What every job in a batch shares. Sent to each worker once.

    Grouped rather than passed alongside: these four travel together, and a
    worker that received three of them would be running against a panel the
    parent did not mean.
    """

    history: pl.DataFrame
    instruments: dict[InstrumentId, Instrument]
    universe: tuple[InstrumentId, ...]
    #: Precomputed factor scores, for factor runs. Computed in the parent so
    #: every worker judges the same signal rather than each recomputing one.
    scores: pl.DataFrame | None = None


def run_jobs(
    jobs: Sequence[JobSpec],
    shared: Workload,
    workers: int = 0,
) -> list[npt.NDArray[np.float64]]:
    """Run every job, in submission order.

    Returns:
        One return series per job, in the order the jobs were given —
        identical, element for element, to running them one at a time.
    """
    if not jobs:
        return []

    # Sized against what a worker will actually hold, so the memory limit is
    # measured rather than assumed. `estimated_size` is Polars' own accounting
    # of the frame's buffers, which is what a second process pays for it.
    plan = worker_plan(
        len(jobs),
        per_worker_bytes=int(shared.history.estimated_size()),
        requested=workers,
    )
    if not plan.parallel:
        return _sequential(jobs, shared)
    chosen = plan.workers

    history_ipc = _to_ipc(shared.history)
    scores_ipc = None if shared.scores is None else _to_ipc(shared.scores)
    try:
        with (
            _no_main_reimport(),
            _bounded_child_threads(chosen),
            ProcessPoolExecutor(
                max_workers=chosen,
                initializer=_initialise,
                initargs=(history_ipc, shared.instruments, shared.universe, scores_ipc),
            ) as pool,
        ):
            # `map` yields in submission order regardless of completion order,
            # which is what keeps the result identical to the sequential run.
            return [
                np.asarray(one, dtype=np.float64) for one in pool.map(_execute, jobs, chunksize=1)
            ]
    except (OSError, RuntimeError, ImportError):
        # A machine that cannot start a pool still has to be able to run the
        # gauntlet. Slower and correct beats fast and unavailable.
        return _sequential(jobs, shared)


def _sequential(jobs: Sequence[JobSpec], shared: Workload) -> list[npt.NDArray[np.float64]]:
    """The same jobs, in this process. The fallback and the reference."""
    _initialise(
        _to_ipc(shared.history),
        shared.instruments,
        shared.universe,
        None if shared.scores is None else _to_ipc(shared.scores),
    )
    return [np.asarray(_execute(job), dtype=np.float64) for job in jobs]


@contextmanager
def _no_main_reimport() -> Iterator[None]:
    """Stop spawned workers re-running whatever `__main__` happens to be.

    **A safety fix, not a tuning one.** On Windows a worker is a fresh
    interpreter that re-imports the parent's `__main__` before it can unpickle
    anything. From `python -m apps.cli.validate` that is harmless. From the
    console — `python -m uvicorn apps.api.main` — every worker would re-run
    *uvicorn's* main and try to start another API server. From a heredoc it
    re-runs `<stdin>`, cannot open it, and the parent waits forever; that is
    how this was found.

    Hiding `__file__` and `__spec__` makes `multiprocessing.spawn` omit the
    main-module step entirely. Everything this pool sends is defined in
    importable modules — `JobSpec` and `_execute` both live here — so there is
    nothing in `__main__` a worker needs.
    """
    import __main__  # noqa: PLC0415 - the module being protected

    file = getattr(__main__, "__file__", None)
    spec = getattr(__main__, "__spec__", None)
    if file is not None:
        del __main__.__file__
    __main__.__spec__ = None
    try:
        yield
    finally:
        if file is not None:
            __main__.__file__ = file
        __main__.__spec__ = spec


def _threads_per_worker(workers: int) -> int:
    """Polars threads to give each worker.

    **Not one, and not eight.** Four workers each opening an eight-thread pool
    on an eight-thread machine is thirty-two threads fighting over four cores.
    One thread each is right there — but on a thirty-two core box running eight
    workers there are three spare cores per worker, and pinning each to a
    single thread leaves most of the machine idle inside every Polars
    operation.

    So the cores are divided among the workers. It is a share of a measured
    machine rather than a constant tuned on one.
    """
    return max(1, _machine().usable_cores // max(1, workers))


@contextmanager
def _bounded_child_threads(workers: int) -> Iterator[None]:
    """Give each worker its share of the cores, for the pool's lifetime.

    Polars sizes its thread pool from the core count at import, so without
    this every worker opens a pool as wide as the whole machine: four workers
    on an eight-thread box create thirty-two threads to run four backtests,
    each contending for cores the *other workers* need.

    Set on the parent's environment because that is what a spawned child
    inherits, and read by the child at import. The parent's own Polars is
    already imported and keeps the pool it built, so nothing outside this
    block changes; the previous value is restored regardless.
    """
    previous = os.environ.get(_POLARS_THREADS)
    os.environ[_POLARS_THREADS] = str(_threads_per_worker(workers))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_POLARS_THREADS, None)
        else:
            os.environ[_POLARS_THREADS] = previous


def _to_ipc(frame: pl.DataFrame) -> bytes:
    """Arrow IPC rather than pickle: same size, and it is the format Polars
    reads back without going through Python objects."""
    buffer = io.BytesIO()
    frame.write_ipc(buffer, compression="uncompressed")
    return buffer.getvalue()
