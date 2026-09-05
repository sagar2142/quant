"""Long work, off the request thread — MASTER_PLAN §12.9.

**A backtest is minutes and an HTTP request is seconds.** The console could
show research that had already been computed and could not start any, because
every endpoint had to answer within a browser's patience. So the two things the
system is actually for — running a backtest and putting a strategy through the
gauntlet — lived only at the command line.

This is the smallest thing that fixes that: submit work, get an id, poll it.

**One worker, deliberately.** A backtest holds the whole panel in memory and
the gauntlet runs dozens of them; two at once would thrash and neither would
finish sooner. Queued work waits, and the queue depth is visible, which is
more honest than pretending to parallelism the machine cannot deliver.

**A running job cannot be cancelled.** Python cannot safely interrupt a thread
mid-computation, and a cancel button that silently does nothing is worse than
no button. Queued jobs can be dropped; running ones are reported as
uncancellable rather than pretending.

**State is in memory and dies with the process.** These are recomputable
results, not records — a gauntlet verdict that matters gets written to the
experiment ledger by the code that produced it, not by this.
"""

from __future__ import annotations

import queue
import threading
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core.clock import utc_now

__all__ = ["Job", "JobQueueFullError", "JobRunner", "JobState", "Progress"]

#: Finished jobs kept for polling. Enough that a console left open all morning
#: can still read what it started; small enough that a day of runs does not
#: accumulate equity curves in memory.
DEFAULT_HISTORY = 40

#: Jobs allowed to wait. Past this the queue is refusing work, which is the
#: honest answer — a submission that sits behind forty backtests is not going
#: to be looked at.
DEFAULT_QUEUE = 12


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: What a worker calls to say where it has got to. Free text, shown verbatim:
#: a gauntlet's twelve checks take minutes apiece and a progress bar that only
#: moves at the end is indistinguishable from a hang.
Progress = Callable[[str], None]


@dataclass
class Job:
    """One unit of submitted work."""

    id: str
    kind: str
    label: str
    state: JobState = JobState.QUEUED
    submitted_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: str = "queued"
    result: dict[str, Any] | None = None
    error: str = ""

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or utc_now()
        return (end - self.started_at).total_seconds()

    @property
    def finished(self) -> bool:
        return self.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED)


class JobQueueFullError(RuntimeError):
    """The queue is at capacity. Refused rather than silently dropped."""


class JobRunner:
    """A single background worker with a bounded queue and bounded memory."""

    def __init__(self, history: int = DEFAULT_HISTORY, max_queued: int = DEFAULT_QUEUE) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: queue.Queue[str] = queue.Queue()
        self._work: dict[str, Callable[[Progress], dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._history = history
        self._max_queued = max_queued
        self._worker: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        """Start the thread on first use rather than at import.

        A daemon thread started at import runs in every test that touches the
        app, and in the CLI, and in anything that merely imports the module.
        Started on demand it exists only where work was actually submitted.
        """
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, daemon=True, name="neutron-jobs")
            self._worker.start()

    def submit(self, kind: str, label: str, work: Callable[[Progress], dict[str, Any]]) -> Job:
        """Queue work and return its job immediately.

        Args:
            kind: Coarse type, for the console to group by.
            label: What this run is, in the operator's terms.
            work: Called with a progress reporter; returns the JSON-able result.

        Raises:
            JobQueueFull: when too much is already waiting.
        """
        with self._lock:
            waiting = sum(1 for j in self._jobs.values() if j.state is JobState.QUEUED)
            if waiting >= self._max_queued:
                raise JobQueueFullError(f"{waiting} job(s) already queued")

            job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
            self._jobs[job.id] = job
            self._work[job.id] = work
            self._trim()

        self._queue.put(job.id)
        self._ensure_worker()
        return job

    def _trim(self) -> None:
        """Drop the oldest finished jobs. Caller holds the lock."""
        finished = [k for k, v in self._jobs.items() if v.finished]
        for key in finished[: max(0, len(finished) - self._history)]:
            self._jobs.pop(key, None)
            self._work.pop(key, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def listing(self, kind: str | None = None) -> list[Job]:
        """Newest first, so a console renders the run you just started at top.

        Insertion order breaks ties. Two jobs submitted inside the clock's
        resolution — which is every batch submit on a fast machine — carry the
        same `submitted_at`, and sorting on that alone put them in an order
        that changed between calls. A list that reshuffles itself under a
        two-second poll is unusable for the thing it exists for: watching a
        run you just started.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        # The insertion index is part of the key, not left to sort stability:
        # a later-inserted job wins a tie explicitly rather than by accident of
        # which order the list happened to be in.
        chosen = [(i, j) for i, j in enumerate(jobs) if kind is None or j.kind == kind]
        chosen.sort(key=lambda pair: (pair[1].submitted_at, pair[0]), reverse=True)
        return [job for _, job in chosen]

    def cancel(self, job_id: str) -> bool:
        """Drop a queued job.

        Returns:
            False if it is already running or finished — a running job cannot
            be interrupted, and saying so beats a button that lies.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state is not JobState.QUEUED:
                return False
            job.state = JobState.CANCELLED
            job.finished_at = utc_now()
            job.progress = "cancelled before it started"
            self._work.pop(job_id, None)
            return True

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._execute(job_id)
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            work = self._work.pop(job_id, None)
            if job is None or work is None or job.state is JobState.CANCELLED:
                return
            job.state = JobState.RUNNING
            job.started_at = utc_now()
            job.progress = "starting"

        def report(message: str) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current.progress = message

        try:
            result = work(report)
        except Exception as exc:  # noqa: BLE001 - a failed job must not kill the worker
            with self._lock:
                if job := self._jobs.get(job_id):
                    job.state = JobState.FAILED
                    job.finished_at = utc_now()
                    # The type and message, not the traceback: the console
                    # shows this to an operator, and the traceback is in the
                    # server log for whoever needs it.
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.progress = "failed"
            traceback.print_exc()
            return

        with self._lock:
            if job := self._jobs.get(job_id):
                job.state = JobState.DONE
                job.finished_at = utc_now()
                job.result = result
                job.progress = "done"
