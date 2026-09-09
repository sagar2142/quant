"""Running the daily jobs from the console — MASTER_PLAN §13.6, §20.

**Everything here already existed as a command.** The ingest and the simulation
cycle were run by hand in a terminal, which is fine for the person who wrote
them and a poor answer for anyone operating the system: the console could show
you that the panel was six days stale and offer no way to fix it, which is the
same failure as a screen that reports a problem it cannot act on.

These endpoints call the CLI's own `run()` rather than reimplementing it. A
second copy of "work out which sessions are missing and fetch them" would
eventually disagree with the first, and the disagreement would be a silently
missing session. What is added here is the plumbing a browser needs — a job id,
progress, and the printed output kept where it can be read afterwards.

**They share the lab's job runner**, which is one worker pulling one queue. That
serialisation is doing real work: two ingests writing the same Parquet
directory, or two cycles racing on one account, are corruption rather than
contention, and the queue makes both impossible without a lock of its own.

**Read access, not write**, on the same reasoning as the lab. `WriteAccess`
gates the kill switch and order entry and is disabled outright without an API
token; the fail-safe direction for money. Neither of these is money. The ingest
spends bandwidth, and the cycle is guarded by `assert_not_live` so it cannot run
at all in a live environment — and by `already_traded`, so a second cycle on a
session already traded is a no-op rather than a duplicate.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apps.api.auth import ReadAccess
from apps.api.jobs import JobQueueFullError, JobRunner, Progress
from apps.api.lab import JobResponse, _as_response

__all__ = ["build_operations_router"]

#: Lines of a command's output kept in the job result.
#:
#: The ingest prints a line per feed and the cycle prints its whole report, so
#: this is generous for both. The tail rather than the head: the summary a
#: reader wants is at the end, and a truncated run should show how it finished.
OUTPUT_LINES = 60


class IngestRequest(BaseModel):
    """Which feeds to fetch, and how politely."""

    #: Fetch the announced board-meeting calendar too. NSE keeps no archive of
    #: it, so a day not fetched is a day nobody can recover.
    events: bool = True
    #: Evaluate the alert rules against the lake this run produced.
    alerts: bool = True
    #: Seconds between requests. The exchanges rate-limit, and an evening run
    #: has time.
    pause: float = Field(default=1.0, ge=0.0, le=10.0)


class CycleRequest(BaseModel):
    """One simulation cycle."""

    top: int = Field(default=30, ge=2, le=200)
    #: Acknowledge a reconciliation halt and continue. A human decision (§9),
    #: which is why it is a field rather than something the console sets for
    #: you — a halt that a retry clears is not a halt.
    clear_halt: bool = False


def _captured(argv: list[str], entry: Any, report: Progress) -> dict[str, Any]:
    """Run a CLI entry point, keeping what it printed.

    A CLI's output *is* its interface — the ingest says which feeds were due
    and which were not served, and the cycle prints its whole report. Throwing
    that away and returning an exit code would leave the console showing
    "failed" with nowhere to look.
    """
    buffer = io.StringIO()
    report("running")
    with contextlib.redirect_stdout(buffer):
        code = int(entry(argv))
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    return {"exit_code": code, "output": lines[-OUTPUT_LINES:]}


def build_operations_router(runner: JobRunner) -> APIRouter:
    """Ingest and simulation, as jobs.

    Args:
        runner: The lab's runner, shared so `/lab/jobs` lists every kind of
            work in one place. A console that had to poll two queues to find
            out whether anything was running would eventually show one of them.
    """
    router = APIRouter(prefix="/system", tags=["operations"])

    def _submit(kind: str, label: str, work: Any) -> JobResponse:
        try:
            return _as_response(runner.submit(kind, label, work), with_result=False)
        except JobQueueFullError as exc:
            # 429, not 500: the request was fine and the machine is busy.
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @router.post("/ingest", response_model=JobResponse, dependencies=[ReadAccess])
    def ingest(request: IngestRequest) -> JobResponse:
        """Fetch whatever every feed is missing.

        Safe to run repeatedly and at the wrong time of day: the planner's
        whole job is to make "too early" a no-op rather than a session recorded
        as unavailable.
        """
        from apps.cli import daily  # noqa: PLC0415 - heavy, and only on demand

        argv = ["--pause", str(request.pause)]
        if request.events:
            argv.append("--events")
        if request.alerts:
            argv.append("--alerts")

        return _submit(
            "ingest",
            "every feed" if request.events else "market feeds",
            lambda report: _captured(argv, daily.run, report),
        )

    @router.post("/cycle", response_model=JobResponse, dependencies=[ReadAccess])
    def cycle(request: CycleRequest) -> JobResponse:
        """Run one simulation cycle against the latest session held.

        Exit code 2 is a halt, and it reaches the result rather than raising:
        a halted cycle ran and produced a finding, which is different from a
        cycle that could not run at all.
        """
        from apps.cli import paper  # noqa: PLC0415 - heavy, and only on demand

        argv = ["--top", str(request.top)]
        if request.clear_halt:
            argv.append("--clear-halt")

        return _submit(
            "cycle",
            f"simulation · top {request.top}",
            lambda report: _captured(argv, paper.run, report),
        )

    return router
