"""Backtest and gauntlet, from the console — MASTER_PLAN §5, §M3, §12.9.

**The two things this system is for could only be done at a command line.** The
console could display research that had already been run and could start none
of it, so every hypothesis meant leaving the screen, remembering the flags, and
reading a wall of text that nothing on the console ever saw again.

Both are minutes of work, so both go through `apps.api.jobs`: submit, poll,
read the result. The endpoints here are thin on purpose — the actual runs call
`apps.cli.backtest` and `apps.cli.validate`, the same functions the terminal
uses. A second implementation of "assemble the twelve gauntlet inputs" would
eventually disagree with the first, and the one on screen would be the one
nobody had checked.

**The gauntlet's verdict is still written to the ledger.** `record_gauntlet_run`
runs here exactly as it does at the command line, so a run started from the
console counts toward the trial count that the Deflated Sharpe check reads
(§5.2). A console that could run the gauntlet without counting the trial would
be a machine for laundering multiple-testing bias.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apps.api.auth import ReadAccess
from apps.api.jobs import Job, JobQueueFullError, JobRunner, Progress
from quant.research.factors import Factor

__all__ = ["build_lab_router"]

STRATEGIES = ("momentum", "sma", "hold")

#: Cap on the equity curve returned to the browser. A seven-year daily backtest
#: is under two thousand points, so this only bites on something unusual — but
#: an endpoint that can return an unbounded array is one bad parameter away
#: from a response nobody can render.
MAX_CURVE_POINTS = 4000


class BacktestRequest(BaseModel):
    strategy: str = Field(default="momentum", pattern="^(momentum|sma|hold)$")
    top: int = Field(default=30, ge=2, le=200)
    lookback: int = Field(default=60, ge=2, le=504)
    skip: int = Field(default=5, ge=0, le=60)
    fast: int = Field(default=20, ge=2, le=200)
    slow: int = Field(default=50, ge=3, le=400)
    cash: float = Field(default=1_000_000, gt=0)
    cost_multiple: float = Field(default=1.0, ge=1.0, le=10.0)
    venue: str = Field(default="NSE", pattern="^(NSE|BSE)$")
    #: Corporate actions cost a network fetch per name. Off by default here
    #: because a console run is exploratory and a 30-name fetch is slow; the
    #: result says which it was, since a split read as a -50% day is invisible.
    actions: bool = False


class GauntletRequest(BaseModel):
    """A gauntlet run. Twelve checks, dozens of backtests, minutes of work."""

    factor: str | None = None
    top: int = Field(default=30, ge=2, le=200)
    lookback: int = Field(default=60, ge=2, le=504)
    skip: int = Field(default=5, ge=0, le=60)
    top_fraction: float = Field(default=0.2, gt=0.0, le=1.0)
    venue: str = Field(default="NSE", pattern="^(NSE|BSE)$")
    sessions: int = Field(default=0, ge=0, le=3000)
    #: The pre-registered hypothesis under test. Without it the Deflated Sharpe
    #: check still reports its number but cannot pass — an unverified trial
    #: count moves a strategy only ever toward accept (§5.2).
    hypothesis: str | None = None
    dropout_samples: int = Field(default=8, ge=2, le=40)
    placebo_samples: int = Field(default=10, ge=2, le=60)
    #: Processes to spread the independent backtests across. 0 picks from the
    #: machine, 1 stays in this process. Results are identical either way — the
    #: sample plan is drawn before anything is dispatched — so this is a speed
    #: knob and never a correctness one.
    workers: int = Field(default=0, ge=0, le=16)


class JobResponse(BaseModel):
    id: str
    kind: str
    label: str
    state: str
    progress: str
    submitted_at: str
    elapsed_seconds: float
    error: str = ""
    result: dict[str, Any] | None = None


def _as_response(job: Job, with_result: bool = True) -> JobResponse:
    return JobResponse(
        id=job.id,
        kind=job.kind,
        label=job.label,
        state=job.state.value,
        progress=job.progress,
        submitted_at=job.submitted_at.isoformat(),
        elapsed_seconds=round(job.elapsed_seconds, 1),
        error=job.error,
        result=job.result if with_result else None,
    )


def _backtest(request: BacktestRequest, report: Progress) -> dict[str, Any]:
    """One backtest, using the CLI's own loaders."""
    import polars as pl  # noqa: PLC0415

    from apps.cli import backtest as cli  # noqa: PLC0415 - heavy, and only needed here
    from core.config import settings  # noqa: PLC0415
    from core.instruments import InstrumentId  # noqa: PLC0415
    from data.store.panel import PanelStore  # noqa: PLC0415
    from engine.backtest import (  # noqa: PLC0415
        BacktestConfig,
        BacktestEngine,
        MarketModel,
        NextOpenFill,
    )
    from engine.costs.india import NseEquityCostModel  # noqa: PLC0415
    from engine.costs.model import ScaledCostModel  # noqa: PLC0415

    store = PanelStore(settings.lake, venue=request.venue)

    report("reading the panel")
    history = cli.load_panel(store)
    if history.is_empty():
        raise ValueError(f"the {request.venue} panel is empty — ingest sessions first")

    report(f"building a {request.top}-name universe")
    universe = cli.build_universe(store, request.top)
    if not universe:
        raise ValueError("universe is empty — ingest more sessions, or lower `top`")

    symbols = dict(history.select("instrument_id", "symbol").unique().iter_rows())
    instruments = {InstrumentId(i): cli.nse_instrument(i, symbols.get(i, i)) for i in universe}

    strategy = cli.build_strategy(
        argparse.Namespace(
            strategy=request.strategy,
            fast=request.fast,
            slow=request.slow,
            lookback=request.lookback,
            skip=request.skip,
        )
    )

    multiple = Decimal(str(request.cost_multiple))
    base_costs = NseEquityCostModel()
    costs = base_costs if multiple == 1 else ScaledCostModel(base_costs, multiple)

    # The CLI prints its corporate-action coverage; captured rather than
    # dropped, because "which names ran unadjusted" is the difference between
    # a real drawdown and a split misread as one (§9).
    report("loading corporate actions" if request.actions else "running")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        actions = cli.load_actions(instruments, symbols, enabled=request.actions)

    report(f"simulating {len(universe)} names")
    engine = BacktestEngine(
        strategy=strategy,
        market=MarketModel(
            cost_model=costs,
            fill_model=NextOpenFill(costs),
            instruments=instruments,
            actions=actions,
        ),
        config=BacktestConfig(initial_cash=Decimal(str(request.cash))),
    )
    result = engine.run(history, universe=universe)

    curve = result.equity_curve
    if curve.is_empty():
        raise ValueError("no equity curve — the panel is shorter than the lookback")

    # Thinned, but never losing the last session. `gather_every` takes indices
    # 0, n, 2n..., which on a curve whose length is not a multiple of the step
    # stops short of the end — so the chart's final point would disagree with
    # the `end_equity` and `total_return` printed beside it, and the chart is
    # the thing people believe.
    # Ceiling division: floor gives step 2 for a 10,001-row curve against a
    # 4,000 cap, which returns 5,001 points and makes the "cap" advisory.
    step = max(1, -(-curve.height // MAX_CURVE_POINTS))
    thinned = curve.gather_every(step) if step > 1 else curve
    if thinned["event_time"][-1] != curve["event_time"][-1]:
        thinned = pl.concat([thinned, curve.tail(1)])

    start_equity = float(curve["equity"][0])
    return {
        "strategy": f"{strategy.name} {strategy.spec.parameters}",
        "cost_model": costs.name,
        "cost_multiple": float(multiple),
        "venue": request.venue,
        "universe": len(universe),
        "sessions": result.bars_processed,
        "start": str(curve["event_time"][0]),
        "end": str(curve["event_time"][-1]),
        "start_equity": start_equity,
        "end_equity": float(curve["equity"][-1]),
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "fees_paid": float(curve["fees_paid"][-1]),
        "orders_generated": result.orders_generated,
        "orders_filled": result.orders_filled,
        "orders_rejected": result.orders_rejected,
        "orders_no_market": result.orders_no_market,
        "orders_unfunded": result.orders_unfunded,
        "liquidity_failures": result.liquidity_failures,
        "trades": result.trades.height,
        # Named so the console can say it, rather than leaving the reader to
        # assume a number that was never adjusted for splits.
        "corporate_actions": request.actions,
        "actions_note": buffer.getvalue().strip(),
        "curve": [
            {"t": str(t), "equity": float(e)}
            for t, e in zip(thinned["event_time"], thinned["equity"], strict=True)
        ],
    }


def _gauntlet(request: GauntletRequest, report: Progress) -> dict[str, Any]:
    """The twelve checks, through the CLI's own assembly.

    Its progress printing is captured and forwarded as job progress, so a run
    that takes four minutes says which of the twelve it is on rather than
    looking indistinguishable from a hang.
    """
    from apps.cli import validate as cli  # noqa: PLC0415 - heavy, and only needed here
    from apps.cli.runners import build_runners  # noqa: PLC0415
    from apps.cli.runs import NSE_SESSIONS  # noqa: PLC0415
    from engine.validation import run_gauntlet  # noqa: PLC0415
    from quant.math.metrics.performance import summarise  # noqa: PLC0415

    args = argparse.Namespace(
        top=request.top,
        lookback=request.lookback,
        skip=request.skip,
        factor=request.factor,
        top_fraction=request.top_fraction,
        lake=None,
        venue=request.venue,
        hypothesis=request.hypothesis,
        sessions=request.sessions,
        dropout_samples=request.dropout_samples,
        placebo_samples=request.placebo_samples,
        workers=request.workers,
    )

    report("loading the panel and universe")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        market = cli.load_market(args)
    if market is None:
        raise ValueError(buffer.getvalue().strip() or "could not load the panel")
    panel, _equity = market

    runners = build_runners(panel, args)
    report(f"baseline: {runners.label} over {len(panel.universe)} names")
    baseline = runners.once(panel, Decimal(1))
    if baseline.size == 0:
        raise ValueError("no returns produced — the panel is shorter than the lookback")

    stats = summarise(baseline, periods_per_year=NSE_SESSIONS)

    report("assembling inputs: sweep, dropout, placebo (this is the slow part)")
    with contextlib.redirect_stdout(buffer):
        inputs, neighbourhood, labels = cli.assemble_inputs(panel, args, baseline, runners)

    report("running the twelve checks")
    # `short_circuit=False`: the console shows every check's verdict. Stopping
    # at the first failure would hide that a candidate also fails four others,
    # which is the difference between "fix this" and "abandon this".
    gauntlet = run_gauntlet(inputs, short_circuit=False)

    report("recording the run")
    with contextlib.redirect_stdout(buffer):
        cli.record_gauntlet_run(args, panel, gauntlet)

    return {
        "label": runners.label,
        "venue": request.venue,
        "universe": len(panel.universe),
        "periods": int(baseline.size),
        "passed": gauntlet.passed,
        "first_failure": gauntlet.first_failure,
        "sharpe": stats.sharpe,
        "implausible": stats.is_implausible,
        "hypothesis": request.hypothesis,
        "results": [
            {
                "test": r.test,
                "passed": r.passed,
                "skipped": r.skipped,
                "statistic": r.statistic,
                "threshold": r.threshold,
                "reason": r.reason,
            }
            for r in gauntlet.results
        ],
        "neighbourhood": [
            {"label": label, "sharpe": float(sharpe)}
            for label, sharpe in zip(labels, neighbourhood, strict=True)
        ],
        "log": buffer.getvalue().strip(),
    }


def _register_runs(router: APIRouter, jobs: JobRunner) -> None:
    """Everything that starts work.

    **Read access, not write.** `WriteAccess` gates the kill switch and order
    entry, and is disabled outright on an install with no API token — the
    fail-safe direction for money. Research is not money: the worst a backtest
    can do is spend CPU. Gating it behind the token would reproduce the exact
    problem this module exists to fix, since a default install would show
    research it could not start and answer 503 without saying why.
    """

    def _submit(kind: str, label: str, work: Callable[[Progress], dict[str, Any]]) -> JobResponse:
        try:
            return _as_response(jobs.submit(kind, label, work), with_result=False)
        except JobQueueFullError as exc:
            # 429, not 500: the request was fine, the machine is busy, and
            # retrying later is the right response.
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @router.get("/strategies", dependencies=[ReadAccess])
    def strategies() -> dict[str, list[str]]:
        """What can be run. A closed set, like the factor library, so a typo
        cannot become a discovery."""
        return {"strategies": list(STRATEGIES), "factors": [f.value for f in Factor]}

    @router.post("/backtest", response_model=JobResponse, dependencies=[ReadAccess])
    def start_backtest(request: BacktestRequest) -> JobResponse:
        label = f"{request.strategy} · top {request.top} · {request.venue}"
        if request.cost_multiple != 1:
            label += f" · {request.cost_multiple:g}x costs"
        return _submit("backtest", label, lambda report: _backtest(request, report))

    @router.post("/gauntlet", response_model=JobResponse, dependencies=[ReadAccess])
    def start_gauntlet(request: GauntletRequest) -> JobResponse:
        subject = request.factor or "momentum"
        label = f"{subject} · top {request.top} · {request.venue}"
        return _submit("gauntlet", label, lambda report: _gauntlet(request, report))


def _register_jobs(router: APIRouter, jobs: JobRunner) -> None:
    """Everything that inspects work already submitted."""

    @router.get("/jobs", response_model=list[JobResponse], dependencies=[ReadAccess])
    def listing(kind: str | None = None) -> list[JobResponse]:
        """Newest first. Results are omitted here — an equity curve per row
        would make the list heavier than the thing it indexes."""
        return [_as_response(j, with_result=False) for j in jobs.listing(kind)]

    @router.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[ReadAccess])
    def one(job_id: str) -> JobResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return _as_response(job)

    @router.delete("/jobs/{job_id}", response_model=JobResponse, dependencies=[ReadAccess])
    def cancel(job_id: str) -> JobResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        if not jobs.cancel(job_id):
            # Refused, not silently ignored: Python cannot interrupt a thread
            # mid-computation, and a cancel that appeared to work would leave
            # the operator believing the machine was free when it was not.
            raise HTTPException(
                status_code=409,
                detail=f"job {job_id} is {job.state.value} and cannot be cancelled",
            )
        return _as_response(jobs.get(job_id) or job)


def build_lab_router(runner: JobRunner | None = None) -> APIRouter:
    router = APIRouter(prefix="/lab", tags=["lab"])
    jobs = runner or JobRunner()
    _register_runs(router, jobs)
    _register_jobs(router, jobs)
    return router
