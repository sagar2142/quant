"""Background jobs and the lab endpoints — MASTER_PLAN §5, §12.9.

A backtest is minutes and an HTTP request is seconds, so the console could
display research and start none of it. These cover the runner that fixes that,
and lean on the cases where a job system lies: a failure that looks like a
result, a cancel that appears to work but does not, a queue that accepts more
than it will ever run.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from apps.api.jobs import JobQueueFullError, JobRunner, JobState
from apps.api.lab import build_lab_router

TIMEOUT = 10.0


def wait_for(runner: JobRunner, job_id: str, timeout: float = TIMEOUT):
    """Poll until the job finishes. Polling, not a sleep: a fixed wait either
    flakes on a slow machine or wastes time on a fast one."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        assert job is not None
        if job.finished:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class TestJobRunner:
    def test_a_job_runs_and_returns_its_result(self) -> None:
        runner = JobRunner()
        job = runner.submit("test", "adds", lambda _report: {"answer": 42})
        assert wait_for(runner, job.id).result == {"answer": 42}

    def test_submit_returns_before_the_work_finishes(self) -> None:
        """The whole point. A submit that blocked would be an endpoint that
        times out, which is what this replaces."""
        runner = JobRunner()
        release = threading.Event()
        job = runner.submit("test", "slow", lambda _r: {"ok": release.wait(TIMEOUT)})
        assert job.state in (JobState.QUEUED, JobState.RUNNING)
        release.set()
        wait_for(runner, job.id)

    def test_a_failure_is_recorded_not_raised(self) -> None:
        """A failed job must not take down the worker, and must not come back
        looking like a result."""

        def explode(_report):
            raise ValueError("the panel is empty")

        runner = JobRunner()
        job = wait_for(runner, runner.submit("test", "boom", explode).id)
        assert job.state is JobState.FAILED
        assert job.result is None
        assert "the panel is empty" in job.error

    def test_the_worker_survives_a_failure(self) -> None:
        def explode(_report):
            raise RuntimeError("first")

        runner = JobRunner()
        wait_for(runner, runner.submit("test", "boom", explode).id)
        after = wait_for(runner, runner.submit("test", "fine", lambda _r: {"n": 1}).id)
        assert after.state is JobState.DONE

    def test_progress_is_visible_while_running(self) -> None:
        """A four-minute gauntlet that says nothing is indistinguishable from
        a hang."""
        runner = JobRunner()
        reported = threading.Event()
        release = threading.Event()

        def work(report):
            report("assembling inputs")
            reported.set()
            release.wait(TIMEOUT)
            return {}

        job = runner.submit("test", "slow", work)
        assert reported.wait(TIMEOUT)
        assert runner.get(job.id).progress == "assembling inputs"
        release.set()
        wait_for(runner, job.id)

    def test_a_queued_job_can_be_cancelled(self) -> None:
        runner = JobRunner()
        release = threading.Event()
        first = runner.submit("test", "blocker", lambda _r: {"ok": release.wait(TIMEOUT)})
        second = runner.submit("test", "waiting", lambda _r: {"never": True})

        assert runner.cancel(second.id) is True
        assert runner.get(second.id).state is JobState.CANCELLED
        release.set()
        wait_for(runner, first.id)
        # And it stays cancelled — the worker must not pick it up afterwards.
        assert runner.get(second.id).result is None

    def test_a_finished_job_cannot_be_cancelled(self) -> None:
        """Python cannot interrupt a thread mid-computation, so a cancel that
        claimed to work would tell the operator the machine was free when it
        was not."""
        runner = JobRunner()
        job = wait_for(runner, runner.submit("test", "quick", lambda _r: {}).id)
        assert runner.cancel(job.id) is False

    def test_the_queue_refuses_rather_than_dropping(self) -> None:
        runner = JobRunner(max_queued=2)
        release = threading.Event()
        runner.submit("test", "blocker", lambda _r: {"ok": release.wait(TIMEOUT)})
        runner.submit("test", "q1", lambda _r: {})
        runner.submit("test", "q2", lambda _r: {})
        with pytest.raises(JobQueueFullError):
            runner.submit("test", "q3", lambda _r: {})
        release.set()

    def test_history_is_bounded(self) -> None:
        """A console left open all day must not accumulate equity curves."""
        runner = JobRunner(history=3)
        for i in range(6):
            wait_for(runner, runner.submit("test", f"j{i}", lambda _r: {"n": 1}).id)
        assert len(runner.listing()) <= 4

    def test_listing_is_newest_first(self) -> None:
        runner = JobRunner()
        for i in range(3):
            wait_for(runner, runner.submit("test", f"j{i}", lambda _r: {}).id)
        assert [j.label for j in runner.listing()] == ["j2", "j1", "j0"]

    def test_listing_filters_by_kind(self) -> None:
        runner = JobRunner()
        wait_for(runner, runner.submit("backtest", "a", lambda _r: {}).id)
        wait_for(runner, runner.submit("gauntlet", "b", lambda _r: {}).id)
        assert [j.label for j in runner.listing("gauntlet")] == ["b"]

    def test_elapsed_is_zero_before_it_starts(self) -> None:
        """Not a fabricated duration for work that has not begun."""
        runner = JobRunner(max_queued=4)
        release = threading.Event()
        runner.submit("test", "blocker", lambda _r: {"ok": release.wait(TIMEOUT)})
        queued = runner.submit("test", "waiting", lambda _r: {})
        assert runner.get(queued.id).elapsed_seconds == 0.0
        release.set()
        wait_for(runner, queued.id)


@pytest.fixture
def lab():
    """A client over just the lab router, with its own runner."""
    from fastapi import FastAPI

    runner = JobRunner()
    app = FastAPI()
    app.include_router(build_lab_router(runner))
    return TestClient(app), runner


class TestLabEndpoints:
    def test_the_strategy_set_is_closed(self, lab) -> None:
        """A free-text strategy field would let a typo become a discovery."""
        client, _ = lab
        body = client.get("/lab/strategies").json()
        assert "momentum" in body["strategies"]
        assert "momentum_12_1" in body["factors"]

    def test_an_unknown_strategy_is_refused(self, lab) -> None:
        client, _ = lab
        assert client.post("/lab/backtest", json={"strategy": "wishful"}).status_code == 422

    def test_a_submitted_backtest_returns_a_job_not_a_result(self, lab) -> None:
        client, runner = lab
        body = client.post("/lab/backtest", json={"strategy": "hold", "top": 2}).json()
        assert body["state"] in ("queued", "running")
        assert body["result"] is None
        assert runner.get(body["id"]) is not None

    def test_an_unknown_job_is_a_404(self, lab) -> None:
        client, _ = lab
        assert client.get("/lab/jobs/nosuchjob").status_code == 404

    def test_the_listing_omits_results(self, lab) -> None:
        """One equity curve per row would make the index heavier than the
        thing it indexes."""
        client, runner = lab
        job = runner.submit("backtest", "done", lambda _r: {"curve": [1, 2, 3]})
        wait_for(runner, job.id)
        rows = client.get("/lab/jobs").json()
        assert rows[0]["result"] is None
        assert client.get(f"/lab/jobs/{job.id}").json()["result"] == {"curve": [1, 2, 3]}

    def test_a_failed_job_reports_its_error_over_http(self, lab) -> None:
        def explode(_report):
            raise ValueError("universe is empty")

        client, runner = lab
        job = wait_for(runner, runner.submit("backtest", "boom", explode).id)
        body = client.get(f"/lab/jobs/{job.id}").json()
        assert body["state"] == "failed"
        assert "universe is empty" in body["error"]

    def test_cancelling_a_finished_job_is_a_conflict_not_a_lie(self, lab) -> None:
        client, runner = lab
        job = wait_for(runner, runner.submit("backtest", "quick", lambda _r: {}).id)
        response = client.delete(f"/lab/jobs/{job.id}")
        assert response.status_code == 409
        assert "cannot be cancelled" in response.json()["detail"]

    def test_a_full_queue_is_a_429(self, lab) -> None:
        """The request was fine and the machine is busy — retrying later is
        the right response, which 500 would not say."""
        client, runner = lab
        release = threading.Event()
        runner.submit("test", "blocker", lambda _r: {"ok": release.wait(TIMEOUT)})
        for _ in range(runner._max_queued):
            runner.submit("test", "filler", lambda _r: {})
        assert client.post("/lab/backtest", json={"strategy": "hold"}).status_code == 429
        release.set()

    def test_gauntlet_parameters_are_bounded(self, lab) -> None:
        """Each dropout sample is a full backtest, so an unbounded count is a
        request that never returns."""
        client, _ = lab
        assert client.post("/lab/gauntlet", json={"dropout_samples": 5000}).status_code == 422


class TestCurveThinning:
    """A long backtest's curve is thinned to fit the browser, and the last
    point must survive that. `gather_every` takes indices 0, n, 2n..., which
    on a curve whose length is not a multiple of the step stops short of the
    end — so the chart's final point would disagree with the `end_equity`
    printed beside it, and the chart is the thing people believe.
    """

    def test_the_final_session_survives_thinning(self) -> None:
        import polars as pl

        from apps.api.lab import MAX_CURVE_POINTS

        curve = pl.DataFrame(
            {
                "event_time": list(range(10_001)),
                "equity": [float(i) for i in range(10_001)],
            }
        )
        step = max(1, -(-curve.height // MAX_CURVE_POINTS))
        thinned = curve.gather_every(step) if step > 1 else curve
        if thinned["event_time"][-1] != curve["event_time"][-1]:
            thinned = pl.concat([thinned, curve.tail(1)])

        assert step > 1, "the fixture must actually be long enough to thin"
        assert thinned["equity"][-1] == curve["equity"][-1]
        assert len(thinned) <= MAX_CURVE_POINTS + 1
