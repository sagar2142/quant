"""Sizing the work to the machine — MASTER_PLAN §13.4.

The parallel plan used to be `min(4, cores - 1, jobs)` — right for the
four-core laptop this was written on and wrong in both directions elsewhere. A
thirty-two-core workstation ran four backtests at a time; a small cloud box ran
four copies of a 131MB panel into swap. Neither failure says anything: the
first looks like "this is slow", the second like "this is *very* slow".

**These tests describe machines this one is not.** That is the point — the
sizing has to be right on a server nobody here can boot, so it is tested
against synthetic hardware rather than against whatever is running the suite.
A test that asserted a number measured from the host would pass everywhere and
prove nothing.
"""

from __future__ import annotations

from ops.hardware import (
    ABSOLUTE_MAX_WORKERS,
    MEMORY_HEADROOM,
    UNKNOWN_MEMORY_WORKERS,
    GpuDevice,
    Hardware,
    detect,
    plan_workers,
)

GB = 1_000_000_000
PANEL = 131_000_000  # a three-year NSE panel, as measured


def machine(
    physical: int | None = 4,
    logical: int = 8,
    total: int | None = 16 * GB,
    available: int | None = 12 * GB,
    gpus: tuple[GpuDevice, ...] = (),
) -> Hardware:
    return Hardware(
        logical_cores=logical,
        physical_cores=physical,
        total_memory_bytes=total,
        available_memory_bytes=available,
        gpus=gpus,
        platform_name="synthetic",
    )


LAPTOP = machine(physical=4, logical=8, total=16 * GB, available=12 * GB)
WORKSTATION = machine(physical=32, logical=64, total=256 * GB, available=200 * GB)
TINY_CLOUD = machine(physical=2, logical=2, total=2 * GB, available=900_000_000)


class TestUsableCores:
    def test_physical_cores_are_preferred(self) -> None:
        """Hyperthreads share an execution unit. The backtest loop is
        arithmetic rather than latency-bound, so a second thread on one core
        buys a fraction of a core and costs a whole worker's memory."""
        assert LAPTOP.usable_cores == 4

    def test_half_the_logical_count_when_physical_is_unknown(self) -> None:
        assert machine(physical=None, logical=16).usable_cores == 8

    def test_a_single_core_machine_still_reports_one(self) -> None:
        assert machine(physical=None, logical=1).usable_cores == 1


class TestScalingUp:
    """The case this machine cannot demonstrate."""

    def test_a_workstation_uses_far_more_than_four(self) -> None:
        plan = plan_workers(WORKSTATION, jobs=200, per_worker_bytes=PANEL)
        assert plan.workers > 4
        assert plan.workers <= ABSOLUTE_MAX_WORKERS

    def test_it_stops_at_the_ceiling(self) -> None:
        """Past this, per-worker startup and shipping the panel dominate."""
        huge = machine(physical=256, logical=512, total=2000 * GB, available=1800 * GB)
        plan = plan_workers(huge, jobs=10_000, per_worker_bytes=PANEL)
        assert plan.workers == ABSOLUTE_MAX_WORKERS
        assert "ceiling" in plan.reason

    def test_a_core_is_left_for_the_parent(self) -> None:
        plan = plan_workers(machine(physical=8, logical=16), jobs=200)
        assert plan.workers == 7


class TestScalingDown:
    """The case that fails silently, by swapping."""

    def test_memory_can_bind_before_cores_do(self) -> None:
        plan = plan_workers(TINY_CLOUD, jobs=200, per_worker_bytes=PANEL)
        # 900MB available, 60% headroom, 131MB each -> 4 by memory, but only
        # 2 physical cores and one is left for the parent.
        assert plan.workers == 1
        assert "allows only one" in plan.reason

    def test_a_big_panel_on_a_small_box_stays_sequential(self) -> None:
        """The failure worth preventing: four workers holding a panel each,
        on a machine that cannot hold four."""
        plan = plan_workers(
            machine(physical=16, logical=32, total=8 * GB, available=1 * GB),
            jobs=200,
            per_worker_bytes=2_000_000_000,
        )
        assert plan.workers == 1
        assert "memory" in plan.reason

    def test_memory_is_the_named_constraint_when_it_binds(self) -> None:
        plan = plan_workers(
            machine(physical=32, logical=64, total=8 * GB, available=4 * GB),
            jobs=200,
            per_worker_bytes=500_000_000,
        )
        assert plan.workers == int(4 * GB * MEMORY_HEADROOM / 500_000_000)
        assert "memory" in plan.reason

    def test_unmeasurable_memory_is_treated_conservatively(self) -> None:
        """An unknown machine is assumed modest. Being too careful costs some
        speed; being too bold costs the run."""
        plan = plan_workers(
            machine(physical=32, logical=64, total=None, available=None),
            jobs=200,
            per_worker_bytes=PANEL,
        )
        assert plan.workers == UNKNOWN_MEMORY_WORKERS
        assert "memory unknown" in plan.reason

    def test_total_memory_stands_in_when_available_is_unknown(self) -> None:
        """macOS reports no cheap "available"; the headroom factor covers it."""
        plan = plan_workers(
            machine(physical=8, logical=8, total=16 * GB, available=None),
            jobs=200,
            per_worker_bytes=PANEL,
        )
        assert plan.workers > 1


class TestExplicitRequests:
    def test_one_worker_is_honoured_whatever_the_machine(self) -> None:
        assert plan_workers(WORKSTATION, jobs=200, requested=1).workers == 1

    def test_a_request_cannot_exceed_what_the_machine_allows(self) -> None:
        """Asking for a hundred on a two-core box does not make it a
        hundred-core box."""
        plan = plan_workers(TINY_CLOUD, jobs=200, requested=100)
        assert plan.workers <= TINY_CLOUD.usable_cores

    def test_a_request_below_the_machine_limit_is_respected(self) -> None:
        plan = plan_workers(WORKSTATION, jobs=200, requested=3)
        assert plan.workers == 3
        assert "requested" in plan.reason

    def test_too_few_jobs_stays_sequential_on_any_machine(self) -> None:
        plan = plan_workers(WORKSTATION, jobs=2, per_worker_bytes=PANEL)
        assert plan.workers == 1
        assert "job" in plan.reason


class TestReasonIsUseful:
    """ "Did it use my cores?" has to be answerable from the output."""

    def test_every_plan_names_its_constraint(self) -> None:
        for hardware in (LAPTOP, WORKSTATION, TINY_CLOUD):
            plan = plan_workers(hardware, jobs=200, per_worker_bytes=PANEL)
            assert plan.reason.strip()
            assert str(plan.workers) in plan.reason or plan.workers == 1

    def test_parallel_matches_the_count(self) -> None:
        assert plan_workers(WORKSTATION, jobs=200).parallel
        assert not plan_workers(WORKSTATION, jobs=200, requested=1).parallel


class TestDetection:
    """Against the real host, asserting only what must hold anywhere."""

    def test_it_measures_this_machine(self) -> None:
        found = detect(probe_gpu=False)
        assert found.logical_cores >= 1
        assert found.physical_cores is None or found.physical_cores >= 1
        assert found.usable_cores >= 1

    def test_unknown_is_none_rather_than_a_guess(self) -> None:
        """A measurement that failed must be distinguishable from one that
        returned zero."""
        blank = machine(physical=None, total=None, available=None)
        assert blank.physical_cores is None
        assert blank.total_memory_bytes is None
        assert "unknown" in blank.format()

    def test_the_report_names_the_gpu_situation_either_way(self) -> None:
        assert "none detected" in machine().format()
        card = machine(gpus=(GpuDevice(name="Tesla T4", memory_mb=16_000, backend="cuda"),))
        assert "Tesla T4" in card.format()
        assert card.has_gpu

    def test_a_plan_can_be_made_from_real_detection(self) -> None:
        plan = plan_workers(detect(probe_gpu=False), jobs=65, per_worker_bytes=PANEL)
        assert plan.workers >= 1
