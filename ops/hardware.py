"""What this machine actually is — MASTER_PLAN §13.4, §M3.

**The system was tuned for the laptop it was written on.** `MAX_WORKERS = 4`
is right for a four-core ultrabook and wrong everywhere else: it wastes
twenty-eight cores on a workstation and, on a small cloud box, four workers
each holding a 131MB panel is the difference between running and swapping.
Neither failure announces itself — the first looks like "this is just slow",
the second looks like "this is *very* slow".

So the parallel plan is derived from the machine rather than declared. Three
inputs decide it:

    cores      how many runs can genuinely proceed at once
    memory     how many copies of the panel fit without swapping
    the work   how many independent jobs there are to do

and the smallest of the three wins.

**Stdlib only.** `psutil` would be a dependency added for a monitoring
concern, and every platform this runs on exposes what is needed already:
`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` on Linux, `sysctl` on macOS.
Each is tried and each is allowed to fail — an unknown quantity is reported as
`None` and the plan falls back to the conservative branch, rather than a guess
being substituted for a measurement.

**GPU is detected and reported, never assumed.** See `gpu_devices`.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "GpuDevice",
    "Hardware",
    "WorkerPlan",
    "detect",
    "gpu_devices",
    "plan_workers",
]

#: Share of *available* memory a parallel run may take. Not all of it: the
#: parent still holds its own panel, the OS needs page cache, and a machine
#: driven to its last megabyte swaps, which is far slower than the sequential
#: run the parallelism was meant to beat.
MEMORY_HEADROOM = 0.6

#: Assumed when memory cannot be measured. Deliberately small — an unknown
#: machine is treated as a modest one, because being too conservative costs
#: some speed and being too aggressive costs the run.
UNKNOWN_MEMORY_WORKERS = 2

#: Hard ceiling regardless of core count.
#:
#: A safety valve rather than a tuning parameter — cores and memory decide the
#: number in practice, and this only catches the absurd case. It is not lower
#: because a large sweep on a large machine is a real workload: sixty-four
#: workers is a 64-core server running one job each, which that machine can do.
#: Past roughly here the per-worker startup and the IPC of shipping a panel to
#: each of them start to dominate whatever is left to parallelise.
ABSOLUTE_MAX_WORKERS = 64


@dataclass(frozen=True)
class GpuDevice:
    """One accelerator, as reported by its own tooling."""

    name: str
    memory_mb: int | None
    #: "cuda" or "unknown". The vendor's own word for what it is.
    backend: str


@dataclass(frozen=True)
class Hardware:
    """What was measured. `None` means "could not tell", never a guess."""

    logical_cores: int
    physical_cores: int | None
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    gpus: tuple[GpuDevice, ...] = ()
    platform_name: str = ""

    @property
    def usable_cores(self) -> int:
        """Cores worth running a CPU-bound job on.

        Physical where known. Hyperthreads share an execution unit, and the
        backtest loop is arithmetic rather than latency-bound, so a second
        thread on the same core buys perhaps a fifth of a core and costs a full
        worker's memory. Counting them would size the pool for hardware that is
        not there.
        """
        return self.physical_cores or max(1, self.logical_cores // 2)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    def format(self) -> str:
        def gb(value: int | None) -> str:
            return "unknown" if value is None else f"{value / 1e9:.1f}GB"

        lines = [
            f"  platform         {self.platform_name or platform.platform()}",
            f"  logical cores    {self.logical_cores}",
            f"  physical cores   {self.physical_cores or 'unknown (assuming half)'}",
            f"  memory total     {gb(self.total_memory_bytes)}",
            f"  memory available {gb(self.available_memory_bytes)}",
        ]
        if self.gpus:
            lines.append(f"  GPUs             {len(self.gpus)}")
            lines.extend(
                f"    {g.name} ({g.backend}" + (f", {g.memory_mb}MB)" if g.memory_mb else ")")
                for g in self.gpus
            )
        else:
            lines.append("  GPUs             none detected")
        return "\n".join(lines)


def _run(command: list[str], timeout: int = 15) -> str:
    """Run a probe and return its output, or an empty string.

    The executable is resolved to a full path first. A bare name is looked up
    in `PATH` at exec time, which on a shared machine is a path an attacker can
    prepend to — and this module runs on an operator's workstation, so the
    lookup happens here where it can be seen.
    """
    resolved = shutil.which(command[0])
    if resolved is None:
        return ""
    try:
        completed = subprocess.run(  # noqa: S603 - resolved path, fixed arguments
            [resolved, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _physical_cores() -> int | None:
    """Physical cores, or None when the platform will not say cheaply."""
    system = platform.system()
    try:
        if system == "Linux":
            return _linux_physical_cores()
        if system == "Darwin":
            return int(_run(["sysctl", "-n", "hw.physicalcpu"]) or 0) or None
        if system == "Windows":
            counted = _run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | "
                    "Measure-Object -Property NumberOfCores -Sum).Sum",
                ]
            )
            return int(counted or 0) or None
    except (OSError, ValueError):
        return None
    return None


def _linux_physical_cores() -> int | None:
    """Distinct (physical id, core id) pairs — cores, not threads."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    pairs: set[tuple[str, str]] = set()
    physical = core = None
    for line in text.splitlines():
        if line.startswith("physical id"):
            physical = line.split(":")[-1].strip()
        elif line.startswith("core id"):
            core = line.split(":")[-1].strip()
        elif not line.strip() and physical is not None and core is not None:
            pairs.add((physical, core))
            physical = core = None
    return len(pairs) or None


class _MemoryStatusEx(ctypes.Structure):
    """Windows `MEMORYSTATUSEX`, as the API expects to be handed it."""

    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def _memory() -> tuple[int | None, int | None]:
    """(total, available) bytes, either of which may be None."""
    system = platform.system()
    try:
        if system == "Windows":
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
            return None, None
        if system == "Linux":
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) * 1024  # kB
            # MemAvailable is the kernel's own estimate of what a new process
            # can have without swapping — a better answer than MemFree, which
            # excludes reclaimable page cache.
            return values.get("MemTotal"), values.get("MemAvailable", values.get("MemFree"))
        if system == "Darwin":
            total = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
            # macOS has no cheap "available"; the total is the honest answer
            # and the headroom factor covers the difference.
            return total, None
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None, None
    return None, None


def _cupy_devices() -> tuple[GpuDevice, ...]:
    """Devices CuPy can open, or none."""
    try:
        import cupy  # type: ignore[import-not-found] # noqa: PLC0415 - optional

        found = []
        for index in range(cupy.cuda.runtime.getDeviceCount()):
            properties = cupy.cuda.runtime.getDeviceProperties(index)
            found.append(
                GpuDevice(
                    name=properties["name"].decode(errors="replace"),
                    memory_mb=int(properties["totalGlobalMem"] / 1e6),
                    backend="cuda",
                )
            )
    except Exception:  # noqa: BLE001 - absent, broken or driverless all mean "no"
        return ()
    return tuple(found)


def _torch_devices() -> tuple[GpuDevice, ...]:
    """Devices Torch can open, or none."""
    try:
        import torch  # type: ignore[import-not-found] # noqa: PLC0415 - optional

        if not torch.cuda.is_available():
            return ()
        found = [
            GpuDevice(
                name=torch.cuda.get_device_name(index),
                memory_mb=int(torch.cuda.get_device_properties(index).total_memory / 1e6),
                backend="cuda",
            )
            for index in range(torch.cuda.device_count())
        ]
    except Exception:  # noqa: BLE001 - as above
        return ()
    return tuple(found)


def _smi_devices() -> tuple[GpuDevice, ...]:
    """What the driver reports, for a machine with no Python GPU stack.

    Reported so the console can say "there is a card here, and nothing
    installed that could use it" — which is actionable, where silence is not.
    """
    output = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    found = []
    for line in output.splitlines():
        name, _, memory = line.partition(",")
        digits = "".join(c for c in memory if c.isdigit())
        if name.strip():
            found.append(
                GpuDevice(
                    name=name.strip(),
                    memory_mb=int(digits) if digits else None,
                    backend="cuda",
                )
            )
    return tuple(found)


def gpu_devices() -> tuple[GpuDevice, ...]:
    """Accelerators this process could actually use.

    **Detected, never assumed, and never used without a verified path.** A GPU
    that exists is not a GPU that helps: the backtest loop is a sequential
    state machine over `Decimal` money and has nothing to offer one. What this
    answers is "could an accelerated path run here", so the console can say so
    and a future kernel has something to gate on.

    The libraries are asked first, because a device CuPy cannot open is not a
    device this process can use; the driver is asked last so a machine with a
    card and no Python GPU stack is still reported rather than looking bare.
    """
    return _cupy_devices() or _torch_devices() or _smi_devices()


def detect(probe_gpu: bool = True) -> Hardware:
    """Measure this machine.

    Args:
        probe_gpu: Skippable because it may shell out to `nvidia-smi`, and a
            hot path that runs per request should not.
    """
    total, available = _memory()
    return Hardware(
        logical_cores=os.cpu_count() or 1,
        physical_cores=_physical_cores(),
        total_memory_bytes=total,
        available_memory_bytes=available,
        gpus=gpu_devices() if probe_gpu else (),
        platform_name=platform.platform(),
    )


@dataclass(frozen=True)
class WorkerPlan:
    """How many processes to run, and why that number.

    The reason is not decoration. A run that is slower than expected is a
    question — "did it use my cores?" — and a plan that can only answer with a
    number leaves the operator to guess whether it was cores, memory or the
    size of the job that decided.
    """

    workers: int
    reason: str

    @property
    def parallel(self) -> bool:
        return self.workers > 1


def plan_workers(
    hardware: Hardware,
    jobs: int,
    per_worker_bytes: int = 0,
    requested: int = 0,
    minimum_jobs: int = 4,
) -> WorkerPlan:
    """Decide the worker count from the machine, the memory and the work.

    Args:
        hardware: What `detect` measured.
        jobs: Independent runs available.
        per_worker_bytes: What one worker will hold — the panel, mostly. Zero
            skips the memory check, which is right only when the payload is
            genuinely small.
        requested: 0 to decide, 1 to force sequential, or an explicit count
            that still respects the job count.
        minimum_jobs: Below this the pool costs more than it saves.

    Returns:
        The count and the constraint that produced it.
    """
    if requested == 1:
        return WorkerPlan(1, "sequential: asked for one worker")
    if jobs < minimum_jobs:
        return WorkerPlan(1, f"sequential: only {jobs} job(s), a pool would cost more")

    by_cores = max(1, hardware.usable_cores - 1)
    limits = [("jobs", jobs), ("cores", by_cores), ("ceiling", ABSOLUTE_MAX_WORKERS)]

    if per_worker_bytes > 0:
        budget = hardware.available_memory_bytes or hardware.total_memory_bytes
        if budget is None:
            limits.append(("memory unknown", UNKNOWN_MEMORY_WORKERS))
        else:
            limits.append(("memory", max(1, int(budget * MEMORY_HEADROOM / per_worker_bytes))))

    if requested > 1:
        limits.append(("requested", requested))

    chosen, name = min((value, label) for label, value in limits)
    if chosen <= 1:
        return WorkerPlan(1, f"sequential: {name} allows only one")
    return WorkerPlan(chosen, f"{chosen} workers, limited by {name}")
