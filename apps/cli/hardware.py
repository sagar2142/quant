"""What this machine is, and how the system will use it — §13.4.

    python -m apps.cli.hardware
    python -m apps.cli.hardware --panel-mb 400

**A speed knob nobody can inspect is a rumour.** The parallel plan is derived
from cores, memory and the size of the work, and when a run is slower than
expected the first question is which of those decided. Printing the answer is
cheaper than every other way of finding out.

It also says plainly what a GPU would and would not do here, because that is
the question every new operator asks and the honest answer is not "yes".
"""

from __future__ import annotations

import argparse
import sys

from apps.cli.pool import MIN_JOBS_FOR_POOL, _threads_per_worker
from ops.hardware import detect, plan_workers

__all__ = ["run"]

#: A three-year NSE panel, as measured. The default because it is the payload
#: a gauntlet actually ships to each worker.
DEFAULT_PANEL_MB = 131

#: Job counts worth showing a plan for: one gauntlet's sweep, its dropout and
#: placebo samples, and a large sweep.
EXAMPLE_JOBS = (1, 4, 16, 65, 200)

RULE = "─" * 74


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the detected hardware and the work plan")
    parser.add_argument(
        "--panel-mb",
        type=float,
        default=DEFAULT_PANEL_MB,
        help="Megabytes each worker will hold. Default is a three-year NSE panel.",
    )
    parser.add_argument(
        "--no-gpu-probe",
        action="store_true",
        help="Skip the GPU probe, which may shell out to nvidia-smi.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hardware = detect(probe_gpu=not args.no_gpu_probe)
    per_worker = int(args.panel_mb * 1_000_000)

    print(RULE)
    print("  HARDWARE")
    print(RULE)
    print(hardware.format())

    print(f"\n{RULE}")
    print(f"  PARALLEL PLAN  (each worker holding {args.panel_mb:.0f}MB)")
    print(RULE)
    for jobs in EXAMPLE_JOBS:
        plan = plan_workers(hardware, jobs=jobs, per_worker_bytes=per_worker)
        threads = _threads_per_worker(plan.workers)
        detail = f"{plan.workers} x {threads} thread(s)"
        print(f"  {jobs:>4} job(s)   {detail:<18}{plan.reason}")
    print(f"\n  Below {MIN_JOBS_FOR_POOL} jobs a pool costs more than it saves.")

    print(f"\n{RULE}")
    print("  ACCELERATION")
    print(RULE)
    if hardware.has_gpu:
        print("  A CUDA device is present.")
        print("  Nothing uses it yet, and the backtest loop never will — see below.")
    else:
        print("  No CUDA device detected.")
    print(
        "\n  The backtest event loop cannot use a GPU, and this is structural\n"
        "  rather than a matter of effort:\n"
        "    - it is a sequential state machine; bar T+1's cash depends on\n"
        "      bar T's fills, so there is nothing to run in parallel across bars\n"
        "    - money is Decimal (14.1.2) and a GPU has no decimal type; float64\n"
        "      would reintroduce the representation error the ledger forbids\n"
        "    - a bar is a few dozen names, well under kernel-launch cost\n"
        "\n  What parallelises is whole *runs* — the gauntlet's sweep, dropout\n"
        "  and placebo samples — and that is what the plan above sizes.\n"
        "\n  Work that would suit a GPU, none of it wired: the factor library\n"
        "  over the full panel, implied-vol inversion across an option surface,\n"
        "  and Monte Carlo resampling. Each would need its result checked\n"
        "  against the CPU path before being trusted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
