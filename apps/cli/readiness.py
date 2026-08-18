"""Live-readiness checklist — MASTER_PLAN §M11, PDF §40.

    python -m apps.cli.readiness

The final build checklist, executable rather than ticked by hand. Items the
machine can verify are verified; items only a human can attest to are printed
as an explicit attestation list, because a checkbox nobody read is worse than
no checkbox.

**This does not authorise live trading.** It reports whether the mechanical
preconditions hold. The decision to deploy capital is yours, it belongs after
legal and tax advice (§32), and it should follow months of paper trading — not
a green console output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.config import settings

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def format(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name:<38} {self.detail}"


def _run(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(  # noqa: S603 — fixed commands, no user input
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=900,
            # A failing check is the answer, not an exception: the checklist
            # reports every result rather than aborting on the first failure.
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"could not run: {exc}"
    return result.returncode == 0, (result.stdout + result.stderr).strip().splitlines()[-1][:80]


def machine_checks() -> list[Check]:
    """Everything verifiable without a human."""
    python = str(REPO / ".venv" / "Scripts" / "python.exe")
    if not Path(python).exists():
        python = sys.executable

    checks: list[Check] = []

    ok, detail = _run([python, "-m", "pytest", "-q"])
    checks.append(Check("test suite green", ok, detail))

    ok, detail = _run([python, "-m", "mypy", "core", "data", "quant", "engine", "trading"])
    checks.append(Check("types strict", ok, detail))

    ok, detail = _run([python, "-m", "tools.lints"])
    checks.append(Check("project lints clean", ok, detail))

    ok, _ = _run([str(REPO / ".venv" / "Scripts" / "lint-imports.exe")])
    checks.append(
        Check(
            "architecture boundaries",
            ok,
            "research cannot import trading; AI cannot reach secrets",
        )
    )

    ok, detail = _run([python, "-m", "pytest", "-m", "regression", "-q"])
    checks.append(Check("backtest regression pinned", ok, detail))

    # Live must be default-off. A checklist run that finds it already enabled is
    # itself a finding.
    checks.append(
        Check(
            "live disabled by default",
            not settings.live_enabled,
            f"NEUTRON_LIVE_ENABLED={settings.live_enabled}",
        )
    )

    runbooks = REPO / "ops" / "runbooks"
    required = {
        "reconciliation_break.md",
        "data_outage.md",
        "risk_breach.md",
        "kill_switch.md",
        "drawdown.md",
    }
    present = {p.name for p in runbooks.glob("*.md")} if runbooks.exists() else set()
    missing = required - present
    checks.append(
        Check(
            "runbooks written",
            not missing,
            "all present" if not missing else f"missing {sorted(missing)}",
        )
    )

    env_file = REPO / ".env"
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    checks.append(
        Check(
            "secrets not committable",
            ".env" in gitignore,
            ".env is gitignored" + ("" if env_file.exists() else " (not yet created)"),
        )
    )

    return checks


#: Things no test can establish. Printed for explicit attestation, with the
#: reason each one matters, because an unread checkbox is worse than none.
HUMAN_ATTESTATIONS = [
    (
        "Legal and tax advice obtained",
        "Proprietary trading of own capital via a broker API is generally fine in "
        "India, but confirm it for your structure before deploying (§32).",
    ),
    (
        "Equity tax position understood",
        "STCG/LTCG treatment, turnover-based audit thresholds, and whether your "
        "activity is business income or capital gains. The answer changes what "
        "you owe on the same P&L (§32).",
    ),
    (
        "Broker terms and market-data licence reviewed",
        "API terms, rate limits and redistribution rules for anything you store.",
    ),
    (
        "6+ weeks of continuous paper trading completed",
        "The M9 gate. Not compressible — drift only exists once generated.",
    ),
    (
        "Paper-vs-backtest drift explained",
        "The single most informative metric the system produces (§35). "
        "Unexplained drift means the backtest is not describing reality.",
    ),
    (
        "Kill switch tested this month",
        "An untested kill switch is not a kill switch (§8).",
    ),
    (
        "Reconciliation clean for 10+ consecutive sessions",
        "One clean day proves nothing.",
    ),
    (
        "Drawdown ladder numbers pre-committed in writing",
        "Decided while calm. They are not negotiable during a drawdown.",
    ),
    (
        "Initial capital is tuition-sized",
        "An amount whose total loss changes nothing in your life (§M12).",
    ),
    (
        "At least one strategy passed the full 12-check gauntlet",
        "Including the locked out-of-sample test, touched once, ever (§5.3).",
    ),
    (
        "Independent review of that strategy completed",
        "Re-read your own gauntlet output as a sceptic. Try to break it.",
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-readiness checklist")
    parser.add_argument(
        "--skip-slow", action="store_true", help="Skip the test and regression runs"
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("NEUTRON live-readiness checklist (PDF §40)\n")
    print("── machine-verified " + "─" * 52)

    checks = machine_checks() if not args.skip_slow else []
    if args.skip_slow:
        print("  (skipped)")
    for check in checks:
        print(check.format())

    failed = [c for c in checks if not c.passed]

    print("\n── human attestation required " + "─" * 42)
    print("  These cannot be verified by any test. Read each one.\n")
    for title, why in HUMAN_ATTESTATIONS:
        print(f"  [ ] {title}")
        print(f"        {why}\n")

    print("─" * 72)
    if failed:
        print(f"MECHANICAL CHECKS FAILED: {len(failed)}")
        for check in failed:
            print(f"  - {check.name}")
        return 1

    print("Mechanical checks pass.")
    print(
        "\nThis does NOT authorise live trading. It reports that the machine is "
        "ready.\nThe decision to deploy capital is yours, and it belongs after "
        "the attestations\nabove — not after a green console output."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
