"""Research report — MASTER_PLAN §5.4, §12.6.

    python -m apps.cli.report --top 30 --out reports/momentum.html

Runs the backtest, runs the gauntlet, and writes **one self-contained HTML
file**: verdict, statistics, equity curve, drawdown, cost drag and the
parameter neighbourhood.

**This is the research surface the plan called for, built as an artifact rather
than a server.** §12.6 sketched it as Streamlit. A Streamlit session cannot be
kept, attached to an experiment row, or opened a year later when the question
is "why did I reject this?" — and §5 requires exactly that traceability. A file
can be committed next to the numbers it explains, and it needs no runtime.

**The verdict is printed above every chart, deliberately.** Rendering an equity
curve first invites "that looks good", which is the bias the twelve checks
exist to kill.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from apps.cli.runs import NSE_SESSIONS, Panel, run_one
from apps.cli.validate import assemble_inputs, load_market
from apps.report.charts import PALETTE, Series, area_chart, bar_chart, line_chart
from apps.report.page import (
    Panel as Block,
)
from apps.report.page import (
    ReportPage,
    render_page,
    table,
    verdict_row,
)
from core.clock import utc_now
from engine.validation import run_gauntlet
from engine.validation.report import GauntletReport
from quant.math.metrics.performance import summarise

#: Where reports land when no path is given. Committed alongside the experiment
#: they explain, or ignored — that is the operator's call, not this module's.
DEFAULT_OUT = Path("reports")


def drawdown_series(equity: list[float]) -> list[float]:
    """Fractional drawdown from the running peak, per bar."""
    peaks = np.maximum.accumulate(np.asarray(equity, dtype=np.float64))
    return list(np.asarray(equity, dtype=np.float64) / peaks - 1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an HTML research report")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--lake", default=None)
    parser.add_argument(
        "--sessions",
        type=int,
        default=0,
        help="Trailing sessions to test over. 0 (default) uses the whole panel.",
    )
    parser.add_argument("--dropout-samples", type=int, default=30)
    parser.add_argument("--placebo-samples", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None, help="Output .html path")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class RunOutcome:
    """Everything one validated run produced. Grouped so the page builder takes
    a result rather than seven loose positional arguments (§14.2)."""

    panel: Panel
    equity: list[float]
    returns: npt.NDArray[np.float64]
    report: GauntletReport
    neighbourhood: list[float]
    sweep_labels: list[str]


def build_page(outcome: RunOutcome, args: argparse.Namespace) -> ReportPage:
    panel, equity, report = outcome.panel, outcome.equity, outcome.report
    stats = summarise(outcome.returns, periods_per_year=NSE_SESSIONS)
    drawdown = drawdown_series(equity)

    failures = [f"{r.test}: {r.reason}" for r in report.failures]
    skipped = [r.test for r in report.results if r.skipped]
    if skipped:
        failures.append(f"not run: {', '.join(skipped)}")

    key = [
        ("total return", f"{equity[-1] / equity[0] - 1:.2%}", ""),
        ("sharpe", f"{stats.sharpe:.2f}", "ok" if stats.sharpe > 0 else "bad"),
        ("max drawdown", f"{min(drawdown):.2%}", "bad"),
        ("volatility", f"{stats.volatility:.2%}", ""),
        ("sessions", f"{len(equity):,}", ""),
        ("universe", f"{len(panel.universe)}", ""),
    ]

    blocks = [
        Block(
            "equity",
            line_chart([Series("equity", equity, PALETTE["accent"])]),
            "Nominal account value per session, costs deducted.",
        ),
        Block(
            "drawdown",
            area_chart(drawdown),
            "Distance below the running peak. The shape matters more than the "
            "depth: one deep trough is survivable, a permanently underwater "
            "curve is not.",
        ),
        Block(
            "parameter neighbourhood (check 6)",
            bar_chart(
                outcome.sweep_labels,
                outcome.neighbourhood,
                lambda v: f"{v:.2f}",
                threshold=0.0,
            ),
            "Sharpe at each swept configuration. A mesa survives; a single "
            "spike with collapse either side is a fitted artefact of one noise "
            "realisation.",
        ),
        Block(
            "the twelve checks",
            table(
                ["", "check", "statistic", "reason"],
                [
                    verdict_row(
                        r.test,
                        r.passed,
                        r.skipped,
                        "—" if r.statistic is None else f"{r.statistic:.4f}",
                        r.reason,
                    )
                    for r in report.results
                ],
            ),
            "All twelve must pass. Not most, not the important ones — all.",
        ),
    ]

    return ReportPage(
        title=f"momentum({args.lookback}/{args.skip}) · {len(panel.universe)} NSE names",
        subtitle=(
            f"generated {utc_now().isoformat(timespec='seconds')} · "
            f"seed fixed · costs = NseEquityCostModel"
        ),
        passed=report.passed,
        verdict_line=(
            "Every check that ran passed."
            if report.passed
            else f"Rejected at {report.first_failure}."
        ),
        failures=failures,
        stats=key,
        panels=blocks,
        footer=(
            "A 90%+ rejection rate is the system working. A strategy that "
            "sails through on the first attempt more likely indicates a broken "
            "gauntlet than a discovered edge."
        ),
    )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    market = load_market(args)
    if market is None:
        return 1
    panel, equity_curve = market

    baseline = run_one(panel, args.lookback, args.skip)
    if baseline.size == 0:
        print("no returns produced — the panel is shorter than the lookback")
        return 1

    print("assembling gauntlet inputs (this runs many backtests)...")
    inputs, neighbourhood, labels = assemble_inputs(panel, args, baseline)
    report = run_gauntlet(inputs, short_circuit=False)

    page = build_page(
        RunOutcome(
            panel=panel,
            equity=equity_curve,
            returns=baseline,
            report=report,
            neighbourhood=neighbourhood,
            sweep_labels=labels,
        ),
        args,
    )
    destination = args.out or DEFAULT_OUT / f"momentum_{args.lookback}_{args.skip}.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_page(page), encoding="utf-8")

    print(report.format())
    print(f"\nreport written: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
