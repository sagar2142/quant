"""Guided tour — MASTER_PLAN §12.6.

    python -m apps.cli.tutorial

**What a new operator lacks is not command names — it is the order of
operations.** This system exists to reject ideas, and someone who starts at the
backtester will read a rising equity curve as a discovery rather than as the
first of twelve questions.

So this prints the *loop*, cheapest stage first, and says at each step what
failing there means. Failure is the expected outcome at most stages, and a tour
that does not say so teaches the wrong lesson.

The same content is on the console's Tutorial screen. Two surfaces, one
sequence — a tour that disagreed with the one on screen would be worse than
having only one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from textwrap import indent

RULE = "─" * 78


@dataclass(frozen=True)
class Stage:
    number: int
    title: str
    cost: str
    what: str
    command: str
    failure: str


STAGES = (
    Stage(
        1,
        "Get data",
        "~40 min cold, seconds daily",
        "NSE bhavcopy lists every security that traded each session, so a stock\n"
        "  delisted in 2022 is still there up to its last day. Unioning the archive\n"
        "  gives a survivorship-bias-free universe from free files.",
        "python -m apps.cli.ingest_nse --start 2019-01-01",
        "Interrupted? Re-run it. Stored sessions are skipped, so a backfill resumes.",
    ),
    Stage(
        2,
        "Check the data",
        "seconds",
        "Gaps, stale prices, impossible bars, unexplained jumps. Run after every\n"
        "  ingest, before trusting any number computed from it.",
        "python -m apps.cli.quality",
        "Bad data does not raise later — it produces a plausible result, which is worse.",
    ),
    Stage(
        3,
        "Score a signal",
        "~6 seconds",
        "Does the idea predict anything? IC at four horizons, quintile buckets,\n"
        "  turnover, and whether the spread survives the 22bp NSE round trip.",
        "python -m apps.cli.factor --all",
        "DIES ON COSTS is normal — 10 of 16 do. Kill it here, not after an afternoon.",
    ),
    Stage(
        4,
        "Check for duplicates",
        "~2 seconds",
        "Combining two factors that measure the same thing counts one effect twice.\n"
        "  The 16-factor library is worth about six independent bets.",
        "python -m apps.cli.factor --overlap",
        "High overlap is not fatal; it says which factor to drop before combining.",
    ),
    Stage(
        5,
        "Combine what survived",
        "~3 seconds",
        "Z-score, remove overlap, weight by historical IC. A good composite scores\n"
        "  higher than any of its parts — that is the reason to combine at all.",
        "python -m apps.cli.factor --combine momentum_12_7,idiosyncratic_vol",
        "Weights are fitted in-sample. This is a candidate, not evidence.",
    ),
    Stage(
        6,
        "Backtest it",
        "minutes",
        "Now ask what it earns, with real Indian costs: STT both legs, DP charges,\n"
        "  market impact. A decision on bar T fills on bar T+1, structurally.",
        "python -m apps.cli.backtest --strategy momentum --top 30",
        "A strong IC routinely loses money. Fees ran 5.3% of capital over three years.",
    ),
    Stage(
        7,
        "Run the gauntlet",
        "~48 backtests",
        "Twelve checks: deflated Sharpe against every trial ever run, PBO,\n"
        "  walk-forward, universe dropout, a placebo of random entries.",
        "python -m apps.cli.validate --top 30",
        "Expect rejection. A 90% rejection rate means the gauntlet is working.",
    ),
    Stage(
        8,
        "Paper trade it",
        "6+ weeks of calendar",
        "One cycle per session, real prices, simulated fills. Measures drift between\n"
        "  what the backtest promised and what happens.",
        "python -m apps.cli.paper --top 30",
        "Exit 2 means HALTED on a reconciliation break. Only you clear it: --clear-halt.",
    ),
    Stage(
        9,
        "Only then, consider capital",
        "human judgement",
        "The pre-live checklist, executable: legal and tax position, kill switch\n"
        "  tested, drawdown ladder written while calm, capital you can lose entirely.",
        "python -m apps.cli.readiness",
        "Nothing trades by default. Four guards stand between this code and an order.",
    ),
)

PRINCIPLES = (
    (
        "The system exists to reject ideas",
        "Its own momentum strategy fails the gauntlet: dropout Sharpe -0.51,\n"
        "  parameter retention 0.44 against a 0.6 floor. A validation suite that\n"
        "  passes everything is not one.",
    ),
    (
        "Cheapest question first",
        "Six seconds, then minutes, then 48 backtests, then six weeks. Running\n"
        "  them in the wrong order is how a month disappears into one bad idea.",
    ),
    (
        "Wrong numbers look plausible",
        "A missed corporate action turned a +1.2% three-year return into -49.4%.\n"
        "  Cash ETFs contaminated a low-volatility factor. Neither raised an error.",
    ),
    (
        "Decimal for money, float for statistics",
        "If a broker could disagree with you about a number, it is Decimal. A lint\n"
        "  rejects float in the ledger, costs, risk and execution paths.",
    ),
)


def _block(text: str, pad: str) -> str:
    """Re-indent a multi-line description.

    Each line is stripped first, so the printer owns the layout and a stray
    space inside a source string cannot shift one line out of alignment with
    the rest.
    """
    return indent("\n".join(line.strip() for line in text.splitlines()), pad)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="How this system works")
    parser.add_argument("--principles", action="store_true", help="Why it is built this way")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print()
    print(RULE)
    if args.principles:
        print("WHY IT IS BUILT THIS WAY")
        print(RULE)
        for title, body in PRINCIPLES:
            print(f"\n  {title}")
            print(_block(body, "  "))
        print()
        return 0

    print("THE RESEARCH LOOP — cheapest question first")
    print(RULE)
    print("\n  Most ideas should die at stage 3, for six seconds of work.")
    print("  Reaching stage 8 with something is the exception, not the plan.")

    for stage in STAGES:
        print()
        print(f"  {stage.number}. {stage.title}".ljust(46) + stage.cost.rjust(30))
        print(_block(stage.what, "     "))
        print(f"\n     $ {stage.command}")
        print(f"     ! {stage.failure}")

    print()
    print(RULE)
    print("  Console: http://127.0.0.1:5173   (Tutorial screen, key 't')")
    print("  Why it is built this way: python -m apps.cli.tutorial --principles")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(run())
