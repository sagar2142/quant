"""Factor research lab — MASTER_PLAN §6.

    python -m apps.cli.factor momentum_12_1
    python -m apps.cli.factor --all --sessions 1000
    python -m apps.cli.factor reversal_5d --horizon 5

**The fast loop.** Scores a signal directly against forward returns instead of
inferring its value from an equity curve. A study takes under a second; a
backtest takes minutes and the gauntlet takes roughly forty-eight of them. Most
ideas should die here.

What the output answers, in order:

    IC          does the signal rank names in the order they later perform
    decay       at which horizon — which sets the holding period
    quantiles   is it monotonic, or one lucky extreme bucket
    turnover    can the edge survive being traded that often (§7.1)

**A strong IC is not a strategy.** It is permission to build one and send it to
the gauntlet. The cost line at the bottom is the first thing that kills a
signal with a real but tiny edge.
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

from apps.cli.backtest import load_panel
from core.config import settings
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from quant.research.factors import FORWARD_HORIZONS, Factor, FactorSpec, build_factor
from quant.research.ic import FactorReport, analyse_factor

RULE = "─" * 78

#: NSE delivery round trip, from the hand-verified cost model (§7.1). Used only
#: to put the quantile spread in context — the real charge comes from
#: NseEquityCostModel during a backtest.
ROUND_TRIP_COST = 0.0022


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a signal against forward returns")
    parser.add_argument(
        "factor",
        nargs="?",
        choices=[f.value for f in Factor],
        help="Signal to score. Omit with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Score every factor in the library")
    parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Forward horizon for the quantile study, in sessions",
    )
    parser.add_argument(
        "--min-adv",
        type=float,
        default=1e7,
        help="Liquidity floor. Illiquid names manufacture IC from stale prices.",
    )
    parser.add_argument("--sessions", type=int, default=0, help="Trailing window. 0 uses all.")
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument("--lake", default=None)
    return parser.parse_args(argv)


def cost_verdict(report: FactorReport) -> str:
    """Whether the quantile spread survives trading at the signal's turnover.

    Crude by construction and useful anyway: it is the difference between a
    signal with a real edge and one with an edge smaller than its own
    transaction costs, which is most of them.
    """
    if report.turnover <= 0 or not report.quantiles:
        return "  cost check: not applicable"

    # Turnover is per session; a holding period of `horizon` sessions trades
    # roughly that fraction of the book each rebalance.
    per_rebalance = min(1.0, report.turnover * report.quantile_horizon)
    charge = per_rebalance * ROUND_TRIP_COST
    net = report.spread - charge
    verdict = "survives" if net > 0 else "DIES ON COSTS"
    return (
        f"  cost check: spread {report.spread:+.3%} - "
        f"{per_rebalance:.0%} turnover x {ROUND_TRIP_COST:.2%} round trip "
        f"= {net:+.3%} net  -> {verdict}"
    )


def study(history: pl.DataFrame, factor: Factor, args: argparse.Namespace) -> FactorReport:
    spec = FactorSpec(factor=factor, min_adv=args.min_adv, window=args.sessions)
    horizons = tuple(sorted({*FORWARD_HORIZONS, args.horizon}))
    scored = build_factor(history, spec, horizons)
    return analyse_factor(scored, factor.value, horizons, args.horizon, args.buckets)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.factor and not args.all:
        print("give a factor name, or --all")
        return 1

    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")
    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}\nRun: python -m apps.cli.ingest_nse --start 2019-01-01")
        return 1

    wanted = list(Factor) if args.all else [Factor(args.factor)]
    for factor in wanted:
        started = time.perf_counter()
        report = study(history, factor, args)
        elapsed = time.perf_counter() - started

        print()
        print(RULE)
        print(report.format())
        print(cost_verdict(report))
        print(f"  {factor.description}")
        print(f"  [{elapsed:.2f}s]")

    print()
    print("A strong IC is permission to build a strategy, not evidence one works.")
    print("Send survivors to: python -m apps.cli.validate")
    return 0


if __name__ == "__main__":
    sys.exit(run())
