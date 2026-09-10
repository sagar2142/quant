"""Data quality report CLI — MASTER_PLAN §M2.

    python -m apps.cli.quality

Runs the quality checks over the NSE panel. Exits non-zero on any CRITICAL
finding, so it can gate a pipeline.

`expect_continuous=False`: NSE is a session market, so overnight and weekend
gaps are expected and are not defects. Which *dates* should exist is the session
calendar's question, not the bar spacing's (§1.2).
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from core.clock import as_decision_time, utc_now
from core.config import settings
from core.events import Timeframe
from data.quality.checks import check_bars
from data.store.bars import NoDataError
from data.store.panel import PanelStore

#: Default sample size. The most liquid names are the ones a strategy actually
#: holds, so their data quality is the quality that matters.
DEFAULT_SAMPLE = 10

#: Names the sector classification is expected to cover. The most liquid ones,
#: for the same reason the bar sample uses them: a classification that misses a
#: name nothing can trade has not missed anything that matters.
TRADED_UNIVERSE = 300


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data quality report for the NSE panel")
    parser.add_argument("--symbols", nargs="*", default=None, help="Default: the most liquid names")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--lake", default=None)
    parser.add_argument(
        "--venue",
        choices=["NSE", "BSE"],
        default="NSE",
        help="Exchange to read. BSE history begins 2024-01-01; NSE begins 2019.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue=args.venue)

    try:
        panel = store.view(as_of=as_decision_time(utc_now()))
    except NoDataError as exc:
        print(f"{exc}")
        print("Run: python -m apps.cli.ingest_nse --start 2024-01-01")
        return 1

    if panel.is_empty():
        print("panel is empty")
        return 1

    sessions = store.sessions()
    print(f"panel: {sessions[0]} -> {sessions[-1]} ({len(sessions)} sessions)\n")

    if args.symbols:
        targets = (
            panel.filter(pl.col("symbol").is_in(args.symbols))["instrument_id"].unique().to_list()
        )
    else:
        targets = (
            panel.with_columns((pl.col("close") * pl.col("volume")).alias("value"))
            .group_by("instrument_id")
            .agg(pl.col("value").median().alias("median_value"))
            .sort("median_value", descending=True)
            .head(args.sample)["instrument_id"]
            .to_list()
        )

    critical = 0
    for instrument_id in targets:
        rows = panel.filter(pl.col("instrument_id") == instrument_id).sort("event_time")
        report = check_bars(instrument_id, Timeframe.D1, rows, expect_continuous=False)
        print(report.format())
        print()
        critical += report.critical_count

    # The feeds that are not bars. Checked here rather than in their own
    # command because a gate an operator has to remember to run twice is a gate
    # that gets run once.
    critical += _check_stores(args, panel)

    verdict = "CLEAN" if critical == 0 else f"{critical} CRITICAL finding(s)"
    print(f"-- quality gate: {verdict}")
    return 1 if critical else 0


def _check_stores(args: argparse.Namespace, panel: pl.DataFrame) -> int:
    """Sectors, quarterly results and the announced calendar.

    Every branch of the research now reads one of these, and none of them was
    covered by a gate that only ever looked at bars.
    """
    from data.quality.stores import check_events, check_fundamentals, check_sectors  # noqa: PLC0415

    lake = args.lake if args.lake is not None else settings.lake

    # Coverage is judged against what is liquid enough to hold, not against the
    # whole panel: most of it is too thin to trade, and counting those names
    # would make an adequate classification look broken.
    traded = (
        panel.with_columns((pl.col("close") * pl.col("volume")).alias("value"))
        .group_by("instrument_id")
        .agg(pl.col("value").median().alias("median_value"))
        .sort("median_value", descending=True)
        .head(TRADED_UNIVERSE)["instrument_id"]
        .to_list()
    )
    universe = {str(i).split(":", 1)[-1] for i in traded}

    critical = 0
    print()
    for report in (check_sectors(lake, universe), check_fundamentals(lake), check_events(lake)):
        print(report.format())
        print()
        critical += report.critical_count
    return critical


if __name__ == "__main__":
    sys.exit(run())
