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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data quality report for the NSE panel")
    parser.add_argument("--symbols", nargs="*", default=None, help="Default: the most liquid names")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--lake", default=None)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")

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

    verdict = "CLEAN" if critical == 0 else f"{critical} CRITICAL finding(s)"
    print(f"-- quality gate: {verdict}")
    return 1 if critical else 0


if __name__ == "__main__":
    sys.exit(run())
