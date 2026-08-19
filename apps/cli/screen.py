"""Universe screener — MASTER_PLAN §6, §253.

    python -m apps.cli.screen --sort liquidity --limit 20
    python -m apps.cli.screen --sort reversal --min-adv 5e7
    python -m apps.cli.screen --stationary-only

Answers *which names*, rather than describing one you already named. That is
where research actually starts.

Two stages: a vectorised pass filters thousands of names on liquidity and
history in about a fifth of a second, then only the shortlist pays for ADF,
KPSS and Hurst. Screening everything deeply would take minutes to describe
names that cannot be traded.
"""

from __future__ import annotations

import argparse
import sys

from apps.cli.backtest import load_panel
from core.config import settings
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from quant.analytics.screener import (
    DEFAULT_MIN_ADV,
    DEFAULT_WINDOW,
    ScreenCriteria,
    ScreenResult,
    SortKey,
)
from quant.analytics.screener import screen_universe as run_screen

RULE = "─" * 86


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen the NSE universe")
    parser.add_argument(
        "--sort",
        choices=[k.value for k in SortKey],
        default=SortKey.LIQUIDITY.value,
        help="What to rank the shortlist by before deep statistics",
    )
    parser.add_argument("--limit", type=int, default=20, help="Names to profile deeply")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Sessions")
    parser.add_argument(
        "--min-adv",
        type=float,
        default=DEFAULT_MIN_ADV,
        help="Minimum median daily traded value, in rupees",
    )
    parser.add_argument(
        "--stationary-only",
        action="store_true",
        help="Only names a mean-reversion strategy could actually fade (§253)",
    )
    parser.add_argument(
        "--include-suspected-actions",
        action="store_true",
        help=(
            "Keep names with a >35%% session move. Off by default: the panel "
            "holds raw prices, so a bonus reads as a crash and would dominate "
            "a reversal screen."
        ),
    )
    parser.add_argument("--lake", default=None)
    return parser.parse_args(argv)


def print_result(result: ScreenResult) -> None:
    print(result.format())
    if not result.rows:
        return

    print()
    print(
        f"  {'symbol':<14}{'ADV Cr':>9}{'return':>9}{'vol':>8}{'sharpe':>8}"
        f"{'maxDD':>9}{'hurst':>8}  process"
    )
    print(f"  {'-' * 82}")
    for row in result.rows:
        p = row.profile
        if p is None:
            continue
        print(
            f"  {row.symbol:<14}{row.adv / 1e7:>9.0f}{row.window_return:>9.1%}"
            f"{p.annual_volatility:>8.1%}{p.sharpe:>8.2f}{p.max_drawdown:>9.1%}"
            f"{p.stationarity.hurst:>8.3f}  {row.verdict}"
            + ("  FADEABLE" if row.fadeable else "")
            + ("  SHARPE>2.5" if p.is_implausible else "")
            + ("  FAT-TAIL" if p.fat_left_tail else "")
        )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")

    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}\nRun: python -m apps.cli.ingest_nse --start 2019-01-01")
        return 1

    criteria = ScreenCriteria(
        window=args.window,
        min_adv=args.min_adv,
        sort_by=SortKey(args.sort),
        limit=args.limit,
        stationary_only=args.stationary_only,
        exclude_suspected_actions=not args.include_suspected_actions,
    )

    print(f"\nSCREEN  sort={criteria.sort_by.value}  window={criteria.window}  ")
    print(RULE)
    result = run_screen(history, criteria)
    print_result(result)

    implausible = sum(1 for r in result.rows if r.profile and r.profile.is_implausible)
    if implausible:
        print()
        print(f"  {implausible} name(s) above the 2.5 Sharpe smell test (§2.1). On a single")
        print("  name that usually means a missed corporate action, not a discovery —")
        print("  profile them individually before believing the number.")

    if criteria.stationary_only and not result.rows:
        print()
        print("  No liquid name is currently stationary enough to fade. That is a")
        print("  result, not an error: a z-score strategy has nothing to trade here,")
        print("  which is the risk §253 flags for strategy family 4.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(run())
