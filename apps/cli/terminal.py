"""Analytics terminal — MASTER_PLAN §12.6.

    python -m apps.cli.terminal RELIANCE
    python -m apps.cli.terminal RELIANCE TCS INFY --sessions 500
    python -m apps.cli.terminal --top 20

Two screens, both dense and both read from the panel:

    security      one name, fully decomposed — returns, risk, distribution,
                  stationarity, volatility regime
    cross-section two or more names — correlation, clusters, effective bets,
                  HRP and ERC weights

**Dense on purpose.** A quant screen is read by scanning columns, not by
clicking. Every number that fits on one screen without scrolling is a number
you compare against its neighbours for free, and the alternative — one figure
per card, generously spaced — makes comparison an act of memory.

**Nothing here recommends a trade.** The profile describes price behaviour;
the gauntlet decides whether a strategy built on it is evidence. Those are
different jobs and this module does only the first.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import numpy.typing as npt
import polars as pl

from apps.cli.backtest import build_universe, load_panel
from core.config import settings
from core.instruments import InstrumentId
from data.corpactions.actions import CorporateAction, back_adjust
from data.feeds.yahoo import YahooActionsLoader, YahooError
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from quant.analytics.crosssection import CrossSection, analyse_cross_section
from quant.analytics.security import SecurityProfile, profile_security
from quant.math.metrics.performance import returns_from_equity

RULE = "─" * 78


def pct(value: float | None, width: int = 9) -> str:
    """Percentages with a sign, or an em dash when the window was too short."""
    return f"{'—':>{width}}" if value is None else f"{value:>{width}.2%}"


def num(value: float, width: int = 9, places: int = 2) -> str:
    return f"{value:>{width}.{places}f}"


def money(value: float | None, width: int = 12) -> str:
    if value is None:
        return f"{'—':>{width}}"
    for unit, size in (("Cr", 1e7), ("L", 1e5), ("K", 1e3)):
        if abs(value) >= size:
            return f"{value / size:>{width - 2}.1f}{unit}"
    return f"{value:>{width}.0f}"


def series_for(
    history: pl.DataFrame,
    symbol: str,
    actions: dict[str, list[CorporateAction]] | None = None,
) -> pl.DataFrame:
    """One name's series, back-adjusted for corporate actions.

    **The panel stores raw prices deliberately** — the backtester applies
    actions to *positions*, which is what happens to a real holding and carries
    no look-ahead. Analytics is the other case: computing a 3-year return or a
    skew from raw closes reads a 1:1 bonus as a -50% day. On RELIANCE that
    produced a -49% three-year return, skew -6.9 and kurtosis 175, all of them
    artefacts of one 2024 corporate action.

    `back_adjust` contains future information by construction and must never
    reach a backtest. Describing what a security *did* is precisely the use it
    exists for (§9).
    """
    rows = (
        history.filter(pl.col("symbol") == symbol)
        .sort("event_time")
        .select("event_time", "open", "high", "low", "close", "volume")
    )
    if not actions or rows.is_empty():
        return rows
    for_symbol = actions.get(symbol, [])
    return back_adjust(rows, for_symbol) if for_symbol else rows


def load_actions(history: pl.DataFrame, symbols: list[str]) -> dict[str, list[CorporateAction]]:
    """Corporate actions per symbol, from the free Yahoo source.

    **Keyed by symbol, never by a single instrument id.** A symbol can carry
    more than one ISIN over its life — HDFCBANK is INE040A01026 until the 2019
    split changes its face value and INE040A01034 after, and 344 symbols in
    this panel have the same shape. Collapsing symbol to *one* id picked an
    arbitrary ISIN, and because the choice was not stable the same command
    returned a -9% three-year return on one run and -54% on the next.

    Yahoo is queried by ticker, so one request covers every ISIN the symbol has
    worn. The panel's price series is likewise continuous by symbol, which is
    what a chart of "HDFCBANK" means. §1.1 keeps ISIN as the stable identity
    for *positions and orders*; this is the display axis, and conflating the
    two is what produced the bug.
    """
    loader = YahooActionsLoader()
    lookup = dict(history.select("symbol", "instrument_id").unique().iter_rows())
    out: dict[str, list[CorporateAction]] = {}
    failed: list[str] = []
    for symbol in symbols:
        if symbol not in lookup:
            continue
        try:
            out[symbol] = loader.fetch(symbol, InstrumentId(lookup[symbol]))
        except YahooError:
            failed.append(symbol)
    if failed:
        print(f"  corporate actions unavailable for: {', '.join(sorted(failed))}")
        print("  those names are shown UNADJUSTED — splits will read as crashes")
    return out


def report_identity_changes(history: pl.DataFrame, symbols: list[str]) -> None:
    """Say when a symbol has worn more than one ISIN in the sample.

    Material rather than trivia: it marks a face-value change, a merger or a
    re-listing, and the price series either side of it may not describe the
    same economic thing.
    """
    counts = (
        history.filter(pl.col("symbol").is_in(symbols))
        .select("symbol", "instrument_id")
        .unique()
        .group_by("symbol")
        .agg(pl.len().alias("ids"))
        .filter(pl.col("ids") > 1)
    )
    for row in counts.sort("symbol").to_dicts():
        print(f"  note: {row['symbol']} carries {row['ids']} ISINs in this sample")
        print("        (face-value change, merger or re-listing)")


def print_security(profile: SecurityProfile) -> None:
    """The single-name screen."""
    p = profile
    print(f"\n{p.symbol:<12} {p.observations:,} sessions      last {p.last_close:,.2f}")
    print(RULE)

    horizons = "  ".join(f"{k:>4}{pct(v, 9)}" for k, v in p.horizon_returns.items())
    print(f"  RETURN     {horizons}")
    print(
        f"             CAGR{pct(p.cagr, 9)}"
        f"   52w hi {p.high_52w:>10,.2f}   lo {p.low_52w:>10,.2f}"
        f"   off hi{pct(p.off_high, 8)}"
    )
    print()
    print(
        f"  RISK       vol {pct(p.annual_volatility, 8)}"
        f"   maxDD {pct(p.max_drawdown, 8)}"
        f"   nowDD {pct(p.current_drawdown, 8)}"
        f"   ADV {money(p.adv_value)}"
    )
    print(
        f"             VaR5 {pct(p.var_5, 8)}"
        f"   CVaR5 {pct(p.cvar_5, 8)}"
        f"   tail {num(p.tail_ratio, 7)}"
        f"   hit {pct(p.hit_rate, 8)}"
    )
    print()
    print(
        f"  RATIO      Sharpe {num(p.sharpe, 7)}"
        f"   Sortino {num(p.sortino, 7)}"
        f"   Calmar {num(p.calmar, 7)}"
    )
    print(
        f"  SHAPE      skew {num(p.skewness, 8)}"
        f"   kurtosis {num(p.kurtosis, 7)}" + ("   FAT LEFT TAIL" if p.fat_left_tail else "")
    )
    print()
    s = p.stationarity
    print(
        f"  PROCESS    {s.verdict.value:<13} ADF p{num(s.adf_pvalue, 7, 4)}"
        f"   KPSS p{num(s.kpss_pvalue, 7, 4)}   Hurst{num(s.hurst, 7, 3)}"
    )
    lags = "  ".join(f"lag{k}{num(v, 7, 3)}" for k, v in p.autocorrelation.items())
    print(f"  AUTOCORR   {lags}")
    print()
    print(
        f"  VOL        realised {pct(p.realised_vol.annualised, 8)}"
        f"   ewma {pct(p.ewma_vol.annualised, 8)}"
        f"   regime {p.vol_regime}"
    )

    if p.is_implausible:
        print("\n  ** Sharpe above the 2.5 smell test (§2.1) — suspect a missed")
        print("     corporate action before believing this number **")
    if not s.tradable_as_mean_reversion:
        print(f"\n  note: not fadeable — {s.verdict.value.lower()} process, mean reversion")
        print("        has no level to revert to here (§253)")


def print_cross_section(section: CrossSection) -> None:
    """The universe screen."""
    print(f"\nCROSS-SECTION   {len(section.names)} names   {section.sessions:,} sessions")
    print(RULE)
    print(
        f"  market  return {pct(section.market_return, 9)}"
        f"   vol {pct(section.market_volatility, 8)}"
        f"   mean corr {num(section.mean_correlation, 6, 3)}"
    )
    print(
        f"  struct  clusters {len(section.clusters):>3}"
        f"   effective bets {num(section.effective_bets, 6, 2)}"
        f"   div ratio {num(section.diversification_ratio, 6, 2)}"
    )
    print(
        f"  matrix  shrinkage {num(section.shrinkage, 6, 3)}"
        f"   condition {section.condition_number:>12,.0f}"
        + ("   ILL-CONDITIONED" if section.is_ill_conditioned else "")
    )
    print()
    print(
        f"  {'symbol':<14}{'return':>9}{'vol':>9}{'sharpe':>8}"
        f"{'beta':>7}{'corr':>7}{'HRP':>8}{'ERC':>8}{'clu':>5}"
    )
    print(f"  {'-' * 75}")
    for n in section.ranked_by("total_return"):
        print(
            f"  {n.symbol:<14}{n.total_return:>8.1%}{n.annual_volatility:>9.1%}"
            f"{n.sharpe:>8.2f}{n.beta:>7.2f}{n.correlation_to_market:>7.2f}"
            f"{n.weight_hrp:>8.1%}{n.weight_erc:>8.1%}{n.cluster:>5}"
        )

    warning = section.concentration_warning
    if warning:
        print(f"\n  ** {warning} **")
    if section.is_ill_conditioned:
        print("\n  ** covariance is near-singular: treat every weight above as")
        print("     arbitrary, and add history or drop names (§268) **")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant analytics terminal")
    parser.add_argument("symbols", nargs="*", help="NSE symbols. Omit to use --top.")
    parser.add_argument("--top", type=int, default=0, help="Use the top-N liquid universe")
    parser.add_argument(
        "--sessions", type=int, default=0, help="Trailing sessions. 0 uses all history."
    )
    parser.add_argument("--lake", default=None)
    parser.add_argument(
        "--raw-prices",
        action="store_true",
        help="Skip corporate-action adjustment. Splits will read as crashes (§9).",
    )
    return parser.parse_args(argv)


def resolve_symbols(
    args: argparse.Namespace, store: PanelStore, history: pl.DataFrame
) -> list[str]:
    if args.symbols:
        return [s.upper() for s in args.symbols]
    if not args.top:
        return []
    universe = build_universe(store, args.top)
    lookup = dict(history.select("instrument_id", "symbol").unique().iter_rows())
    return [lookup[i] for i in universe if i in lookup]


def aligned_returns(
    history: pl.DataFrame,
    symbols: list[str],
    actions: dict[str, list[CorporateAction]] | None = None,
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Return matrix over sessions where *every* name traded.

    Inner join rather than per-name cleaning: a correlation computed across
    misaligned dates is noise that looks like structure.

    **Repeated symbols are collapsed, first occurrence winning.** Each name
    becomes a column keyed by its own symbol, so a repeat previously produced a
    second column that polars auto-suffixed — a cross-section holding the same
    instrument twice, correlating 1.0 with itself and understating the
    effective-bet count. That returned a plausible 200 rather than an error.
    Above two repeats the join collided outright and raised.
    """
    frames = []
    kept: list[str] = []
    for symbol in dict.fromkeys(symbols):
        rows = series_for(history, symbol, actions)
        if rows.height:
            frames.append(rows.select("event_time", pl.col("close").alias(symbol)))
            kept.append(symbol)
    if not frames:
        return [], np.empty((0, 0))

    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.join(frame, on="event_time", how="inner")
    joined = joined.sort("event_time")

    columns = [returns_from_equity(joined[s].to_list()) for s in kept]
    return kept, np.column_stack(columns) if columns else np.empty((0, 0))


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")

    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}\nRun: python -m apps.cli.ingest_nse --start 2019-01-01")
        return 1

    symbols = resolve_symbols(args, store, history)
    if not symbols:
        print("give one or more symbols, or --top N")
        return 1

    actions = {} if args.raw_prices else load_actions(history, symbols)
    report_identity_changes(history, symbols)

    if args.sessions:
        recent = history["event_time"].unique().sort().tail(args.sessions)
        history = history.filter(pl.col("event_time").is_in(recent.implode()))

    profiled = 0
    for symbol in symbols:
        rows = series_for(history, symbol, actions)
        if rows.is_empty():
            print(f"\n{symbol}: not in the panel")
            continue
        try:
            print_security(
                profile_security(symbol, rows["close"].to_list(), rows["volume"].to_list())
            )
            profiled += 1
        except ValueError as exc:
            print(f"\n{exc}")

    if len(symbols) > 1:
        kept, matrix = aligned_returns(history, symbols, actions)
        if matrix.size:
            try:
                print()
                print_cross_section(analyse_cross_section(kept, matrix))
            except ValueError as exc:
                print(f"\ncross-section unavailable: {exc}")

    print()
    return 0 if profiled else 1


if __name__ == "__main__":
    sys.exit(run())
