"""Signal library and forward returns — MASTER_PLAN §6.

**The fast loop.** Until now the only way to evaluate an idea was to build a
strategy and backtest it: minutes per attempt, and the gauntlet costs roughly
forty-eight runs. That is the wrong tool for the first question, which is not
*"how much would this have made"* but *"does this predict anything at all"*.

A signal is scored here in about four hundred milliseconds across seventeen
hundred names. Most ideas should die in this loop; only survivors deserve a
backtest.

**Every factor is computed cross-sectionally and point-in-time.** A signal at
bar *t* uses only closes up to *t*, and forward returns start at *t+1*. Polars
`shift(...).over("symbol")` does the alignment per name, so a short series
produces nulls rather than borrowing another instrument's history.

**Signals here are price and volume only.** No earnings, no book value, no
shares outstanding — there is no clean free source for Indian fundamentals, so
value and quality factors are absent rather than approximated badly. What
remains is the technical sleeve most systematic equity books run anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import polars as pl

from core.config import settings

__all__ = [
    "FORWARD_HORIZONS",
    "Factor",
    "FactorSpec",
    "add_forward_returns",
    "build_factor",
    "prepare_panel",
]

#: Forward horizons scored by default, in sessions: a week, a month, a quarter,
#: and one day for the short-horizon reversal case.
FORWARD_HORIZONS: tuple[int, ...] = (1, 5, 21, 63)

#: Minimum bars a name needs before any factor is defined for it. One trading
#: year plus a month, so the 12-1 window has a full lookback.
MIN_BARS = 273

#: Indian ISINs encode the issuer type in their third character: INE is a
#: company, INF a mutual fund or ETF, IN9 a depositary receipt. The panel holds
#: whatever traded on NSE, which includes cash and liquid ETFs.
EQUITY_ISIN_MARKER = ":INE"


class Factor(str, Enum):
    """The signals this library can compute from OHLCV.

    Deliberately a closed set. A free-text formula field would let a typo
    become a discovery, and every member here is a documented published effect
    rather than something found by searching.
    """

    MOMENTUM_12_1 = "momentum_12_1"
    MOMENTUM_6_1 = "momentum_6_1"
    MOMENTUM_1M = "momentum_1m"
    REVERSAL_1D = "reversal_1d"
    REVERSAL_5D = "reversal_5d"
    VOLATILITY_60 = "volatility_60"
    HIGH_52W_PROXIMITY = "high_52w_proximity"
    VOLUME_SHOCK = "volume_shock"
    ILLIQUIDITY = "illiquidity"

    # ── residual factors (§6) ───────────────────────────────────────────────
    # Computed after stripping out market beta. Consistently stronger than
    # their raw equivalents, because the raw version pays partly for market
    # exposure and calls it alpha.
    RESIDUAL_MOMENTUM = "residual_momentum"
    IDIOSYNCRATIC_VOL = "idiosyncratic_vol"
    BETA = "beta"
    DOWNSIDE_BETA = "downside_beta"

    # ── published price/volume effects ──────────────────────────────────────
    MOMENTUM_12_7 = "momentum_12_7"
    MAX_RETURN = "max_return"
    SEASONALITY = "seasonality"

    # ── pre-registered questions (§5.1, apps.cli.preregister) ───────────────
    # Registered with success criteria fixed before any of them was computed.
    # None was in the library when the hypotheses were written, which is the
    # only reason they could be pre-registered at all: the sixteen above have
    # known results and cannot honestly be registered after the fact.
    OVERNIGHT_MOMENTUM = "overnight_momentum"
    INTRADAY_MOMENTUM = "intraday_momentum"
    AVG_TRADE_SIZE = "avg_trade_size"
    TRADE_COUNT_SHOCK = "trade_count_shock"
    RANGE_VOL_RATIO = "range_vol_ratio"
    MOMENTUM_CONSISTENCY = "momentum_consistency"
    MOMENTUM_ACCELERATION = "momentum_acceleration"
    MA200_DISTANCE = "ma200_distance"
    VOL_OF_VOL = "vol_of_vol"
    SEMI_DEVIATION_RATIO = "semi_deviation_ratio"
    LOW_52W_PROXIMITY = "low_52w_proximity"
    RESIDUAL_REVERSAL = "residual_reversal"

    # ── fundamental factors (§6) ────────────────────────────────────────────
    #
    # The first signals here that are not price or volume. Every other member
    # of this enum is derived from the bhavcopy, and so was every hypothesis
    # the register has rejected -- which bounded what the research could find
    # rather than merely what it had tried.
    EARNINGS_YIELD = "earnings_yield"
    NET_MARGIN = "net_margin"

    @property
    def description(self) -> str:
        """What this factor is, and the published effect it comes from.

        Kept in `factor_docs` rather than inline: the prose is longer than the
        enum it annotates, and a reader looking for which factors exist should
        not have to scroll past three hundred lines of citation to find out.
        """
        from quant.research.factor_docs import DESCRIPTIONS  # noqa: PLC0415 - cycle

        return DESCRIPTIONS[self]

    @property
    def needs_fundamentals(self) -> bool:
        """Whether the factor requires quarterly results joined to the panel.

        Checked so the point-in-time join runs once per study and only when
        something reads it, exactly as `needs_residuals` gates the regression.
        """
        return self in {Factor.EARNINGS_YIELD, Factor.NET_MARGIN}

    @property
    def needs_residuals(self) -> bool:
        """Whether the factor requires the beta regression.

        Checked so the regression runs once per study and only when something
        actually reads it — it is the expensive part of building a factor.
        """
        return self in {
            Factor.RESIDUAL_MOMENTUM,
            Factor.RESIDUAL_REVERSAL,
            Factor.IDIOSYNCRATIC_VOL,
            Factor.BETA,
            Factor.DOWNSIDE_BETA,
        }


@dataclass(frozen=True)
class FactorSpec:
    """One factor and the universe filter it is scored on."""

    factor: Factor
    #: Median daily traded value below which a name is excluded. Applied before
    #: scoring, because an illiquid name produces a spectacular IC from stale
    #: prices and none of it is capturable.
    min_adv: float = 1e7
    #: Sessions of history used. 0 uses everything available.
    window: int = 0
    #: Make prices continuous across inferred splits before scoring.
    #:
    #: **On by default.** The panel stores raw closes, and 593 of 2,575 equity
    #: names carry at least one session move above 35% in the last thousand
    #: sessions. Every one enters momentum as a real return: a 1:1 bonus reads
    #: as -50%. Off only to measure what the contamination was worth.
    adjust_splits: bool = True
    #: Restrict to listed companies, excluding ETFs and mutual funds.
    #:
    #: **On by default, and it is not cosmetic.** NSE lists cash and liquid
    #: ETFs — LIQUID1, CASHIETF, LIQUIDPLUS — whose volatility is near zero by
    #: construction rather than by anomaly. Left in, they dominate any
    #: low-volatility factor: 48 of the top 60 names of a momentum plus
    #: low-volatility composite were money-market funds, whose "edge" is that
    #: they are not equities. A backtest on them would show a wonderful Sharpe
    #: and describe a savings account.
    equities_only: bool = True

    def __post_init__(self) -> None:
        if self.min_adv < 0:
            raise ValueError("min_adv cannot be negative")


def prepare_panel(history: pl.DataFrame, spec: FactorSpec) -> pl.DataFrame:
    """Liquid, sorted, windowed panel ready for factor construction.

    Liquidity is filtered first and on purpose: a thinly traded name has stale
    closes, stale closes autocorrelate, and autocorrelation manufactures an
    Information Coefficient that no order could ever capture.
    """
    frame = history
    if spec.window > 0:
        recent = frame["event_time"].unique().sort().tail(spec.window)
        frame = frame.filter(pl.col("event_time").is_in(recent.implode()))

    # Sorted once, here, rather than on the way out: the split inference needs
    # the order anyway, and sorting a panel this size is the most expensive
    # single step in the function. Every filter below preserves row order.
    frame = frame.sort(["symbol", "event_time"])

    if spec.adjust_splits:
        # Before the liquidity filter: a split moves both price and volume, so
        # an unadjusted traded-value median straddling one is wrong too.
        from data.corpactions.inferred import (  # noqa: PLC0415 - keeps data off the import path
            adjust_for_inferred_splits,
        )

        frame = adjust_for_inferred_splits(frame, already_sorted=True)

    if spec.equities_only:
        frame = frame.filter(pl.col("instrument_id").str.contains(EQUITY_ISIN_MARKER, literal=True))

    if spec.min_adv > 0:
        liquid = (
            frame.group_by("symbol")
            .agg((pl.col("close") * pl.col("volume")).median().alias("adv"))
            .filter(pl.col("adv") >= spec.min_adv)["symbol"]
        )
        frame = frame.filter(pl.col("symbol").is_in(liquid.implode()))

    # Enforced rather than merely declared: a name with fifty bars produces a
    # null momentum score that is silently dropped later, and the count of
    # names in a report would then include instruments that contributed
    # nothing.
    long_enough = (
        frame.group_by("symbol").agg(pl.len().alias("bars")).filter(pl.col("bars") >= MIN_BARS)
    )["symbol"]
    return frame.filter(pl.col("symbol").is_in(long_enough.implode()))


#: Prior years averaged by the seasonality factor. Three is what a seven-year
#: panel supports; more would drop most names for want of history.
def add_forward_returns(
    panel: pl.DataFrame, horizons: tuple[int, ...] = FORWARD_HORIZONS
) -> pl.DataFrame:
    """Attach forward returns at each horizon.

    **Forward means strictly after the decision bar.** `shift(-h)` reads bar
    `t+h` against bar `t`, so the return being predicted begins after the
    signal is observable. Using `t` in both would score a signal against a
    return it already contains, which is the most common way a factor study
    reports an edge that does not exist.
    """
    return panel.with_columns(
        [
            (pl.col("close").shift(-h).over("symbol") / pl.col("close") - 1).alias(f"fwd_{h}")
            for h in horizons
        ]
    )


def build_factor(
    history: pl.DataFrame,
    spec: FactorSpec,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pl.DataFrame:
    """Panel with the signal and every forward return attached.

    Returns:
        Columns `event_time`, `symbol`, `signal`, `fwd_<h>`. Rows where the
        signal is undefined are dropped; rows where only *some* forward
        horizons exist are kept, because the tail of the sample legitimately
        has a 5-day forward return and not a 63-day one, and discarding it
        would throw away the most recent evidence.
    """
    panel = prepare_panel(history, spec)
    if spec.factor.needs_fundamentals:
        # The point-in-time join, once per study. Deliberately after
        # `prepare_panel`: the liquidity filter runs first, so this joins
        # filings onto the names a book could actually hold rather than onto
        # every listing on the exchange.
        from quant.research.fundamentals import attach_fundamentals  # noqa: PLC0415 - cycle

        panel = attach_fundamentals(panel, settings.lake)
    if spec.factor.needs_residuals:
        # One regression per study, not per factor: it is the expensive part.
        from quant.research.residual import add_residuals  # noqa: PLC0415 - cycle

        panel = add_residuals(panel)
    from quant.research.expressions import (  # noqa: PLC0415 - breaks a cycle
        signal_expression,
    )

    scored = panel.with_columns(signal_expression(spec.factor).alias("signal"))
    with_forward = add_forward_returns(scored, horizons)
    return (
        with_forward.select(
            "event_time",
            # Identity travels with the signal. A symbol is not identity
            # (§3.3) — 344 of them map to more than one ISIN on this panel —
            # and anything joining a factor score back to a position needs
            # the key that cannot collide.
            "instrument_id",
            "symbol",
            "signal",
            *[f"fwd_{h}" for h in horizons],
        )
        .drop_nulls("signal")
        .filter(pl.col("signal").is_finite())
    )
