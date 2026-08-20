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

    @property
    def description(self) -> str:
        return {
            Factor.MOMENTUM_12_1: (
                "Return from 252 to 21 sessions ago. The classic 12-1 window: "
                "the most recent month is skipped because short-horizon "
                "reversal runs against momentum there and including it "
                "measurably degrades the signal."
            ),
            Factor.MOMENTUM_6_1: (
                "Return from 126 to 21 sessions ago. The same construction as "
                "12-1 over half the lookback, so the two together show whether "
                "the effect is a long-horizon one or a recent-trend one."
            ),
            Factor.MOMENTUM_1M: (
                "Trailing one-month return, unskipped. Included precisely so "
                "the skip in 12-1 can be shown to matter."
            ),
            Factor.REVERSAL_1D: "Negated prior-session return. Fades one-day moves.",
            Factor.REVERSAL_5D: "Negated trailing week. Fades short-horizon moves.",
            Factor.VOLATILITY_60: (
                "Negated 60-session realised volatility, so a high score means "
                "low volatility — the direction the low-volatility anomaly pays."
            ),
            Factor.HIGH_52W_PROXIMITY: (
                "Close over the 252-session high. Near 1.0 means at highs; the "
                "anchoring effect says those keep running."
            ),
            Factor.VOLUME_SHOCK: ("Session volume over its 21-session average. Attention proxy."),
            Factor.ILLIQUIDITY: (
                "Amihud: |return| per rupee traded, negated so a high score is "
                "liquid. Illiquid names pay a premium that a retail book "
                "cannot actually collect, which is why the sign is worth seeing."
            ),
        }[self]


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

    if spec.equities_only:
        frame = frame.filter(pl.col("instrument_id").str.contains(EQUITY_ISIN_MARKER, literal=True))

    if spec.min_adv > 0:
        liquid = (
            frame.group_by("symbol")
            .agg((pl.col("close") * pl.col("volume")).median().alias("adv"))
            .filter(pl.col("adv") >= spec.min_adv)["symbol"]
        )
        frame = frame.filter(pl.col("symbol").is_in(liquid.implode()))

    return frame.sort(["symbol", "event_time"])


def _signal_expression(factor: Factor) -> pl.Expr:
    """The factor as a Polars expression over a symbol-sorted panel.

    Every `shift` is `.over("symbol")`, so a name with too little history
    yields null rather than silently reaching into the previous instrument's
    rows — the alignment bug that would otherwise be invisible.
    """
    close = pl.col("close")
    by = "symbol"

    if factor is Factor.MOMENTUM_12_1:
        return close.shift(21).over(by) / close.shift(252).over(by) - 1
    if factor is Factor.MOMENTUM_6_1:
        return close.shift(21).over(by) / close.shift(126).over(by) - 1
    if factor is Factor.MOMENTUM_1M:
        return close / close.shift(21).over(by) - 1
    if factor is Factor.REVERSAL_1D:
        return -(close / close.shift(1).over(by) - 1)
    if factor is Factor.REVERSAL_5D:
        return -(close / close.shift(5).over(by) - 1)
    if factor is Factor.VOLATILITY_60:
        daily = close / close.shift(1).over(by) - 1
        return -daily.rolling_std(60).over(by)
    if factor is Factor.HIGH_52W_PROXIMITY:
        return close / close.rolling_max(252).over(by)
    if factor is Factor.VOLUME_SHOCK:
        return pl.col("volume") / pl.col("volume").rolling_mean(21).over(by)
    # Amihud illiquidity, negated so that a high score means liquid.
    daily_return = (close / close.shift(1).over(by) - 1).abs()
    traded = close * pl.col("volume")
    return -(daily_return / traded).rolling_mean(21).over(by)


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
    scored = panel.with_columns(_signal_expression(spec.factor).alias("signal"))
    with_forward = add_forward_returns(scored, horizons)
    return (
        with_forward.select("event_time", "symbol", "signal", *[f"fwd_{h}" for h in horizons])
        .drop_nulls("signal")
        .filter(pl.col("signal").is_finite())
    )
