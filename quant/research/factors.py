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
            Factor.RESIDUAL_MOMENTUM: (
                "Momentum of the return that market beta does not explain "
                "(Blitz, Huij and Martens). Cleaner than raw momentum, which "
                "pays partly for having held high-beta names in a rising market."
            ),
            Factor.IDIOSYNCRATIC_VOL: (
                "Negated volatility of the residual, so a high score is a name "
                "that is quiet for its own reasons (Ang, Hodrick, Xing and "
                "Zhang). This is the low-volatility anomaly as documented; raw "
                "volatility instead rewards anything that barely moves."
            ),
            Factor.BETA: (
                "Negated trailing beta: betting against beta (Frazzini and "
                "Pedersen). Low-beta names have historically outperformed on a "
                "risk-adjusted basis, which the CAPM says should not happen."
            ),
            Factor.DOWNSIDE_BETA: (
                "Negated beta measured only on sessions when the market fell "
                "(Ang, Chen and Xing). A name can carry ordinary beta and far "
                "worse downside beta, and only the second one hurts."
            ),
            Factor.MOMENTUM_12_7: (
                "Return from 252 to 126 sessions ago — the intermediate horizon "
                "only (Novy-Marx). The claim is that momentum lives in the "
                "older half of the window, not the recent half."
            ),
            Factor.MAX_RETURN: (
                "Negated largest single-session return of the past month: "
                "lottery demand (Bali, Cakici and Whitelaw). Investors overpay "
                "for names that recently spiked, and those subsequently "
                "underperform."
            ),
            Factor.SEASONALITY: (
                "Average return in this calendar month across prior years "
                "(Heston and Sadka). Same-month returns persist far more than "
                "a random walk allows."
            ),
        }[self]

    @property
    def needs_residuals(self) -> bool:
        """Whether the factor requires the beta regression.

        Checked so the regression runs once per study and only when something
        actually reads it — it is the expensive part of building a factor.
        """
        return self in {
            Factor.RESIDUAL_MOMENTUM,
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


def _daily_return() -> pl.Expr:
    return pl.col("close") / pl.col("close").shift(1).over("symbol") - 1


def _price_expressions() -> dict[Factor, pl.Expr]:
    """Factors computed from price and volume alone.

    A table rather than a branch chain: each entry is one line of arithmetic
    and reads as a definition, which is what these are.
    """
    close = pl.col("close")
    by = "symbol"
    daily = _daily_return()

    return {
        Factor.MOMENTUM_12_1: close.shift(21).over(by) / close.shift(252).over(by) - 1,
        Factor.MOMENTUM_6_1: close.shift(21).over(by) / close.shift(126).over(by) - 1,
        Factor.MOMENTUM_1M: close / close.shift(21).over(by) - 1,
        Factor.MOMENTUM_12_7: close.shift(126).over(by) / close.shift(252).over(by) - 1,
        Factor.REVERSAL_1D: -(close / close.shift(1).over(by) - 1),
        Factor.REVERSAL_5D: -(close / close.shift(5).over(by) - 1),
        Factor.VOLATILITY_60: -daily.rolling_std(60).over(by),
        Factor.MAX_RETURN: -daily.rolling_max(21).over(by),
        Factor.HIGH_52W_PROXIMITY: close / close.rolling_max(252).over(by),
        # Shifted a full year, so the current month never scores itself.
        Factor.SEASONALITY: (close / close.shift(21).over(by) - 1).shift(252).over(by),
        Factor.VOLUME_SHOCK: pl.col("volume") / pl.col("volume").rolling_mean(21).over(by),
        # Amihud, negated so that a high score means liquid.
        Factor.ILLIQUIDITY: -(daily.abs() / (close * pl.col("volume"))).rolling_mean(21).over(by),
    }


def _residual_expression(factor: Factor) -> pl.Expr:
    """Factors read from the columns `add_residuals` attached.

    Kept apart from the price factors because they have a precondition the
    others do not: the beta regression must already have run.
    """
    by = "symbol"

    if factor is Factor.BETA:
        return -pl.col("beta")
    if factor is Factor.RESIDUAL_MOMENTUM:
        # Cumulative residual return over the 12-1 window. Summed rather than
        # compounded: residuals are already excess of the market and small, and
        # compounding them implies a portfolio nobody holds.
        return pl.col("residual").rolling_sum(231).over(by).shift(21).over(by)
    if factor is Factor.IDIOSYNCRATIC_VOL:
        return -pl.col("residual").rolling_std(60).over(by)

    # Downside beta: co-movement on sessions when the market fell. The residual
    # is unused — what matters here is the exposure, not what is left after it.
    down = pl.when(pl.col("market_ret") < 0).then(pl.col("ret")).otherwise(None)
    down_market = pl.when(pl.col("market_ret") < 0).then(pl.col("market_ret")).otherwise(None)
    mean_down = down.rolling_mean(252, min_samples=30).over(by)
    mean_market = down_market.rolling_mean(252, min_samples=30).over(by)
    covariance = (down * down_market).rolling_mean(252, min_samples=30).over(by) - (
        mean_down * mean_market
    )
    variance = (down_market * down_market).rolling_mean(252, min_samples=30).over(by) - (
        mean_market**2
    )
    return -pl.when(variance > 0).then(covariance / variance).otherwise(None)


def _signal_expression(factor: Factor) -> pl.Expr:
    """The factor as a Polars expression over a symbol-sorted panel.

    Every `shift` is `.over("symbol")`, so a name with too little history
    yields null rather than silently reaching into the previous instrument's
    rows — the alignment bug that would otherwise be invisible.
    """
    price = _price_expressions().get(factor)
    return price if price is not None else _residual_expression(factor)


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
    if spec.factor.needs_residuals:
        # One regression per study, not per factor: it is the expensive part.
        from quant.research.residual import add_residuals  # noqa: PLC0415 - cycle

        panel = add_residuals(panel)
    scored = panel.with_columns(_signal_expression(spec.factor).alias("signal"))
    with_forward = add_forward_returns(scored, horizons)
    return (
        with_forward.select("event_time", "symbol", "signal", *[f"fwd_{h}" for h in horizons])
        .drop_nulls("signal")
        .filter(pl.col("signal").is_finite())
    )
