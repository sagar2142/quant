"""Factor arithmetic — MASTER_PLAN §6.

Split from `factors` because the two answer different questions. That module
says *what signals exist* and how a panel is prepared for them; this one is the
arithmetic of each, and it grew past the point where both fitted in one file
without the reader having to hold two subjects at once.

**Each entry is a definition, not a branch.** The price factors live in one
dispatch table so a factor reads as a line of arithmetic beside its name, which
is what these are. The residual factors are kept separate because they have a
precondition the others do not: the beta regression must already have run.
"""

from __future__ import annotations

import polars as pl

from quant.research.factors import Factor

__all__ = ["SEASONALITY_YEARS", "signal_expression"]


SEASONALITY_YEARS = 3


def _seasonality_expression() -> pl.Expr:
    """Mean return in this calendar month across prior years (Heston-Sadka).

    Averaged over several years, not read from one. A single year-ago monthly
    return is one noisy observation, and calling it a seasonality factor claims
    a statistic it is not.
    """
    by = "symbol"
    monthly = pl.col("close") / pl.col("close").shift(21).over(by) - 1
    # 252 sessions is a year; the same calendar month one, two and three years
    # back. Shifted so the current month never contributes to its own score.
    lagged = [monthly.shift(252 * (year + 1)).over(by) for year in range(SEASONALITY_YEARS)]
    return sum(lagged[1:], start=lagged[0]) / SEASONALITY_YEARS


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
        Factor.SEASONALITY: _seasonality_expression(),
        Factor.VOLUME_SHOCK: pl.col("volume") / pl.col("volume").rolling_mean(21).over(by),
        # Amihud, negated so that a high score means liquid.
        Factor.ILLIQUIDITY: -(daily.abs() / (close * pl.col("volume"))).rolling_mean(21).over(by),
        # ── pre-registered (§5.1) ───────────────────────────────────────────
        # Overnight is close-to-open, intraday open-to-close. The split matters
        # because the two are set by different participants: overnight prices
        # information arriving while the exchange is shut, intraday is largely
        # liquidity provision to the flow that creates.
        Factor.OVERNIGHT_MOMENTUM: (
            (pl.col("open") / close.shift(1).over(by) - 1).rolling_mean(21).over(by)
        ),
        Factor.INTRADAY_MOMENTUM: (close / pl.col("open") - 1).rolling_mean(21).over(by),
        # Traded value per transaction: the institutional footprint retail
        # churn does not leave. `trades` is in the panel and no other factor
        # reads it.
        Factor.AVG_TRADE_SIZE: (
            (close * pl.col("volume") / pl.col("trades").clip(1)).rolling_mean(21).over(by)
        ),
        # Transactions against their own recent norm, net of the value shock —
        # many small orders rather than repositioning.
        Factor.TRADE_COUNT_SHOCK: -(
            pl.col("trades") / pl.col("trades").rolling_mean(21).over(by).clip(1)
        ),
        # Parkinson high-low estimator over close-to-close. Above one means the
        # session churned intraday and closed near where it opened.
        Factor.RANGE_VOL_RATIO: (
            ((pl.col("high") / pl.col("low")).log()).rolling_mean(21).over(by)
            / daily.abs().rolling_mean(21).over(by).clip(1e-9)
        ),
        # How much of the lookback was spent rising, not how far it travelled.
        # A smooth path attracts less arbitrage attention than one jump.
        Factor.MOMENTUM_CONSISTENCY: (daily > 0).cast(pl.Float64).rolling_mean(252).over(by),
        # Change in trailing return: the second derivative turns before the
        # level does.
        Factor.MOMENTUM_ACCELERATION: (
            (close / close.shift(21).over(by) - 1)
            - (close.shift(21).over(by) / close.shift(42).over(by) - 1)
        ),
        # A different anchor from the 52-week high, watched by different people.
        Factor.MA200_DISTANCE: close / close.rolling_mean(200).over(by) - 1,
        # Volatility of volatility: uncertainty about how risky the position
        # will be, distinct from the risk itself.
        Factor.VOL_OF_VOL: -daily.rolling_std(21).over(by).rolling_std(63).over(by),
        # Downside dispersion against upside. Investors pay for protection, so
        # asymmetric names should trade at a discount.
        # `min_samples` is load-bearing: masking to one side leaves nulls
        # scattered through the window, and polars' default demands a full
        # window of non-nulls, so every value would be null. Ten of sixty-three
        # is enough for a dispersion and few enough that a quiet name still
        # scores.
        Factor.SEMI_DEVIATION_RATIO: (
            pl.when(daily < 0).then(daily).otherwise(None).rolling_std(63, min_samples=10).over(by)
            / pl.when(daily > 0)
            .then(daily)
            .otherwise(None)
            .rolling_std(63, min_samples=10)
            .over(by)
            .clip(1e-9)
        ),
        # The mirror of high proximity, and not assumed to be its negative: the
        # disposition effect near lows is a different bias from anchoring near
        # highs.
        Factor.LOW_52W_PROXIMITY: -(close / close.rolling_min(252).over(by)),
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
    if factor is Factor.RESIDUAL_REVERSAL:
        # Five-day reversal on residuals rather than raw returns. Raw reversal
        # mixes name-specific liquidity provision with market-wide reversal,
        # and only the first is a payment for absorbing an imbalance.
        return -pl.col("residual").rolling_sum(5).over(by)

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


def signal_expression(factor: Factor) -> pl.Expr:
    """The factor as a Polars expression over a symbol-sorted panel.

    Every `shift` is `.over("symbol")`, so a name with too little history
    yields null rather than silently reaching into the previous instrument's
    rows — the alignment bug that would otherwise be invisible.
    """
    price = _price_expressions().get(factor)
    return price if price is not None else _residual_expression(factor)
