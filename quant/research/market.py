"""Market beta against a real index — MASTER_PLAN §6, §8.

**Why this exists.** `build_risk_model` gave every name a market exposure of
exactly one. That is the Barra country-factor construction and it is not wrong,
but it makes the market factor's return the *equal-weighted* cross-sectional
average of everything that traded — roughly two thousand NSE names, most of
them small. That is not the market anyone hedges against, and a beta measured
against it is a beta to an equal-weight smallcap basket wearing the word
"market".

The research so far turns on a signal being 72.7% market beta. This module is
what makes that number mean what it says.

**Exposure is beta, not one.** With a real index the honest market exposure is
each name's sensitivity to it, estimated over a trailing window. The regression
slope on that column is then the return of a unit-beta portfolio — a market
return — and a long-only book's risk lands on the market factor in proportion
to how market-sensitive its holdings actually are.

**Beta is deliberately not z-scored.** Every style exposure in the risk model is
standardised cross-sectionally, which forces it to sum to zero. Doing that to
beta would turn "moves with the market" into "moves more with the market than
its peers do" and reintroduce exactly the bug this replaces: an equal-weight
book of everything would once again show no market exposure at all.

**Estimated point-in-time.** The beta for a session uses only returns up to and
including that session, so a backtest cannot borrow a beta computed from its
own future.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "BETA_CLIP",
    "BETA_WINDOW",
    "MIN_BETA_SESSIONS",
    "index_returns",
    "market_betas",
]

#: Sessions of history a beta is estimated over. A year is the usual choice:
#: long enough that a single volatile week does not dominate, short enough that
#: a company which changed what it is does not carry its old sensitivity
#: forever.
BETA_WINDOW = 252

#: Fewest observations that will produce a beta at all. Below this the estimate
#: is noise, and a newly listed name is left without one rather than being
#: assigned a fabricated exposure. It drops out of the regression for those
#: sessions, which is the truth: nothing yet says how it moves with the market.
MIN_BETA_SESSIONS = 60

#: Betas are clipped to this range. A beta of 14 is not a stock that moves
#: fourteen times the market; it is a stock whose returns are dominated by
#: something idiosyncratic — a corporate action, a thin book, a price band —
#: and left unclipped it would drag a whole session's regression with it.
BETA_CLIP = 3.0


def index_returns(series: pl.DataFrame, price: str = "close") -> pl.DataFrame:
    """Session returns of one index series.

    Args:
        series: Frame with `event_time` and a price column, one row per session.
        price: The price column to difference.

    Returns:
        `event_time`, `market_return`, sorted, with the first session dropped —
        it has no prior close and therefore no return.
    """
    if series.is_empty():
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "market_return": pl.Float64(),
            }
        )

    return (
        series.sort("event_time")
        .select(
            "event_time",
            (pl.col(price) / pl.col(price).shift(1) - 1.0).alias("market_return"),
        )
        .drop_nulls()
    )


def market_betas(
    history: pl.DataFrame,
    index_series: pl.DataFrame,
    window: int = BETA_WINDOW,
    min_periods: int = MIN_BETA_SESSIONS,
) -> pl.DataFrame:
    """Trailing beta of every instrument against the index.

    Args:
        history: Panel frame with `event_time`, `instrument_id`, `close`.
        index_series: The benchmark, with `event_time` and `close`.
        window: Trailing sessions the covariance is measured over.
        min_periods: Fewest observations that yield a beta.

    Returns:
        `event_time`, `instrument_id`, `beta` — one row per instrument per
        session, with sessions that have no estimable beta absent rather than
        filled. Empty if either input is empty or they do not overlap.

    Note:
        The join is inner on `event_time`, so a session the index does not cover
        yields no betas at all. That is the intended failure: a partially
        backfilled index should visibly shrink the model's history rather than
        silently estimating some names against a market and others against
        nothing.
    """
    if history.is_empty() or index_series.is_empty():
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "instrument_id": pl.String(),
                "beta": pl.Float64(),
            }
        )

    market = index_returns(index_series)
    if market.is_empty():
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "instrument_id": pl.String(),
                "beta": pl.Float64(),
            }
        )

    returns = (
        history.select("event_time", "instrument_id", "close")
        .sort(["instrument_id", "event_time"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1.0).alias(
                "asset_return"
            )
        )
        .drop_nulls("asset_return")
        .join(market, on="event_time", how="inner")
    )
    if returns.is_empty():
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                "instrument_id": pl.String(),
                "beta": pl.Float64(),
            }
        )

    # Rolling cov / rolling var, per instrument. Polars has no rolling
    # covariance, so it is written out: E[xy] - E[x]E[y] over the same window
    # that produces E[x] and E[y], which keeps numerator and denominator on
    # identical samples even where a name is missing sessions the index has.
    def trailing(expr: pl.Expr, name: str) -> pl.Expr:
        return (
            expr.rolling_mean(window_size=window, min_samples=min_periods)
            .over("instrument_id")
            .alias(name)
        )

    return (
        returns.sort(["instrument_id", "event_time"])
        .with_columns(
            trailing(pl.col("asset_return"), "_ma"),
            trailing(pl.col("market_return"), "_mm"),
            trailing(pl.col("asset_return") * pl.col("market_return"), "_mxy"),
            trailing(pl.col("market_return") ** 2, "_mxx"),
        )
        .with_columns(
            (pl.col("_mxx") - pl.col("_mm") ** 2).alias("_var"),
            (pl.col("_mxy") - pl.col("_ma") * pl.col("_mm")).alias("_cov"),
        )
        .filter(pl.col("_var") > 0)
        .select(
            "event_time",
            "instrument_id",
            (pl.col("_cov") / pl.col("_var")).clip(-BETA_CLIP, BETA_CLIP).alias("beta"),
        )
        .drop_nulls("beta")
        .sort(["event_time", "instrument_id"])
    )
