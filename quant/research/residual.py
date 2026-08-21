"""Residual factors — MASTER_PLAN §6.

**The distinction between a retail technical signal and a desk-grade one.**

A raw price factor measures two things at once: what is specific to the name,
and how much market the name happens to carry. Sorting on raw volatility ranks
instruments by how little they move — which is why a momentum-plus-low-vol
composite on this panel selected cash ETFs, whose stillness is structural
rather than an anomaly. Sorting on *residual* volatility ranks them by how
much they move for reasons of their own, which is the effect the literature
actually documents.

One regression produces four published factors:

    beta                 Frazzini-Pedersen, betting-against-beta
    residual momentum    Blitz, Huij and Martens — momentum after beta
    idiosyncratic vol    Ang, Hodrick, Xing and Zhang — the real low-vol effect
    downside beta        Ang, Chen and Xing — beta when the market falls

**The market is the equal-weight universe, not an index.** The panel holds no
index, and an external one would describe a different universe than the book
trades. §5.4's regime work makes the same choice for the same reason.

**Rolling and trailing.** Beta at bar *t* uses the window ending at *t*, so a
name's exposure is what was estimable then rather than what it turned out to
be. A full-sample beta would leak the future into every residual computed from
it.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = [
    "BETA_WINDOW",
    "add_market_return",
    "add_residuals",
    "rolling_beta_column",
]

#: Sessions used to estimate beta. A year: long enough for the slope to settle,
#: short enough that a name changing character shows up within a year.
BETA_WINDOW = 252

#: Below this a beta estimate is noise, and the residual built on it is worse
#: than the raw series it replaced.
MIN_BETA_BARS = 60


def add_market_return(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach the equal-weight cross-sectional return of each session.

    Equal weight rather than value weight: the panel carries no shares
    outstanding, and a traded-value weighting would let the few largest names
    define "the market" for a book that holds thirty of them.
    """
    with_returns = panel.sort(["symbol", "event_time"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret")
    )
    market = (
        with_returns.drop_nulls("ret")
        .group_by("event_time")
        .agg(pl.col("ret").mean().alias("market_ret"))
    )
    return with_returns.join(market, on="event_time", how="left")


def _rolling_beta(
    stock: npt.NDArray[np.float64], market: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Trailing beta, NaN until the window fills.

    Written as cumulative sums rather than a sliding-window regression: one
    pass over each name instead of one regression per bar, which is the
    difference between a factor that builds in seconds and one that does not
    build at all on 1,700 names.
    """
    size = stock.size
    out = np.full(size, np.nan, dtype=np.float64)
    if size < window:
        return out

    # Pad so every window is a difference of two prefix sums.
    ones = np.r_[0.0, np.cumsum(np.ones(size))]
    sx = np.r_[0.0, np.cumsum(market)]
    sy = np.r_[0.0, np.cumsum(stock)]
    sxx = np.r_[0.0, np.cumsum(market * market)]
    sxy = np.r_[0.0, np.cumsum(market * stock)]

    hi = np.arange(window, size + 1)
    lo = hi - window
    n = ones[hi] - ones[lo]
    mx = (sx[hi] - sx[lo]) / n
    my = (sy[hi] - sy[lo]) / n
    covariance = (sxy[hi] - sxy[lo]) / n - mx * my
    variance = (sxx[hi] - sxx[lo]) / n - mx * mx

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(variance > 0, covariance / variance, np.nan)
    out[window - 1 :] = beta
    return out


def rolling_beta_column(panel: pl.DataFrame, window: int = BETA_WINDOW) -> pl.DataFrame:
    """Attach trailing beta and the residual return per name.

    Residual = actual return - beta * market return. What is left is the part
    of the move the market does not explain, and every factor computed on it
    describes the name rather than its exposure.
    """
    if window < MIN_BETA_BARS:
        raise ValueError(f"beta window {window} is below the {MIN_BETA_BARS} minimum")

    framed = add_market_return(panel)
    ordered = framed.sort(["symbol", "event_time"])

    stock = ordered["ret"].fill_null(0.0).to_numpy().astype(np.float64)
    market = ordered["market_ret"].fill_null(0.0).to_numpy().astype(np.float64)
    symbols = ordered["symbol"].to_numpy()

    # Per-name slices, so a beta never regresses one instrument on another's
    # history at the boundary between them.
    betas = np.full(stock.size, np.nan, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, symbols[1:] != symbols[:-1]])
    bounds = np.r_[starts, stock.size]
    for lo, hi in pairwise(bounds):
        betas[lo:hi] = _rolling_beta(stock[lo:hi], market[lo:hi], window)

    return ordered.with_columns(
        pl.Series("beta", betas),
        (pl.col("ret") - pl.Series("beta", betas) * pl.col("market_ret")).alias("residual"),
    )


def add_residuals(panel: pl.DataFrame, window: int = BETA_WINDOW) -> pl.DataFrame:
    """Panel with `ret`, `market_ret`, `beta` and `residual` attached.

    The single entry point the factor library uses. Computed once per study
    rather than per factor, because the regression is the expensive part and
    four factors read the same output.
    """
    return rolling_beta_column(panel, window)
