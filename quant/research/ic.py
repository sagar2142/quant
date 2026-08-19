"""Scoring a signal — MASTER_PLAN §5.4, §6.

Four questions, asked in the order that kills ideas fastest:

    IC          does the signal rank names in the order they later perform?
    quantiles   is the relationship monotonic, or driven by one extreme bucket?
    decay       over what horizon does the edge exist — which sets the holding
                period, rather than the holding period being chosen first?
    turnover    how fast does the signal churn, and does the edge survive the
                round-trip cost of trading it that often?

**Rank correlation, not Pearson.** A cross-section of equity returns has fat
tails, and one name up 300% would dominate a linear correlation. Spearman asks
only whether the ordering was right, which is the question a long-short book
actually depends on.

**IC is computed per session, then summarised.** The mean alone says nothing
about reliability — a mean IC of 0.03 built from wild swings is not the same
signal as a steady 0.03, and the information ratio and t-statistic are what
separate them.

**Nothing here is a verdict.** A strong IC is a reason to build a strategy and
send it to the gauntlet; it is not evidence the strategy will survive costs.
The turnover column exists to make that gap visible early.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "MIN_NAMES_PER_SESSION",
    "FactorReport",
    "ICSummary",
    "ICWindow",
    "QuantileRow",
    "analyse_factor",
    "information_coefficient",
    "quantile_returns",
    "rolling_ic",
    "signal_turnover",
]

#: A cross-section thinner than this cannot support a rank correlation worth
#: reading. Sessions below it are dropped rather than contributing noise.
MIN_NAMES_PER_SESSION = 20

#: Buckets used for the monotonicity check. Deciles need a wide cross-section;
#: five is readable and stable on a universe of a few hundred names.
DEFAULT_QUANTILES = 5

TRADING_DAYS = 252

#: Conventional two-sigma bar for the IC t-statistic.
SIGNIFICANT_T = 2.0

#: A spread needs a top and a bottom bucket to exist.
MIN_BUCKETS = 2

#: Two sessions is the minimum from which a standard deviation can be formed.
MIN_IC_SESSIONS = 2

#: A complete reshuffle of the ranking averages a half-unit move per name, so
#: doubling makes 1.0 mean "fully reordered every session".
TURNOVER_SCALE = 2

#: Sessions per rolling IC window. About a year: long enough for the mean to
#: settle, short enough to show a factor dying while it is happening.
ROLLING_WINDOW = 252


@dataclass(frozen=True)
class ICSummary:
    """The Information Coefficient at one forward horizon."""

    horizon: int
    mean: float
    std: float
    #: mean / std. The information ratio of the signal itself, before any
    #: portfolio construction — how reliably it ranks, not how much it earns.
    information_ratio: float
    t_stat: float
    hit_rate: float
    sessions: int

    @property
    def is_significant(self) -> bool:
        """|t| above 2, the conventional bar.

        Not a licence to trade: with thousands of sessions almost any
        persistent tilt clears t=2, which is why the magnitude of the mean IC
        and the cost check matter more than the significance flag.
        """
        return abs(self.t_stat) > SIGNIFICANT_T

    def format(self) -> str:
        return (
            f"  {self.horizon:>3}d   IC {self.mean:>+8.4f}   IR {self.information_ratio:>+7.3f}"
            f"   t {self.t_stat:>+7.2f}   hit {self.hit_rate:>6.1%}   n {self.sessions:>5,}"
        )


@dataclass(frozen=True)
class QuantileRow:
    """Forward return of one signal bucket."""

    quantile: int
    mean_forward_return: float
    names: int


@dataclass(frozen=True)
class FactorReport:
    """Everything the fast loop produces for one signal."""

    factor: str
    horizons: list[ICSummary]
    quantiles: list[QuantileRow]
    quantile_horizon: int
    turnover: float
    names: int
    sessions: int

    @property
    def primary(self) -> ICSummary | None:
        """The horizon the quantile study was run at."""
        return next((h for h in self.horizons if h.horizon == self.quantile_horizon), None)

    @property
    def spread(self) -> float:
        """Top bucket minus bottom bucket, at the primary horizon.

        The number a long-short book would earn per period before costs, and
        the one to compare against the round-trip cost of the turnover below.
        """
        if len(self.quantiles) < MIN_BUCKETS:
            return 0.0
        return self.quantiles[-1].mean_forward_return - self.quantiles[0].mean_forward_return

    @property
    def is_monotonic(self) -> bool:
        """Whether bucket returns rise with the signal.

        A monotonic staircase means the signal orders the whole cross-section.
        A lumpy chart with one strong extreme bucket usually means a handful of
        outliers, and it will not survive the universe-dropout check (§5.4).
        """
        values = [q.mean_forward_return for q in self.quantiles]
        return all(a <= b for a, b in pairwise(values))

    def format(self) -> str:
        lines = [
            f"{self.factor}   {self.names:,} names   {self.sessions:,} sessions",
            "",
            "  INFORMATION COEFFICIENT (rank, per session)",
        ]
        lines.extend(h.format() for h in self.horizons)
        lines.append("")
        lines.append(f"  QUANTILE FORWARD RETURN ({self.quantile_horizon}d)")
        for row in self.quantiles:
            bar = "#" * max(0, int(abs(row.mean_forward_return) * 400))
            lines.append(f"  Q{row.quantile}  {row.mean_forward_return:>+8.3%}  {bar[:40]}")
        lines.append("")
        lines.append(
            f"  spread Q{len(self.quantiles)}-Q1 {self.spread:>+.3%}"
            f"   monotonic {'yes' if self.is_monotonic else 'no'}"
            f"   turnover {self.turnover:.1%}/session"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class ICWindow:
    """Mean IC over one trailing window."""

    end: str
    ic: float
    sessions: int


def rolling_ic(scored: pl.DataFrame, horizon: int, window: int = ROLLING_WINDOW) -> list[ICWindow]:
    """Mean IC over trailing windows — does the factor still work?

    **A full-sample IC cannot answer that.** Momentum with an IC of +0.035 over
    seven years could have been +0.08 for five and zero for two, which is a
    factor that has been arbitraged away, and the average reports it as
    healthy. Factor decay is the normal life cycle of a published effect, not
    an exception.

    Trailing windows only: each value labels the session it ends on.
    """
    column = f"fwd_{horizon}"
    per_session = (
        scored.drop_nulls(["signal", column])
        .group_by("event_time")
        .agg(
            pl.corr(pl.col("signal").rank(), pl.col(column).rank()).alias("ic"),
            pl.len().alias("names"),
        )
        .filter((pl.col("names") >= MIN_NAMES_PER_SESSION) & pl.col("ic").is_finite())
        .sort("event_time")
    )
    if per_session.height < window:
        return []

    values = per_session["ic"].to_numpy()
    stamps = per_session["event_time"].to_list()
    frames = sliding_window_view(values, window)
    means = frames.mean(axis=1)
    return [
        ICWindow(end=stamps[i + window - 1].date().isoformat(), ic=float(m), sessions=window)
        for i, m in enumerate(means)
    ]


def information_coefficient(scored: pl.DataFrame, horizon: int) -> ICSummary:
    """Spearman IC between the signal and the forward return, per session.

    Computed cross-sectionally within each session and then summarised across
    sessions. Pooling every (name, session) pair instead would let a few
    high-volatility periods dominate, and would treat observations from the
    same day as independent when they are not.
    """
    column = f"fwd_{horizon}"
    per_session = (
        scored.drop_nulls(["signal", column])
        .group_by("event_time")
        .agg(
            pl.corr(pl.col("signal").rank(), pl.col(column).rank()).alias("ic"),
            pl.len().alias("names"),
        )
        .filter((pl.col("names") >= MIN_NAMES_PER_SESSION) & pl.col("ic").is_finite())
        .sort("event_time")
    )

    values = per_session["ic"].to_numpy()
    if values.size < MIN_IC_SESSIONS:
        return ICSummary(horizon, 0.0, 0.0, 0.0, 0.0, 0.0, int(values.size))

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    ratio = mean / std if std > 0 else 0.0
    t_stat = mean / (std / np.sqrt(values.size)) if std > 0 else 0.0
    return ICSummary(
        horizon=horizon,
        mean=mean,
        std=std,
        information_ratio=ratio,
        t_stat=float(t_stat),
        hit_rate=float(np.mean(values > 0)),
        sessions=int(values.size),
    )


def quantile_returns(
    scored: pl.DataFrame, horizon: int, buckets: int = DEFAULT_QUANTILES
) -> list[QuantileRow]:
    """Mean forward return by signal bucket.

    Buckets are formed *within each session*, not across the pooled sample. A
    global cut would put whole calm periods in the bottom bucket and whole
    volatile ones in the top, measuring the calendar rather than the signal.
    """
    column = f"fwd_{horizon}"
    usable = scored.drop_nulls(["signal", column])
    if usable.is_empty():
        return []

    bucketed = usable.with_columns(
        (
            (pl.col("signal").rank("ordinal").over("event_time") - 1)
            * buckets
            // pl.len().over("event_time")
        ).alias("bucket")
    )
    grouped = (
        bucketed.group_by("bucket")
        .agg(pl.col(column).mean().alias("fwd"), pl.len().alias("names"))
        .sort("bucket")
    )
    return [
        QuantileRow(
            quantile=int(row["bucket"]) + 1,
            mean_forward_return=float(row["fwd"]),
            names=int(row["names"]),
        )
        for row in grouped.to_dicts()
        if row["fwd"] is not None
    ]


def signal_turnover(scored: pl.DataFrame) -> float:
    """Average fraction of the cross-sectional ranking that changes per session.

    The cost side of the trade-off. A signal with a strong IC that reorders the
    universe every day is not tradable against a 22bp NSE round trip, and this
    is the number that says so before a backtest is run.

    Measured as the mean absolute change in normalised rank, doubled so that a
    complete reshuffle reads as 1.0.
    """
    ranked = scored.sort(["symbol", "event_time"]).with_columns(
        (
            (pl.col("signal").rank("ordinal").over("event_time") - 1)
            / (pl.len().over("event_time") - 1).clip(lower_bound=1)
        ).alias("pct_rank")
    )
    changes = ranked.with_columns(
        (pl.col("pct_rank") - pl.col("pct_rank").shift(1).over("symbol")).abs().alias("delta")
    )["delta"].drop_nulls()
    if not changes.len():
        return 0.0
    # Via numpy: polars types a scalar aggregate as a broad union that float()
    # will not accept.
    mean_change = float(np.nanmean(changes.to_numpy()))
    return mean_change * TURNOVER_SCALE


def analyse_factor(
    scored: pl.DataFrame,
    factor_name: str,
    horizons: tuple[int, ...],
    quantile_horizon: int = 21,
    buckets: int = DEFAULT_QUANTILES,
) -> FactorReport:
    """Full report: IC at every horizon, quantiles, spread and turnover."""
    primary = quantile_horizon if quantile_horizon in horizons else horizons[-1]
    return FactorReport(
        factor=factor_name,
        horizons=[information_coefficient(scored, h) for h in horizons],
        quantiles=quantile_returns(scored, primary, buckets),
        quantile_horizon=primary,
        turnover=signal_turnover(scored),
        names=scored["symbol"].n_unique(),
        sessions=scored["event_time"].n_unique(),
    )
