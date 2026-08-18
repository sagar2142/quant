"""Point-in-time universe construction — MASTER_PLAN §M2 gate.

The single highest-value component in the data layer, and the one whose absence
quietly invalidates the most research.

**The failure it prevents.** Ask "the 100 most liquid NSE stocks" of a naive
system and you get today's list, then backtest it over 2015-2025. Every name in
that list survived to today. The companies that were liquid in 2016 and then
collapsed are absent. The backtest therefore only ever holds survivors, and
reports a return no one could have earned. That is survivorship bias, and it is
worth several percent a year of pure fiction.

**What this does instead.** Membership is computed from the cross-section as it
existed on the selection date, using only bars already received by then. A
company that was in the top 100 in 2016 and delisted in 2019 is in the 2016
universe and absent from the 2020 one — which is exactly what a real portfolio
would have experienced.

**Rebalance dates are explicit.** Membership is fixed between rebalances, so a
strategy cannot silently benefit from a name entering the universe mid-holding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from core.clock import UTC, DecisionTime, as_decision_time, require_utc
from data.store.panel import PanelStore

__all__ = ["Universe", "UniverseBuilder", "UniverseSpec"]


@dataclass(frozen=True)
class UniverseSpec:
    """Selection rules. Versioned alongside the experiment that used them.

    Attributes:
        top_n: How many names to keep, ranked by median traded value.
        lookback_days: Trailing window for the liquidity estimate.
        min_price: Floor to exclude penny stocks, where percentage returns are
            dominated by tick size and fills are not realistic.
        min_sessions: Minimum observations in the window. Excludes names that
            only just listed, whose liquidity estimate is noise.
        min_median_value: Absolute liquidity floor in currency units.
    """

    top_n: int = 100
    lookback_days: int = 60
    min_price: float = 20.0
    min_sessions: int = 40
    min_median_value: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if self.min_sessions > self.lookback_days:
            raise ValueError(
                f"min_sessions ({self.min_sessions}) exceeds lookback_days "
                f"({self.lookback_days}); no name could ever qualify"
            )


@dataclass(frozen=True)
class Universe:
    """Membership as of one selection date."""

    as_of: datetime
    members: tuple[str, ...]
    spec: UniverseSpec
    #: Median daily traded value per member, the ranking statistic.
    liquidity: dict[str, float]

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, instrument_id: str) -> bool:
        return instrument_id in self.members

    def turnover_against(self, previous: Universe) -> float:
        """Fraction of membership that changed since `previous`.

        Universe turnover is itself a cost: every entry and exit is a trade.
        """
        if not previous.members:
            return 1.0 if self.members else 0.0
        current, prior = set(self.members), set(previous.members)
        return len(current ^ prior) / len(current | prior)


class UniverseBuilder:
    """Builds point-in-time universes from a cross-sectional panel."""

    def __init__(self, panel: PanelStore) -> None:
        self.panel = panel

    def build(self, as_of: DecisionTime, spec: UniverseSpec | None = None) -> Universe:
        """Membership as it would have been computed on `as_of`.

        Only rows with ``receive_time <= as_of`` participate, so the selection
        could genuinely have been made on the day.
        """
        spec = spec or UniverseSpec()
        cutoff = require_utc(as_of)
        window_start = (cutoff - timedelta(days=spec.lookback_days * 2)).date()

        rows = self.panel.view(as_of=as_of, start=window_start)
        if rows.is_empty():
            return Universe(cutoff, (), spec, {})

        # Trailing calendar window, then per-name liquidity statistics.
        sessions = rows.select("event_time").unique().sort("event_time")
        keep_from = sessions.tail(spec.lookback_days)["event_time"].min()
        window = rows.filter(pl.col("event_time") >= keep_from)

        stats = (
            window.with_columns((pl.col("close") * pl.col("volume")).alias("traded_value"))
            .group_by("instrument_id")
            .agg(
                pl.col("traded_value").median().alias("median_value"),
                pl.col("close").last().alias("last_close"),
                pl.len().alias("sessions"),
            )
        )

        eligible = (
            stats.filter(
                (pl.col("sessions") >= spec.min_sessions)
                & (pl.col("last_close") >= spec.min_price)
                & (pl.col("median_value") >= spec.min_median_value)
            )
            # instrument_id breaks ties deterministically: two names with equal
            # liquidity must not reorder between runs (§14.1.1).
            .sort(["median_value", "instrument_id"], descending=[True, False])
            .head(spec.top_n)
        )

        members = tuple(eligible["instrument_id"].to_list())
        liquidity = dict(zip(members, eligible["median_value"].to_list(), strict=True))
        return Universe(cutoff, members, spec, liquidity)

    def build_schedule(
        self,
        rebalance_dates: list[date],
        spec: UniverseSpec | None = None,
        decision_hour_utc: int = 12,
    ) -> dict[date, Universe]:
        """Universes for a series of rebalance dates.

        Bhavcopy publishes at ~12:30 UTC (18:00 IST). The default decision hour
        of 12:00 UTC is deliberately *before* that, so a rebalance dated D is
        computed from sessions up to D-1 and cannot use a file that had not yet
        appeared. Set it later only if you intend to model an evening decision.
        """
        out: dict[date, Universe] = {}
        for day in sorted(rebalance_dates):
            stamp = datetime(day.year, day.month, day.day, decision_hour_utc, tzinfo=UTC)
            out[day] = self.build(as_decision_time(stamp), spec)
        return out
