"""Universe screener — MASTER_PLAN §6, §253.

The system could describe a security you named. It could not answer *which
ones*, and that is the question research actually starts from: which names are
liquid enough to trade, which are stationary enough to fade, which are trending.

**Two stages, because the cheap one is 200x faster.** Measured on this panel:

    vectorised pass over every name       ~220 ms
    full statistical profile per name     ~47 ms  -> ~100 s for 2,200 names

So the cheap pass — computed entirely in Polars, no Python loop — narrows
thousands of names to a shortlist on liquidity, history and return, and only
the survivors pay for ADF, KPSS, Hurst and the rest. Screening everything
deeply would take minutes and answer a question nobody asked, because a name
that trades ₹2 lakh a day is not investable regardless of how stationary it is.

**Liquidity is the first filter, always.** An illiquid name will show
spectacular statistics — thin trading produces stale prices, stale prices
produce low volatility and high autocorrelation — and none of it survives
contact with a real order. Filtering it last means computing beautiful
statistics for names you can never trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import polars as pl

from quant.analytics.security import SecurityProfile, profile_security

__all__ = [
    "ScreenCriteria",
    "ScreenResult",
    "ScreenRow",
    "SortKey",
    "screen_universe",
]

#: Sessions used for the cheap pass. About a trading year: long enough for a
#: liquidity median to be stable, short enough to describe the current market.
DEFAULT_WINDOW = 250

#: A name below this is not investable for a retail book, whatever its
#: statistics say. ₹1 crore of daily traded value.
DEFAULT_MIN_ADV = 1e7

#: Below this many bars in the window the name is too thinly listed to profile.
DEFAULT_MIN_BARS = 200

#: How many survivors get the full statistical treatment. Each costs ~47ms.
DEFAULT_DEEP_LIMIT = 60

#: A single-session move this large is almost never a price move. It is a
#: split, a bonus or a bad print. Matches the data-quality check's threshold so
#: the two agree about what looks wrong.
SUSPECT_MOVE = 0.35


class SortKey(str, Enum):
    """What the shortlist is ranked by before deep statistics are computed."""

    LIQUIDITY = "liquidity"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"

    @property
    def column(self) -> str:
        return {
            SortKey.LIQUIDITY: "adv",
            SortKey.MOMENTUM: "window_return",
            SortKey.REVERSAL: "window_return",
            SortKey.VOLATILITY: "volatility",
        }[self]

    @property
    def descending(self) -> bool:
        """Reversal wants the worst performers; everything else wants the best."""
        return self is not SortKey.REVERSAL


@dataclass(frozen=True)
class ScreenCriteria:
    """Filters applied in the cheap pass."""

    window: int = DEFAULT_WINDOW
    min_adv: float = DEFAULT_MIN_ADV
    min_bars: int = DEFAULT_MIN_BARS
    min_price: float = 5.0
    sort_by: SortKey = SortKey.LIQUIDITY
    limit: int = DEFAULT_DEEP_LIMIT
    #: Drop names whose window contains a move too large to be a price move.
    #:
    #: **On by default, and the default matters.** The panel stores raw closes,
    #: so a 1:1 bonus appears as a -50% session. A reversal screen sorted on
    #: window return would then rank every recently-split name as a top loser —
    #: HDFCBANK showed -62.7% on this panel purely from its August 2025 bonus.
    #: A screen dominated by corporate actions is worse than no screen, because
    #: its output looks like a list of opportunities.
    #:
    #: Adjusting every name instead would mean a network fetch per symbol
    #: across a thousand names. Excluding them is one vectorised comparison,
    #: and anything excluded here can be profiled properly by name.
    exclude_suspected_actions: bool = True
    #: Restrict deep statistics to names that can actually be faded. Costs
    #: nothing extra — the verdict is computed either way — but it turns the
    #: screener into the tool a mean-reversion strategy needs (§253).
    stationary_only: bool = False

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be at least 1")
        if self.window < self.min_bars:
            raise ValueError(
                f"window {self.window} is shorter than min_bars {self.min_bars}; "
                "no name could ever qualify"
            )


@dataclass(frozen=True)
class ScreenRow:
    """One name that survived the cheap pass, with its deep statistics."""

    symbol: str
    adv: float
    bars: int
    last_close: float
    window_return: float
    profile: SecurityProfile | None = None

    @property
    def verdict(self) -> str:
        return self.profile.stationarity.verdict.value if self.profile else "—"

    @property
    def fadeable(self) -> bool:
        return bool(self.profile and self.profile.stationarity.tradable_as_mean_reversion)


@dataclass(frozen=True)
class ScreenResult:
    """What the screen found, and what it had to discard to get there."""

    rows: list[ScreenRow] = field(default_factory=list)
    considered: int = 0
    passed_filters: int = 0
    profiled: int = 0
    #: Names dropped for containing a move too large to be a price move.
    suspected_actions: int = 0
    criteria: ScreenCriteria = field(default_factory=ScreenCriteria)

    def format(self) -> str:
        lines = [
            f"screened {self.considered:,} names -> {self.passed_filters:,} liquid "
            f"-> {self.profiled} profiled"
        ]
        if self.suspected_actions:
            lines.append(
                f"  {self.suspected_actions:,} excluded for an unexplained "
                f"move above {SUSPECT_MOVE:.0%} (likely a split or bonus)"
            )
        if not self.rows:
            lines.append("  nothing met the criteria")
        return "\n".join(lines)


def cheap_pass(history: pl.DataFrame, criteria: ScreenCriteria) -> pl.DataFrame:
    """Vectorised filter over every name. No Python loop, no per-name cost.

    Returns one row per surviving symbol with the columns the deep stage needs
    to rank on.
    """
    sessions = history["event_time"].unique().sort().tail(criteria.window)
    window = history.filter(pl.col("event_time").is_in(sessions.implode()))

    return (
        window.sort("event_time")
        .group_by("symbol")
        .agg(
            pl.len().alias("bars"),
            pl.col("close").last().alias("last_close"),
            pl.col("close").first().alias("first_close"),
            (pl.col("close") * pl.col("volume")).median().alias("adv"),
            # Standard deviation of daily returns, annualised. Cheap because
            # Polars computes it column-wise rather than per name in Python.
            (pl.col("close") / pl.col("close").shift(1) - 1).std().alias("daily_vol"),
            (pl.col("close") / pl.col("close").shift(1) - 1).abs().max().alias("max_move"),
        )
        .with_columns(
            (pl.col("last_close") / pl.col("first_close") - 1).alias("window_return"),
            (pl.col("daily_vol") * (252**0.5)).alias("volatility"),
        )
        .filter(
            (pl.col("bars") >= criteria.min_bars)
            & (pl.col("adv") >= criteria.min_adv)
            & (pl.col("last_close") >= criteria.min_price)
            & pl.col("window_return").is_finite()
        )
        .filter(
            pl.col("max_move") < SUSPECT_MOVE
            if criteria.exclude_suspected_actions
            else pl.lit(value=True)
        )
    )


def screen_universe(history: pl.DataFrame, criteria: ScreenCriteria | None = None) -> ScreenResult:
    """Screen every name in the panel, deeply profiling only the shortlist.

    Args:
        history: Long-format panel. Prices should already be back-adjusted if
            the caller cares about corporate actions — this does not adjust,
            because adjusting thousands of names would cost more than the
            screen itself. Liquidity and ranking are robust to it; the deep
            profile of a shortlisted name is not, so the caller re-profiles
            adjusted series for anything it intends to trade.
    """
    active = criteria or ScreenCriteria()
    considered = history["symbol"].n_unique()

    survivors = cheap_pass(history, active)
    # Count what the action filter removed, so an empty screen is explainable.
    unfiltered = cheap_pass(history, replace(active, exclude_suspected_actions=False))
    suspected = unfiltered.height - survivors.height
    shortlist = survivors.sort(
        active.sort_by.column, descending=active.sort_by.descending, nulls_last=True
    )

    # Over-fetch when filtering on stationarity, since the verdict is only
    # known after profiling and most names are not stationary.
    depth = active.limit * (4 if active.stationary_only else 1)
    candidates = shortlist.head(depth).to_dicts()

    rows: list[ScreenRow] = []
    for candidate in candidates:
        symbol = candidate["symbol"]
        closes = (
            history.filter(pl.col("symbol") == symbol)
            .sort("event_time")
            .tail(active.window)["close"]
            .to_list()
        )
        try:
            profile = profile_security(symbol, closes)
        except ValueError:
            # Too little history for the statistics to mean anything. The name
            # already passed the liquidity filter, so this is rare and not
            # worth failing the whole screen over.
            continue

        if active.stationary_only and not profile.stationarity.tradable_as_mean_reversion:
            continue

        rows.append(
            ScreenRow(
                symbol=symbol,
                adv=float(candidate["adv"]),
                bars=int(candidate["bars"]),
                last_close=float(candidate["last_close"]),
                window_return=float(candidate["window_return"]),
                profile=profile,
            )
        )
        if len(rows) >= active.limit:
            break

    return ScreenResult(
        rows=rows,
        considered=considered,
        passed_filters=survivors.height,
        profiled=len(rows),
        suspected_actions=suspected,
        criteria=active,
    )
