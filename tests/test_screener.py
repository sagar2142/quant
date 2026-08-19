"""Universe screener (§6, §253).

The screener's job is to be *selective correctly*: keep what is tradable, drop
what only looks tradable. Each test builds a panel where the right answer is
known and asserts the filter reaches it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from core.clock import UTC
from quant.analytics.screener import (
    SUSPECT_MOVE,
    ScreenCriteria,
    SortKey,
    cheap_pass,
    screen_universe,
)

SEED = 20260819


def panel(series: dict[str, list[float]], volume: float = 1e6) -> pl.DataFrame:
    """Long-format panel from per-symbol close series of equal length."""
    length = len(next(iter(series.values())))
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]
    frames = [
        pl.DataFrame(
            {
                "event_time": times,
                "symbol": [symbol] * length,
                "instrument_id": [f"NSE:{symbol}"] * length,
                "close": closes,
                "volume": [volume] * length,
            },
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )
        for symbol, closes in series.items()
    ]
    return pl.concat(frames)


def walk(n: int = 300, drift: float = 0.0, seed: int = SEED, start: float = 100.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(start * np.exp(np.cumsum(rng.normal(drift, 0.015, n))))


class TestCheapPass:
    def test_illiquid_names_are_dropped(self):
        """Liquidity first, always: thin trading makes stale prices, and stale
        prices manufacture flattering statistics that no order can capture."""
        frame = pl.concat(
            [
                panel({"LIQUID": walk()}, volume=1e6),
                panel({"THIN": walk(seed=SEED + 1)}, volume=1.0),
            ]
        )
        kept = cheap_pass(frame, ScreenCriteria(min_adv=1e7))
        assert kept["symbol"].to_list() == ["LIQUID"]

    def test_short_history_is_dropped(self):
        frame = pl.concat([panel({"LONG": walk(300)}), panel({"SHORT": walk(50)})])
        kept = cheap_pass(frame, ScreenCriteria(min_bars=200))
        assert kept["symbol"].to_list() == ["LONG"]

    def test_penny_stocks_are_dropped(self):
        frame = pl.concat([panel({"NORMAL": walk()}), panel({"PENNY": walk(start=1.0)})])
        kept = cheap_pass(frame, ScreenCriteria(min_price=5.0))
        assert kept["symbol"].to_list() == ["NORMAL"]

    def test_it_computes_the_ranking_columns(self):
        kept = cheap_pass(panel({"A": walk()}), ScreenCriteria())
        assert set(kept.columns) >= {"symbol", "adv", "bars", "window_return", "volatility"}


class TestCorporateActionExclusion:
    def split_affected(self) -> list[float]:
        """A clean series with one 1:1 bonus in the middle — a -50% session in
        raw prices."""
        closes = walk(300)
        return [c if i < 150 else c / 2 for i, c in enumerate(closes)]

    def test_a_split_like_move_is_excluded_by_default(self):
        """The panel holds raw prices, so a bonus reads as a crash. Left in, it
        would top every reversal screen — HDFCBANK showed -62.7% on the real
        panel purely from its 2025 bonus."""
        frame = pl.concat([panel({"CLEAN": walk()}), panel({"SPLIT": self.split_affected()})])
        kept = cheap_pass(frame, ScreenCriteria())
        assert kept["symbol"].to_list() == ["CLEAN"]

    def test_it_can_be_disabled_deliberately(self):
        frame = pl.concat([panel({"CLEAN": walk()}), panel({"SPLIT": self.split_affected()})])
        kept = cheap_pass(frame, ScreenCriteria(exclude_suspected_actions=False))
        assert set(kept["symbol"].to_list()) == {"CLEAN", "SPLIT"}

    def test_the_exclusion_is_counted_and_reported(self):
        """An empty screen has to be explainable."""
        frame = pl.concat([panel({"CLEAN": walk()}), panel({"SPLIT": self.split_affected()})])
        result = screen_universe(frame, ScreenCriteria(limit=5))
        assert result.suspected_actions == 1
        assert "split or bonus" in result.format()

    def test_a_move_just_under_the_threshold_survives(self):
        """One genuine large move is not a corporate action.

        The whole tail is rescaled, not a single point: moving one close
        creates a *second* jump when the series returns to trend, and that
        recovery would trip the filter for the wrong reason.
        """
        closes = walk(300)
        drop = 1 - SUSPECT_MOVE * 0.9
        closes = [c if i < 150 else c * drop for i, c in enumerate(closes)]
        kept = cheap_pass(panel({"BUMPY": closes}), ScreenCriteria())
        assert kept["symbol"].to_list() == ["BUMPY"]


class TestSorting:
    def universe(self) -> pl.DataFrame:
        return pl.concat(
            [
                panel({"WINNER": walk(drift=0.004, seed=1)}),
                panel({"LOSER": walk(drift=-0.004, seed=2)}),
                panel({"FLAT": walk(drift=0.0, seed=3)}),
            ]
        )

    def test_momentum_ranks_the_winner_first(self):
        result = screen_universe(self.universe(), ScreenCriteria(sort_by=SortKey.MOMENTUM, limit=1))
        assert result.rows[0].symbol == "WINNER"

    def test_reversal_ranks_the_loser_first(self):
        """Reversal wants the worst performers — the sort has to invert."""
        result = screen_universe(self.universe(), ScreenCriteria(sort_by=SortKey.REVERSAL, limit=1))
        assert result.rows[0].symbol == "LOSER"

    def test_liquidity_ranks_by_traded_value(self):
        frame = pl.concat(
            [panel({"BIG": walk()}, volume=1e7), panel({"SMALL": walk(seed=2)}, volume=1e6)]
        )
        result = screen_universe(frame, ScreenCriteria(sort_by=SortKey.LIQUIDITY, limit=1))
        assert result.rows[0].symbol == "BIG"


class TestDeepStage:
    def test_only_the_shortlist_is_profiled(self):
        """The deep stage costs ~47ms a name. Profiling everything would take
        minutes to describe names the filter already rejected."""
        frame = pl.concat([panel({f"N{i}": walk(seed=i)}) for i in range(12)])
        result = screen_universe(frame, ScreenCriteria(limit=3))
        assert result.passed_filters == 12
        assert result.profiled == 3

    def test_profiles_are_attached(self):
        result = screen_universe(panel({"A": walk()}), ScreenCriteria(limit=1))
        row = result.rows[0]
        assert row.profile is not None
        assert row.verdict in {"STATIONARY", "UNIT_ROOT", "INCONCLUSIVE"}

    def test_stationary_only_filters_random_walks(self):
        """A random walk has no level to revert to, so a fadeable screen must
        return nothing rather than the closest thing available."""
        frame = pl.concat([panel({f"W{i}": walk(seed=i)}) for i in range(6)])
        result = screen_universe(frame, ScreenCriteria(limit=5, stationary_only=True))
        assert result.rows == []

    def test_stationary_only_keeps_a_reverting_series(self):
        rng = np.random.default_rng(SEED)
        ou = np.zeros(300)
        for i in range(1, 300):
            ou[i] = ou[i - 1] + 0.3 * (0.0 - ou[i - 1]) + rng.normal(0, 1.0)
        result = screen_universe(
            panel({"OU": list(ou + 100.0)}), ScreenCriteria(limit=1, stationary_only=True)
        )
        assert [r.symbol for r in result.rows] == ["OU"]
        assert result.rows[0].fadeable


class TestCriteriaValidation:
    def test_a_zero_limit_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            ScreenCriteria(limit=0)

    def test_a_window_shorter_than_min_bars_is_refused(self):
        """No name could ever qualify, so the screen would silently return
        nothing for a reason the caller cannot see."""
        with pytest.raises(ValueError, match="shorter than"):
            ScreenCriteria(window=100, min_bars=200)


class TestResultReporting:
    def test_it_reports_the_funnel(self):
        frame = pl.concat([panel({f"N{i}": walk(seed=i)}) for i in range(5)])
        result = screen_universe(frame, ScreenCriteria(limit=2))
        text = result.format()
        assert "5" in text
        assert "profiled" in text

    def test_an_empty_screen_says_so(self):
        frame = panel({"THIN": walk()}, volume=1.0)
        result = screen_universe(frame, ScreenCriteria(min_adv=1e12))
        assert result.rows == []
        assert "nothing met the criteria" in result.format()
