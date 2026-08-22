"""Inferring splits from the price series (§9).

The risk runs both ways and the tests are split accordingly: a missed split
leaves a fake -50% return in every factor computed from it, and an over-eager
inference silently rewrites a real crash into something that never happened.
The second is worse, so most of these test what the inference declines to do.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from core.clock import UTC
from data.corpactions.inferred import (
    RATIO_TOLERANCE,
    SUSPECT_MOVE,
    adjust_for_inferred_splits,
    inferred_split_factors,
)


def series(closes: dict[str, list[float]], volume: float = 1e6) -> pl.DataFrame:
    length = len(next(iter(closes.values())))
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "event_time": times,
                    "symbol": [symbol] * length,
                    "close": prices,
                    "volume": [volume] * length,
                },
                schema_overrides={"event_time": pl.Datetime("us", "UTC")},
            )
            for symbol, prices in closes.items()
        ]
    )


def with_split(ratio: float, n: int = 40, at: int = 20) -> list[float]:
    """A flat series that splits once."""
    return [100.0 if i < at else 100.0 * ratio for i in range(n)]


class TestDetection:
    def test_a_one_for_one_bonus_is_found(self):
        """The commonest case, and the one that reads as -50%."""
        adjusted = adjust_for_inferred_splits(series({"AAA": with_split(0.5)}))
        closes = adjusted["close"].to_list()
        # Continuous across the event: no -50% step remains.
        assert closes[19] == pytest.approx(closes[20])

    @pytest.mark.parametrize("ratio", [0.5, 0.2, 0.25, 1 / 3, 0.1])
    def test_common_ratios_are_found(self, ratio):
        adjusted = adjust_for_inferred_splits(series({"AAA": with_split(ratio)}))
        closes = adjusted["close"].to_list()
        assert closes[19] == pytest.approx(closes[20], rel=1e-6)

    def test_a_reverse_split_is_found(self):
        adjusted = adjust_for_inferred_splits(series({"AAA": with_split(5.0)}))
        closes = adjusted["close"].to_list()
        assert closes[19] == pytest.approx(closes[20], rel=1e-6)

    def test_the_series_becomes_continuous_end_to_end(self):
        """The number that matters: total return across the event."""
        raw = series({"AAA": with_split(0.5)})
        adjusted = adjust_for_inferred_splits(raw)
        total = adjusted["close"].to_list()[-1] / adjusted["close"].to_list()[0] - 1
        assert total == pytest.approx(0.0, abs=1e-9)


class TestRestraint:
    """What it declines to do. A rewritten real crash is worse than a fake one
    left in, because nothing downstream can tell it happened."""

    def test_a_genuine_crash_is_left_alone(self):
        """-38% matches no plausible split ratio."""
        crash = [100.0 if i < 20 else 62.0 for i in range(40)]
        adjusted = adjust_for_inferred_splits(series({"AAA": crash}))
        assert adjusted["close"].to_list() == crash

    def test_a_move_below_the_threshold_is_left_alone(self):
        gentle = [100.0 if i < 20 else 100.0 * (1 - SUSPECT_MOVE / 2) for i in range(40)]
        adjusted = adjust_for_inferred_splits(series({"AAA": gentle}))
        assert adjusted["close"].to_list() == gentle

    def test_a_near_miss_ratio_is_left_alone(self):
        """Just outside the tolerance band around 1:2."""
        ratio = 0.5 * (1 - RATIO_TOLERANCE * 4)
        prices = with_split(ratio)
        adjusted = adjust_for_inferred_splits(series({"AAA": prices}))
        assert adjusted["close"].to_list() == prices

    def test_an_untouched_name_keeps_every_price(self):
        rising = [100.0 * (1.01**i) for i in range(40)]
        adjusted = adjust_for_inferred_splits(series({"AAA": rising}))
        assert adjusted["close"].to_list() == pytest.approx(rising)

    def test_factors_are_one_where_nothing_happened(self):
        framed = inferred_split_factors(series({"AAA": [100.0] * 30}))
        assert set(framed["split_factor"].to_list()) == {1.0}


class TestMultipleNames:
    def test_names_do_not_contaminate_each_other(self):
        """The step between the last bar of one name and the first of the next
        is not a price move."""
        frame = series({"AAA": [100.0] * 30, "BBB": [500.0] * 30})
        adjusted = adjust_for_inferred_splits(frame)
        by_symbol = adjusted.group_by("symbol").agg(pl.col("close").n_unique().alias("distinct"))
        assert set(by_symbol["distinct"].to_list()) == {1}

    def test_one_name_splitting_leaves_the_other_alone(self):
        frame = series({"SPLIT": with_split(0.5), "CLEAN": [100.0] * 40})
        adjusted = adjust_for_inferred_splits(frame)
        clean = adjusted.filter(pl.col("symbol") == "CLEAN")["close"].to_list()
        assert clean == [100.0] * 40


class TestVolume:
    def test_volume_moves_the_other_way(self):
        """More shares outstanding, more traded. Traded *value* is what stays
        invariant, and a liquidity filter straddling a split depends on it."""
        adjusted = adjust_for_inferred_splits(series({"AAA": with_split(0.5)}))
        volumes = adjusted["volume"].to_list()
        assert volumes[0] == pytest.approx(volumes[-1] * 2)

    def test_traded_value_is_continuous_on_a_realistic_split(self):
        """Real volume roughly doubles at a 1:2 split — twice the shares
        changing hands for the same rupees. A fixture holding volume constant
        across the event is not a split, and asserting value invariance on it
        would be testing the fixture.
        """
        length = 40
        times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]
        frame = pl.DataFrame(
            {
                "event_time": times,
                "symbol": ["AAA"] * length,
                "close": with_split(0.5, length),
                "volume": [1e6 if i < 20 else 2e6 for i in range(length)],
            },
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )
        adjusted = adjust_for_inferred_splits(frame)
        values = (adjusted["close"] * adjusted["volume"]).to_list()
        assert values[0] == pytest.approx(values[-1])


class TestEdges:
    def test_an_empty_panel_is_safe(self):
        empty = pl.DataFrame(
            {"event_time": [], "symbol": [], "close": [], "volume": []},
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "symbol": pl.String,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )
        assert adjust_for_inferred_splits(empty).height == 0

    def test_a_single_bar_is_safe(self):
        adjusted = adjust_for_inferred_splits(series({"AAA": [100.0]}))
        assert adjusted["close"].to_list() == [100.0]

    def test_two_splits_compound(self):
        """A name that splits twice is scaled by both, not the later one only."""
        prices = [100.0] * 10 + [50.0] * 10 + [25.0] * 10
        adjusted = adjust_for_inferred_splits(series({"AAA": prices}))
        closes = adjusted["close"].to_list()
        assert closes[0] == pytest.approx(closes[-1])
