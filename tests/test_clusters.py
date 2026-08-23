"""Correlation groups for the concentration limit — §8.

`max_cluster_pct` was configured, enforced by the risk engine and evaluated on
no order ever placed, because nothing set a cluster to check. These cover the
join, and the restraint: a group invented from noise is a risk limit invented
from noise.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from core.clock import UTC
from core.instruments import InstrumentId
from quant.analytics.clusters import MIN_OBS_PER_NAME, assign_clusters


def panel(series: dict[str, np.ndarray]) -> pl.DataFrame:
    length = len(next(iter(series.values())))
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "event_time": times,
                    "instrument_id": [name] * length,
                    "close": list(closes),
                },
                schema_overrides={"event_time": pl.Datetime("us", "UTC")},
            )
            for name, closes in series.items()
        ]
    )


def walk(steps: np.ndarray) -> np.ndarray:
    return 100.0 * np.exp(np.cumsum(steps))


class TestGrouping:
    def test_names_that_move_together_land_in_one_group(self):
        rng = np.random.default_rng(0)
        shared = rng.normal(0, 0.02, 400)
        pair = {
            "NSE:INEAAA": walk(shared + rng.normal(0, 0.002, 400)),
            "NSE:INEBBB": walk(shared + rng.normal(0, 0.002, 400)),
            "NSE:INECCC": walk(rng.normal(0, 0.02, 400)),
        }
        labels = assign_clusters(panel(pair), tuple(InstrumentId(k) for k in pair))
        assert labels[InstrumentId("NSE:INEAAA")] == labels[InstrumentId("NSE:INEBBB")]
        assert labels[InstrumentId("NSE:INECCC")] != labels[InstrumentId("NSE:INEAAA")]

    def test_independent_names_stay_apart(self):
        rng = np.random.default_rng(1)
        names = {f"NSE:INE{i:03d}": walk(rng.normal(0, 0.02, 400)) for i in range(6)}
        labels = assign_clusters(panel(names), tuple(InstrumentId(k) for k in names))
        assert len(set(labels.values())) == len(names)

    def test_every_instrument_is_labelled(self):
        rng = np.random.default_rng(2)
        names = {f"NSE:INE{i:03d}": walk(rng.normal(0, 0.02, 400)) for i in range(5)}
        labels = assign_clusters(panel(names), tuple(InstrumentId(k) for k in names))
        assert set(labels) == {InstrumentId(k) for k in names}


class TestRestraint:
    """What it refuses to group. An invented cluster is an invented limit."""

    def test_too_few_observations_yields_no_clusters(self):
        """Fewer rows than names is estimation error, not correlation."""
        rng = np.random.default_rng(3)
        names = {f"NSE:INE{i:03d}": walk(rng.normal(0, 0.02, 12)) for i in range(10)}
        assert assign_clusters(panel(names), tuple(InstrumentId(k) for k in names)) == {}

    def test_the_observation_floor_is_per_name(self):
        """Widening the universe demands proportionally more history."""
        assert MIN_OBS_PER_NAME >= 1.0

    def test_a_single_instrument_has_no_group(self):
        rng = np.random.default_rng(4)
        one = {"NSE:INEAAA": walk(rng.normal(0, 0.02, 400))}
        assert assign_clusters(panel(one), (InstrumentId("NSE:INEAAA"),)) == {}

    def test_an_empty_panel_is_safe(self):
        empty = pl.DataFrame(
            {"event_time": [], "instrument_id": [], "close": []},
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "instrument_id": pl.String,
                "close": pl.Float64,
            },
        )
        assert assign_clusters(empty, (InstrumentId("NSE:INEAAA"),)) == {}

    def test_names_absent_from_the_panel_are_not_invented(self):
        rng = np.random.default_rng(5)
        names = {f"NSE:INE{i:03d}": walk(rng.normal(0, 0.02, 400)) for i in range(3)}
        wanted = (*[InstrumentId(k) for k in names], InstrumentId("NSE:INEZZZ"))
        labels = assign_clusters(panel(names), wanted)
        assert InstrumentId("NSE:INEZZZ") not in labels


class TestLookAhead:
    def test_only_the_trailing_window_is_used(self):
        """Clusters computed over all history would size today's order with
        tomorrow's correlations."""
        rng = np.random.default_rng(6)
        shared = rng.normal(0, 0.02, 600)
        # Correlated only in the distant past; independent across the window.
        a = np.r_[shared[:300], rng.normal(0, 0.02, 300)]
        b = np.r_[shared[:300], rng.normal(0, 0.02, 300)]
        frame = panel({"NSE:INEAAA": walk(a), "NSE:INEBBB": walk(b)})
        labels = assign_clusters(
            frame, (InstrumentId("NSE:INEAAA"), InstrumentId("NSE:INEBBB")), window=250
        )
        assert labels.get(InstrumentId("NSE:INEAAA")) != labels.get(InstrumentId("NSE:INEBBB"))
