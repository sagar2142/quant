"""Bar store (§14.1.4). The central claim under test: the future is unreachable."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from core.events import Timeframe
from core.instruments import InstrumentId
from data.store.bars import BAR_SCHEMA, BarStore, NoDataError

IID = InstrumentId("NSE:INE002A01018")
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def frame(n: int = 10, lag: timedelta = timedelta(milliseconds=100)) -> pl.DataFrame:
    event = [T0 + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "event_time": event,
            "receive_time": [t + lag for t in event],
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0] * n,
            "trades": [50] * n,
        },
        schema_overrides={
            "event_time": BAR_SCHEMA["event_time"],
            "receive_time": BAR_SCHEMA["receive_time"],
        },
    )


@pytest.fixture
def store(tmp_path) -> BarStore:
    return BarStore(tmp_path)


class TestWriteRead:
    def test_roundtrip(self, store):
        store.write(IID, Timeframe.D1, frame(10))
        got = store.view(IID, Timeframe.D1, as_of=as_decision_time(T0 + timedelta(days=30)))
        assert got.height == 10
        assert got["close"][0] == 100.5

    def test_result_is_sorted(self, store):
        shuffled = frame(10).sample(fraction=1.0, shuffle=True, seed=42)
        store.write(IID, Timeframe.D1, shuffled)
        got = store.view(IID, Timeframe.D1, as_of=as_decision_time(T0 + timedelta(days=30)))
        assert got["event_time"].is_sorted()

    def test_rewrite_is_idempotent(self, store):
        store.write(IID, Timeframe.D1, frame(10))
        store.write(IID, Timeframe.D1, frame(10))
        _, _, n = store.coverage(IID, Timeframe.D1)
        assert n == 10

    def test_overlapping_backfill_converges(self, store):
        store.write(IID, Timeframe.D1, frame(5))
        store.write(IID, Timeframe.D1, frame(10))
        _, _, n = store.coverage(IID, Timeframe.D1)
        assert n == 10

    def test_spans_year_partitions(self, store):
        store.write(IID, Timeframe.D1, frame(400))
        first, last, n = store.coverage(IID, Timeframe.D1)
        assert n == 400
        assert first.year == 2024
        assert last.year == 2025

    def test_missing_data_raises_not_empty(self, store):
        # §14.1.5: an empty frame becomes a silently skipped universe member.
        with pytest.raises(NoDataError):
            store.view(IID, Timeframe.D1, as_of=as_decision_time(T0))

    def test_coverage_missing_raises(self, store):
        with pytest.raises(NoDataError):
            store.coverage(IID, Timeframe.D1)

    def test_instruments_listing(self, store):
        store.write(IID, Timeframe.D1, frame(3))
        assert store.instruments(Timeframe.D1) == ["NSE_INE002A01018"]
        assert store.instruments(Timeframe.H1) == []


class TestPointInTime:
    """The reason this class exists."""

    def test_future_bars_excluded(self, store):
        store.write(IID, Timeframe.D1, frame(10))
        # Decision on day 5 must not see days 5..9.
        got = store.view(IID, Timeframe.D1, as_of=as_decision_time(T0 + timedelta(days=5)))
        assert got.height == 5
        assert got["event_time"].max() < T0 + timedelta(days=5)

    def test_publication_lag_is_respected(self, store):
        # A bar closing at T is not observable at T; it arrives at T + lag.
        store.write(IID, Timeframe.D1, frame(10, lag=timedelta(hours=6)))
        at_close = store.view(IID, Timeframe.D1, as_of=as_decision_time(T0))
        assert at_close.height == 0
        after_lag = store.view(IID, Timeframe.D1, as_of=as_decision_time(T0 + timedelta(hours=6)))
        assert after_lag.height == 1

    def test_start_filter_trims_history(self, store):
        store.write(IID, Timeframe.D1, frame(10))
        got = store.view(
            IID,
            Timeframe.D1,
            as_of=as_decision_time(T0 + timedelta(days=30)),
            start=T0 + timedelta(days=7),
        )
        assert got.height == 3

    def test_naive_as_of_rejected(self, store):
        store.write(IID, Timeframe.D1, frame(3))
        with pytest.raises(ValueError, match="naive"):
            store.view(IID, Timeframe.D1, as_of=datetime(2024, 6, 1))  # type: ignore[arg-type]


class TestWriteValidation:
    def test_missing_column_rejected(self, store):
        with pytest.raises(ValueError, match="missing columns"):
            store.write(IID, Timeframe.D1, frame(5).drop("volume"))

    def test_empty_frame_rejected(self, store):
        with pytest.raises(ValueError, match="empty"):
            store.write(IID, Timeframe.D1, frame(5).head(0))

    def test_causality_violation_rejected(self, store):
        bad = frame(5).with_columns(
            (pl.col("event_time") - pl.duration(hours=1)).alias("receive_time")
        )
        with pytest.raises(ValueError, match="received before"):
            store.write(IID, Timeframe.D1, bad)

    @pytest.mark.parametrize(
        ("column", "value"),
        [("high", 1.0), ("low", 1e9), ("close", 1e9), ("open", 1e9)],
    )
    def test_ohlc_violation_rejected(self, store, column, value):
        bad = frame(5).with_columns(pl.lit(value).alias(column))
        with pytest.raises(ValueError, match="OHLC"):
            store.write(IID, Timeframe.D1, bad)

    def test_non_positive_price_rejected(self, store):
        bad = frame(5).with_columns(pl.lit(0.0).alias("low"))
        with pytest.raises(ValueError, match="OHLC"):
            store.write(IID, Timeframe.D1, bad)

    def test_negative_volume_rejected(self, store):
        bad = frame(5).with_columns(pl.lit(-1.0).alias("volume"))
        with pytest.raises(ValueError, match="OHLC"):
            store.write(IID, Timeframe.D1, bad)


def test_colon_in_instrument_id_is_path_safe(store):
    # Windows forbids ':' in filenames; ids like NSE:INE002A01018 must still work.
    store.write(IID, Timeframe.D1, frame(3))
    assert (store.root / "bars" / "1d" / "NSE_INE002A01018").is_dir()
