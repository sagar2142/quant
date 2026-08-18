"""Data quality checks (§M1/§M2).

Each check gets a positive case (it fires on the defect) and a negative case
(clean data stays clean). A check that never fires is worse than no check: it
manufactures confidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from core.clock import UTC
from core.events import Timeframe
from data.quality.checks import Finding, QualityReport, Severity, check_bars

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def bars(n: int = 20, step: timedelta = timedelta(days=1)) -> pl.DataFrame:
    event = [T0 + step * i for i in range(n)]
    return pl.DataFrame(
        {
            "event_time": event,
            "receive_time": [t + timedelta(milliseconds=100) for t in event],
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
            "trades": [50] * n,
        },
        schema_overrides={
            "event_time": pl.Datetime("us", "UTC"),
            "receive_time": pl.Datetime("us", "UTC"),
        },
    )


def run(frame: pl.DataFrame, **kwargs) -> QualityReport:
    return check_bars("TEST:X", Timeframe.D1, frame, **kwargs)


def checks_fired(report: QualityReport) -> set[str]:
    return {f.check for f in report.findings}


class TestCleanData:
    def test_clean_data_has_no_findings(self):
        report = run(bars())
        assert report.findings == []
        assert report.is_clean

    def test_clean_report_formats(self):
        assert "no findings" in run(bars()).format()


class TestCriticalFindings:
    def test_empty_frame(self):
        report = run(bars().head(0))
        assert not report.is_clean
        assert "empty" in checks_fired(report)

    def test_duplicate_timestamps(self):
        frame = pl.concat([bars(5), bars(5)]).sort("event_time")
        report = run(frame)
        assert "duplicate_timestamps" in checks_fired(report)
        assert not report.is_clean

    def test_out_of_order(self):
        report = run(bars(10).reverse())
        assert "out_of_order" in checks_fired(report)
        assert not report.is_clean

    def test_receive_before_event(self):
        frame = bars(10).with_columns(
            (pl.col("event_time") - pl.duration(hours=1)).alias("receive_time")
        )
        report = run(frame)
        assert "receive_before_event" in checks_fired(report)
        assert not report.is_clean

    def test_invalid_ohlc(self):
        frame = bars(10).with_columns(pl.lit(50.0).alias("high"))
        report = run(frame)
        assert "invalid_ohlc" in checks_fired(report)
        assert not report.is_clean

    def test_non_positive_price(self):
        frame = bars(10).with_columns(pl.lit(0.0).alias("low"))
        report = run(frame)
        assert "invalid_ohlc" in checks_fired(report)

    def test_critical_count(self):
        frame = pl.concat([bars(5), bars(5)]).sort("event_time")
        assert run(frame).critical_count >= 1


class TestExtremeMoves:
    """The primary unadjusted-split detector for equities."""

    def test_split_shaped_drop_flagged(self):
        frame = bars(10)
        closes = [100.0] * 5 + [50.0] * 5  # a clean 2:1 split
        frame = frame.with_columns(pl.Series("close", closes)).with_columns(
            pl.min_horizontal("low", "close").alias("low"),
            pl.max_horizontal("high", "close").alias("high"),
        )
        report = run(frame)
        assert "extreme_move" in checks_fired(report)

    def test_normal_moves_not_flagged(self):
        frame = bars(10)
        closes = [100.0 + i for i in range(10)]
        frame = frame.with_columns(pl.Series("close", closes)).with_columns(
            pl.max_horizontal("high", "close").alias("high")
        )
        assert "extreme_move" not in checks_fired(run(frame))

    def test_extreme_move_is_warn_not_critical(self):
        # A 40% daily move can be real. Flag it; do not fail the gate on it.
        frame = bars(10)
        closes = [100.0] * 5 + [50.0] * 5
        frame = frame.with_columns(pl.Series("close", closes)).with_columns(
            pl.min_horizontal("low", "close").alias("low")
        )
        report = run(frame)
        assert report.is_clean


class TestGaps:
    def test_missing_interval_flagged(self):
        frame = bars(20).filter(pl.col("event_time") != T0 + timedelta(days=10))
        report = run(frame)
        assert "missing_bars" in checks_fired(report)

    def test_continuous_data_has_no_gaps(self):
        assert "missing_bars" not in checks_fired(run(bars(20)))

    def test_gap_check_skipped_for_session_markets(self):
        # Weekends are not gaps on NSE; the session calendar governs instead.
        frame = bars(20).filter(pl.col("event_time") != T0 + timedelta(days=10))
        report = run(frame, expect_continuous=False)
        assert "missing_bars" not in checks_fired(report)


class TestMissingHistory:
    """A hole longer than any holiday is missing history, not a closure.

    This is the failure that produced a 103% annualised volatility and a
    kurtosis of 116 on a real NSE panel: two eras stitched together with a
    five-year hole, so a held position banked the whole intervening move in one
    step. It surfaced downstream as absurd statistics rather than as an error,
    which is exactly how such a thing reaches a results table.
    """

    def two_eras(self) -> pl.DataFrame:
        early = bars(30)
        late = bars(30)
        shifted = late.with_columns(
            (pl.col("event_time") + pl.duration(days=1800)).alias("event_time"),
            (pl.col("receive_time") + pl.duration(days=1800)).alias("receive_time"),
        )
        return pl.concat([early, shifted]).sort("event_time")

    def test_multi_year_hole_is_critical(self):
        report = run(self.two_eras(), expect_continuous=False)
        assert "missing_history" in checks_fired(report)
        assert not report.is_clean

    def test_message_says_what_to_do(self):
        report = run(self.two_eras(), expect_continuous=False)
        finding = next(f for f in report.findings if f.check == "missing_history")
        assert "backfill" in finding.detail
        assert "consecutive" in finding.detail

    def test_normal_weekend_gaps_are_not_flagged(self):
        # Mon-Fri only: three-day weekend gaps must stay silent.
        weekdays = [d for i in range(40) if (d := T0 + timedelta(days=i)).weekday() < 5]
        frame = bars(len(weekdays)).with_columns(
            pl.Series("event_time", weekdays).cast(pl.Datetime("us", "UTC")),
            pl.Series("receive_time", weekdays).cast(pl.Datetime("us", "UTC")),
        )
        assert "missing_history" not in checks_fired(run(frame, expect_continuous=False))

    def test_a_long_holiday_break_is_tolerated(self):
        # A week-long exchange closure is unusual but real; 10 days is the line.
        frame = bars(20).filter(
            ~pl.col("event_time").is_between(T0 + timedelta(days=5), T0 + timedelta(days=11))
        )
        assert "missing_history" not in checks_fired(run(frame, expect_continuous=False))


class TestInformationalFindings:
    def test_flat_bars_detected(self):
        frame = bars(10).with_columns(pl.lit(100.0).alias("high"), pl.lit(100.0).alias("low"))
        report = run(frame)
        assert "flat_bars" in checks_fired(report)
        # All bars flat is a stalled feed — worth a WARN, not a CRITICAL.
        assert report.is_clean

    def test_zero_volume_detected(self):
        frame = bars(10).with_columns(pl.lit(0.0).alias("volume"))
        report = run(frame)
        assert "zero_volume" in checks_fired(report)
        assert report.is_clean


class TestFindingFormatting:
    def test_finding_renders(self):
        text = Finding("check_x", Severity.WARN, 3, "something odd").format()
        assert "WARN" in text
        assert "check_x" in text

    def test_finding_includes_samples(self):
        text = Finding("c", Severity.INFO, 1, "d", sample=("2024-01-01",)).format()
        assert "2024-01-01" in text

    @pytest.mark.parametrize("severity", list(Severity))
    def test_all_severities_render(self, severity):
        assert severity.value in Finding("c", severity, 1, "d").format()
