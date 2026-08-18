"""Data quality checks — MASTER_PLAN §M1/§M2, PDF §9-10.

A sophisticated model cannot compensate for wrong timestamps, missing
corporate actions, duplicates or survivorship bias. These checks run on every
ingest and their findings are persisted, because a data problem you discover
six months into a backtest has already wasted six months.

Findings are graded, not fatal. Some are genuinely benign (a zero-volume bar in
an illiquid name); the point is that every one is *visible* rather than
silently absorbed into a return series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum

import polars as pl

from core.events import Timeframe

__all__ = ["Finding", "QualityReport", "Severity", "check_bars"]

#: A single bar moving more than this is almost always a split, a bad print, or
#: a genuine crash — all three need a human to look (§9).
EXTREME_RETURN = 0.35

#: Flat OHLC across a whole bar. Legitimate when halted, suspicious otherwise.
FLAT_BAR_RATIO_WARN = 0.05

#: Longest defensible hole in a session market's history. No exchange holiday
#: runs this long; anything beyond it is missing data, not a closure.
MAX_SESSION_GAP_DAYS = 10


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    count: int
    detail: str
    sample: tuple[str, ...] = ()

    def format(self) -> str:
        head = f"[{self.severity.value:<8}] {self.check:<22} {self.count:>6}  {self.detail}"
        if not self.sample:
            return head
        return (
            head
            + "\n"
            + "\n".join(f"                                       {s}" for s in self.sample)
        )


@dataclass
class QualityReport:
    instrument_id: str
    timeframe: Timeframe
    rows: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Clean means no CRITICAL findings. WARNs are for the operator to judge."""
        return not any(f.severity is Severity.CRITICAL for f in self.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity is Severity.CRITICAL)

    def format(self) -> str:
        status = "CLEAN" if self.is_clean else "CRITICAL FINDINGS"
        lines = [f"{self.instrument_id} @ {self.timeframe.value} — {self.rows} rows — {status}"]
        lines.extend(f.format() for f in self.findings)
        if not self.findings:
            lines.append("           no findings")
        return "\n".join(lines)


def _sample_times(frame: pl.DataFrame, limit: int = 3) -> tuple[str, ...]:
    if frame.is_empty():
        return ()
    return tuple(str(t) for t in frame["event_time"].head(limit).to_list())


def check_bars(
    instrument_id: str,
    timeframe: Timeframe,
    frame: pl.DataFrame,
    *,
    expect_continuous: bool = True,
) -> QualityReport:
    """Run every bar-level check.

    Args:
        expect_continuous: True for 24/7 venues, where any missing interval is a
            real gap. False for session markets, where overnight and weekend
            gaps are expected and the session calendar governs instead.
    """
    report = QualityReport(instrument_id, timeframe, frame.height)
    if frame.is_empty():
        report.findings.append(Finding("empty", Severity.CRITICAL, 0, "no rows at all"))
        return report

    _check_duplicates(frame, report)
    _check_ordering(frame, report)
    _check_causality(frame, report)
    _check_prices(frame, report)
    _check_extreme_moves(frame, report)
    _check_flat_bars(frame, report)
    _check_zero_volume(frame, report)
    if expect_continuous:
        _check_gaps(frame, timeframe, report)
    else:
        _check_session_holes(frame, report)
    return report


def _check_duplicates(frame: pl.DataFrame, report: QualityReport) -> None:
    dupes = frame.height - frame["event_time"].n_unique()
    if dupes:
        report.findings.append(
            Finding(
                "duplicate_timestamps",
                Severity.CRITICAL,
                dupes,
                "same event_time appears more than once; returns will double-count",
            )
        )


def _check_ordering(frame: pl.DataFrame, report: QualityReport) -> None:
    if not frame["event_time"].is_sorted():
        report.findings.append(
            Finding(
                "out_of_order",
                Severity.CRITICAL,
                1,
                "event_time is not monotonically increasing",
            )
        )


def _check_causality(frame: pl.DataFrame, report: QualityReport) -> None:
    bad = frame.filter(pl.col("receive_time") < pl.col("event_time"))
    if bad.height:
        report.findings.append(
            Finding(
                "receive_before_event",
                Severity.CRITICAL,
                bad.height,
                "bar received before it occurred — clock drift or adapter bug",
                _sample_times(bad),
            )
        )


def _check_prices(frame: pl.DataFrame, report: QualityReport) -> None:
    bad = frame.filter(
        (pl.col("low") <= 0)
        | (pl.col("high") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
    )
    if bad.height:
        report.findings.append(
            Finding(
                "invalid_ohlc",
                Severity.CRITICAL,
                bad.height,
                "non-positive price or OHLC inconsistency",
                _sample_times(bad),
            )
        )


def _check_extreme_moves(frame: pl.DataFrame, report: QualityReport) -> None:
    """Bar-to-bar jumps large enough to be a split or a bad print.

    For equities this is the primary unadjusted-corporate-action detector: a
    2:1 split shows up as a clean -50% that no news explains.
    """
    moves = frame.select(
        pl.col("event_time"),
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret"),
    ).drop_nulls()
    extreme = moves.filter(pl.col("ret").abs() > EXTREME_RETURN)
    if extreme.height:
        # Polars aggregates are loosely typed; coerce at the boundary.
        worst = float(extreme["ret"].abs().max())  # type: ignore[arg-type]
        report.findings.append(
            Finding(
                "extreme_move",
                Severity.WARN,
                extreme.height,
                f"|return| > {EXTREME_RETURN:.0%} (max {worst:.1%}); "
                "check for splits, bad prints or genuine crashes",
                _sample_times(extreme),
            )
        )


def _check_flat_bars(frame: pl.DataFrame, report: QualityReport) -> None:
    flat = frame.filter(pl.col("high") == pl.col("low"))
    if not flat.height:
        return
    ratio = flat.height / frame.height
    severity = Severity.WARN if ratio > FLAT_BAR_RATIO_WARN else Severity.INFO
    report.findings.append(
        Finding(
            "flat_bars",
            severity,
            flat.height,
            f"{ratio:.1%} of bars have zero range; illiquidity or a stalled feed",
        )
    )


def _check_zero_volume(frame: pl.DataFrame, report: QualityReport) -> None:
    zero = frame.filter(pl.col("volume") == 0)
    if zero.height:
        report.findings.append(
            Finding(
                "zero_volume",
                Severity.INFO,
                zero.height,
                "bars with no traded volume; fills there are not realistic",
            )
        )


def _check_session_holes(frame: pl.DataFrame, report: QualityReport) -> None:
    """Multi-week holes in a session market's history.

    Weekend and overnight gaps are expected on an exchange, which is why the
    continuous gap check is skipped — but a hole longer than any holiday is not
    a gap, it is *missing history*, and it is CRITICAL.

    A backtest across such a hole treats the bars either side as consecutive, so
    a held position banks the entire intervening move in a single step. That
    shows up downstream as absurd skew and kurtosis rather than as an error,
    which is exactly how it survives to the results table.
    """
    deltas = frame.select(
        pl.col("event_time"),
        pl.col("event_time").diff().alias("delta"),
    ).drop_nulls()

    holes = deltas.filter(pl.col("delta") > timedelta(days=MAX_SESSION_GAP_DAYS))
    if not holes.height:
        return

    biggest = str(holes["delta"].max())
    report.findings.append(
        Finding(
            "missing_history",
            Severity.CRITICAL,
            holes.height,
            f"gap(s) longer than {MAX_SESSION_GAP_DAYS} days (largest {biggest}). "
            "Bars either side will be treated as consecutive — backfill the "
            "missing sessions before trusting any result over this window",
            _sample_times(holes),
        )
    )


def _check_gaps(frame: pl.DataFrame, timeframe: Timeframe, report: QualityReport) -> None:
    """Missing intervals on a venue that never closes."""
    expected = timedelta(seconds=timeframe.seconds)
    deltas = frame.select(
        pl.col("event_time"),
        pl.col("event_time").diff().alias("delta"),
    ).drop_nulls()

    gaps = deltas.filter(pl.col("delta") > expected)
    if gaps.height:
        biggest = str(gaps["delta"].max())
        report.findings.append(
            Finding(
                "missing_bars",
                Severity.WARN,
                gaps.height,
                f"gaps larger than one {timeframe.value} interval (largest {biggest}); "
                "venue outage or an interrupted backfill",
                _sample_times(gaps),
            )
        )
