"""Paper against backtest, and whether the cycle is running — §M9, §35.

Two failures matter here and both are silent:

  - a drift number computed from **misaligned sessions**, which is a plausible
    tracking error from a comparison of unrelated days
  - a stalled scheduler that looks exactly like a **quiet market**, because
    "no cycle ran" and "a cycle ran and did nothing" render identically unless
    something separates them
"""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from apps.api.paper_status import paper_status_of
from quant.math.metrics.drift import MIN_SESSIONS, compare_curves
from trading.paper.state import PaperStateStore

START = date(2026, 8, 3)


def curve(values: list[float], start: date = START) -> dict[date, float]:
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


def flat(n: int, base: float = 1_000_000.0, step: float = 1000.0) -> dict[date, float]:
    return curve([base + i * step for i in range(n)])


class TestDriftRefusesShortSamples:
    def test_no_overlap_is_unmeasurable(self) -> None:
        far = curve([1_000_000.0] * 12, START + timedelta(days=365))
        found = compare_curves(flat(12), far)
        assert not found.measurable
        assert found.sessions == 0

    def test_below_the_minimum_is_unmeasurable(self) -> None:
        """A tracking error from three points is noise wearing a decimal point."""
        found = compare_curves(flat(4), flat(4))
        assert not found.measurable
        assert str(MIN_SESSIONS) in found.reason

    def test_unmeasurable_is_not_zero_drift(self) -> None:
        """Absence is not agreement. It has not looked."""
        found = compare_curves(flat(3), flat(3))
        assert found.tracking_error == 0.0
        assert not found.measurable

    def test_at_the_minimum_it_measures(self) -> None:
        assert compare_curves(flat(MIN_SESSIONS), flat(MIN_SESSIONS)).measurable


class TestDriftAlignment:
    """Sessions are matched by date, never by position."""

    def test_a_missed_cycle_costs_one_session_not_the_rest(self) -> None:
        model = flat(14)
        paper = {d: v for d, v in model.items() if d != START + timedelta(days=3)}
        found = compare_curves(paper, model)
        assert found.sessions == 13
        assert found.measurable

    def test_identical_curves_have_no_drift(self) -> None:
        found = compare_curves(flat(14), flat(14))
        assert found.tracking_error == pytest.approx(0.0, abs=1e-12)
        assert found.gap == pytest.approx(0.0, abs=1e-12)

    def test_offset_dates_never_silently_compare(self) -> None:
        """Same shape, different dates. Zipping by position would compare them
        happily and report a tracking error from unrelated days."""
        aligned = compare_curves(flat(12), flat(12))
        shifted = compare_curves(flat(12), curve([1_000_000.0] * 12, START + timedelta(days=60)))
        assert aligned.measurable
        assert not shifted.measurable
        assert shifted.sessions == 0


class TestDriftDirection:
    def test_paper_trailing_the_model_is_a_negative_gap(self) -> None:
        """The expected direction: paper pays costs the backtest estimated."""
        found = compare_curves(flat(14, step=900.0), flat(14, step=1000.0))
        assert found.gap < 0
        assert found.mean_drift < 0

    def test_paper_ahead_of_the_model_is_reported_not_hidden(self) -> None:
        """Worse than trailing — it usually means they are different strategies."""
        found = compare_curves(flat(14, step=1200.0), flat(14, step=1000.0))
        assert found.gap > 0

    def test_curves_can_agree_at_the_end_and_disagree_throughout(self) -> None:
        """Why tracking error is the headline rather than total return."""
        model = flat(14)
        noisy = dict(model)
        for i, day in enumerate(sorted(noisy)[1:-1], start=1):
            noisy[day] = model[day] + (5000.0 if i % 2 else -5000.0)
        found = compare_curves(noisy, model)
        assert found.gap == pytest.approx(0.0, abs=1e-9)
        assert found.tracking_error > 0.01

    def test_the_worst_session_is_named(self) -> None:
        """A level shift from one session onward, so exactly one return differs
        and the reported session is unambiguous. Depressing a single point
        instead would produce two large differences — the drop and the rebound —
        and the rebound is the larger of them, which is correct and is not what
        a reader means by "the worst session"."""
        model = flat(14)
        days = sorted(model)
        bad = days[7]
        paper = {d: (v * 0.9 if d >= bad else v) for d, v in model.items()}
        found = compare_curves(paper, model)
        assert found.worst_session == bad
        assert found.worst_gap < 0

    def test_a_zero_start_is_refused(self) -> None:
        found = compare_curves(curve([0.0] * 12), flat(12))
        assert not found.measurable
        assert "zero" in found.reason


class TestPaperStatus:
    """Whether the cycle is running — the question no screen could answer."""

    def store(self, cycles: int = 0, strategy: str = "CrossSectionalMomentum"):
        store = PaperStateStore(Path(tempfile.mkdtemp()))
        for i in range(cycles):
            store.append_equity(
                START + timedelta(days=i),
                Decimal(1_000_000 + i * 900),
                Decimal(50_000),
                Decimal(120 + i),
                strategy=strategy,
                parameters={"lookback_bars": 60},
            )
        return store

    def test_nothing_run_is_not_started(self) -> None:
        """Distinct from a cycle that ran and did nothing."""
        found = paper_status_of(self.store(0))
        assert not found.started
        assert found.cycles == 0
        assert found.last_session is None

    def test_a_running_clock_reports_its_progress(self) -> None:
        found = paper_status_of(self.store(12))
        assert found.started
        assert found.cycles == 12
        assert found.sessions_required == 30
        assert found.first_session == START.isoformat()

    def test_the_strategy_comes_from_the_log(self) -> None:
        assert paper_status_of(self.store(12)).strategies == ["CrossSectionalMomentum"]

    def test_a_changed_strategy_mid_run_is_visible(self) -> None:
        """Two experiments, not one — and the curve should not be read as one."""
        store = self.store(6)
        for i in range(6, 12):
            store.append_equity(
                START + timedelta(days=i),
                Decimal(1_000_000),
                Decimal(0),
                Decimal(0),
                strategy="MeanReversion",
                parameters={"lookback_bars": 5},
            )
        assert len(paper_status_of(store).strategies) == 2

    def test_days_since_last_is_reported(self) -> None:
        """The number that says a scheduler stopped firing."""
        found = paper_status_of(self.store(3))
        assert found.days_since_last is not None
        assert found.days_since_last > 0

    def test_drift_is_refused_below_the_minimum(self) -> None:
        found = paper_status_of(self.store(3))
        assert found.drift_reason
        assert str(MIN_SESSIONS) in found.drift_reason

    def test_drift_becomes_computable_with_enough_cycles(self) -> None:
        assert paper_status_of(self.store(MIN_SESSIONS)).drift_reason == ""

    def test_cycles_without_a_spec_cannot_be_compared(self) -> None:
        """A drift number against guessed parameters is a different strategy."""
        found = paper_status_of(self.store(MIN_SESSIONS, strategy=""))
        assert "no strategy spec" in found.drift_reason
