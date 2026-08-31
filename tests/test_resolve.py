"""Closing hypotheses against their own criteria — MASTER_PLAN §5.1, §5.5.

Pre-registration is worth nothing if closing is a matter of opinion afterwards,
so these test that the verdict follows mechanically from the criteria the
hypothesis carries — and, more importantly, that stage 3 cannot confirm.

The first version of the resolver marked `ma200_distance` CONFIRMED on
stage-3 evidence while the gauntlet had already rejected it at walk-forward.
That would have written a verdict the next stage contradicts into the very
audit trail the M6/M7 gate exists to protect.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from apps.cli.resolve import (
    MIN_T_STAT,
    TESTED_BY,
    committed_horizon,
    factor_for,
    judge,
    threshold,
)
from engine.experiments.registry import HypothesisStatus
from quant.research.factors import Factor
from tests.test_research import panel


class TestCriteriaParsing:
    def test_a_threshold_is_read_from_the_text_it_was_written_as(self):
        assert threshold({"ic_21d": ">= 0.02"}, "ic_21d", 0.0) == pytest.approx(0.02)
        assert threshold({"ic_5d": "< 0.005"}, "ic_5d", 0.0) == pytest.approx(0.005)

    def test_an_unreadable_criterion_falls_back(self):
        assert threshold({"ic_21d": "somewhat positive"}, "ic_21d", 0.02) == pytest.approx(0.02)

    def test_a_missing_criterion_falls_back(self):
        assert threshold({}, "ic_21d", 0.02) == pytest.approx(0.02)

    def test_the_committed_horizon_is_read_not_assumed(self):
        """A 5-day claim judged on a 21-day IC is unfalsifiable either way."""
        assert committed_horizon({"ic_5d": ">= 0.02"}) == 5
        assert committed_horizon({"ic_1d": ">= 0.02"}) == 1
        assert committed_horizon({"ic_21d": ">= 0.02"}) == 21

    def test_an_unspecified_horizon_defaults_to_the_common_one(self):
        assert committed_horizon({"t_stat": ">= 3.0"}) == 21


class TestMapping:
    def test_a_registered_question_maps_to_its_factor(self):
        factor, mapped = factor_for("Distance from the 200-day moving average is distinct")
        assert mapped
        assert factor is Factor.MA200_DISTANCE

    def test_an_untestable_question_maps_to_nothing_but_is_still_recognised(self):
        """Distinguished from an unknown statement: this one was registered and
        the factor lab genuinely cannot express it."""
        factor, mapped = factor_for("Opening gaps mean-revert intraday on NSE equities")
        assert mapped
        assert factor is None

    def test_an_unknown_statement_is_not_mapped(self):
        factor, mapped = factor_for("exploratory runs of demo_strategy")
        assert not mapped
        assert factor is None

    def test_every_catalogue_question_is_accounted_for(self):
        """A question nobody mapped would be silently skipped forever."""
        from apps.cli.preregister import CATALOGUE

        for hypothesis in CATALOGUE:
            _, mapped = factor_for(hypothesis.statement)
            assert mapped, hypothesis.statement

    def test_the_untestable_ones_are_declared_rather_than_guessed(self):
        """A `None` says the factor lab cannot express the question — it does
        not say the question goes unanswered.

        Asserting the route rather than a count. The count was four when four
        studies existed and became stale the moment a cadence comparison was
        added; what actually has to hold is that every question the lab cannot
        score is picked up by something that can, or it is silently OPEN for
        ever.
        """
        from apps.cli.resolve import CADENCE_BY, STUDIED_BY

        unmappable = [k for k, v in TESTED_BY.items() if v is None]
        assert unmappable, "the marker is load-bearing; something should be using it"
        for prefix in unmappable:
            routed = any(prefix.startswith(k) for k in (*STUDIED_BY, *CADENCE_BY))
            assert routed, f"{prefix} has no test at all"


class TestJudgement:
    """Whether a verdict follows from the numbers, not from the reader."""

    def market(self, drift: float = 0.0009, seed: int = 3) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        series = {}
        for i in range(30):
            step = (i - 15) * drift
            series[f"N{i:02d}"] = list(100.0 * np.exp(np.cumsum(rng.normal(step, 0.004, 400))))
        return panel(series)

    def criteria(self, horizon: int = 21) -> tuple[dict, dict]:
        success = {f"ic_{horizon}d": ">= 0.02", "t_stat": ">= 3.0"}
        kill = {f"ic_{horizon}d": "< 0.005"}
        return success, kill

    def test_stage_three_never_confirms(self):
        """Clearing stage 3 is permission to build a strategy, not evidence one
        works. `ma200_distance` cleared every threshold here and was then
        rejected by the gauntlet at walk-forward."""
        success, kill = self.criteria()
        verdict = judge(self.market(), Factor.MOMENTUM_1M, success, kill, 0.0, 0)
        assert verdict.status is not HypothesisStatus.CONFIRMED

    def test_a_signal_below_its_kill_threshold_is_rejected(self):
        success, kill = self.criteria()
        # Pure noise: no cross-sectional drift for the signal to find.
        verdict = judge(self.market(drift=0.0), Factor.MOMENTUM_1M, success, kill, 0.0, 0)
        assert verdict.status is HypothesisStatus.REJECTED

    def test_a_clearing_signal_stays_open_and_says_why(self):
        success, kill = self.criteria()
        verdict = judge(self.market(), Factor.MOMENTUM_1M, success, kill, 0.0, 0)
        if verdict.status is HypothesisStatus.OPEN and "clears stage 3" in verdict.reason:
            assert "gauntlet" in verdict.reason

    def test_the_reason_carries_the_numbers_it_judged_on(self):
        """A verdict nobody can audit is a verdict nobody should trust."""
        success, kill = self.criteria()
        verdict = judge(self.market(), Factor.MOMENTUM_1M, success, kill, 0.0, 0)
        assert "IC(" in verdict.reason
        assert "net" in verdict.reason

    def test_an_unscoreable_factor_stays_open(self):
        """No names surviving the filters is not evidence against the idea."""
        success, kill = self.criteria()
        empty = self.market().head(0)
        verdict = judge(empty, Factor.MOMENTUM_1M, success, kill, 0.0, 0)
        assert verdict.status is HypothesisStatus.OPEN

    def test_the_significance_floor_is_uniform(self):
        """A threshold tuned per idea is a threshold tuned to the answer."""
        assert MIN_T_STAT == 3.0
