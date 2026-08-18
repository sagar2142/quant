"""Hypothesis registry (§5.1, §5.3, §5.5).

The decisive test is `TestMechanismIsMandatory`. Everything else in the
research protocol assumes a hypothesis had a reason before it had a backtest.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from engine.experiments.registry import (
    MIN_MECHANISM_CHARS,
    DataPeriod,
    ExperimentRecord,
    Hypothesis,
    HypothesisStatus,
    MechanismTooThinError,
    Rejection,
)

GOOD_MECHANISM = (
    "Index funds mechanically buy at rebalance dates, creating temporary price "
    "pressure that reverts within five sessions as liquidity providers unwind."
)


def hypothesis(**overrides) -> Hypothesis:
    defaults = dict(
        statement="Rebalance-date pressure in NSE index members reverts",
        economic_mechanism=GOOD_MECHANISM,
        prediction="Long-short decile portfolio, net Sharpe > 0.5",
        success_criteria={"sharpe": 0.5},
        kill_criteria={"sharpe": 0.0},
        dev_start=date(2015, 1, 1),
        dev_end=date(2021, 1, 1),
        val_start=date(2021, 1, 1),
        val_end=date(2023, 1, 1),
        test_start=date(2023, 1, 1),
        test_end=date(2026, 1, 1),
    )
    return Hypothesis(**{**defaults, **overrides})


class TestMechanismIsMandatory:
    """§5.1 — the highest-value field in the system."""

    def test_real_mechanism_accepted(self):
        assert hypothesis().economic_mechanism == GOOD_MECHANISM

    def test_thin_mechanism_rejected(self):
        # The classic non-mechanism.
        with pytest.raises(MechanismTooThinError):
            hypothesis(economic_mechanism="The z-score reverts.")

    def test_whitespace_padding_does_not_count(self):
        with pytest.raises(MechanismTooThinError):
            hypothesis(economic_mechanism="  " * 100)

    def test_error_says_what_is_wanted(self):
        with pytest.raises(MechanismTooThinError, match="who is on the other side"):
            hypothesis(economic_mechanism="momentum works")

    def test_threshold_boundary(self):
        assert len("x" * MIN_MECHANISM_CHARS) == MIN_MECHANISM_CHARS
        hypothesis(economic_mechanism="x" * MIN_MECHANISM_CHARS)
        with pytest.raises(MechanismTooThinError):
            hypothesis(economic_mechanism="x" * (MIN_MECHANISM_CHARS - 1))

    def test_empty_statement_rejected(self):
        with pytest.raises(ValueError, match="statement"):
            hypothesis(statement="   ")


class TestDataPartitions:
    """§5.3 — development, validation, locked test, in that order."""

    def test_periods_classified(self):
        h = hypothesis()
        assert h.period_for(date(2018, 6, 1)) is DataPeriod.DEVELOPMENT
        assert h.period_for(date(2022, 6, 1)) is DataPeriod.VALIDATION
        assert h.period_for(date(2024, 6, 1)) is DataPeriod.LOCKED_TEST

    def test_date_outside_all_periods(self):
        assert hypothesis().period_for(date(2010, 1, 1)) is None

    def test_overlapping_periods_rejected(self):
        with pytest.raises(ValueError, match="must precede"):
            hypothesis(dev_end=date(2022, 1, 1))  # runs into validation

    def test_validation_must_precede_locked_test(self):
        with pytest.raises(ValueError, match="must precede"):
            hypothesis(val_end=date(2024, 1, 1))  # runs into the locked period

    def test_empty_locked_period_rejected(self):
        with pytest.raises(ValueError, match="locked test period is empty"):
            hypothesis(test_start=date(2026, 1, 1), test_end=date(2026, 1, 1))

    def test_locked_flag(self):
        assert DataPeriod.LOCKED_TEST.is_locked
        assert not DataPeriod.DEVELOPMENT.is_locked
        assert not DataPeriod.VALIDATION.is_locked


class TestExperimentReproducibility:
    """§M3 — same inputs, same fingerprint, same numbers."""

    def record(self, **overrides) -> ExperimentRecord:
        defaults = dict(
            hypothesis_id=uuid.UUID(int=1),
            strategy_name="xs_momentum",
            period=DataPeriod.DEVELOPMENT,
            dataset_version="abc123",
            parameters={"lookback": 252, "skip": 21},
            cost_model="NSE_EQUITY_DELIVERY",
            universe=["NSE:A", "NSE:B"],
            code_commit="deadbeef",
            seed=42,
        )
        return ExperimentRecord(**{**defaults, **overrides})

    def test_identical_inputs_share_a_fingerprint(self):
        assert self.record().fingerprint() == self.record().fingerprint()

    def test_parameter_order_does_not_matter(self):
        a = self.record(parameters={"lookback": 252, "skip": 21})
        b = self.record(parameters={"skip": 21, "lookback": 252})
        assert a.fingerprint() == b.fingerprint()

    def test_universe_order_does_not_matter(self):
        a = self.record(universe=["NSE:A", "NSE:B"])
        b = self.record(universe=["NSE:B", "NSE:A"])
        assert a.fingerprint() == b.fingerprint()

    def test_different_seed_changes_fingerprint(self):
        assert self.record().fingerprint() != self.record(seed=43).fingerprint()

    def test_different_data_version_changes_fingerprint(self):
        assert self.record().fingerprint() != self.record(dataset_version="x").fingerprint()

    def test_different_commit_changes_fingerprint(self):
        # A result you cannot tie to a commit is not a result (§14.10).
        assert self.record().fingerprint() != self.record(code_commit="other").fingerprint()

    def test_different_cost_model_changes_fingerprint(self):
        assert self.record().fingerprint() != self.record(cost_model="x3").fingerprint()

    def test_each_experiment_gets_a_unique_id(self):
        assert self.record().experiment_id != self.record().experiment_id


class TestRejectionLog:
    """§5.5 — the closest thing a solo quant has to institutional memory."""

    def test_records_stage_and_reason(self):
        rejection = Rejection(
            hypothesis_id=uuid.UUID(int=1),
            killed_at_stage="7_cost_sensitivity",
            reason="Sharpe turns negative at 3x modelled costs",
            lesson="Weekly rebalancing cannot survive NSE delivery costs",
        )
        row = rejection.to_row()
        assert row["killed_at_stage"] == "7_cost_sensitivity"
        assert "3x" in row["reason"]

    def test_formats_with_lesson(self):
        text = Rejection(
            hypothesis_id=uuid.UUID(int=1),
            killed_at_stage="3_deflated_sharpe",
            reason="DSR 0.18 after 18 trials",
            lesson="Sweeping parameters is itself a cost",
        ).format()
        assert "lesson:" in text

    def test_formats_without_lesson(self):
        text = Rejection(
            hypothesis_id=uuid.UUID(int=1),
            killed_at_stage="12_locked_oos",
            reason="failed on locked data",
        ).format()
        assert "lesson:" not in text

    def test_experiment_link_optional(self):
        assert (
            Rejection(hypothesis_id=uuid.UUID(int=1), killed_at_stage="x", reason="y").to_row()[
                "experiment_id"
            ]
            is None
        )


class TestSerialisation:
    def test_hypothesis_row_has_every_column(self):
        row = hypothesis().to_row()
        for column in (
            "hypothesis_id",
            "statement",
            "economic_mechanism",
            "prediction",
            "success_criteria",
            "kill_criteria",
            "dev_start",
            "test_end",
            "status",
        ):
            assert column in row

    def test_status_serialises_as_value(self):
        assert hypothesis().to_row()["status"] == "OPEN"
        assert HypothesisStatus.REJECTED.value == "REJECTED"

    def test_criteria_serialise_as_json(self):
        row = hypothesis().to_row()
        assert row["success_criteria"] == '{"sharpe": 0.5}'
