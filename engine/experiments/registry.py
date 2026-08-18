"""Hypothesis and experiment registry — MASTER_PLAN §5.1, §5.2, §5.3, §5.5.

Where pre-registration stops being a good intention and becomes a data
structure.

**The mechanism is mandatory and it is the point.** A hypothesis cannot be
registered without an economic explanation of at least 80 characters, enforced
both here and by a database CHECK constraint. "The z-score reverts" is not a
mechanism. "Index funds mechanically buy at rebalance dates, creating temporary
price pressure that reverts within five sessions as liquidity providers unwind"
is. The length floor is a deliberate obstacle: it is hard to write 80 characters
of causal explanation for an idea you found by searching parameters.

**The trial counter is maintained by a database trigger**, not by this code
(§5.2). A counter you can forget to increment always reads 1, and a Deflated
Sharpe computed with N=1 is not deflated at all. Anything that inserts an
experiment row bumps it, including a stray script.

**The locked test set has a UNIQUE constraint** (§5.3). Second access raises a
database error rather than a warning someone overrides at 2am while convinced
that this time is different.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from core.clock import utc_now

__all__ = [
    "MIN_MECHANISM_CHARS",
    "DataPeriod",
    "ExperimentRecord",
    "Hypothesis",
    "HypothesisStatus",
    "Rejection",
]

#: Mirrors the database CHECK. Kept in sync deliberately: the constraint is the
#: enforcement, this is the fast feedback.
MIN_MECHANISM_CHARS = 80


class DataPeriod(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    LOCKED_TEST = "LOCKED_TEST"

    @property
    def is_locked(self) -> bool:
        return self is DataPeriod.LOCKED_TEST


class HypothesisStatus(str, Enum):
    OPEN = "OPEN"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    ABANDONED = "ABANDONED"


class MechanismTooThinError(ValueError):
    """The economic mechanism is absent or too thin to be one.

    Raised rather than warned. §5.1 calls this the highest-value field in the
    system, and an idea without a mechanism is data mining with extra steps.
    """

    def __init__(self, given: int) -> None:
        super().__init__(
            f"economic_mechanism is {given} characters; at least "
            f"{MIN_MECHANISM_CHARS} are required (§5.1). State *why* this edge "
            "should exist and who is on the other side of the trade."
        )


@dataclass(frozen=True)
class Hypothesis:
    """A pre-registered research question. Written BEFORE any backtest runs."""

    statement: str
    economic_mechanism: str
    prediction: str
    success_criteria: dict[str, Any]
    kill_criteria: dict[str, Any]

    dev_start: date
    dev_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date

    hypothesis_id: uuid.UUID = field(default_factory=uuid.uuid4)
    status: HypothesisStatus = HypothesisStatus.OPEN
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        mechanism = self.economic_mechanism.strip()
        if len(mechanism) < MIN_MECHANISM_CHARS:
            raise MechanismTooThinError(len(mechanism))
        if not self.statement.strip():
            raise ValueError("statement cannot be empty")
        self._check_periods()

    def _check_periods(self) -> None:
        """Periods must be ordered and non-overlapping (§5.3)."""
        if not self.dev_start < self.dev_end <= self.val_start:
            raise ValueError(
                f"development {self.dev_start}..{self.dev_end} must precede "
                f"validation starting {self.val_start}"
            )
        if not self.val_start < self.val_end <= self.test_start:
            raise ValueError(
                f"validation {self.val_start}..{self.val_end} must precede "
                f"the locked test starting {self.test_start}"
            )
        if not self.test_start < self.test_end:
            raise ValueError("locked test period is empty")

    def period_for(self, day: date) -> DataPeriod | None:
        """Which partition a date belongs to, or None if outside all of them."""
        if self.dev_start <= day < self.dev_end:
            return DataPeriod.DEVELOPMENT
        if self.val_start <= day < self.val_end:
            return DataPeriod.VALIDATION
        if self.test_start <= day < self.test_end:
            return DataPeriod.LOCKED_TEST
        return None

    def to_row(self) -> dict[str, Any]:
        """Column mapping for the `hypotheses` table."""
        return {
            "hypothesis_id": str(self.hypothesis_id),
            "statement": self.statement,
            "economic_mechanism": self.economic_mechanism,
            "prediction": self.prediction,
            "success_criteria": json.dumps(self.success_criteria),
            "kill_criteria": json.dumps(self.kill_criteria),
            "dev_start": self.dev_start,
            "dev_end": self.dev_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ExperimentRecord:
    """One backtest run, with everything needed to reproduce it exactly (§M3)."""

    hypothesis_id: uuid.UUID
    strategy_name: str
    period: DataPeriod

    dataset_version: str
    parameters: dict[str, Any]
    cost_model: str
    universe: list[str]
    code_commit: str
    seed: int

    experiment_id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=utc_now)

    def fingerprint(self) -> str:
        """Stable identity of the inputs.

        Two experiments sharing a fingerprint must produce identical results.
        If they do not, something non-deterministic slipped in and the M3 gate
        is broken.
        """
        payload = {
            "strategy": self.strategy_name,
            "dataset": self.dataset_version,
            "parameters": dict(sorted(self.parameters.items())),
            "cost_model": self.cost_model,
            "universe": sorted(self.universe),
            "commit": self.code_commit,
            "seed": self.seed,
        }
        return json.dumps(payload, sort_keys=True)

    def to_row(self) -> dict[str, Any]:
        return {
            "experiment_id": str(self.experiment_id),
            "hypothesis_id": str(self.hypothesis_id),
            "period": self.period.value,
            "parameters": json.dumps(self.parameters),
            "cost_model": json.dumps({"name": self.cost_model}),
            "universe": json.dumps({"members": self.universe}),
            "code_commit": self.code_commit,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Rejection:
    """A killed idea, and why — MASTER_PLAN §5.5.

    The closest thing a solo quant has to institutional memory. It prevents
    re-testing dead ideas, and its *patterns* are more valuable than any single
    entry: "every mean-reversion idea I have dies on cost sensitivity" says
    something about the cost model or the holding period, not about mean
    reversion.
    """

    hypothesis_id: uuid.UUID
    killed_at_stage: str
    reason: str
    lesson: str = ""
    experiment_id: uuid.UUID | None = None
    rejected_at: datetime = field(default_factory=utc_now)

    def to_row(self) -> dict[str, Any]:
        return {
            "hypothesis_id": str(self.hypothesis_id),
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "killed_at_stage": self.killed_at_stage,
            "reason": self.reason,
            "lesson": self.lesson,
        }

    def format(self) -> str:
        line = f"  {self.killed_at_stage:<22} {self.reason}"
        return f"{line}\n    lesson: {self.lesson}" if self.lesson else line
