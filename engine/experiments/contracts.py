"""Contracts of the research persistence layer — MASTER_PLAN §5.1, §5.3.

The types and rules a caller must hold to *before* touching the database:
what a connection must look like, what a dataset version claims, which errors
name protocol violations, and the guard that keeps a run inside the period it
declared. `repository.py` executes SQL; everything here is meaningful without
a database at all, which is why it lives apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from engine.experiments.registry import DataPeriod, Hypothesis

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Connection",
    "Cursor",
    "DatasetVersion",
    "LockedTestReusedError",
    "UnregisteredHypothesisError",
    "is_unique_violation",
    "period_guard",
]


class Cursor(Protocol):
    """The slice of DB-API this layer uses.

    Structural rather than concrete so tests need no driver, and so swapping
    psycopg for anything else touches nothing here.
    """

    def execute(self, query: str, params: Sequence[Any] | None = ...) -> Any: ...
    def fetchone(self) -> tuple[Any, ...] | None: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...


class Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


@dataclass(frozen=True)
class DatasetVersion:
    """The exact data an experiment consumed.

    `content_hash` is what makes reproducibility checkable: "the NSE panel" is
    not a dataset, "the NSE panel with this hash" is. Two experiments claiming
    the same dataset and producing different numbers are only detectable if the
    claim was this specific.
    """

    dataset_id: str
    content_hash: str
    row_count: int
    coverage_start: datetime
    coverage_end: datetime
    storage_uri: str


class LockedTestReusedError(RuntimeError):
    """A strategy tried to touch the locked test set twice (§5.3).

    Not recoverable and not overridable. Once a strategy has seen the locked
    period, any subsequent number it produces on that period is contaminated by
    the first look, and no amount of good intent uncontaminates it.
    """

    def __init__(self, strategy_id: str) -> None:
        super().__init__(
            f"strategy {strategy_id!r} has already accessed the locked test set. "
            "There is no second access. Fork a new strategy_id only if the "
            "change is a genuinely new hypothesis, not a retune of this one."
        )


class UnregisteredHypothesisError(RuntimeError):
    """An experiment referenced a hypothesis that was never written down.

    The foreign key would catch this anyway; catching it here says *why* it is
    forbidden rather than reporting a constraint name.
    """

    def __init__(self, hypothesis_id: UUID) -> None:
        super().__init__(
            f"hypothesis {hypothesis_id} is not registered. Pre-registration "
            "comes first (§5.1) — an experiment whose hypothesis was written "
            "afterwards is a result in search of a question."
        )


def is_unique_violation(exc: BaseException) -> bool:
    """Whether a driver exception is a UNIQUE violation.

    Matched on SQLSTATE 23505 rather than on an exception class, so nothing
    here imports psycopg and the layer stays substitutable.
    """
    code = getattr(exc, "sqlstate", None) or getattr(getattr(exc, "diag", None), "sqlstate", None)
    return str(code) == "23505"


def period_guard(hypothesis: Hypothesis, day: datetime, allowed: DataPeriod) -> None:
    """Fail if `day` is not in the period the caller claims to be using.

    Cheap, and it catches the most expensive mistake in the protocol: running
    what you believe is a development backtest over dates that are actually
    inside the locked test set.
    """
    actual = hypothesis.period_for(day.date())
    if actual is not allowed:
        raise ValueError(
            f"{day.date()} falls in {actual.value if actual else 'no registered period'}, "
            f"but this run declared {allowed.value}. Fix the dates, not the declaration."
        )
