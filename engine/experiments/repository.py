"""Persistence for the research protocol — MASTER_PLAN §5.1, §5.2, §5.3, §5.5.

**Why this module has to exist.** `registry.py` defines `Hypothesis`,
`ExperimentRecord` and `Rejection` and gives each a `to_row()`. Until something
executes those inserts, the protocol is a set of dataclasses that validate
themselves and then evaporate. Two guarantees in particular live in the
*database*, not in Python, and are worth nothing unless rows actually arrive:

1. **The trial counter** (§5.2). `experiments_bump_trials` fires on INSERT and
   increments `hypotheses.n_trials`. Nothing in Python can forget to do it, and
   nothing can be exempted from it — including a throwaway script. A Deflated
   Sharpe computed against an N that was never incremented is not deflated, it
   is just a Sharpe ratio wearing a disguise.
2. **The locked test set** (§5.3). `test_set_access` is UNIQUE on `strategy_id`,
   so a second peek raises an integrity error rather than a warning.

**Locked-test writes are atomic with each other, deliberately.** The experiment
row and its access record are inserted inside one savepoint, so a refused second
access takes *both* with it. You cannot end up with a recorded locked-test result
whose access record failed to insert — the one arrangement that would let the
test set be reused while the audit trail still looks clean. It is a savepoint
rather than a full rollback so that the refusal punishes only the offending
write, instead of destroying whatever else the caller had already done.

**Connections are injected, never constructed here.** The caller owns the
transaction boundary; this module only knows how to write rows. That keeps
`ops` in charge of connection policy and lets tests roll everything back.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from engine.experiments.contracts import (
    Connection,
    Cursor,
    DatasetVersion,
    LockedTestReusedError,
    UnregisteredHypothesisError,
    is_unique_violation,
    period_guard,
)
from engine.experiments.registry import (
    ExperimentRecord,
    Hypothesis,
    HypothesisStatus,
    Rejection,
)
from engine.validation.report import GauntletResult

__all__ = [
    "DatasetVersion",
    "ExperimentRepository",
    "LockedTestReusedError",
    "UnregisteredHypothesisError",
    "period_guard",
]

#: Fixed identifier, never interpolated from input. Savepoint names cannot be
#: parameterised, so the only safe name is a constant one.
_EXPERIMENT_SAVEPOINT = "neutron_experiment_write"


class ExperimentRepository:
    """Reads and writes the research tables.

    Args:
        connection: An open DB-API connection. The caller owns commit/rollback
            except where a method documents otherwise.
    """

    def __init__(self, connection: Connection) -> None:
        self._conn = connection

    # ── hypotheses ──────────────────────────────────────────────────────────

    def register_hypothesis(self, hypothesis: Hypothesis) -> uuid.UUID:
        """Write a pre-registration. Returns its id.

        The 80-character mechanism floor is enforced twice — by
        `Hypothesis.__post_init__` for fast feedback and by a CHECK constraint
        for actual enforcement. This method deliberately does not soften either.
        """
        row = hypothesis.to_row()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO hypotheses (
                    hypothesis_id, statement, economic_mechanism, prediction,
                    success_criteria, kill_criteria,
                    dev_start, dev_end, val_start, val_end, test_start, test_end,
                    status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    row["hypothesis_id"],
                    row["statement"],
                    row["economic_mechanism"],
                    row["prediction"],
                    row["success_criteria"],
                    row["kill_criteria"],
                    row["dev_start"],
                    row["dev_end"],
                    row["val_start"],
                    row["val_end"],
                    row["test_start"],
                    row["test_end"],
                    row["status"],
                ],
            )
        return hypothesis.hypothesis_id

    def trials_for(self, hypothesis_id: uuid.UUID) -> int:
        """Current trial count, as maintained by the database trigger.

        **Read this, never a Python counter.** This is the N that goes into the
        Deflated Sharpe Ratio, and its whole value is that no code path can
        forget to increment it.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT n_trials FROM hypotheses WHERE hypothesis_id = %s",
                [str(hypothesis_id)],
            )
            found = cur.fetchone()
        if found is None:
            raise UnregisteredHypothesisError(hypothesis_id)
        return int(found[0])

    def resolve_hypothesis(self, hypothesis_id: uuid.UUID, status: HypothesisStatus) -> None:
        """Close a hypothesis. `resolved_at` is set by the same statement.

        A CHECK constraint ties the two together, so a resolved hypothesis
        without a timestamp cannot exist.
        """
        if status is HypothesisStatus.OPEN:
            raise ValueError("resolving a hypothesis requires a terminal status")
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE hypotheses SET status = %s, resolved_at = now() WHERE hypothesis_id = %s",
                [status.value, str(hypothesis_id)],
            )

    # ── experiments ─────────────────────────────────────────────────────────

    def record_experiment(
        self,
        experiment: ExperimentRecord,
        dataset_version_id: uuid.UUID,
        strategy_id: str | None = None,
    ) -> uuid.UUID:
        """Insert an experiment, bumping the trial counter as a side effect.

        Args:
            experiment: The run being recorded.
            dataset_version_id: The exact data the run consumed. Required by the
                schema and by reproducibility — "the NSE panel" is not a
                dataset, "the NSE panel with this content hash" is.
            strategy_id: Required when `experiment.period` is LOCKED_TEST,
                because a locked-test run must also write its access record.

        Raises:
            LockedTestReusedError: The strategy already spent its one access.
            ValueError: A locked-test run arrived without a `strategy_id`.
        """
        if experiment.period.is_locked and strategy_id is None:
            raise ValueError(
                "a LOCKED_TEST experiment needs strategy_id so its one-and-only "
                "access can be recorded in the same transaction (§5.3)"
            )

        row = experiment.to_row()
        with self._conn.cursor() as cur:
            # A SAVEPOINT, not a full rollback. The two inserts must be atomic
            # with respect to each other, but a refused second access must not
            # also destroy whatever else the caller had already written in this
            # transaction — that would turn a guard rail into data loss.
            cur.execute(f"SAVEPOINT {_EXPERIMENT_SAVEPOINT}")
            try:
                self._insert_experiment(cur, row, dataset_version_id)
                if experiment.period.is_locked:
                    assert strategy_id is not None  # narrowed by the guard above
                    self._insert_test_access(cur, strategy_id, experiment)
            except Exception as exc:
                cur.execute(f"ROLLBACK TO SAVEPOINT {_EXPERIMENT_SAVEPOINT}")
                # The locked-test UNIQUE violation is the one failure worth
                # naming. Everything else propagates untranslated (§14.1.5).
                if experiment.period.is_locked and is_unique_violation(exc):
                    assert strategy_id is not None
                    raise LockedTestReusedError(strategy_id) from exc
                raise
            cur.execute(f"RELEASE SAVEPOINT {_EXPERIMENT_SAVEPOINT}")
        return experiment.experiment_id

    @staticmethod
    def _insert_experiment(cur: Cursor, row: dict[str, Any], dataset_version_id: uuid.UUID) -> None:
        cur.execute(
            """
            INSERT INTO experiments (
                experiment_id, hypothesis_id, dataset_version_id, period,
                parameters, cost_model, universe, code_commit, seed
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                row["experiment_id"],
                row["hypothesis_id"],
                str(dataset_version_id),
                row["period"],
                row["parameters"],
                row["cost_model"],
                row["universe"],
                row["code_commit"],
                row["seed"],
            ],
        )

    @staticmethod
    def _insert_test_access(cur: Cursor, strategy_id: str, experiment: ExperimentRecord) -> None:
        cur.execute(
            """
            INSERT INTO test_set_access (strategy_id, experiment_id, outcome)
            VALUES (%s, %s, %s)
            """,
            [
                strategy_id,
                str(experiment.experiment_id),
                json.dumps({"fingerprint": experiment.fingerprint()}),
            ],
        )

    def record_metrics(self, experiment_id: uuid.UUID, metrics: dict[str, Any]) -> None:
        """Attach performance metrics to an experiment.

        `deflated_sharpe` and `pbo` are left NULL until the gauntlet runs; the
        schema allows that on purpose, because a metrics row that pretends to
        carry overfitting statistics it never computed is worse than a NULL.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO backtest_metrics (
                    experiment_id, total_return, cagr, sharpe, sortino,
                    max_drawdown, volatility, turnover, hit_rate, n_trades,
                    cost_drag_bps, deflated_sharpe, pbo
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    str(experiment_id),
                    metrics["total_return"],
                    metrics["cagr"],
                    metrics["sharpe"],
                    metrics.get("sortino"),
                    metrics["max_drawdown"],
                    metrics["volatility"],
                    metrics["turnover"],
                    metrics.get("hit_rate"),
                    metrics["n_trades"],
                    metrics["cost_drag_bps"],
                    metrics.get("deflated_sharpe"),
                    metrics.get("pbo"),
                ],
            )

    def record_gauntlet(self, experiment_id: uuid.UUID, result: GauntletResult) -> None:
        """Record one gauntlet check. Re-running a check overwrites its result.

        Takes the check's own result object rather than loose fields, so the
        persisted row cannot drift from what the gauntlet actually reported.

        The UNIQUE constraint makes the upsert explicit rather than letting a
        second run quietly append a second, contradictory verdict for the same
        check. A skipped check is stored as a failure with its reason: an
        unfilled slot must never read as a pass (§5.4).
        """
        detail: dict[str, Any] = {"reason": result.reason, "skipped": result.skipped}
        test_name, passed = result.test, result.passed and not result.skipped
        statistic, threshold = result.statistic, result.threshold
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gauntlet_results (
                    experiment_id, test_name, passed, statistic, threshold, detail
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id, test_name) DO UPDATE SET
                    passed = EXCLUDED.passed,
                    statistic = EXCLUDED.statistic,
                    threshold = EXCLUDED.threshold,
                    detail = EXCLUDED.detail,
                    run_at = now()
                """,
                [
                    str(experiment_id),
                    test_name,
                    passed,
                    statistic,
                    threshold,
                    json.dumps(detail),
                ],
            )

    # ── rejection log ───────────────────────────────────────────────────────

    def record_rejection(self, rejection: Rejection) -> None:
        """Write a killed idea to the log (§5.5).

        Patterns across these rows are worth more than any single entry: if
        every mean-reversion idea dies on cost sensitivity, that says something
        about the holding period, not about mean reversion.
        """
        row = rejection.to_row()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rejection_log (
                    hypothesis_id, experiment_id, killed_at_stage, reason, lesson
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    row["hypothesis_id"],
                    row["experiment_id"],
                    row["killed_at_stage"],
                    row["reason"],
                    row["lesson"],
                ],
            )

    # ── dataset versions ────────────────────────────────────────────────────

    def register_dataset_version(self, version: DatasetVersion) -> uuid.UUID:
        """Register the exact bytes an experiment ran against.

        Idempotent on `(dataset_id, content_hash)`: re-registering identical
        content returns the existing id rather than minting a second one, so
        two runs over the same data are provably over the same data.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataset_versions (
                    dataset_id, content_hash, row_count,
                    coverage_start, coverage_end, storage_uri
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id, content_hash) DO UPDATE
                    SET row_count = dataset_versions.row_count
                RETURNING version_id
                """,
                [
                    version.dataset_id,
                    version.content_hash,
                    version.row_count,
                    version.coverage_start,
                    version.coverage_end,
                    version.storage_uri,
                ],
            )
            found = cur.fetchone()
        if found is None:  # pragma: no cover — RETURNING always yields a row
            raise RuntimeError("dataset_versions insert returned nothing")
        return uuid.UUID(str(found[0]))

    # ── queries ─────────────────────────────────────────────────────────────

    def locked_test_leaks(self) -> list[tuple[str, datetime]]:
        """Locked-test experiments with no access record.

        Should always be empty: `record_experiment` writes both in one
        transaction. A non-empty result means something wrote to `experiments`
        without going through this repository, which is exactly the thing the
        view exists to make visible (§5.3).
        """
        with self._conn.cursor() as cur:
            cur.execute("SELECT experiment_id, created_at FROM locked_test_without_record")
            return [(str(a), b) for a, b in cur.fetchall()]
