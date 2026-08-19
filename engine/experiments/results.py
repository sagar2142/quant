"""Recording what a run produced — MASTER_PLAN §5.4.

Separated from `repository.py` so that "how do I reach the research tables"
and "what does a finished run look like" are different files. The repository
owns identity and the protocol guarantees — the trial counter, the locked test
set. This owns the numbers a run leaves behind.

Mixed into `ExperimentRepository` at class level rather than passed around,
because every method here needs the same connection and the same transaction.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from engine.experiments.registry import Rejection
from engine.validation.report import GauntletResult

__all__ = ["ResultsMixin"]


class ResultsMixin:
    """Metrics, gauntlet verdicts and rejections.

    Expects `self._conn` from `ExperimentRepository`.
    """

    _conn: Any

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
