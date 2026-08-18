"""The research protocol must be enforced by the database, not by discipline.

MASTER_PLAN §5. Each test here proves a rule cannot be bypassed by an
application bug, a stray script, or the operator at 2am who is certain that
this time is different.
"""

from __future__ import annotations

import json
import uuid

import pytest

pytestmark = pytest.mark.integration

MECHANISM = (
    "Index funds mechanically buy at rebalance dates, creating temporary price "
    "pressure that reverts within five sessions as liquidity providers unwind."
)


def _new_hypothesis(cur, mechanism: str = MECHANISM) -> uuid.UUID:
    cur.execute(
        """
        INSERT INTO hypotheses (
            statement, economic_mechanism, prediction,
            success_criteria, kill_criteria,
            dev_start, dev_end, val_start, val_end, test_start, test_end
        ) VALUES (
            'test hypothesis', %s, 'net Sharpe > 0.5',
            %s, %s,
            '2015-01-01', '2020-12-31', '2021-01-01', '2022-12-31',
            '2023-01-01', '2025-12-31'
        ) RETURNING hypothesis_id
        """,
        (mechanism, json.dumps({"sharpe": 0.5}), json.dumps({"sharpe": 0.0})),
    )
    return cur.fetchone()[0]


def _scaffold(cur) -> tuple[uuid.UUID, str, uuid.UUID]:
    """A hypothesis, a strategy and a dataset version to hang experiments on."""
    hid = _new_hypothesis(cur)
    sid = f"test_strat_{uuid.uuid4().hex[:8]}"
    cur.execute(
        "INSERT INTO strategies (strategy_id, name, hypothesis_id, family) "
        "VALUES (%s, 'Test', %s, 'momentum')",
        (sid, hid),
    )
    did = f"ds_{uuid.uuid4().hex[:8]}"
    cur.execute(
        "INSERT INTO datasets (dataset_id, name, source) VALUES (%s, 'test', 'nse_bhavcopy')",
        (did,),
    )
    cur.execute(
        """
        INSERT INTO dataset_versions (
            dataset_id, content_hash, row_count,
            coverage_start, coverage_end, storage_uri
        ) VALUES (%s, %s, 100, '2015-01-01', '2025-12-31', 'lake://test')
        RETURNING version_id
        """,
        (did, uuid.uuid4().hex),
    )
    return hid, sid, cur.fetchone()[0]


def _add_experiment(cur, hid, sid, dvid, period: str = "DEVELOPMENT") -> uuid.UUID:
    cur.execute(
        """
        INSERT INTO experiments (
            hypothesis_id, dataset_version_id, period,
            cost_model, universe, code_commit, seed
        ) VALUES (%s, %s, %s, %s, %s, 'abc123', 42)
        RETURNING experiment_id
        """,
        (hid, dvid, period, json.dumps({"model": "nse_delivery"}), json.dumps({"n": 100})),
    )
    return cur.fetchone()[0]


class TestEconomicMechanism:
    """§5.1 — the highest-value column in the database."""

    def test_short_mechanism_rejected(self, db):
        with db.cursor() as cur, pytest.raises(Exception, match="economic_mechanism"):
            # "The z-score reverts" is not a mechanism.
            _new_hypothesis(cur, "The z-score reverts.")

    def test_blank_mechanism_rejected(self, db):
        with db.cursor() as cur, pytest.raises(Exception, match="economic_mechanism"):
            _new_hypothesis(cur, "   " * 40)

    def test_real_mechanism_accepted(self, db):
        with db.cursor() as cur:
            assert _new_hypothesis(cur) is not None


class TestTrialCounter:
    """§5.2 — DSR needs N, so N must be automatic."""

    def test_counter_starts_at_zero(self, db):
        with db.cursor() as cur:
            hid = _new_hypothesis(cur)
            cur.execute("SELECT n_trials FROM hypotheses WHERE hypothesis_id = %s", (hid,))
            assert cur.fetchone()[0] == 0

    def test_each_experiment_increments_counter(self, db):
        with db.cursor() as cur:
            hid, sid, dvid = _scaffold(cur)
            for expected in (1, 2, 3):
                _add_experiment(cur, hid, sid, dvid)
                cur.execute("SELECT n_trials FROM hypotheses WHERE hypothesis_id = %s", (hid,))
                assert cur.fetchone()[0] == expected

    def test_counter_survives_without_application_cooperation(self, db):
        # The trigger fires on raw INSERT — no ORM, no service layer involved.
        with db.cursor() as cur:
            hid, sid, dvid = _scaffold(cur)
            _add_experiment(cur, hid, sid, dvid)
            cur.execute("SELECT n_trials FROM hypotheses WHERE hypothesis_id = %s", (hid,))
            assert cur.fetchone()[0] == 1


class TestLockedTestSet:
    """§5.3 — one access per strategy, ever."""

    def test_first_access_recorded(self, db):
        with db.cursor() as cur:
            hid, sid, dvid = _scaffold(cur)
            eid = _add_experiment(cur, hid, sid, dvid, period="LOCKED_TEST")
            cur.execute(
                "INSERT INTO test_set_access (strategy_id, experiment_id, outcome) "
                "VALUES (%s, %s, %s)",
                (sid, eid, json.dumps({"sharpe": 0.8})),
            )

    def test_second_access_raises(self, db):
        with db.cursor() as cur:
            hid, sid, dvid = _scaffold(cur)
            first = _add_experiment(cur, hid, sid, dvid, period="LOCKED_TEST")
            cur.execute(
                "INSERT INTO test_set_access (strategy_id, experiment_id, outcome) "
                "VALUES (%s, %s, %s)",
                (sid, first, json.dumps({"sharpe": 0.8})),
            )
            second = _add_experiment(cur, hid, sid, dvid, period="LOCKED_TEST")
            # A failed strategy may not be tweaked and re-run against the test set.
            with pytest.raises(Exception, match="one_locked_test_per_strategy"):
                cur.execute(
                    "INSERT INTO test_set_access (strategy_id, experiment_id, outcome) "
                    "VALUES (%s, %s, %s)",
                    (sid, second, json.dumps({"sharpe": 1.9})),
                )

    def test_unrecorded_locked_test_is_visible(self, db):
        with db.cursor() as cur:
            hid, sid, dvid = _scaffold(cur)
            eid = _add_experiment(cur, hid, sid, dvid, period="LOCKED_TEST")
            cur.execute(
                "SELECT experiment_id FROM locked_test_without_record WHERE experiment_id = %s",
                (eid,),
            )
            assert cur.fetchone() is not None


class TestAuditImmutability:
    """§21 — an audit log that can be edited is not an audit log."""

    def _insert(self, cur) -> int:
        cur.execute(
            "INSERT INTO audit_events (actor, action, entity_type, entity_id) "
            "VALUES ('tester', 'CREATE', 'order', 'o-1') RETURNING event_id"
        )
        return cur.fetchone()[0]

    def test_insert_allowed(self, db):
        with db.cursor() as cur:
            assert self._insert(cur) > 0

    def test_update_rejected(self, db):
        with db.cursor() as cur:
            eid = self._insert(cur)
            with pytest.raises(Exception, match="append-only"):
                cur.execute(
                    "UPDATE audit_events SET action = 'TAMPERED' WHERE event_id = %s", (eid,)
                )

    def test_delete_rejected(self, db):
        with db.cursor() as cur:
            eid = self._insert(cur)
            with pytest.raises(Exception, match="append-only"):
                cur.execute("DELETE FROM audit_events WHERE event_id = %s", (eid,))


class TestDataIntegrityConstraints:
    def test_period_ordering_enforced(self, db):
        with db.cursor() as cur, pytest.raises(Exception, match="periods_ordered"):
            cur.execute(
                """
                INSERT INTO hypotheses (
                    statement, economic_mechanism, prediction,
                    success_criteria, kill_criteria,
                    dev_start, dev_end, val_start, val_end, test_start, test_end
                ) VALUES ('x', %s, 'y', '{}', '{}',
                    '2015-01-01', '2020-12-31',
                    '2019-01-01', '2022-12-31',   -- overlaps development
                    '2023-01-01', '2025-12-31')
                """,
                (MECHANISM,),
            )

    def test_kill_switch_requires_context(self, db):
        with db.cursor() as cur, pytest.raises(Exception, match="engaged_has_context"):
            cur.execute("UPDATE kill_switch SET engaged = TRUE WHERE id")

    def test_kill_switch_engages_with_context(self, db):
        with db.cursor() as cur:
            cur.execute(
                "UPDATE kill_switch SET engaged = TRUE, engaged_by = 'operator', "
                "reason = 'data staleness', engaged_at = now() WHERE id"
            )
            cur.execute("SELECT engaged FROM kill_switch")
            assert cur.fetchone()[0] is True

    def test_order_cannot_overfill(self, db):
        with db.cursor() as cur:
            _hid, sid, _dvid = _scaffold(cur)
            cur.execute(
                "INSERT INTO instruments (instrument_id, symbol, asset_class, exchange, "
                "currency, tick_size) VALUES ('T:X', 'X', 'EQUITY', 'NSE', 'INR', 0.05)"
            )
            with pytest.raises(Exception, match="no_overfill"):
                cur.execute(
                    """
                    INSERT INTO orders (
                        idempotency_key, mode, strategy_id, instrument_id,
                        side, order_type, quantity, filled_quantity, decision_time
                    ) VALUES (%s, 'PAPER', %s, 'T:X', 'BUY', 'MARKET', 100, 150, now())
                    """,
                    (uuid.uuid4().hex, sid),
                )

    def test_idempotency_key_is_unique(self, db):
        with db.cursor() as cur:
            _hid, sid, _dvid = _scaffold(cur)
            cur.execute(
                "INSERT INTO instruments (instrument_id, symbol, asset_class, exchange, "
                "currency, tick_size) VALUES ('T:Y', 'Y', 'EQUITY', 'NSE', 'INR', 0.05)"
            )
            key = uuid.uuid4().hex
            sql = """
                INSERT INTO orders (
                    idempotency_key, mode, strategy_id, instrument_id,
                    side, order_type, quantity, decision_time
                ) VALUES (%s, 'PAPER', %s, 'T:Y', 'BUY', 'MARKET', 100, now())
            """
            cur.execute(sql, (key, sid))
            # A retry after a network timeout must never double-fill.
            with pytest.raises(Exception, match="idempotency_key"):
                cur.execute(sql, (key, sid))

    def test_max_drawdown_must_be_negative(self, db):
        with db.cursor() as cur:
            hid, sid, dvid = _scaffold(cur)
            eid = _add_experiment(cur, hid, sid, dvid)
            with pytest.raises(Exception, match="max_drawdown"):
                cur.execute(
                    """
                    INSERT INTO backtest_metrics (
                        experiment_id, total_return, cagr, sharpe, max_drawdown,
                        volatility, turnover, n_trades, cost_drag_bps
                    ) VALUES (%s, 0.2, 0.1, 1.0, 0.15, 0.12, 2.0, 50, 30)
                    """,
                    (eid,),
                )
