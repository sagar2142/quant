"""Research-protocol persistence (§5.1-§5.5).

These tests need Postgres, because the two guarantees under test live in the
database and nowhere else: the trial-counter trigger and the locked-test UNIQUE
constraint. Asserting them against a Python stand-in would test the stand-in.

    docker compose up -d postgres
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

import pytest

from core.clock import UTC
from engine.experiments.registry import (
    DataPeriod,
    ExperimentRecord,
    Hypothesis,
    HypothesisStatus,
    Rejection,
)
from engine.experiments.repository import (
    DatasetVersion,
    ExperimentRepository,
    LockedTestReusedError,
    UnregisteredHypothesisError,
    period_guard,
)
from engine.validation.report import GauntletResult

pytestmark = pytest.mark.integration

MECHANISM = (
    "Index funds mechanically buy at rebalance dates, creating temporary price "
    "pressure that reverts within five sessions as liquidity providers unwind "
    "their inventory against the flow."
)


def make_hypothesis() -> Hypothesis:
    return Hypothesis(
        statement="Index rebalance pressure reverts",
        economic_mechanism=MECHANISM,
        prediction="Positive 5-day reversal after rebalance dates",
        success_criteria={"sharpe": 1.0},
        kill_criteria={"sharpe": 0.3},
        dev_start=date(2019, 1, 1),
        dev_end=date(2021, 1, 1),
        val_start=date(2021, 1, 1),
        val_end=date(2023, 1, 1),
        test_start=date(2023, 1, 1),
        test_end=date(2024, 1, 1),
    )


def make_experiment(hypothesis_id: uuid.UUID, period: DataPeriod) -> ExperimentRecord:
    return ExperimentRecord(
        hypothesis_id=hypothesis_id,
        strategy_name="rebalance_reversal",
        period=period,
        dataset_version="nse-panel-v1",
        parameters={"lookback": 5},
        cost_model="NseEquityCostModel",
        universe=["NSE:INE002A01018"],
        code_commit="abc1234",
        seed=7,
    )


@pytest.fixture
def repo(db):
    return ExperimentRepository(db)


@pytest.fixture
def dataset_version(repo, db):
    """A dataset version to hang experiments off. `datasets` needs a parent row."""
    dataset_id = f"test-{uuid.uuid4().hex[:8]}"
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO datasets (dataset_id, name, source) VALUES (%s, %s, %s)",
            [dataset_id, "test panel", "pytest"],
        )
    return repo.register_dataset_version(
        DatasetVersion(
            dataset_id=dataset_id,
            content_hash=uuid.uuid4().hex,
            row_count=100,
            coverage_start=datetime(2019, 1, 1, tzinfo=UTC),
            coverage_end=datetime(2024, 1, 1, tzinfo=UTC),
            storage_uri="file:///lake/nse",
        )
    )


class TestHypotheses:
    def test_register_and_read_back(self, repo):
        hid = repo.register_hypothesis(make_hypothesis())
        assert repo.trials_for(hid) == 0

    def test_unregistered_hypothesis_is_named(self, repo):
        with pytest.raises(UnregisteredHypothesisError, match="Pre-registration"):
            repo.trials_for(uuid.uuid4())

    def test_thin_mechanism_never_reaches_the_database(self):
        """The CHECK constraint is the enforcement; this is the fast feedback."""
        with pytest.raises(ValueError, match="economic_mechanism"):
            Hypothesis(
                statement="x",
                economic_mechanism="it goes up",
                prediction="up",
                success_criteria={},
                kill_criteria={},
                dev_start=date(2019, 1, 1),
                dev_end=date(2021, 1, 1),
                val_start=date(2021, 1, 1),
                val_end=date(2023, 1, 1),
                test_start=date(2023, 1, 1),
                test_end=date(2024, 1, 1),
            )

    def test_resolving_requires_a_terminal_status(self, repo):
        hid = repo.register_hypothesis(make_hypothesis())
        with pytest.raises(ValueError, match="terminal status"):
            repo.resolve_hypothesis(hid, HypothesisStatus.OPEN)

    def test_resolution_sets_a_timestamp(self, repo, db):
        hid = repo.register_hypothesis(make_hypothesis())
        repo.resolve_hypothesis(hid, HypothesisStatus.REJECTED)
        with db.cursor() as cur:
            cur.execute(
                "SELECT status, resolved_at FROM hypotheses WHERE hypothesis_id = %s", [str(hid)]
            )
            status, resolved = cur.fetchone()
        assert status == "REJECTED"
        assert resolved is not None


class TestTrialCounter:
    """§5.2 — the N in the Deflated Sharpe Ratio."""

    def test_each_experiment_bumps_the_counter(self, repo, dataset_version):
        hid = repo.register_hypothesis(make_hypothesis())
        for expected in (1, 2, 3):
            repo.record_experiment(make_experiment(hid, DataPeriod.DEVELOPMENT), dataset_version)
            assert repo.trials_for(hid) == expected

    def test_the_counter_is_not_maintained_by_this_code(self, repo, dataset_version, db):
        """A raw INSERT bypassing the repository still increments.

        That is the whole point: a counter you can route around is not a
        counter, and a stray tuning script is exactly what routes around it.
        """
        hid = repo.register_hypothesis(make_hypothesis())
        experiment = make_experiment(hid, DataPeriod.DEVELOPMENT)
        row = experiment.to_row()
        with db.cursor() as cur:
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
                    str(dataset_version),
                    row["period"],
                    row["parameters"],
                    row["cost_model"],
                    row["universe"],
                    row["code_commit"],
                    row["seed"],
                ],
            )
        assert repo.trials_for(hid) == 1


class TestLockedTestSet:
    """§5.3 — one access per strategy, ever."""

    def strategy(self, db, hid: uuid.UUID) -> str:
        strategy_id = f"strat-{uuid.uuid4().hex[:8]}"
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO strategies (strategy_id, name, hypothesis_id, family) "
                "VALUES (%s, %s, %s, %s)",
                [strategy_id, "rebalance", str(hid), "reversal"],
            )
        return strategy_id

    def test_locked_run_without_strategy_id_is_refused(self, repo, dataset_version):
        hid = repo.register_hypothesis(make_hypothesis())
        with pytest.raises(ValueError, match="strategy_id"):
            repo.record_experiment(make_experiment(hid, DataPeriod.LOCKED_TEST), dataset_version)

    def test_first_access_succeeds(self, repo, dataset_version, db):
        hid = repo.register_hypothesis(make_hypothesis())
        sid = self.strategy(db, hid)
        repo.record_experiment(
            make_experiment(hid, DataPeriod.LOCKED_TEST), dataset_version, strategy_id=sid
        )
        assert repo.locked_test_leaks() == []

    def test_second_access_is_refused(self, repo, dataset_version, db):
        hid = repo.register_hypothesis(make_hypothesis())
        sid = self.strategy(db, hid)
        repo.record_experiment(
            make_experiment(hid, DataPeriod.LOCKED_TEST), dataset_version, strategy_id=sid
        )
        with pytest.raises(LockedTestReusedError, match="already accessed"):
            repo.record_experiment(
                make_experiment(hid, DataPeriod.LOCKED_TEST), dataset_version, strategy_id=sid
            )

    def test_refused_access_leaves_no_experiment_behind(self, repo, dataset_version, db):
        """The savepoint must take the second experiment row with it.

        A recorded locked-test result whose access record failed to insert is
        the one arrangement that lets the test set be reused while the audit
        trail still looks clean.
        """
        hid = repo.register_hypothesis(make_hypothesis())
        sid = self.strategy(db, hid)
        first = make_experiment(hid, DataPeriod.LOCKED_TEST)
        repo.record_experiment(first, dataset_version, strategy_id=sid)

        second = make_experiment(hid, DataPeriod.LOCKED_TEST)
        with pytest.raises(LockedTestReusedError):
            repo.record_experiment(second, dataset_version, strategy_id=sid)

        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM experiments WHERE experiment_id = %s",
                [str(second.experiment_id)],
            )
            assert cur.fetchone()[0] == 0

    def test_refusal_spares_the_rest_of_the_transaction(self, repo, dataset_version, db):
        """Rolling back to the savepoint, not to the start of the transaction.

        A full rollback would make the guard rail destructive: the refused write
        would also erase the legitimate first access and the hypothesis itself,
        which is a far worse outcome than the mistake it is punishing.
        """
        hid = repo.register_hypothesis(make_hypothesis())
        sid = self.strategy(db, hid)
        first = make_experiment(hid, DataPeriod.LOCKED_TEST)
        repo.record_experiment(first, dataset_version, strategy_id=sid)

        with pytest.raises(LockedTestReusedError):
            repo.record_experiment(
                make_experiment(hid, DataPeriod.LOCKED_TEST), dataset_version, strategy_id=sid
            )

        # The hypothesis, the first experiment and its access record all survive.
        assert repo.trials_for(hid) == 1
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM experiments WHERE experiment_id = %s",
                [str(first.experiment_id)],
            )
            assert cur.fetchone()[0] == 1
        assert repo.locked_test_leaks() == []

    def test_the_connection_stays_usable_after_a_refusal(self, repo, dataset_version, db):
        """An aborted transaction would make every later statement fail.

        Without the savepoint the caller's next query raises
        "current transaction is aborted", and the real error gets buried.
        """
        hid = repo.register_hypothesis(make_hypothesis())
        sid = self.strategy(db, hid)
        repo.record_experiment(
            make_experiment(hid, DataPeriod.LOCKED_TEST), dataset_version, strategy_id=sid
        )
        with pytest.raises(LockedTestReusedError):
            repo.record_experiment(
                make_experiment(hid, DataPeriod.LOCKED_TEST), dataset_version, strategy_id=sid
            )
        # Development runs continue to work on the same connection.
        repo.record_experiment(make_experiment(hid, DataPeriod.DEVELOPMENT), dataset_version)
        assert repo.trials_for(hid) == 2


class TestMetricsAndGauntlet:
    def test_metrics_attach_to_an_experiment(self, repo, dataset_version, db):
        hid = repo.register_hypothesis(make_hypothesis())
        exp = make_experiment(hid, DataPeriod.DEVELOPMENT)
        repo.record_experiment(exp, dataset_version)
        repo.record_metrics(
            exp.experiment_id,
            {
                "total_return": 0.12,
                "cagr": 0.08,
                "sharpe": 0.9,
                "max_drawdown": -0.15,
                "volatility": 0.18,
                "turnover": 2.4,
                "n_trades": 60,
                "cost_drag_bps": 22.0,
            },
        )
        with db.cursor() as cur:
            cur.execute(
                "SELECT deflated_sharpe, pbo FROM backtest_metrics WHERE experiment_id = %s",
                [str(exp.experiment_id)],
            )
            dsr, pbo = cur.fetchone()
        # NULL until the gauntlet runs — better than a fabricated number.
        assert dsr is None
        assert pbo is None

    def test_rerunning_a_check_overwrites_rather_than_appends(self, repo, dataset_version, db):
        hid = repo.register_hypothesis(make_hypothesis())
        exp = make_experiment(hid, DataPeriod.DEVELOPMENT)
        repo.record_experiment(exp, dataset_version)
        repo.record_gauntlet(
            exp.experiment_id,
            GauntletResult("3_deflated_sharpe", passed=False, statistic=0.08),
        )
        repo.record_gauntlet(
            exp.experiment_id,
            GauntletResult("3_deflated_sharpe", passed=True, statistic=0.97),
        )
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*), bool_and(passed) FROM gauntlet_results "
                "WHERE experiment_id = %s AND test_name = %s",
                [str(exp.experiment_id), "3_deflated_sharpe"],
            )
            count, passed = cur.fetchone()
        assert (count, passed) == (1, True)

    def test_a_skipped_check_is_never_stored_as_a_pass(self, repo, dataset_version, db):
        """An unfilled slot must not look like evidence.

        `GauntletResult` defaults `passed` alongside `skipped`, so a skip that
        happened to carry passed=True would silently become a green row.
        """
        hid = repo.register_hypothesis(make_hypothesis())
        exp = make_experiment(hid, DataPeriod.DEVELOPMENT)
        repo.record_experiment(exp, dataset_version)
        repo.record_gauntlet(
            exp.experiment_id,
            GauntletResult("10_placebo", passed=True, skipped=True, reason="no placebo_sharpes"),
        )
        with db.cursor() as cur:
            cur.execute(
                "SELECT passed, detail->>'skipped' FROM gauntlet_results "
                "WHERE experiment_id = %s AND test_name = %s",
                [str(exp.experiment_id), "10_placebo"],
            )
            passed, skipped = cur.fetchone()
        assert passed is False
        assert skipped == "true"


class TestRejectionLog:
    def test_rejection_is_written(self, repo, db):
        hid = repo.register_hypothesis(make_hypothesis())
        repo.record_rejection(
            Rejection(
                hypothesis_id=hid,
                killed_at_stage="cost_sensitivity",
                reason="dies at 2x modelled costs",
                lesson="holding period too short for NSE round-trip costs",
            )
        )
        with db.cursor() as cur:
            cur.execute(
                "SELECT killed_at_stage FROM rejection_log WHERE hypothesis_id = %s", [str(hid)]
            )
            assert cur.fetchone()[0] == "cost_sensitivity"


class TestDatasetVersions:
    def test_identical_content_reuses_the_version(self, repo, db):
        """Two runs over the same bytes must be provably over the same bytes."""
        dataset_id = f"test-{uuid.uuid4().hex[:8]}"
        digest = uuid.uuid4().hex
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (dataset_id, name, source) VALUES (%s, %s, %s)",
                [dataset_id, "panel", "pytest"],
            )
        version = DatasetVersion(
            dataset_id=dataset_id,
            content_hash=digest,
            row_count=100,
            coverage_start=datetime(2019, 1, 1, tzinfo=UTC),
            coverage_end=datetime(2024, 1, 1, tzinfo=UTC),
            storage_uri="file:///lake/nse",
        )
        assert repo.register_dataset_version(version) == repo.register_dataset_version(version)


class TestPeriodGuard:
    """Catches the most expensive mistake in the protocol."""

    def test_development_date_passes(self):
        period_guard(make_hypothesis(), datetime(2020, 6, 1, tzinfo=UTC), DataPeriod.DEVELOPMENT)

    def test_locked_date_declared_as_development_is_caught(self):
        with pytest.raises(ValueError, match="LOCKED_TEST"):
            period_guard(
                make_hypothesis(), datetime(2023, 6, 1, tzinfo=UTC), DataPeriod.DEVELOPMENT
            )

    def test_date_outside_every_period_is_caught(self):
        with pytest.raises(ValueError, match="no registered period"):
            period_guard(
                make_hypothesis(), datetime(2018, 6, 1, tzinfo=UTC), DataPeriod.DEVELOPMENT
            )
