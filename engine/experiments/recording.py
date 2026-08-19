"""Recording a validation run — MASTER_PLAN §5.1, §5.2, §5.4.

**Why this exists.** `ExperimentRepository` could write every research table and
nothing called it. The tables stayed empty, so the trial-counter trigger never
fired, so the Deflated Sharpe Ratio was computed against whatever the current
sweep happened to count — a number that resets to fifteen every time you run
the command.

That is not a small inaccuracy. DSR deflates the observed Sharpe by the
*expected maximum* over N trials. A 2.0 Sharpe passes comfortably at N=15 and
fails at N=225. Counting only the current run means the deflation never grows,
which is the same as not deflating at all.

**The trial count comes back from the database, never from Python.** The
`experiments_bump_trials` trigger increments on INSERT, so a stray script, a
forgotten sweep and a deliberate run all count identically. Reading the counter
back is what makes N honest.

**Connections are injected.** This module knows what a run is worth recording;
`ops` decides how to reach Postgres.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from dataclasses import dataclass
from datetime import date

import polars as pl

from engine.experiments.contracts import Connection, DatasetVersion
from engine.experiments.registry import DataPeriod, ExperimentRecord, Hypothesis
from engine.experiments.repository import ExperimentRepository
from engine.validation.report import GauntletReport

__all__ = [
    "RunInputs",
    "RunRecord",
    "code_commit",
    "dataset_version_for",
    "exploratory_hypothesis",
    "record_run",
]

#: Identifies the NSE panel in `datasets`. One logical dataset, many versions.
NSE_DATASET_ID = "nse-eod-panel"

#: Rows sampled when fingerprinting the panel. Hashing 3.3M rows on every run
#: costs seconds for no extra safety — the shape plus a deterministic sample
#: distinguishes any two panels that differ in a way a backtest would notice.
FINGERPRINT_ROWS = 2000


@dataclass(frozen=True)
class RunRecord:
    """What a recorded run produced, and the honest trial count behind it."""

    experiment_id: uuid.UUID
    hypothesis_id: uuid.UUID
    #: Read back from the database *after* this run was inserted, so it
    #: includes this run. This is the N that belongs in the DSR.
    trials: int


def code_commit() -> str:
    """Current git SHA, or a marker when the tree is not a repository.

    Recorded on every experiment: a number you cannot tie to the code that
    produced it is not reproducible, and §M3 turns on exactly that.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git resolved from PATH
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def dataset_version_for(history: pl.DataFrame, storage_uri: str) -> DatasetVersion:
    """Fingerprint the exact panel a run consumed.

    "The NSE panel" is not a dataset; "the NSE panel with this hash" is. Two
    runs claiming the same data and producing different numbers are only
    detectable if the claim was this specific.
    """
    stamps = history["event_time"]
    ordered = history.sort(["event_time", "instrument_id"])
    step = max(1, ordered.height // FINGERPRINT_ROWS)
    sample = ordered.gather_every(step).select("event_time", "instrument_id", "close")

    digest = hashlib.sha256()
    digest.update(str(history.height).encode())
    digest.update(str(history.columns).encode())
    for row in sample.iter_rows():
        digest.update(str(row).encode())

    return DatasetVersion(
        dataset_id=NSE_DATASET_ID,
        content_hash=digest.hexdigest(),
        row_count=history.height,
        coverage_start=stamps.min(),  # type: ignore[arg-type]
        coverage_end=stamps.max(),  # type: ignore[arg-type]
        storage_uri=storage_uri,
    )


def exploratory_hypothesis(strategy_name: str) -> Hypothesis:
    """A standing hypothesis for runs made without pre-registration.

    **Not a loophole — the opposite.** §5.1 wants an economic mechanism written
    before the backtest, and most exploratory runs will not have one. The wrong
    response is to let those runs go uncounted, because they are exactly the
    trials that inflate the maximum Sharpe you eventually find. They are booked
    against this shared hypothesis so the counter keeps rising, and the mechanism
    text says plainly that no mechanism was offered.

    Every exploratory run of a given strategy shares one hypothesis id, so the
    trial count accumulates across sessions rather than resetting.
    """
    return Hypothesis(
        statement=f"exploratory runs of {strategy_name}",
        economic_mechanism=(
            "No economic mechanism was pre-registered for this run. It is "
            "recorded so that the trial counter reflects every backtest "
            "attempted, because an uncounted trial is what inflates the "
            "maximum Sharpe eventually discovered (see MASTER_PLAN 5.2)."
        ),
        prediction="none registered; exploratory",
        success_criteria={"note": "exploratory"},
        kill_criteria={"note": "exploratory"},
        dev_start=date(2019, 1, 1),
        dev_end=date(2023, 1, 1),
        val_start=date(2023, 1, 1),
        val_end=date(2025, 1, 1),
        test_start=date(2025, 1, 1),
        test_end=date(2027, 1, 1),
        # Deterministic id from the strategy name: every exploratory run of the
        # same strategy books against one row, so trials accumulate across
        # sessions instead of resetting.
        hypothesis_id=uuid.uuid5(uuid.NAMESPACE_URL, f"neutron:exploratory:{strategy_name}"),
    )


@dataclass(frozen=True)
class RunInputs:
    """Everything that identifies one run, grouped so it travels together.

    These fields are the reproducibility contract (§M3): strategy, parameters,
    universe, data, seed and cost model. Splitting them across a long argument
    list invites a caller to record five of six and produce a row that cannot
    be replayed.
    """

    hypothesis: Hypothesis
    strategy_name: str
    parameters: dict[str, object]
    universe: list[str]
    history: pl.DataFrame
    storage_uri: str
    seed: int
    cost_model: str
    period: DataPeriod = DataPeriod.DEVELOPMENT


def record_run(
    connection: Connection,
    inputs: RunInputs,
    metrics: dict[str, object] | None = None,
    gauntlet: GauntletReport | None = None,
) -> RunRecord:
    """Insert one run and return the trial count including it.

    The order matters and is the point: the hypothesis is registered (or found
    already present), the experiment is inserted — firing the counter trigger —
    and only then is the counter read. A count taken before the insert would
    omit the run being judged.
    """
    repository = ExperimentRepository(connection)
    repository.ensure_dataset(NSE_DATASET_ID, "NSE end-of-day panel", "nse-bhavcopy")
    version_id = repository.register_dataset_version(
        dataset_version_for(inputs.history, inputs.storage_uri)
    )
    repository.ensure_hypothesis(inputs.hypothesis)

    experiment = ExperimentRecord(
        hypothesis_id=inputs.hypothesis.hypothesis_id,
        strategy_name=inputs.strategy_name,
        period=inputs.period,
        dataset_version=str(version_id),
        parameters={k: str(v) for k, v in inputs.parameters.items()},
        cost_model=inputs.cost_model,
        universe=inputs.universe,
        code_commit=code_commit(),
        seed=inputs.seed,
    )
    repository.record_experiment(experiment, version_id)

    if metrics is not None:
        repository.record_metrics(experiment.experiment_id, metrics)
    if gauntlet is not None:
        for result in gauntlet.results:
            repository.record_gauntlet(experiment.experiment_id, result)

    connection.commit()
    return RunRecord(
        experiment_id=experiment.experiment_id,
        hypothesis_id=inputs.hypothesis.hypothesis_id,
        trials=repository.trials_for(inputs.hypothesis.hypothesis_id),
    )
