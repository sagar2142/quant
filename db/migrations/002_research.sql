-- ─────────────────────────────────────────────────────────────────────────────
-- 002 — the research protocol: hypotheses, trial counting, locked test set,
--       experiments, backtests, the gauntlet, the rejection log
-- MASTER_PLAN §5 — the machinery that stops you fooling yourself
-- ─────────────────────────────────────────────────────────────────────────────

SET timezone = 'UTC';

CREATE TYPE hypothesis_status AS ENUM (
    'OPEN', 'CONFIRMED', 'REJECTED', 'ABANDONED'
);

CREATE TYPE strategy_state AS ENUM (
    'RESEARCH', 'VALIDATED', 'PAPER_APPROVED', 'LIVE_PENDING',
    'LIVE_APPROVED', 'ACTIVE', 'PAUSED', 'KILLED', 'RETIRED'
);

CREATE TYPE run_status AS ENUM (
    'DRAFT', 'RUNNING', 'COMPLETED', 'FAILED', 'REJECTED', 'NEEDS_REVIEW'
);

CREATE TYPE data_period AS ENUM ('DEVELOPMENT', 'VALIDATION', 'LOCKED_TEST');


-- ── hypotheses (§5.1) ───────────────────────────────────────────────────────
-- Pre-registration. Written BEFORE any backtest runs.
CREATE TABLE hypotheses (
    hypothesis_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    statement          TEXT NOT NULL,

    -- The highest-value column in the whole database. "The z-score reverts" is
    -- not a mechanism. "Index funds mechanically buy at rebalance dates,
    -- creating temporary price pressure that reverts within 5 days" is.
    -- Ideas without mechanisms are data mining with extra steps, so the
    -- minimum length is a deliberate obstacle, not a formality.
    economic_mechanism TEXT NOT NULL CHECK (length(trim(economic_mechanism)) >= 80),

    prediction         TEXT NOT NULL,
    success_criteria   JSONB NOT NULL,
    kill_criteria      JSONB NOT NULL,

    dev_start          DATE NOT NULL,
    dev_end            DATE NOT NULL,
    val_start          DATE NOT NULL,
    val_end            DATE NOT NULL,
    test_start         DATE NOT NULL,
    test_end           DATE NOT NULL,

    -- Feeds the Deflated Sharpe Ratio (§5.2). Maintained by trigger, never by
    -- hand: a trial counter you can forget to increment is not a trial counter.
    n_trials           INTEGER NOT NULL DEFAULT 0 CHECK (n_trials >= 0),

    status             hypothesis_status NOT NULL DEFAULT 'OPEN',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at        TIMESTAMPTZ,

    CONSTRAINT periods_ordered CHECK (
        dev_start < dev_end AND dev_end <= val_start
        AND val_start < val_end AND val_end <= test_start
        AND test_start < test_end
    ),
    CONSTRAINT resolved_has_timestamp CHECK (
        (status = 'OPEN') = (resolved_at IS NULL)
    )
);

CREATE INDEX hypotheses_status_idx ON hypotheses (status, created_at DESC);


-- ── strategies ──────────────────────────────────────────────────────────────
CREATE TABLE strategies (
    strategy_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    hypothesis_id UUID REFERENCES hypotheses (hypothesis_id),
    family        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE strategy_versions (
    version_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id   TEXT NOT NULL REFERENCES strategies (strategy_id),
    version       INTEGER NOT NULL CHECK (version >= 1),
    spec          JSONB NOT NULL,
    code_commit   TEXT NOT NULL,
    state         strategy_state NOT NULL DEFAULT 'RESEARCH',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT strategy_version_unique UNIQUE (strategy_id, version)
);

-- Lifecycle transitions are recorded, never silently overwritten. "Who moved
-- this to LIVE_APPROVED and when" must always be answerable (§21).
CREATE TABLE strategy_state_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    version_id    UUID NOT NULL REFERENCES strategy_versions (version_id),
    from_state    strategy_state,
    to_state      strategy_state NOT NULL,
    reason        TEXT NOT NULL,
    actor         TEXT NOT NULL,
    occurred_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ── experiments and backtests ───────────────────────────────────────────────
-- Everything needed to reproduce a number exactly (§M3 gate).
CREATE TABLE experiments (
    experiment_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id      UUID NOT NULL REFERENCES hypotheses (hypothesis_id),
    strategy_version_id UUID REFERENCES strategy_versions (version_id),

    dataset_version_id UUID NOT NULL REFERENCES dataset_versions (version_id),
    period             data_period NOT NULL,
    parameters         JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_model         JSONB NOT NULL,
    universe           JSONB NOT NULL,

    code_commit        TEXT NOT NULL,
    seed               BIGINT NOT NULL,
    environment        JSONB NOT NULL DEFAULT '{}'::jsonb,

    status             run_status NOT NULL DEFAULT 'DRAFT',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at         TIMESTAMPTZ,
    finished_at        TIMESTAMPTZ,

    CONSTRAINT finish_after_start CHECK (
        finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at
    )
);

CREATE INDEX experiments_hypothesis_idx ON experiments (hypothesis_id, created_at DESC);
CREATE INDEX experiments_period_idx ON experiments (period);

CREATE TABLE backtest_metrics (
    experiment_id  UUID PRIMARY KEY REFERENCES experiments (experiment_id),

    -- float8 is correct here: these are statistics, not money (§14.1.2).
    total_return   DOUBLE PRECISION NOT NULL,
    cagr           DOUBLE PRECISION NOT NULL,
    sharpe         DOUBLE PRECISION NOT NULL,
    sortino        DOUBLE PRECISION,
    max_drawdown   DOUBLE PRECISION NOT NULL CHECK (max_drawdown <= 0),
    volatility     DOUBLE PRECISION NOT NULL CHECK (volatility >= 0),
    turnover       DOUBLE PRECISION NOT NULL CHECK (turnover >= 0),
    hit_rate       DOUBLE PRECISION CHECK (hit_rate BETWEEN 0 AND 1),
    n_trades       INTEGER NOT NULL CHECK (n_trades >= 0),

    -- Realised cost drag. If this is small, the cost model is probably wrong.
    cost_drag_bps  DOUBLE PRECISION NOT NULL,

    -- Overfitting statistics (§5.4). NULL until the gauntlet runs.
    deflated_sharpe DOUBLE PRECISION,
    pbo             DOUBLE PRECISION CHECK (pbo BETWEEN 0 AND 1),

    equity_curve_uri TEXT,
    trades_uri       TEXT,
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ── trial counter (§5.2) ────────────────────────────────────────────────────
-- If you test 100 variants at p<0.05, ~5 pass by chance. The Sharpe you observe
-- is the maximum of N draws, not a single draw. DSR needs N; N must be
-- automatic, because a counter you can forget to increment always ends at 1.
CREATE FUNCTION bump_hypothesis_trials() RETURNS TRIGGER AS $$
BEGIN
    UPDATE hypotheses
       SET n_trials = n_trials + 1
     WHERE hypothesis_id = NEW.hypothesis_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER experiments_bump_trials
    AFTER INSERT ON experiments
    FOR EACH ROW EXECUTE FUNCTION bump_hypothesis_trials();


-- ── locked test set (§5.3) ──────────────────────────────────────────────────
-- One access per strategy, ever. The UNIQUE constraint is the enforcement: a
-- second attempt raises a database error rather than a warning someone
-- overrides at 2am while convinced this time is different.
CREATE TABLE test_set_access (
    access_id     BIGSERIAL PRIMARY KEY,
    strategy_id   TEXT NOT NULL REFERENCES strategies (strategy_id),
    experiment_id UUID NOT NULL REFERENCES experiments (experiment_id),
    accessed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    outcome       JSONB NOT NULL,

    CONSTRAINT one_locked_test_per_strategy UNIQUE (strategy_id)
);

-- An experiment against LOCKED_TEST must be accompanied by an access record.
-- Enforced from the application side on the write path; this view makes any
-- violation trivially visible in review.
CREATE VIEW locked_test_without_record AS
SELECT e.experiment_id, e.created_at
  FROM experiments e
  LEFT JOIN test_set_access t ON t.experiment_id = e.experiment_id
 WHERE e.period = 'LOCKED_TEST' AND t.access_id IS NULL;


-- ── gauntlet results (§5.4) ─────────────────────────────────────────────────
CREATE TABLE gauntlet_results (
    result_id     BIGSERIAL PRIMARY KEY,
    experiment_id UUID NOT NULL REFERENCES experiments (experiment_id),
    test_name     TEXT NOT NULL,
    passed        BOOLEAN NOT NULL,
    statistic     DOUBLE PRECISION,
    threshold     DOUBLE PRECISION,
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb,
    run_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT one_result_per_test UNIQUE (experiment_id, test_name)
);

CREATE INDEX gauntlet_failures_idx ON gauntlet_results (experiment_id) WHERE NOT passed;


-- ── rejection log (§5.5) ────────────────────────────────────────────────────
-- The closest thing a solo quant has to institutional memory. Prevents
-- re-testing dead ideas and reveals patterns in how ideas die.
CREATE TABLE rejection_log (
    rejection_id     BIGSERIAL PRIMARY KEY,
    hypothesis_id    UUID NOT NULL REFERENCES hypotheses (hypothesis_id),
    experiment_id    UUID REFERENCES experiments (experiment_id),
    killed_at_stage  TEXT NOT NULL,
    reason           TEXT NOT NULL,
    lesson           TEXT NOT NULL DEFAULT '',
    rejected_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX rejection_log_stage_idx ON rejection_log (killed_at_stage);


-- ── approvals (§21) ─────────────────────────────────────────────────────────
CREATE TABLE approvals (
    approval_id   BIGSERIAL PRIMARY KEY,
    version_id    UUID NOT NULL REFERENCES strategy_versions (version_id),
    target_state  strategy_state NOT NULL,
    approver      TEXT NOT NULL,
    rationale     TEXT NOT NULL CHECK (length(trim(rationale)) >= 20),
    checklist     JSONB NOT NULL,
    approved_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
