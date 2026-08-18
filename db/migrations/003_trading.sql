-- ─────────────────────────────────────────────────────────────────────────────
-- 003 — trading plane: orders, fills, positions, risk limits, kill switch
-- MASTER_PLAN §8, §9, §19
--
-- Every money column here is NUMERIC, never DOUBLE PRECISION (§14.1.2).
-- If a broker could disagree with you about the number, it is exact.
-- ─────────────────────────────────────────────────────────────────────────────

SET timezone = 'UTC';

CREATE TYPE order_side AS ENUM ('BUY', 'SELL');

CREATE TYPE order_kind AS ENUM ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT');

-- The state machine from §19, verbatim. UNKNOWN is the important one: networks
-- fail mid-submit, and "did that order reach the broker?" must be answerable
-- without guessing.
CREATE TYPE order_state AS ENUM (
    'CREATED', 'RISK_CHECKED', 'SUBMITTED', 'ACKNOWLEDGED',
    'PARTIALLY_FILLED', 'FILLED', 'REJECTED',
    'CANCEL_REQUESTED', 'CANCELLED', 'UNKNOWN'
);

CREATE TYPE trading_mode AS ENUM ('BACKTEST', 'PAPER', 'LIVE');

CREATE TYPE risk_decision AS ENUM ('ALLOW', 'BLOCK');


-- ── orders ──────────────────────────────────────────────────────────────────
CREATE TABLE orders (
    order_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Client-generated, unique. A retry after a timeout must never double-fill,
    -- so idempotency is a database constraint rather than an intention (§9).
    idempotency_key TEXT NOT NULL UNIQUE,

    mode            trading_mode NOT NULL,
    strategy_id     TEXT NOT NULL REFERENCES strategies (strategy_id),
    instrument_id   TEXT NOT NULL REFERENCES instruments (instrument_id),

    side            order_side NOT NULL,
    order_type      order_kind NOT NULL,
    quantity        NUMERIC(28, 10) NOT NULL CHECK (quantity > 0),
    limit_price     NUMERIC(28, 10) CHECK (limit_price IS NULL OR limit_price > 0),
    stop_price      NUMERIC(28, 10) CHECK (stop_price IS NULL OR stop_price > 0),

    state           order_state NOT NULL DEFAULT 'CREATED',
    filled_quantity NUMERIC(28, 10) NOT NULL DEFAULT 0 CHECK (filled_quantity >= 0),
    broker_order_id TEXT,

    decision_time   TIMESTAMPTZ NOT NULL,
    submitted_at    TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ,
    terminal_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT limit_orders_have_price CHECK (
        order_type NOT IN ('LIMIT', 'STOP_LIMIT') OR limit_price IS NOT NULL
    ),
    CONSTRAINT stop_orders_have_stop CHECK (
        order_type NOT IN ('STOP', 'STOP_LIMIT') OR stop_price IS NOT NULL
    ),
    CONSTRAINT no_overfill CHECK (filled_quantity <= quantity)
);

CREATE INDEX orders_state_idx ON orders (state) WHERE state NOT IN ('FILLED', 'CANCELLED', 'REJECTED');
CREATE INDEX orders_strategy_idx ON orders (strategy_id, created_at DESC);
CREATE INDEX orders_instrument_idx ON orders (instrument_id, created_at DESC);

-- Every transition, appended. Reconstructing an order's history is how you
-- diagnose a fill you did not expect.
CREATE TABLE order_transitions (
    transition_id BIGSERIAL PRIMARY KEY,
    order_id      UUID NOT NULL REFERENCES orders (order_id),
    from_state    order_state,
    to_state      order_state NOT NULL,
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX order_transitions_order_idx ON order_transitions (order_id, occurred_at);


-- ── fills ───────────────────────────────────────────────────────────────────
CREATE TABLE fills (
    fill_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id       UUID NOT NULL REFERENCES orders (order_id),
    broker_fill_id TEXT,

    quantity       NUMERIC(28, 10) NOT NULL CHECK (quantity > 0),
    price          NUMERIC(28, 10) NOT NULL CHECK (price > 0),

    -- Cost breakdown kept itemised, not summed. Weekly comparison of realised
    -- against modelled costs is what keeps the backtest honest (§9).
    commission     NUMERIC(28, 10) NOT NULL DEFAULT 0 CHECK (commission >= 0),
    taxes          NUMERIC(28, 10) NOT NULL DEFAULT 0 CHECK (taxes >= 0),
    exchange_fees  NUMERIC(28, 10) NOT NULL DEFAULT 0 CHECK (exchange_fees >= 0),
    other_fees     NUMERIC(28, 10) NOT NULL DEFAULT 0 CHECK (other_fees >= 0),

    -- Execution-quality inputs: realised slippage recalibrates the cost model.
    intended_price NUMERIC(28, 10) CHECK (intended_price IS NULL OR intended_price > 0),
    event_time     TIMESTAMPTZ NOT NULL,
    receive_time   TIMESTAMPTZ NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fill_causality CHECK (receive_time >= event_time),
    CONSTRAINT broker_fill_unique UNIQUE (broker_fill_id)
);

CREATE INDEX fills_order_idx ON fills (order_id);
CREATE INDEX fills_time_idx ON fills (event_time);


-- ── positions ───────────────────────────────────────────────────────────────
CREATE TABLE positions (
    position_id     BIGSERIAL PRIMARY KEY,
    mode            trading_mode NOT NULL,
    strategy_id     TEXT NOT NULL REFERENCES strategies (strategy_id),
    instrument_id   TEXT NOT NULL REFERENCES instruments (instrument_id),

    quantity        NUMERIC(28, 10) NOT NULL,          -- signed: negative is short
    average_price   NUMERIC(28, 10) NOT NULL CHECK (average_price >= 0),
    realised_pnl    NUMERIC(28, 10) NOT NULL DEFAULT 0,
    fees_paid       NUMERIC(28, 10) NOT NULL DEFAULT 0 CHECK (fees_paid >= 0),

    opened_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT position_unique UNIQUE (mode, strategy_id, instrument_id)
);


-- ── risk limits (§8) ────────────────────────────────────────────────────────
-- The risk engine owns this table. Strategies have no write path to it: a
-- strategy that can raise its own limit has no limit.
CREATE TABLE risk_limits (
    limit_id     BIGSERIAL PRIMARY KEY,
    mode         trading_mode NOT NULL,
    scope        TEXT NOT NULL,          -- 'GLOBAL' | strategy_id | instrument_id
    limit_name   TEXT NOT NULL,
    limit_value  NUMERIC(28, 10) NOT NULL,
    unit         TEXT NOT NULL,          -- 'INR' | 'PCT_NAV' | 'COUNT' | 'BPS'
    active       BOOLEAN NOT NULL DEFAULT TRUE,
    set_by       TEXT NOT NULL,
    set_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT risk_limit_unique UNIQUE (mode, scope, limit_name)
);

CREATE TABLE risk_events (
    event_id      BIGSERIAL PRIMARY KEY,
    mode          trading_mode NOT NULL,
    order_id      UUID REFERENCES orders (order_id),
    strategy_id   TEXT REFERENCES strategies (strategy_id),
    decision      risk_decision NOT NULL,
    limit_name    TEXT,
    observed      NUMERIC(28, 10),
    threshold     NUMERIC(28, 10),
    reasons       JSONB NOT NULL DEFAULT '[]'::jsonb,
    occurred_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX risk_events_blocks_idx ON risk_events (occurred_at DESC) WHERE decision = 'BLOCK';


-- ── kill switch (§8) ────────────────────────────────────────────────────────
-- Single row. Deliberately trivial to read and trivial to set: the mechanism
-- that stops everything must not itself be complicated.
CREATE TABLE kill_switch (
    id           BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    engaged      BOOLEAN NOT NULL DEFAULT FALSE,
    engaged_by   TEXT,
    reason       TEXT,
    engaged_at   TIMESTAMPTZ,
    released_by  TEXT,
    released_at  TIMESTAMPTZ,

    CONSTRAINT engaged_has_context CHECK (
        NOT engaged OR (engaged_by IS NOT NULL AND reason IS NOT NULL)
    )
);

INSERT INTO kill_switch (id, engaged) VALUES (TRUE, FALSE);


-- ── drawdown ladder (§8) ────────────────────────────────────────────────────
-- Pre-committed in code and in the database, before going live. The purpose is
-- to remove the operator from the decision at the exact moment they are least
-- capable of making it.
CREATE TABLE drawdown_ladder (
    rung_id        SERIAL PRIMARY KEY,
    mode           trading_mode NOT NULL,
    drawdown_pct   NUMERIC(6, 4) NOT NULL CHECK (drawdown_pct < 0),
    scale_to_pct   NUMERIC(6, 4) NOT NULL CHECK (scale_to_pct BETWEEN 0 AND 1),
    halt           BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT ladder_rung_unique UNIQUE (mode, drawdown_pct)
);


-- ── reconciliation (§9) ─────────────────────────────────────────────────────
-- Any non-empty break is a system-down event, not a rounding issue.
CREATE TABLE reconciliation_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    mode          trading_mode NOT NULL,
    as_of         TIMESTAMPTZ NOT NULL,
    breaks_found  INTEGER NOT NULL DEFAULT 0 CHECK (breaks_found >= 0),
    resolved      BOOLEAN NOT NULL DEFAULT FALSE,
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb,
    run_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE reconciliation_breaks (
    break_id      BIGSERIAL PRIMARY KEY,
    run_id        BIGINT NOT NULL REFERENCES reconciliation_runs (run_id),
    instrument_id TEXT REFERENCES instruments (instrument_id),
    field         TEXT NOT NULL,           -- 'quantity' | 'cash' | 'fill'
    internal      NUMERIC(28, 10),
    broker        NUMERIC(28, 10),
    explanation   TEXT,
    resolved_at   TIMESTAMPTZ
);
