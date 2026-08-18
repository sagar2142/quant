-- ─────────────────────────────────────────────────────────────────────────────
-- 001 — core reference data: instruments, symbol history, datasets
-- MASTER_PLAN §1.1, §3.3, §M2
-- ─────────────────────────────────────────────────────────────────────────────

SET timezone = 'UTC';

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Every timestamp in this database is timestamptz. A naive timestamp column is
-- a latent 5.5-hour bug (§14.1.3), so the convention is enforced by review and
-- by the fact that no table below declares a bare `timestamp`.

-- ── enums ───────────────────────────────────────────────────────────────────
-- Equities and their derivatives only (MASTER_PLAN 0.0). An unused enum
-- value is an invitation to write code paths nobody tests.
CREATE TYPE asset_class AS ENUM (
    'EQUITY', 'ETF', 'INDEX', 'FUTURE', 'OPTION'
);

CREATE TYPE exchange_code AS ENUM (
    'NSE', 'BSE', 'NASDAQ', 'NYSE', 'ARCA'
);

CREATE TYPE currency_code AS ENUM ('INR', 'USD');

CREATE TYPE option_kind AS ENUM ('CE', 'PE');


-- ── instruments ─────────────────────────────────────────────────────────────
-- instrument_id is internal and NEVER reused. Exchange symbols are not
-- identity: they get recycled after delistings and renamed on rebranding.
CREATE TABLE instruments (
    instrument_id   TEXT PRIMARY KEY,
    symbol          TEXT        NOT NULL,
    name            TEXT        NOT NULL DEFAULT '',
    asset_class     asset_class NOT NULL,
    exchange        exchange_code NOT NULL,
    currency        currency_code NOT NULL,

    tick_size       NUMERIC(20, 10) NOT NULL CHECK (tick_size > 0),
    lot_size        INTEGER     NOT NULL DEFAULT 1 CHECK (lot_size >= 1),
    multiplier      NUMERIC(20, 10) NOT NULL DEFAULT 1 CHECK (multiplier > 0),

    -- derivatives only
    expiry          TIMESTAMPTZ,
    strike          NUMERIC(20, 10) CHECK (strike IS NULL OR strike > 0),
    option_type     option_kind,
    underlying_id   TEXT REFERENCES instruments (instrument_id),

    -- lifecycle: what makes survivorship-bias-free universes possible (§M2 gate)
    listed_on       TIMESTAMPTZ,
    delisted_on     TIMESTAMPTZ,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT derivatives_have_expiry CHECK (
        asset_class NOT IN ('FUTURE', 'OPTION') OR expiry IS NOT NULL
    ),
    CONSTRAINT options_have_strike_and_type CHECK (
        (asset_class = 'OPTION' AND strike IS NOT NULL AND option_type IS NOT NULL)
        OR (asset_class <> 'OPTION' AND strike IS NULL AND option_type IS NULL)
    ),
    CONSTRAINT lifecycle_ordered CHECK (
        listed_on IS NULL OR delisted_on IS NULL OR delisted_on >= listed_on
    )
);

CREATE INDEX instruments_class_exchange_idx ON instruments (asset_class, exchange);
CREATE INDEX instruments_lifecycle_idx ON instruments (listed_on, delisted_on);


-- ── symbol history ──────────────────────────────────────────────────────────
-- Point-in-time symbol resolution. Today's symbol table must never answer a
-- question about 2010 (§3.3).
CREATE TABLE symbol_aliases (
    alias_id      BIGSERIAL PRIMARY KEY,
    instrument_id TEXT        NOT NULL REFERENCES instruments (instrument_id),
    symbol        TEXT        NOT NULL,
    exchange      exchange_code NOT NULL,
    valid_from    TIMESTAMPTZ NOT NULL,
    valid_to      TIMESTAMPTZ,

    CONSTRAINT alias_window_ordered CHECK (valid_to IS NULL OR valid_to > valid_from)
);

CREATE INDEX symbol_aliases_lookup_idx
    ON symbol_aliases (upper(symbol), exchange, valid_from);

-- No instrument may carry two live aliases for the same symbol on one venue at
-- the same instant, or resolution becomes ambiguous.
CREATE UNIQUE INDEX symbol_aliases_no_overlap_idx
    ON symbol_aliases (upper(symbol), exchange, valid_from);


-- ── datasets ────────────────────────────────────────────────────────────────
-- Every experiment references an immutable dataset version by content hash.
-- Without this, "reproducible" is a claim rather than a fact (§M3 gate).
CREATE TABLE datasets (
    dataset_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE dataset_versions (
    version_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id     TEXT NOT NULL REFERENCES datasets (dataset_id),
    content_hash   TEXT NOT NULL,
    row_count      BIGINT NOT NULL CHECK (row_count >= 0),
    coverage_start TIMESTAMPTZ NOT NULL,
    coverage_end   TIMESTAMPTZ NOT NULL,
    storage_uri    TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    quality_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT coverage_ordered CHECK (coverage_end >= coverage_start),
    CONSTRAINT dataset_version_unique UNIQUE (dataset_id, content_hash)
);

CREATE INDEX dataset_versions_dataset_idx ON dataset_versions (dataset_id, created_at DESC);


-- ── observed trading sessions ───────────────────────────────────────────────
-- The empirical calendar (§1.2): sessions derived from data actually present,
-- reconciled against the declared holiday rules by the M2 quality suite.
CREATE TABLE observed_sessions (
    exchange     exchange_code NOT NULL,
    session_date DATE          NOT NULL,
    instrument_count INTEGER   NOT NULL CHECK (instrument_count > 0),
    first_seen   TIMESTAMPTZ   NOT NULL DEFAULT now(),

    PRIMARY KEY (exchange, session_date)
);
