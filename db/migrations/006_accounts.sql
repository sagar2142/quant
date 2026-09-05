-- 006 — operator accounts and their settings.
--
-- WHAT THIS IS, AND WHAT IT IS NOT. This is a credential store and a place to
-- keep per-operator preferences. It is NOT, on its own, a security boundary:
-- the API binds to 127.0.0.1 and anything already running on this machine can
-- reach it directly. `NEUTRON_API_TOKEN` is what closes that door (§13.7); an
-- account layer in front of an open API would be theatre, and theatre on a
-- system that can place real orders is worse than nothing.
--
-- So sessions here are for identifying WHICH operator is acting and carrying
-- their settings, and the token stays responsible for whether the API answers
-- at all.
--
-- **No broker secrets live here.** Kite keys stay in the environment where the
-- process reads them once, because a database row is backed up, replicated and
-- read by whoever can query it, and a key that can move money should not be in
-- any of those places.

CREATE TABLE users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL DEFAULT '',

    -- scrypt, from the standard library. Memory-hard, so a stolen table cannot
    -- be brute-forced at the rate a plain hash allows. The salt is per user and
    -- the parameters are stored beside the digest so they can be raised later
    -- without invalidating every existing password.
    password_hash BYTEA NOT NULL,
    password_salt BYTEA NOT NULL,
    scrypt_n      INTEGER NOT NULL,
    scrypt_r      INTEGER NOT NULL,
    scrypt_p      INTEGER NOT NULL,

    -- Display and workflow preferences only. Anything that decides money —
    -- position limits, the kill switch — belongs to the risk engine, not to a
    -- per-user settings blob that an account could quietly widen.
    settings      JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ
);

CREATE TABLE sessions (
    -- The token itself is never stored. What is stored is its SHA-256, so a
    -- leaked table cannot be replayed as a live session.
    token_hash  BYTEA PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT session_expires_after_creation CHECK (expires_at > created_at)
);

CREATE INDEX sessions_user_idx ON sessions (user_id, expires_at DESC);
