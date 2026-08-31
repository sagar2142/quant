-- 005 — a statement is a hypothesis's identity, enforced by the database.
--
-- WHY THIS EXISTS. `hypotheses.hypothesis_id` defaulted to a fresh UUID per
-- Python object, so the pre-registration CATALOGUE was assigned new ids on
-- every import. `ensure_hypothesis` deduplicates on ON CONFLICT (hypothesis_id)
-- and therefore never fired: running `preregister --register` a second time
-- inserted a duplicate OPEN copy of every question, sitting beside the
-- resolved original. Eleven REJECTED verdicts were shadowed by eleven fresh
-- OPEN rows, and the M6/M7 rejection rate was computed over a denominator
-- twice its true size.
--
-- A pre-registration ledger whose verdicts can be un-made by re-running a
-- command is not a ledger, which is the whole reason §5.1 exists. The
-- application-side fix derives the id from the statement; this is the half
-- that does not depend on remembering to.
--
-- Idempotent, and a no-op on a fresh database that has no rows yet.

-- ── deduplicate before constraining ─────────────────────────────────────────
-- Which copy survives, in order: a resolved verdict outranks an unresolved
-- one, because losing a REJECTED row loses the finding; then the copy that
-- burned trials, because the trial counter is evidence; then the oldest,
-- because it is the one that was actually pre-registered.
--
-- Rows referenced by a strategy, experiment or rejection are never deleted:
-- the foreign key would stop it anyway, and a deletion that silently orphaned
-- a result would be a worse bug than the one being fixed.
WITH ranked AS (
    SELECT
        hypothesis_id,
        row_number() OVER (
            PARTITION BY statement
            ORDER BY (status = 'OPEN'), n_trials DESC, created_at
        ) AS rank
    FROM hypotheses
)
DELETE FROM hypotheses h
USING ranked r
WHERE h.hypothesis_id = r.hypothesis_id
  AND r.rank > 1
  AND NOT EXISTS (SELECT 1 FROM strategies    s WHERE s.hypothesis_id = h.hypothesis_id)
  AND NOT EXISTS (SELECT 1 FROM experiments   e WHERE e.hypothesis_id = h.hypothesis_id)
  AND NOT EXISTS (SELECT 1 FROM rejection_log l WHERE l.hypothesis_id = h.hypothesis_id);

-- ── then make it impossible again ───────────────────────────────────────────
-- On the statement rather than the id, so that a caller which invents its own
-- id still cannot register the same claim twice. Whitespace is normalised in
-- the application; two statements differing only in a reflowed line are the
-- same hypothesis and must collide here too.
ALTER TABLE hypotheses
    ADD CONSTRAINT hypotheses_statement_key UNIQUE (statement);
