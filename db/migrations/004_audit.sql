-- ─────────────────────────────────────────────────────────────────────────────
-- 004 — audit trail and incidents
-- MASTER_PLAN §21, §24 — every production action is auditable
-- ─────────────────────────────────────────────────────────────────────────────

SET timezone = 'UTC';

CREATE TYPE incident_severity AS ENUM ('INFO', 'WARN', 'CRITICAL');

CREATE TYPE incident_status AS ENUM ('OPEN', 'MITIGATED', 'RESOLVED');


-- ── audit events ────────────────────────────────────────────────────────────
-- Append-only by construction: UPDATE and DELETE are revoked below, and a
-- trigger rejects them even for the owner. An audit log that can be edited is
-- not an audit log.
CREATE TABLE audit_events (
    event_id    BIGSERIAL PRIMARY KEY,
    actor       TEXT NOT NULL,          -- human, service name, or agent id
    action      TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX audit_events_entity_idx ON audit_events (entity_type, entity_id, occurred_at DESC);
CREATE INDEX audit_events_actor_idx ON audit_events (actor, occurred_at DESC);

CREATE FUNCTION reject_audit_mutation() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only (MASTER_PLAN §21)';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_events_no_update
    BEFORE UPDATE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation();

CREATE TRIGGER audit_events_no_delete
    BEFORE DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation();


-- ── incidents (§24) ─────────────────────────────────────────────────────────
CREATE TABLE system_incidents (
    incident_id  BIGSERIAL PRIMARY KEY,
    severity     incident_severity NOT NULL,
    status       incident_status NOT NULL DEFAULT 'OPEN',
    title        TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '',
    runbook      TEXT,
    detected_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    mitigated_at TIMESTAMPTZ,
    resolved_at  TIMESTAMPTZ,
    root_cause   TEXT,

    CONSTRAINT resolution_ordered CHECK (
        resolved_at IS NULL OR mitigated_at IS NULL OR resolved_at >= mitigated_at
    ),
    CONSTRAINT resolved_has_root_cause CHECK (
        status <> 'RESOLVED' OR root_cause IS NOT NULL
    )
);

CREATE INDEX incidents_open_idx ON system_incidents (detected_at DESC) WHERE status <> 'RESOLVED';


-- ── data quality findings (§M2) ─────────────────────────────────────────────
CREATE TABLE data_quality_findings (
    finding_id    BIGSERIAL PRIMARY KEY,
    dataset_version_id UUID REFERENCES dataset_versions (version_id),
    instrument_id TEXT REFERENCES instruments (instrument_id),
    check_name    TEXT NOT NULL,
    severity      incident_severity NOT NULL,
    observed_at   TIMESTAMPTZ,
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb,
    found_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX dq_findings_check_idx ON data_quality_findings (check_name, found_at DESC);
CREATE INDEX dq_findings_critical_idx ON data_quality_findings (found_at DESC)
    WHERE severity = 'CRITICAL';
