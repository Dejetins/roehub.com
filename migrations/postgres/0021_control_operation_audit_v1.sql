BEGIN;

CREATE TABLE IF NOT EXISTS control_operation_audit_events (
    entry_hash TEXT PRIMARY KEY
        CHECK (entry_hash ~ '^[0-9a-f]{64}$'),
    sequence BIGINT NOT NULL UNIQUE
        CHECK (sequence > 0),
    operation_id UUID NOT NULL,
    action TEXT NOT NULL
        CHECK (action ~ '^[a-z][a-z0-9.-]{0,63}$'),
    state TEXT NOT NULL
        CHECK (state IN ('accepted', 'running', 'succeeded', 'failed', 'unknown', 'rejected')),
    detail_code TEXT NOT NULL
        CHECK (detail_code ~ '^[a-z0-9][a-z0-9._-]{0,127}$'),
    recorded_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL
        CHECK (jsonb_typeof(payload) = 'object'),
    reconciled_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS control_operation_audit_cursor (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE
        CHECK (singleton),
    sequence BIGINT NOT NULL DEFAULT 0
        CHECK (sequence >= 0),
    entry_hash TEXT NULL
        REFERENCES control_operation_audit_events(entry_hash)
        ON DELETE RESTRICT
);

INSERT INTO control_operation_audit_cursor (singleton, sequence, entry_hash)
VALUES (TRUE, 0, NULL)
ON CONFLICT (singleton) DO NOTHING;

COMMIT;
