BEGIN;
CREATE TABLE IF NOT EXISTS backtest_artifact_storage_commands (
    installation_id UUID NOT NULL REFERENCES identity_installations ON DELETE RESTRICT,
    actor_id UUID NOT NULL REFERENCES identity_users ON DELETE RESTRICT,
    idempotency_key TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    response JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(installation_id,actor_id,idempotency_key)
);
CREATE TABLE IF NOT EXISTS backtest_artifact_storage_previews (
    preview_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL REFERENCES identity_installations ON DELETE RESTRICT,
    actor_id UUID NOT NULL REFERENCES identity_users ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK(kind IN ('validate','cleanup_preview')),
    control_version BIGINT NOT NULL,
    root_id TEXT NOT NULL,
    root_identity JSONB NOT NULL,
    target JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed BOOLEAN NOT NULL DEFAULT FALSE
);
COMMIT;
