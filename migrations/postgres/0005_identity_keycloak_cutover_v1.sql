BEGIN;

ALTER TABLE identity_users
    ADD COLUMN IF NOT EXISTS keycloak_subject TEXT NULL;

ALTER TABLE identity_users
    ALTER COLUMN telegram_user_id DROP NOT NULL;

DROP INDEX IF EXISTS idx_identity_users_telegram_user_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_identity_users_keycloak_subject
    ON identity_users (keycloak_subject)
    WHERE keycloak_subject IS NOT NULL;

CREATE TABLE IF NOT EXISTS identity_sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES identity_users (user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    idle_expires_at TIMESTAMPTZ NOT NULL,
    absolute_expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_sessions_last_seen_chk
        CHECK (last_seen_at >= created_at),
    CONSTRAINT identity_sessions_idle_expiry_chk
        CHECK (idle_expires_at >= last_seen_at),
    CONSTRAINT identity_sessions_absolute_expiry_chk
        CHECK (absolute_expires_at >= idle_expires_at),
    CONSTRAINT identity_sessions_revoked_at_chk
        CHECK (revoked_at IS NULL OR revoked_at >= created_at)
);

CREATE INDEX IF NOT EXISTS idx_identity_sessions_user_id
    ON identity_sessions (user_id);

CREATE INDEX IF NOT EXISTS idx_identity_sessions_active_lookup
    ON identity_sessions (session_id, revoked_at, idle_expires_at, absolute_expires_at);

DROP TABLE IF EXISTS identity_2fa;

COMMIT;
