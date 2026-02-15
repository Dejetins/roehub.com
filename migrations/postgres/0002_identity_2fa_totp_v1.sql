BEGIN;

CREATE TABLE IF NOT EXISTS identity_2fa (
    user_id UUID PRIMARY KEY REFERENCES identity_users (user_id) ON DELETE CASCADE,
    totp_secret_enc BYTEA NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    enabled_at TIMESTAMPTZ NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_2fa_enabled_state_chk
        CHECK (
            (enabled = TRUE AND enabled_at IS NOT NULL)
            OR
            (enabled = FALSE AND enabled_at IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS idx_identity_2fa_enabled
    ON identity_2fa (enabled);

COMMIT;
