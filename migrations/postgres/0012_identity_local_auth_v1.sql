BEGIN;

CREATE TABLE identity_local_accounts (
    user_id UUID PRIMARY KEY REFERENCES identity_users(user_id) ON DELETE CASCADE,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_local_accounts_username_chk
        CHECK (username ~ '^[a-z][a-z0-9._-]{2,63}$'),
    CONSTRAINT identity_local_accounts_display_name_chk
        CHECK (char_length(trim(display_name)) BETWEEN 2 AND 120),
    CONSTRAINT identity_local_accounts_password_hash_chk
        CHECK (password_hash IS NULL OR password_hash LIKE '$argon2id$%'),
    CONSTRAINT identity_local_accounts_updated_chk CHECK (updated_at >= created_at)
);

CREATE TABLE identity_webauthn_credentials (
    credential_id TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES identity_local_accounts(user_id) ON DELETE CASCADE,
    public_key BYTEA NOT NULL,
    sign_count BIGINT NOT NULL DEFAULT 0,
    transports TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ NOT NULL,
    last_used_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_webauthn_credentials_id_chk
        CHECK (
            credential_id ~ '^[A-Za-z0-9_-]+$'
            AND char_length(credential_id) BETWEEN 16 AND 1024
        ),
    CONSTRAINT identity_webauthn_credentials_public_key_chk
        CHECK (octet_length(public_key) BETWEEN 16 AND 4096),
    CONSTRAINT identity_webauthn_credentials_sign_count_chk CHECK (sign_count >= 0),
    CONSTRAINT identity_webauthn_credentials_last_used_chk
        CHECK (last_used_at >= created_at)
);

CREATE INDEX idx_identity_webauthn_credentials_user
    ON identity_webauthn_credentials(user_id, created_at, credential_id);

CREATE TABLE identity_local_bootstrap_tickets (
    ticket_id UUID PRIMARY KEY,
    token_sha256 TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_local_bootstrap_token_hash_chk
        CHECK (token_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_local_bootstrap_expiry_chk CHECK (expires_at > created_at),
    CONSTRAINT identity_local_bootstrap_consumed_chk
        CHECK (consumed_at IS NULL OR consumed_at >= created_at)
);

CREATE UNIQUE INDEX idx_identity_local_bootstrap_single_active
    ON identity_local_bootstrap_tickets((TRUE)) WHERE consumed_at IS NULL;

CREATE TABLE identity_local_auth_challenges (
    challenge_id UUID PRIMARY KEY,
    purpose TEXT NOT NULL,
    challenge_sha256 TEXT NOT NULL,
    user_id UUID NULL REFERENCES identity_users(user_id) ON DELETE CASCADE,
    context_json JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_local_auth_challenges_purpose_chk
        CHECK (purpose IN ('bootstrap', 'login', 'register', 'recent_auth')),
    CONSTRAINT identity_local_auth_challenges_hash_chk
        CHECK (challenge_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_local_auth_challenges_context_chk
        CHECK (jsonb_typeof(context_json) = 'object'),
    CONSTRAINT identity_local_auth_challenges_expiry_chk CHECK (expires_at > created_at),
    CONSTRAINT identity_local_auth_challenges_consumed_chk
        CHECK (consumed_at IS NULL OR consumed_at >= created_at),
    CONSTRAINT identity_local_auth_challenges_user_chk CHECK (
        (purpose IN ('bootstrap', 'login') AND user_id IS NULL)
        OR (purpose IN ('register', 'recent_auth') AND user_id IS NOT NULL)
    ),
    CONSTRAINT identity_local_auth_challenges_bootstrap_context_chk CHECK (
        purpose <> 'bootstrap'
        OR context_json->>'bootstrap_user_id' ~
            '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    )
);

CREATE INDEX idx_identity_local_auth_challenges_expiry
    ON identity_local_auth_challenges(expires_at) WHERE consumed_at IS NULL;

CREATE TABLE identity_local_recovery_codes (
    recovery_code_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES identity_local_accounts(user_id) ON DELETE CASCADE,
    code_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_local_recovery_codes_hash_chk
        CHECK (code_hash LIKE '$argon2id$%'),
    CONSTRAINT identity_local_recovery_codes_consumed_chk
        CHECK (consumed_at IS NULL OR consumed_at >= created_at)
);

CREATE INDEX idx_identity_local_recovery_codes_user_active
    ON identity_local_recovery_codes(user_id) WHERE consumed_at IS NULL;

CREATE TABLE identity_local_auth_rate_limits (
    subject_sha256 TEXT PRIMARY KEY,
    failed_count INTEGER NOT NULL,
    window_started_at TIMESTAMPTZ NOT NULL,
    locked_until TIMESTAMPTZ NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_local_auth_rate_subject_chk
        CHECK (subject_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_local_auth_rate_count_chk CHECK (failed_count > 0),
    CONSTRAINT identity_local_auth_rate_updated_chk
        CHECK (updated_at >= window_started_at),
    CONSTRAINT identity_local_auth_rate_lock_chk
        CHECK (locked_until IS NULL OR locked_until >= window_started_at)
);

CREATE TABLE identity_local_auth_events (
    event_id UUID PRIMARY KEY,
    user_id UUID NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    subject_sha256 TEXT NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_local_auth_events_subject_chk
        CHECK (subject_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_local_auth_events_action_chk
        CHECK (action ~ '^local_auth\.[a-z_]+$'),
    CONSTRAINT identity_local_auth_events_outcome_chk
        CHECK (outcome IN ('succeeded', 'rejected')),
    CONSTRAINT identity_local_auth_events_reason_chk
        CHECK (reason_code ~ '^[a-z][a-z0-9_]{1,63}$')
);

CREATE INDEX idx_identity_local_auth_events_user_created
    ON identity_local_auth_events(user_id, created_at DESC, event_id DESC);

CREATE OR REPLACE FUNCTION identity_reject_local_auth_event_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'local auth events are immutable' USING ERRCODE = '55000';
END
$$;

CREATE TRIGGER identity_local_auth_events_immutable
BEFORE UPDATE OR DELETE ON identity_local_auth_events
FOR EACH ROW EXECUTE FUNCTION identity_reject_local_auth_event_mutation();

COMMIT;
