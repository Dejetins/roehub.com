BEGIN;

CREATE TABLE identity_oidc_login_attempts (
    attempt_id UUID PRIMARY KEY,
    provider_id TEXT NOT NULL,
    issuer TEXT NOT NULL,
    purpose TEXT NOT NULL,
    state_sha256 TEXT NOT NULL UNIQUE,
    nonce_sha256 TEXT NOT NULL,
    code_verifier TEXT NOT NULL,
    linking_user_id UUID NULL REFERENCES identity_users(user_id) ON DELETE CASCADE,
    next_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    exchange_started_at TIMESTAMPTZ NULL,
    consumed_at TIMESTAMPTZ NULL,
    rejection_reason TEXT NULL,
    CONSTRAINT identity_oidc_attempts_provider_chk
        CHECK (provider_id ~ '^[a-z][a-z0-9._-]{2,63}$'),
    CONSTRAINT identity_oidc_attempts_issuer_chk
        CHECK (issuer ~ '^https?://[^[:space:]]+$' AND char_length(issuer) <= 2048),
    CONSTRAINT identity_oidc_attempts_purpose_chk CHECK (purpose IN ('login', 'link')),
    CONSTRAINT identity_oidc_attempts_state_chk CHECK (state_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_oidc_attempts_nonce_chk CHECK (nonce_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_oidc_attempts_verifier_chk
        CHECK (code_verifier ~ '^[A-Za-z0-9_-]{43,128}$'),
    CONSTRAINT identity_oidc_attempts_next_chk
        CHECK (next_path ~ '^/[^\\]*$' AND char_length(next_path) <= 1024),
    CONSTRAINT identity_oidc_attempts_expiry_chk CHECK (expires_at > created_at),
    CONSTRAINT identity_oidc_attempts_consumed_chk
        CHECK (consumed_at IS NULL OR consumed_at >= created_at),
    CONSTRAINT identity_oidc_attempts_exchange_started_chk CHECK (
        exchange_started_at IS NULL
        OR (exchange_started_at >= created_at AND exchange_started_at < expires_at)
    ),
    CONSTRAINT identity_oidc_attempts_linking_chk CHECK (
        (purpose = 'link' AND linking_user_id IS NOT NULL)
        OR (purpose = 'login' AND linking_user_id IS NULL)
    ),
    CONSTRAINT identity_oidc_attempts_rejection_chk CHECK (
        rejection_reason IS NULL
        OR rejection_reason ~ '^[a-z][a-z0-9_]{1,63}$'
    )
);

CREATE INDEX idx_identity_oidc_attempts_active
    ON identity_oidc_login_attempts(expires_at, attempt_id)
    WHERE consumed_at IS NULL;

CREATE TABLE identity_external_identities (
    external_identity_id UUID PRIMARY KEY,
    provider_id TEXT NOT NULL,
    issuer TEXT NOT NULL,
    subject_sha256 TEXT NOT NULL,
    user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    verified_email_sha256 TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    last_login_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_external_provider_chk
        CHECK (provider_id ~ '^[a-z][a-z0-9._-]{2,63}$'),
    CONSTRAINT identity_external_issuer_chk
        CHECK (issuer ~ '^https?://[^[:space:]]+$' AND char_length(issuer) <= 2048),
    CONSTRAINT identity_external_subject_chk CHECK (subject_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_external_email_chk
        CHECK (verified_email_sha256 IS NULL OR verified_email_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_external_login_chk CHECK (last_login_at >= created_at),
    UNIQUE (provider_id, issuer, subject_sha256),
    UNIQUE (provider_id, issuer, user_id)
);

CREATE INDEX idx_identity_external_identities_user
    ON identity_external_identities(user_id, provider_id, issuer);

CREATE TABLE identity_oidc_auth_events (
    event_id UUID PRIMARY KEY,
    attempt_id UUID NULL REFERENCES identity_oidc_login_attempts(attempt_id) ON DELETE SET NULL,
    provider_id TEXT NOT NULL,
    user_id UUID NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    subject_sha256 TEXT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_oidc_events_provider_chk
        CHECK (provider_id ~ '^[a-z][a-z0-9._-]{2,63}$'),
    CONSTRAINT identity_oidc_events_subject_chk
        CHECK (subject_sha256 IS NULL OR subject_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_oidc_events_action_chk CHECK (action ~ '^oidc\.[a-z_]+$'),
    CONSTRAINT identity_oidc_events_outcome_chk
        CHECK (outcome IN ('succeeded', 'rejected')),
    CONSTRAINT identity_oidc_events_reason_chk
        CHECK (reason_code ~ '^[a-z][a-z0-9_]{1,63}$')
);

CREATE INDEX idx_identity_oidc_events_provider_created
    ON identity_oidc_auth_events(provider_id, created_at DESC, event_id DESC);

CREATE OR REPLACE FUNCTION identity_reject_oidc_event_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'OIDC auth events are immutable' USING ERRCODE = '55000';
END
$$;

CREATE TRIGGER identity_oidc_auth_events_immutable
BEFORE UPDATE OR DELETE ON identity_oidc_auth_events
FOR EACH ROW EXECUTE FUNCTION identity_reject_oidc_event_mutation();

COMMIT;
