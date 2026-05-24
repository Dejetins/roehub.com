BEGIN;

CREATE TABLE IF NOT EXISTS exchange_connections (
    connection_id UUID PRIMARY KEY,
    owner_user_id UUID NOT NULL REFERENCES identity_users (user_id) ON DELETE CASCADE,
    exchange_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    environment TEXT NOT NULL,
    label TEXT NULL,
    active_credential_version_id UUID NULL,
    status TEXT NOT NULL,
    status_reason TEXT NULL,
    permission_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ip_restriction_status TEXT NOT NULL DEFAULT 'unknown',
    remote_account_fingerprint BYTEA NULL,
    last_validated_at TIMESTAMPTZ NULL,
    last_used_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    disabled_at TIMESTAMPTZ NULL,
    CONSTRAINT exchange_connections_exchange_name_chk
        CHECK (exchange_name IN ('binance', 'bybit')),
    CONSTRAINT exchange_connections_market_type_chk
        CHECK (market_type IN ('spot', 'futures')),
    CONSTRAINT exchange_connections_environment_chk
        CHECK (environment IN ('mainnet', 'testnet')),
    CONSTRAINT exchange_connections_status_chk
        CHECK (status IN ('active', 'disabled')),
    CONSTRAINT exchange_connections_disabled_state_chk
        CHECK (
            (status = 'disabled' AND disabled_at IS NOT NULL)
            OR
            (status = 'active' AND disabled_at IS NULL)
        )
);

CREATE TABLE IF NOT EXISTS exchange_credential_versions (
    credential_version_id UUID PRIMARY KEY,
    connection_id UUID NOT NULL REFERENCES exchange_connections (connection_id)
        ON DELETE CASCADE,
    api_key_ciphertext TEXT NOT NULL,
    api_secret_ciphertext TEXT NOT NULL,
    passphrase_ciphertext TEXT NULL,
    api_key_last4 TEXT NOT NULL,
    api_key_fingerprint_hmac BYTEA NOT NULL,
    secret_cipher TEXT NOT NULL,
    transit_key_id TEXT NOT NULL,
    credential_scheme TEXT NOT NULL,
    status TEXT NOT NULL,
    created_by_user_id UUID NOT NULL REFERENCES identity_users (user_id),
    created_by_session_id UUID NULL,
    created_at TIMESTAMPTZ NOT NULL,
    rotated_at TIMESTAMPTZ NULL,
    disabled_at TIMESTAMPTZ NULL,
    CONSTRAINT exchange_credential_versions_api_key_last4_len_chk
        CHECK (char_length(api_key_last4) BETWEEN 1 AND 4),
    CONSTRAINT exchange_credential_versions_fingerprint_len_chk
        CHECK (octet_length(api_key_fingerprint_hmac) >= 16),
    CONSTRAINT exchange_credential_versions_status_chk
        CHECK (status IN ('active', 'rotated', 'disabled'))
);

ALTER TABLE exchange_connections
    DROP CONSTRAINT IF EXISTS exchange_connections_active_credential_version_id_fk;

ALTER TABLE exchange_connections
    ADD CONSTRAINT exchange_connections_active_credential_version_id_fk
        FOREIGN KEY (active_credential_version_id)
        REFERENCES exchange_credential_versions (credential_version_id)
        DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX IF NOT EXISTS idx_exchange_connections_owner_status_created
    ON exchange_connections (owner_user_id, status, created_at, connection_id);

CREATE INDEX IF NOT EXISTS idx_exchange_credential_versions_connection_created
    ON exchange_credential_versions (connection_id, created_at, credential_version_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_exchange_connections_active_unique_fingerprint
    ON exchange_connections (
        owner_user_id,
        exchange_name,
        market_type,
        environment,
        active_credential_version_id
    )
    WHERE status = 'active';

INSERT INTO exchange_connections (
    connection_id,
    owner_user_id,
    exchange_name,
    market_type,
    environment,
    label,
    active_credential_version_id,
    status,
    status_reason,
    permission_summary_json,
    ip_restriction_status,
    created_at,
    updated_at,
    disabled_at
)
SELECT
    key_id AS connection_id,
    user_id AS owner_user_id,
    exchange_name,
    market_type,
    'mainnet' AS environment,
    label,
    NULL AS active_credential_version_id,
    CASE WHEN is_deleted THEN 'disabled' ELSE 'active' END AS status,
    CASE WHEN is_deleted THEN 'legacy_deleted' ELSE NULL END AS status_reason,
    jsonb_build_object('permissions', permissions) AS permission_summary_json,
    'unknown' AS ip_restriction_status,
    created_at,
    updated_at,
    deleted_at AS disabled_at
FROM identity_exchange_keys
ON CONFLICT (connection_id) DO NOTHING;

INSERT INTO exchange_credential_versions (
    credential_version_id,
    connection_id,
    api_key_ciphertext,
    api_secret_ciphertext,
    passphrase_ciphertext,
    api_key_last4,
    api_key_fingerprint_hmac,
    secret_cipher,
    transit_key_id,
    credential_scheme,
    status,
    created_by_user_id,
    created_by_session_id,
    created_at,
    rotated_at,
    disabled_at
)
SELECT
    key_id AS credential_version_id,
    key_id AS connection_id,
    'legacy:identity_exchange_keys_v2:' || encode(api_key_enc, 'hex') AS api_key_ciphertext,
    'legacy:identity_exchange_keys_v2:' || encode(api_secret_enc, 'hex') AS api_secret_ciphertext,
    CASE
        WHEN passphrase_enc IS NULL THEN NULL
        ELSE 'legacy:identity_exchange_keys_v2:' || encode(passphrase_enc, 'hex')
    END AS passphrase_ciphertext,
    api_key_last4,
    api_key_hash AS api_key_fingerprint_hmac,
    'identity_exchange_keys_v2_legacy' AS secret_cipher,
    'legacy_identity_exchange_keys_v2' AS transit_key_id,
    'legacy_backfill_v1' AS credential_scheme,
    CASE WHEN is_deleted THEN 'disabled' ELSE 'active' END AS status,
    user_id AS created_by_user_id,
    NULL AS created_by_session_id,
    created_at,
    NULL AS rotated_at,
    deleted_at AS disabled_at
FROM identity_exchange_keys
ON CONFLICT (credential_version_id) DO NOTHING;

UPDATE exchange_connections AS connection
SET active_credential_version_id = legacy.key_id
FROM identity_exchange_keys AS legacy
WHERE connection.connection_id = legacy.key_id
  AND connection.active_credential_version_id IS NULL;

COMMIT;
