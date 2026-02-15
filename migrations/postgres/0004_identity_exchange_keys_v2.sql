BEGIN;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM identity_exchange_keys LIMIT 1) THEN
        RAISE EXCEPTION USING
            MESSAGE = '0004_identity_exchange_keys_v2.sql requires empty identity_exchange_keys table',
            HINT = 'Run explicit re-encryption migration for existing rows before applying v2 schema.';
    END IF;
END $$;

ALTER TABLE identity_exchange_keys
    ADD COLUMN api_key_enc BYTEA,
    ADD COLUMN api_key_hash BYTEA,
    ADD COLUMN api_key_last4 TEXT;

DROP INDEX IF EXISTS idx_identity_exchange_keys_active_unique_key;

ALTER TABLE identity_exchange_keys
    DROP COLUMN api_key;

ALTER TABLE identity_exchange_keys
    ALTER COLUMN api_key_enc SET NOT NULL,
    ALTER COLUMN api_key_hash SET NOT NULL,
    ALTER COLUMN api_key_last4 SET NOT NULL;

ALTER TABLE identity_exchange_keys
    ADD CONSTRAINT identity_exchange_keys_api_key_hash_len_chk
        CHECK (octet_length(api_key_hash) = 32),
    ADD CONSTRAINT identity_exchange_keys_api_key_last4_len_chk
        CHECK (char_length(api_key_last4) BETWEEN 1 AND 4);

CREATE UNIQUE INDEX IF NOT EXISTS idx_identity_exchange_keys_active_unique_key
    ON identity_exchange_keys (user_id, exchange_name, market_type, api_key_hash)
    WHERE is_deleted = FALSE;

COMMIT;
