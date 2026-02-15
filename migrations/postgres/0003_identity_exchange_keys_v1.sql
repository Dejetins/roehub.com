BEGIN;

CREATE TABLE IF NOT EXISTS identity_exchange_keys (
    key_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES identity_users (user_id) ON DELETE CASCADE,
    exchange_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    label TEXT NULL,
    permissions TEXT NOT NULL,
    api_key TEXT NOT NULL,
    api_secret_enc BYTEA NOT NULL,
    passphrase_enc BYTEA NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_exchange_keys_exchange_name_chk
        CHECK (exchange_name IN ('binance', 'bybit')),
    CONSTRAINT identity_exchange_keys_market_type_chk
        CHECK (market_type IN ('spot', 'futures')),
    CONSTRAINT identity_exchange_keys_permissions_chk
        CHECK (permissions IN ('read', 'trade')),
    CONSTRAINT identity_exchange_keys_deleted_state_chk
        CHECK (
            (is_deleted = TRUE AND deleted_at IS NOT NULL)
            OR
            (is_deleted = FALSE AND deleted_at IS NULL)
        )
);

CREATE INDEX IF NOT EXISTS idx_identity_exchange_keys_user_deleted_created
    ON identity_exchange_keys (user_id, is_deleted, created_at, key_id);

CREATE INDEX IF NOT EXISTS idx_identity_exchange_keys_active_by_user
    ON identity_exchange_keys (user_id, created_at, key_id)
    WHERE is_deleted = FALSE;

CREATE UNIQUE INDEX IF NOT EXISTS idx_identity_exchange_keys_active_unique_key
    ON identity_exchange_keys (user_id, exchange_name, market_type, api_key)
    WHERE is_deleted = FALSE;

COMMIT;
