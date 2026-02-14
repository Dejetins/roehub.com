BEGIN;

CREATE TABLE IF NOT EXISTS identity_users (
    user_id UUID PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL,
    paid_level TEXT NOT NULL DEFAULT 'free',
    created_at TIMESTAMPTZ NOT NULL,
    last_login_at TIMESTAMPTZ NULL,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT identity_users_paid_level_chk
        CHECK (paid_level IN ('base', 'free', 'pro', 'ultra'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_identity_users_telegram_user_id
    ON identity_users (telegram_user_id);

CREATE INDEX IF NOT EXISTS idx_identity_users_is_deleted
    ON identity_users (is_deleted);

CREATE TABLE IF NOT EXISTS identity_telegram_channels (
    user_id UUID NOT NULL REFERENCES identity_users (user_id),
    chat_id BIGINT NOT NULL,
    is_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    confirmed_at TIMESTAMPTZ NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_identity_telegram_channels_chat_id
    ON identity_telegram_channels (chat_id);

COMMIT;
