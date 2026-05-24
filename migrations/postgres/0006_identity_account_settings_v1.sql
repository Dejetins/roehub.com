BEGIN;

CREATE TABLE IF NOT EXISTS identity_user_preferences (
    owner_user_id UUID PRIMARY KEY REFERENCES identity_users(user_id) ON DELETE CASCADE,
    theme TEXT NOT NULL DEFAULT 'terminal-orange',
    locale TEXT NOT NULL DEFAULT 'en',
    density TEXT NOT NULL DEFAULT 'compact',
    autorefresh_preset TEXT NOT NULL DEFAULT '15s',
    refresh_interval_seconds INTEGER NOT NULL DEFAULT 15,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_user_preferences_theme_check
        CHECK (theme IN ('terminal-orange', 'graphite', 'matrix-green', 'high-contrast')),
    CONSTRAINT identity_user_preferences_locale_check
        CHECK (locale IN ('en', 'ru')),
    CONSTRAINT identity_user_preferences_density_check
        CHECK (density IN ('compact', 'comfortable')),
    CONSTRAINT identity_user_preferences_autorefresh_preset_check
        CHECK (autorefresh_preset IN ('off', '10s', '15s', '30s', '1m', '5m', 'custom')),
    CONSTRAINT identity_user_preferences_refresh_interval_check
        CHECK (
            (autorefresh_preset = 'off' AND refresh_interval_seconds = 0)
            OR (autorefresh_preset <> 'off' AND refresh_interval_seconds >= 10)
        )
);

CREATE INDEX IF NOT EXISTS idx_identity_user_preferences_updated_at
    ON identity_user_preferences(updated_at DESC);

ALTER TABLE identity_user_preferences
    ADD COLUMN IF NOT EXISTS autorefresh_preset TEXT NOT NULL DEFAULT '15s';

ALTER TABLE identity_user_preferences
    ADD COLUMN IF NOT EXISTS refresh_interval_seconds INTEGER NOT NULL DEFAULT 15;

CREATE TABLE IF NOT EXISTS identity_user_profile_overrides (
    owner_user_id UUID PRIMARY KEY REFERENCES identity_users(user_id) ON DELETE CASCADE,
    username TEXT,
    email TEXT,
    timezone TEXT NOT NULL DEFAULT 'Europe/Moscow',
    telegram_discord TEXT,
    updated_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE identity_user_profile_overrides
    ADD COLUMN IF NOT EXISTS username TEXT;

ALTER TABLE identity_user_profile_overrides
    ADD COLUMN IF NOT EXISTS email TEXT;

ALTER TABLE identity_user_profile_overrides
    ADD COLUMN IF NOT EXISTS telegram_discord TEXT;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'identity_user_profile_overrides'
          AND column_name = 'display_name'
    ) THEN
        UPDATE identity_user_profile_overrides
        SET username = COALESCE(username, display_name)
        WHERE username IS NULL;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'identity_user_profile_overrides'
          AND column_name = 'created_at'
    ) THEN
        ALTER TABLE identity_user_profile_overrides
            ALTER COLUMN created_at SET DEFAULT '1970-01-01 00:00:00+00'::timestamptz;
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS identity_integrations (
    owner_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE CASCADE,
    integration_key TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'off',
    webhook_url_masked TEXT,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (owner_user_id, integration_key),
    CONSTRAINT identity_integrations_key_check
        CHECK (integration_key IN ('telegram', 'discord', 'slack')),
    CONSTRAINT identity_integrations_mode_check
        CHECK (mode IN ('off', 'alerts', 'critical'))
);

CREATE INDEX IF NOT EXISTS idx_identity_integrations_updated_at
    ON identity_integrations(owner_user_id, updated_at DESC);

ALTER TABLE identity_integrations
    ADD COLUMN IF NOT EXISTS integration_key TEXT;

ALTER TABLE identity_integrations
    ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'off';

ALTER TABLE identity_integrations
    ADD COLUMN IF NOT EXISTS webhook_url_masked TEXT;

DO $$
DECLARE
    pkey_definition TEXT;
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'identity_integrations'
          AND column_name = 'provider'
    ) THEN
        UPDATE identity_integrations
        SET integration_key = COALESCE(
                integration_key,
                CASE provider
                    WHEN 'telegram' THEN 'telegram'
                    WHEN 'webhook_alerts' THEN 'slack'
                    ELSE 'discord'
                END
            ),
            mode = CASE
                WHEN integration_key IS NULL AND provider = 'email_digest' THEN 'off'
                WHEN integration_key IS NULL AND enabled THEN 'alerts'
                WHEN integration_key IS NULL THEN 'off'
                WHEN mode IS NOT NULL THEN mode
                ELSE 'off'
            END
        WHERE integration_key IS NULL
           OR mode IS NULL;

        ALTER TABLE identity_integrations
            ALTER COLUMN provider SET DEFAULT 'telegram';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'identity_integrations'
          AND column_name = 'created_at'
    ) THEN
        ALTER TABLE identity_integrations
            ALTER COLUMN created_at SET DEFAULT '1970-01-01 00:00:00+00'::timestamptz;
    END IF;

    UPDATE identity_integrations
    SET integration_key = 'telegram'
    WHERE integration_key IS NULL;

    ALTER TABLE identity_integrations
        ALTER COLUMN integration_key SET NOT NULL;

    SELECT pg_get_constraintdef(oid)
    INTO pkey_definition
    FROM pg_constraint
    WHERE conrelid = 'identity_integrations'::regclass
      AND conname = 'identity_integrations_pkey';

    IF pkey_definition LIKE '%provider%' THEN
        ALTER TABLE identity_integrations
            DROP CONSTRAINT identity_integrations_pkey;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'identity_integrations'::regclass
          AND conname = 'identity_integrations_pkey'
    ) THEN
        ALTER TABLE identity_integrations
            ADD CONSTRAINT identity_integrations_pkey
            PRIMARY KEY (owner_user_id, integration_key);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'identity_integrations'::regclass
          AND conname = 'identity_integrations_key_check'
    ) THEN
        ALTER TABLE identity_integrations
            ADD CONSTRAINT identity_integrations_key_check
            CHECK (integration_key IN ('telegram', 'discord', 'slack'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'identity_integrations'::regclass
          AND conname = 'identity_integrations_mode_check'
    ) THEN
        ALTER TABLE identity_integrations
            ADD CONSTRAINT identity_integrations_mode_check
            CHECK (mode IN ('off', 'alerts', 'critical'));
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS identity_notification_preferences (
    owner_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE CASCADE,
    channel_key TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'on',
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (owner_user_id, channel_key),
    CONSTRAINT identity_notification_preferences_channel_check
        CHECK (
            channel_key IN (
                'telegram',
                'email',
                'push',
                'trade_fills',
                'risk_alerts',
                'daily_report',
                'system'
            )
        ),
    CONSTRAINT identity_notification_preferences_mode_check
        CHECK (mode IN ('off', 'on', 'critical'))
);

CREATE TABLE IF NOT EXISTS identity_audit_events (
    event_id UUID PRIMARY KEY,
    owner_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT identity_audit_events_type_check
        CHECK (
            event_type IN (
                'profile_updated',
                'preferences_updated',
                'integration_updated',
                'notifications_updated'
            )
        )
);

CREATE INDEX IF NOT EXISTS idx_identity_audit_events_owner_created
    ON identity_audit_events(owner_user_id, created_at DESC, event_id DESC);

ALTER TABLE identity_audit_events
    ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT 'Account event';

COMMIT;
