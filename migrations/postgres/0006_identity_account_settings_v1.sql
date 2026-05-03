BEGIN;

CREATE TABLE IF NOT EXISTS identity_user_preferences (
    owner_user_id UUID PRIMARY KEY REFERENCES identity_users (user_id) ON DELETE CASCADE,
    theme TEXT NOT NULL DEFAULT 'terminal-orange',
    locale TEXT NOT NULL DEFAULT 'en',
    density TEXT NOT NULL DEFAULT 'compact',
    email_notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    trade_alerts_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    product_updates_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_user_preferences_theme_chk
        CHECK (theme IN ('terminal-orange', 'graphite', 'matrix-green', 'high-contrast')),
    CONSTRAINT identity_user_preferences_locale_chk
        CHECK (locale IN ('en', 'ru')),
    CONSTRAINT identity_user_preferences_density_chk
        CHECK (density IN ('compact', 'comfortable')),
    CONSTRAINT identity_user_preferences_updated_at_chk
        CHECK (updated_at >= created_at)
);

CREATE INDEX IF NOT EXISTS idx_identity_user_preferences_updated_at
    ON identity_user_preferences (updated_at DESC);

CREATE TABLE IF NOT EXISTS identity_user_profile_overrides (
    owner_user_id UUID PRIMARY KEY REFERENCES identity_users (user_id) ON DELETE CASCADE,
    display_name TEXT NULL,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_user_profile_display_name_len_chk
        CHECK (display_name IS NULL OR char_length(display_name) <= 80),
    CONSTRAINT identity_user_profile_timezone_chk
        CHECK (timezone <> '' AND char_length(timezone) <= 64),
    CONSTRAINT identity_user_profile_updated_at_chk
        CHECK (updated_at >= created_at)
);

CREATE TABLE IF NOT EXISTS identity_integrations (
    owner_user_id UUID NOT NULL REFERENCES identity_users (user_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    settings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (owner_user_id, provider),
    CONSTRAINT identity_integrations_provider_chk
        CHECK (provider IN ('telegram', 'email_digest', 'webhook_alerts')),
    CONSTRAINT identity_integrations_settings_object_chk
        CHECK (jsonb_typeof(settings_json) = 'object'),
    CONSTRAINT identity_integrations_updated_at_chk
        CHECK (updated_at >= created_at)
);

CREATE INDEX IF NOT EXISTS idx_identity_integrations_owner_enabled
    ON identity_integrations (owner_user_id, enabled);

CREATE TABLE IF NOT EXISTS identity_audit_events (
    event_id UUID PRIMARY KEY,
    owner_user_id UUID NOT NULL REFERENCES identity_users (user_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    event_version INTEGER NOT NULL DEFAULT 1,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_audit_events_type_chk
        CHECK (event_type <> '' AND char_length(event_type) <= 96),
    CONSTRAINT identity_audit_events_version_chk
        CHECK (event_version >= 1),
    CONSTRAINT identity_audit_events_metadata_object_chk
        CHECK (jsonb_typeof(metadata_json) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_identity_audit_events_owner_cursor
    ON identity_audit_events (owner_user_id, created_at DESC, event_id DESC);

CREATE INDEX IF NOT EXISTS idx_identity_sessions_owner_cursor
    ON identity_sessions (user_id, created_at DESC, session_id DESC);

COMMIT;
