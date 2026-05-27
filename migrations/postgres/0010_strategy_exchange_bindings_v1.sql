BEGIN;

CREATE TABLE IF NOT EXISTS strategy_exchange_bindings (
    binding_id UUID PRIMARY KEY,
    owner_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE CASCADE,
    strategy_id UUID NOT NULL REFERENCES strategy_strategies(strategy_id) ON DELETE CASCADE,
    exchange_connection_id UUID NOT NULL REFERENCES exchange_connections(connection_id) ON DELETE RESTRICT,
    usage_mode TEXT NOT NULL,
    binding_status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    disabled_at TIMESTAMPTZ,
    archived_at TIMESTAMPTZ,
    CONSTRAINT strategy_exchange_bindings_usage_mode_check
        CHECK (usage_mode IN ('trading')),
    CONSTRAINT strategy_exchange_bindings_status_check
        CHECK (binding_status IN ('active', 'paused', 'disabled', 'archived')),
    CONSTRAINT strategy_exchange_bindings_lifecycle_check
        CHECK (
            (binding_status IN ('active', 'paused') AND disabled_at IS NULL AND archived_at IS NULL)
            OR (binding_status = 'disabled' AND disabled_at IS NOT NULL AND archived_at IS NULL)
            OR (binding_status = 'archived' AND disabled_at IS NOT NULL AND archived_at IS NOT NULL)
        )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_exchange_bindings_active_unique
    ON strategy_exchange_bindings(
        owner_user_id,
        strategy_id,
        exchange_connection_id,
        usage_mode
    )
    WHERE binding_status = 'active';

CREATE INDEX IF NOT EXISTS idx_strategy_exchange_bindings_connection_active
    ON strategy_exchange_bindings(owner_user_id, exchange_connection_id)
    WHERE usage_mode = 'trading' AND binding_status = 'active';

CREATE INDEX IF NOT EXISTS idx_strategy_exchange_bindings_strategy_created
    ON strategy_exchange_bindings(owner_user_id, strategy_id, created_at DESC);

ALTER TABLE identity_audit_events
    DROP CONSTRAINT IF EXISTS identity_audit_events_type_check;

ALTER TABLE identity_audit_events
    ADD CONSTRAINT identity_audit_events_type_check
        CHECK (
            event_type IN (
                'profile_updated',
                'preferences_updated',
                'integration_updated',
                'notifications_updated',
                'exchange_key_created',
                'exchange_key_deleted',
                'exchange_connection_created',
                'exchange_connection_validated',
                'exchange_connection_validation_failed',
                'exchange_credential_rotated',
                'exchange_connection_disabled',
                'exchange_connection_archived',
                'exchange_connection_deleted',
                'exchange_connection_reclassified',
                'exchange_connection_disconnect_blocked',
                'strategy_exchange_binding_created',
                'strategy_exchange_binding_disabled',
                'strategy_exchange_binding_archived'
            )
        );

COMMIT;
