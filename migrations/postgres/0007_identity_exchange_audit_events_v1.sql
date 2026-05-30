BEGIN;

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
