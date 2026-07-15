BEGIN;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM execution_intents LIMIT 1)
       OR EXISTS (SELECT 1 FROM execution_orders LIMIT 1) THEN
        RAISE EXCEPTION
            'execution gateway safety migration requires empty greenfield execution tables';
    END IF;
END
$$;

ALTER TABLE execution_intents
    ADD COLUMN constraints_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN canonical_intent_hash TEXT NOT NULL,
    ADD CONSTRAINT execution_intents_constraints_object_chk
        CHECK (jsonb_typeof(constraints_json) = 'object'),
    ADD CONSTRAINT execution_intents_canonical_hash_chk
        CHECK (canonical_intent_hash ~ '^[0-9a-f]{64}$');

CREATE TABLE execution_gateway_audit_events (
    event_id UUID PRIMARY KEY,
    organization_id UUID NULL,
    owner_user_id UUID NULL,
    exchange_connection_id UUID NULL,
    intent_id UUID NULL,
    approval_id UUID NULL,
    event_type TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    actor_user_id UUID NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT execution_gateway_audit_org_fk
        FOREIGN KEY (organization_id)
        REFERENCES identity_organizations(organization_id) ON DELETE RESTRICT,
    CONSTRAINT execution_gateway_audit_owner_fk
        FOREIGN KEY (owner_user_id)
        REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    CONSTRAINT execution_gateway_audit_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id)
        ON DELETE RESTRICT,
    CONSTRAINT execution_gateway_audit_intent_fk
        FOREIGN KEY (organization_id, intent_id)
        REFERENCES execution_intents(organization_id, intent_id) ON DELETE RESTRICT,
    CONSTRAINT execution_gateway_audit_decision_chk
        CHECK (decision IN ('accepted', 'rejected')),
    CONSTRAINT execution_gateway_audit_reason_chk
        CHECK (char_length(trim(reason)) > 0),
    CONSTRAINT execution_gateway_audit_metadata_chk
        CHECK (jsonb_typeof(metadata_json) = 'object')
);

CREATE TABLE execution_provider_allowlist (
    provider_id TEXT PRIMARY KEY,
    provider_version TEXT NOT NULL,
    provider_kind TEXT NOT NULL,
    exchange_name TEXT NOT NULL,
    revision_hash TEXT NOT NULL,
    order_submit_capability BOOLEAN NOT NULL,
    enabled BOOLEAN NOT NULL,
    approved_by_user_id UUID NOT NULL
        REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    updated_at TIMESTAMPTZ NOT NULL,
    audit_event_id UUID NOT NULL
        REFERENCES execution_gateway_audit_events(event_id) ON DELETE RESTRICT,
    CONSTRAINT execution_provider_kind_chk
        CHECK (provider_kind IN ('core', 'verified')),
    CONSTRAINT execution_provider_exchange_chk
        CHECK (exchange_name IN ('binance', 'bybit')),
    CONSTRAINT execution_provider_revision_chk
        CHECK (revision_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT execution_provider_id_chk
        CHECK (provider_id ~ '^[a-z0-9][a-z0-9:._-]{2,127}$')
);

CREATE TABLE execution_account_safety_state (
    organization_id UUID NOT NULL,
    owner_user_id UUID NOT NULL,
    exchange_connection_id UUID NOT NULL,
    mode TEXT NOT NULL,
    risk_revision_hash TEXT NOT NULL,
    account_revision_hash TEXT NOT NULL,
    secret_reference_hash TEXT NOT NULL,
    risk_allows_submit BOOLEAN NOT NULL,
    max_order_notional NUMERIC NOT NULL,
    daily_notional_limit NUMERIC NOT NULL,
    max_account_exposure_notional NUMERIC NOT NULL,
    risk_valid_until TIMESTAMPTZ NOT NULL,
    updated_by_user_id UUID NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    audit_event_id UUID NOT NULL
        REFERENCES execution_gateway_audit_events(event_id) ON DELETE RESTRICT,
    PRIMARY KEY (organization_id, exchange_connection_id),
    CONSTRAINT execution_account_safety_owner_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    CONSTRAINT execution_account_safety_connection_fk
        FOREIGN KEY (organization_id, owner_user_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, owner_user_id, connection_id)
        ON DELETE RESTRICT,
    CONSTRAINT execution_account_safety_actor_fk
        FOREIGN KEY (organization_id, updated_by_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    CONSTRAINT execution_account_safety_mode_chk
        CHECK (mode IN ('research', 'paper', 'testnet', 'mainnet')),
    CONSTRAINT execution_account_safety_risk_hash_chk
        CHECK (risk_revision_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT execution_account_safety_account_hash_chk
        CHECK (account_revision_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT execution_account_safety_secret_ref_hash_chk
        CHECK (secret_reference_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT execution_account_safety_limits_chk CHECK (
        max_order_notional > 0
        AND daily_notional_limit >= max_order_notional
        AND max_account_exposure_notional >= max_order_notional
    ),
    CONSTRAINT execution_account_safety_freshness_chk
        CHECK (updated_at < risk_valid_until)
);

CREATE TABLE execution_kill_switch_state (
    scope_type TEXT NOT NULL,
    organization_id UUID NULL,
    exchange_connection_id UUID NULL,
    active BOOLEAN NOT NULL,
    reason TEXT NOT NULL,
    updated_by_user_id UUID NOT NULL
        REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    updated_at TIMESTAMPTZ NOT NULL,
    audit_event_id UUID NOT NULL
        REFERENCES execution_gateway_audit_events(event_id) ON DELETE RESTRICT,
    CONSTRAINT execution_kill_switch_scope_unique
        UNIQUE NULLS NOT DISTINCT
        (scope_type, organization_id, exchange_connection_id),
    CONSTRAINT execution_kill_switch_scope_chk CHECK (
        (scope_type = 'installation'
            AND organization_id IS NULL
            AND exchange_connection_id IS NULL)
        OR (scope_type = 'organization'
            AND organization_id IS NOT NULL
            AND exchange_connection_id IS NULL)
        OR (scope_type = 'account'
            AND organization_id IS NOT NULL
            AND exchange_connection_id IS NOT NULL)
    ),
    CONSTRAINT execution_kill_switch_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id)
        ON DELETE RESTRICT,
    CONSTRAINT execution_kill_switch_reason_chk
        CHECK (char_length(trim(reason)) > 0)
);

CREATE TABLE execution_mainnet_approvals (
    approval_id UUID PRIMARY KEY,
    organization_id UUID NOT NULL,
    owner_user_id UUID NOT NULL,
    exchange_connection_id UUID NOT NULL,
    exchange_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    provider_id TEXT NOT NULL
        REFERENCES execution_provider_allowlist(provider_id) ON DELETE RESTRICT,
    risk_revision_hash TEXT NOT NULL,
    account_revision_hash TEXT NOT NULL,
    provider_revision_hash TEXT NOT NULL,
    recent_auth_session_id UUID NOT NULL
        REFERENCES identity_sessions(session_id) ON DELETE RESTRICT,
    recent_auth_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    audit_event_id UUID NOT NULL
        REFERENCES execution_gateway_audit_events(event_id) ON DELETE RESTRICT,
    revoked_at TIMESTAMPTZ NULL,
    revocation_reason TEXT NULL,
    revocation_audit_event_id UUID NULL
        REFERENCES execution_gateway_audit_events(event_id) ON DELETE RESTRICT,
    CONSTRAINT execution_mainnet_approval_owner_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    CONSTRAINT execution_mainnet_approval_connection_fk
        FOREIGN KEY (organization_id, owner_user_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, owner_user_id, connection_id)
        ON DELETE RESTRICT,
    CONSTRAINT execution_mainnet_approval_exchange_chk
        CHECK (exchange_name IN ('binance', 'bybit')),
    CONSTRAINT execution_mainnet_approval_market_chk
        CHECK (market_type IN ('spot', 'futures')),
    CONSTRAINT execution_mainnet_approval_window_chk
        CHECK (
            recent_auth_at <= approved_at
            AND approved_at - recent_auth_at <= INTERVAL '10 minutes'
            AND approved_at < expires_at
            AND expires_at <= approved_at + INTERVAL '15 minutes'
        ),
    CONSTRAINT execution_mainnet_approval_hashes_chk CHECK (
        risk_revision_hash ~ '^[0-9a-f]{64}$'
        AND account_revision_hash ~ '^[0-9a-f]{64}$'
        AND provider_revision_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT execution_mainnet_approval_revocation_chk CHECK (
        (revoked_at IS NULL
            AND revocation_reason IS NULL
            AND revocation_audit_event_id IS NULL)
        OR (revoked_at IS NOT NULL
            AND revoked_at >= approved_at
            AND char_length(trim(revocation_reason)) > 0
            AND revocation_audit_event_id IS NOT NULL)
    )
);

CREATE INDEX execution_mainnet_approvals_active_lookup
    ON execution_mainnet_approvals (
        organization_id,
        exchange_connection_id,
        exchange_name,
        market_type,
        provider_id
    )
    WHERE revoked_at IS NULL;

ALTER TABLE execution_orders
    ADD COLUMN submit_claim_id UUID NULL,
    ADD COLUMN submit_claimed_at TIMESTAMPTZ NULL,
    ADD COLUMN submit_claim_expires_at TIMESTAMPTZ NULL,
    ADD COLUMN submit_guard_audit_event_id UUID NULL
        REFERENCES execution_gateway_audit_events(event_id) ON DELETE RESTRICT,
    ADD COLUMN mainnet_approval_id UUID NULL
        REFERENCES execution_mainnet_approvals(approval_id) ON DELETE RESTRICT,
    ADD CONSTRAINT execution_orders_submit_claim_chk CHECK (
        (submit_claim_id IS NULL
            AND submit_claimed_at IS NULL
            AND submit_claim_expires_at IS NULL)
        OR (submit_claim_id IS NOT NULL
            AND submit_claimed_at IS NOT NULL
            AND submit_claim_expires_at > submit_claimed_at)
    );

ALTER TABLE execution_orders
    DROP CONSTRAINT IF EXISTS execution_orders_environment_chk;

ALTER TABLE execution_orders
    ADD CONSTRAINT execution_orders_environment_chk CHECK (
        environment = 'testnet'
        OR (
            environment = 'mainnet'
            AND (
                status = 'guard_rejected'
                OR (
                    submit_guard_audit_event_id IS NOT NULL
                    AND mainnet_approval_id IS NOT NULL
                )
            )
        )
    );

ALTER TABLE exchange_execution_process_heartbeats
    DROP CONSTRAINT IF EXISTS exchange_execution_process_heartbeats_adapter_mode_chk;
ALTER TABLE exchange_execution_process_heartbeats
    ADD CONSTRAINT exchange_execution_process_heartbeats_adapter_mode_chk
        CHECK (adapter_mode IN ('disabled', 'testnet', 'emulator'));

ALTER TABLE exchange_execution_request_observations
    DROP CONSTRAINT IF EXISTS exchange_execution_request_observations_adapter_mode_chk;
ALTER TABLE exchange_execution_request_observations
    ADD CONSTRAINT exchange_execution_request_observations_adapter_mode_chk
        CHECK (adapter_mode IN ('disabled', 'testnet', 'emulator'));

ALTER TABLE exchange_execution_request_observations
    DROP CONSTRAINT IF EXISTS exchange_execution_request_observations_status_chk;
ALTER TABLE exchange_execution_request_observations
    ADD CONSTRAINT exchange_execution_request_observations_status_chk CHECK (
        status IN (
            'adapter_disabled',
            'adapter_error',
            'guard_rejected',
            'quarantined',
            'skipped',
            'reconciled',
            'testnet_submitted',
            'emulator_submitted'
        )
    );

ALTER TABLE exchange_private_stream_sessions
    DROP CONSTRAINT IF EXISTS exchange_private_stream_sessions_environment_chk;
ALTER TABLE exchange_private_stream_sessions
    ADD CONSTRAINT exchange_private_stream_sessions_environment_chk
        CHECK (environment IN ('testnet', 'mainnet'));

CREATE FUNCTION execution_gateway_audit_immutable_guard()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'execution gateway audit events are immutable';
END
$$;

CREATE TRIGGER execution_gateway_audit_immutable
BEFORE UPDATE OR DELETE ON execution_gateway_audit_events
FOR EACH ROW EXECUTE FUNCTION execution_gateway_audit_immutable_guard();

CREATE FUNCTION execution_gateway_policy_delete_guard()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'execution gateway policy rows cannot be deleted';
END
$$;

CREATE TRIGGER execution_provider_allowlist_no_delete
BEFORE DELETE ON execution_provider_allowlist
FOR EACH ROW EXECUTE FUNCTION execution_gateway_policy_delete_guard();

CREATE TRIGGER execution_account_safety_no_delete
BEFORE DELETE ON execution_account_safety_state
FOR EACH ROW EXECUTE FUNCTION execution_gateway_policy_delete_guard();

CREATE TRIGGER execution_kill_switch_no_delete
BEFORE DELETE ON execution_kill_switch_state
FOR EACH ROW EXECUTE FUNCTION execution_gateway_policy_delete_guard();

CREATE TRIGGER execution_mainnet_approval_no_delete
BEFORE DELETE ON execution_mainnet_approvals
FOR EACH ROW EXECUTE FUNCTION execution_gateway_policy_delete_guard();

CREATE FUNCTION execution_gateway_policy_update_guard()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    expected_actor UUID;
BEGIN
    expected_actor := COALESCE(
        NULLIF(to_jsonb(NEW) ->> 'approved_by_user_id', '')::UUID,
        NULLIF(to_jsonb(NEW) ->> 'updated_by_user_id', '')::UUID
    );
    IF OLD.audit_event_id IS NOT DISTINCT FROM NEW.audit_event_id
       OR NEW.updated_at <= OLD.updated_at THEN
        RAISE EXCEPTION 'execution gateway policy update requires a new audit event';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM execution_gateway_audit_events AS audit
        WHERE audit.event_id = NEW.audit_event_id
          AND audit.decision = 'accepted'
          AND audit.actor_user_id = expected_actor
          AND audit.created_at = NEW.updated_at
    ) THEN
        RAISE EXCEPTION 'execution gateway policy update audit binding is invalid';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER execution_provider_allowlist_audited_update
BEFORE UPDATE ON execution_provider_allowlist
FOR EACH ROW EXECUTE FUNCTION execution_gateway_policy_update_guard();

CREATE TRIGGER execution_account_safety_audited_update
BEFORE UPDATE ON execution_account_safety_state
FOR EACH ROW EXECUTE FUNCTION execution_gateway_policy_update_guard();

CREATE TRIGGER execution_kill_switch_audited_update
BEFORE UPDATE ON execution_kill_switch_state
FOR EACH ROW EXECUTE FUNCTION execution_gateway_policy_update_guard();

CREATE FUNCTION execution_mainnet_approval_update_guard()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.approval_id IS DISTINCT FROM NEW.approval_id
       OR OLD.organization_id IS DISTINCT FROM NEW.organization_id
       OR OLD.owner_user_id IS DISTINCT FROM NEW.owner_user_id
       OR OLD.exchange_connection_id IS DISTINCT FROM NEW.exchange_connection_id
       OR OLD.exchange_name IS DISTINCT FROM NEW.exchange_name
       OR OLD.market_type IS DISTINCT FROM NEW.market_type
       OR OLD.provider_id IS DISTINCT FROM NEW.provider_id
       OR OLD.risk_revision_hash IS DISTINCT FROM NEW.risk_revision_hash
       OR OLD.account_revision_hash IS DISTINCT FROM NEW.account_revision_hash
       OR OLD.provider_revision_hash IS DISTINCT FROM NEW.provider_revision_hash
       OR OLD.recent_auth_session_id IS DISTINCT FROM NEW.recent_auth_session_id
       OR OLD.recent_auth_at IS DISTINCT FROM NEW.recent_auth_at
       OR OLD.approved_at IS DISTINCT FROM NEW.approved_at
       OR OLD.expires_at IS DISTINCT FROM NEW.expires_at
       OR OLD.audit_event_id IS DISTINCT FROM NEW.audit_event_id
       OR OLD.revoked_at IS NOT NULL THEN
        RAISE EXCEPTION 'mainnet approval is immutable except first revocation';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM execution_gateway_audit_events AS audit
        WHERE audit.event_id = NEW.revocation_audit_event_id
          AND audit.organization_id = NEW.organization_id
          AND audit.approval_id = NEW.approval_id
          AND audit.decision = 'accepted'
          AND audit.event_type = 'mainnet_approval_revoked'
          AND audit.created_at = NEW.revoked_at
    ) THEN
        RAISE EXCEPTION 'mainnet approval revocation audit binding is invalid';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER execution_mainnet_approval_update_only_revoke
BEFORE UPDATE ON execution_mainnet_approvals
FOR EACH ROW EXECUTE FUNCTION execution_mainnet_approval_update_guard();

COMMIT;
