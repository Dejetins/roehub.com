BEGIN;

-- Stage 05 is intentionally greenfield-only. Existing product resources are not
-- inferred, imported or backfilled into organizations.
DO $$
DECLARE
    populated_table TEXT;
    has_rows BOOLEAN;
BEGIN
    FOREACH populated_table IN ARRAY ARRAY[
        'exchange_connections',
        'strategy_strategies',
        'backtest_jobs',
        'strategy_backtest_variant_provenance',
        'strategy_position_ownership',
        'exchange_account_snapshots',
        'exchange_position_snapshots',
        'strategy_exchange_bindings'
    ]
    LOOP
        IF to_regclass('public.' || populated_table) IS NULL THEN
            RAISE EXCEPTION 'required greenfield table is missing: %', populated_table;
        END IF;
        EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I)', populated_table)
            INTO STRICT has_rows;
        IF has_rows THEN
            RAISE EXCEPTION
                'organization schema requires empty greenfield table: %',
                populated_table;
        END IF;
    END LOOP;
END
$$;

CREATE TABLE identity_installations (
    installation_id UUID PRIMARY KEY,
    singleton_key BOOLEAN NOT NULL DEFAULT TRUE UNIQUE,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_installations_singleton_chk CHECK (singleton_key),
    CONSTRAINT identity_installations_display_name_chk
        CHECK (char_length(trim(display_name)) BETWEEN 2 AND 120)
);

CREATE TABLE identity_installation_owners (
    installation_id UUID NOT NULL REFERENCES identity_installations(installation_id)
        ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    granted_by_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    granted_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (installation_id, user_id)
);

CREATE TABLE identity_organizations (
    organization_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL REFERENCES identity_installations(installation_id)
        ON DELETE CASCADE,
    slug TEXT NOT NULL,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL,
    archived_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_organizations_slug_chk
        CHECK (slug ~ '^[a-z][a-z0-9-]{1,62}[a-z0-9]$'),
    CONSTRAINT identity_organizations_display_name_chk
        CHECK (char_length(trim(display_name)) BETWEEN 2 AND 120),
    CONSTRAINT identity_organizations_status_chk CHECK (status IN ('active', 'archived')),
    CONSTRAINT identity_organizations_archive_state_chk CHECK (
        (status = 'active' AND archived_at IS NULL)
        OR (status = 'archived' AND archived_at IS NOT NULL)
    ),
    UNIQUE (installation_id, slug),
    UNIQUE (installation_id, organization_id)
);

CREATE TABLE identity_memberships (
    organization_id UUID NOT NULL REFERENCES identity_organizations(organization_id)
        ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    role TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, user_id),
    CONSTRAINT identity_memberships_role_chk
        CHECK (role IN ('owner', 'admin', 'operator', 'trader', 'viewer')),
    CONSTRAINT identity_memberships_status_chk CHECK (status IN ('active', 'suspended'))
);

CREATE INDEX idx_identity_memberships_user_active
    ON identity_memberships(user_id, organization_id)
    WHERE status = 'active';

CREATE TABLE identity_invitations (
    invitation_id UUID PRIMARY KEY,
    organization_id UUID NOT NULL REFERENCES identity_organizations(organization_id)
        ON DELETE CASCADE,
    recipient_email_sha256 TEXT NOT NULL,
    role TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_by_user_id UUID NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    accepted_by_user_id UUID NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    accepted_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_invitations_email_hash_chk
        CHECK (recipient_email_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT identity_invitations_role_chk
        CHECK (role IN ('owner', 'admin', 'operator', 'trader', 'viewer')),
    CONSTRAINT identity_invitations_status_chk
        CHECK (status IN ('pending', 'accepted', 'revoked', 'expired')),
    CONSTRAINT identity_invitations_expiry_chk CHECK (expires_at > created_at),
    CONSTRAINT identity_invitations_acceptance_chk CHECK (
        (status = 'accepted' AND accepted_by_user_id IS NOT NULL AND accepted_at IS NOT NULL)
        OR (status <> 'accepted' AND accepted_by_user_id IS NULL AND accepted_at IS NULL)
    ),
    FOREIGN KEY (organization_id, created_by_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX idx_identity_invitations_pending_recipient
    ON identity_invitations(organization_id, recipient_email_sha256)
    WHERE status = 'pending';

CREATE TABLE identity_plugin_permissions (
    organization_id UUID NOT NULL,
    plugin_id TEXT NOT NULL,
    user_id UUID NOT NULL,
    permission TEXT NOT NULL,
    granted_by_user_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, plugin_id, user_id),
    CONSTRAINT identity_plugin_permissions_plugin_id_chk
        CHECK (plugin_id ~ '^[a-z][a-z0-9._-]{2,127}$'),
    CONSTRAINT identity_plugin_permissions_permission_chk
        CHECK (permission IN ('read', 'configure', 'operate')),
    FOREIGN KEY (organization_id, user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id, granted_by_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT
);

CREATE TABLE identity_support_access_grants (
    grant_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL REFERENCES identity_installations(installation_id)
        ON DELETE CASCADE,
    support_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    granted_by_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    reason TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NULL,
    CONSTRAINT identity_support_access_reason_chk
        CHECK (char_length(trim(reason)) BETWEEN 8 AND 240),
    CONSTRAINT identity_support_access_expiry_chk
        CHECK (expires_at > created_at AND expires_at <= created_at + INTERVAL '24 hours'),
    CONSTRAINT identity_support_access_revoke_chk
        CHECK (revoked_at IS NULL OR revoked_at >= created_at),
    FOREIGN KEY (installation_id, granted_by_user_id)
        REFERENCES identity_installation_owners(installation_id, user_id) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX idx_identity_support_access_one_active
    ON identity_support_access_grants(installation_id, support_user_id)
    WHERE revoked_at IS NULL;

CREATE TABLE identity_administrative_audit_events (
    event_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL REFERENCES identity_installations(installation_id)
        ON DELETE RESTRICT,
    organization_id UUID NULL REFERENCES identity_organizations(organization_id)
        ON DELETE RESTRICT,
    actor_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT identity_admin_audit_action_chk CHECK (char_length(trim(action)) > 0),
    CONSTRAINT identity_admin_audit_target_chk
        CHECK (char_length(trim(target_type)) > 0 AND char_length(trim(target_id)) > 0),
    CONSTRAINT identity_admin_audit_outcome_chk
        CHECK (outcome IN ('succeeded', 'rejected')),
    CONSTRAINT identity_admin_audit_metadata_shape_chk
        CHECK (jsonb_typeof(metadata_json) = 'object'),
    CONSTRAINT identity_admin_audit_no_sensitive_keys_chk CHECK (
        metadata_json::TEXT !~* '"(password|token|secret|credential|cookie|authorization|dsn|api[_-]?key|private[_-]?key)"[[:space:]]*:'
    )
);

CREATE INDEX idx_identity_admin_audit_org_created
    ON identity_administrative_audit_events(organization_id, created_at DESC, event_id DESC);

CREATE OR REPLACE FUNCTION identity_reject_last_owner()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.role = 'owner' AND OLD.status = 'active'
       AND (TG_OP = 'DELETE' OR NEW.role <> 'owner' OR NEW.status <> 'active')
    THEN
        PERFORM pg_advisory_xact_lock(hashtextextended(OLD.organization_id::TEXT, 0));
        IF (
            SELECT count(*)
            FROM identity_memberships
            WHERE organization_id = OLD.organization_id
              AND role = 'owner'
              AND status = 'active'
        ) <= 1
        THEN
            RAISE EXCEPTION 'last organization owner cannot be removed or demoted'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END
$$;

CREATE TRIGGER identity_memberships_last_owner_guard
BEFORE UPDATE OR DELETE ON identity_memberships
FOR EACH ROW EXECUTE FUNCTION identity_reject_last_owner();

CREATE OR REPLACE FUNCTION identity_reject_last_installation_owner()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(OLD.installation_id::TEXT, 0));
    IF (
        SELECT count(*)
        FROM identity_installation_owners
        WHERE installation_id = OLD.installation_id
    ) <= 1
    THEN
        RAISE EXCEPTION 'last installation owner cannot be removed'
            USING ERRCODE = '23514';
    END IF;
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END
$$;

CREATE TRIGGER identity_installation_owners_last_owner_guard
BEFORE UPDATE OR DELETE ON identity_installation_owners
FOR EACH ROW EXECUTE FUNCTION identity_reject_last_installation_owner();

CREATE OR REPLACE FUNCTION identity_reject_admin_audit_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'administrative audit events are immutable'
        USING ERRCODE = '55000';
END
$$;

CREATE TRIGGER identity_admin_audit_immutable
BEFORE UPDATE OR DELETE ON identity_administrative_audit_events
FOR EACH ROW EXECUTE FUNCTION identity_reject_admin_audit_mutation();

-- Existing product tables become organization-scoped immediately for the
-- greenfield v1 lifecycle. Application adapters are migrated in Stages 09/10.
ALTER TABLE exchange_connections ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE strategy_strategies ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE backtest_jobs ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE strategy_backtest_variant_provenance ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE strategy_position_ownership ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE exchange_account_snapshots ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE exchange_position_snapshots ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE strategy_exchange_bindings ADD COLUMN organization_id UUID NOT NULL;

ALTER TABLE exchange_connections
    ADD CONSTRAINT exchange_connections_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT exchange_connections_org_id_unique
        UNIQUE (organization_id, connection_id);

ALTER TABLE strategy_strategies
    ADD CONSTRAINT strategy_strategies_org_member_fk
        FOREIGN KEY (organization_id, user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_strategies_org_id_unique
        UNIQUE (organization_id, strategy_id);

ALTER TABLE backtest_jobs
    ADD CONSTRAINT backtest_jobs_org_member_fk
        FOREIGN KEY (organization_id, user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT backtest_jobs_org_id_unique
        UNIQUE (organization_id, job_id);

ALTER TABLE strategy_backtest_variant_provenance
    ADD CONSTRAINT strategy_provenance_org_member_fk
        FOREIGN KEY (organization_id, user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_provenance_org_strategy_fk
        FOREIGN KEY (organization_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, strategy_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_provenance_org_job_fk
        FOREIGN KEY (organization_id, source_job_id)
        REFERENCES backtest_jobs(organization_id, job_id) ON DELETE RESTRICT;

ALTER TABLE strategy_position_ownership
    ADD CONSTRAINT strategy_position_ownership_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_position_ownership_org_strategy_fk
        FOREIGN KEY (organization_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, strategy_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_position_ownership_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE RESTRICT;

ALTER TABLE exchange_account_snapshots
    ADD CONSTRAINT exchange_account_snapshots_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT exchange_account_snapshots_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE RESTRICT,
    ADD CONSTRAINT exchange_account_snapshots_org_id_unique
        UNIQUE (organization_id, account_snapshot_id);

ALTER TABLE exchange_position_snapshots
    ADD CONSTRAINT exchange_position_snapshots_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT exchange_position_snapshots_org_account_fk
        FOREIGN KEY (organization_id, account_snapshot_id)
        REFERENCES exchange_account_snapshots(organization_id, account_snapshot_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT exchange_position_snapshots_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE RESTRICT;

ALTER TABLE strategy_exchange_bindings
    ADD CONSTRAINT strategy_exchange_bindings_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT strategy_exchange_bindings_org_strategy_fk
        FOREIGN KEY (organization_id, strategy_id)
        REFERENCES strategy_strategies(organization_id, strategy_id) ON DELETE CASCADE,
    ADD CONSTRAINT strategy_exchange_bindings_org_connection_fk
        FOREIGN KEY (organization_id, exchange_connection_id)
        REFERENCES exchange_connections(organization_id, connection_id) ON DELETE RESTRICT;

COMMIT;
