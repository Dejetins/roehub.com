BEGIN;

-- Stage 11 is greenfield-only. Existing notification rows, bindings, provider
-- credentials, recipients, cursors, and delivery state are never inferred.
DO $$
DECLARE
    populated_table TEXT;
    has_rows BOOLEAN;
BEGIN
    FOREACH populated_table IN ARRAY ARRAY[
        'notification_events',
        'notification_routes',
        'notification_report_runs',
        'notification_deliveries',
        'notification_delivery_attempts',
        'notification_telegram_updates'
    ]
    LOOP
        IF to_regclass('public.' || populated_table) IS NULL THEN
            RAISE EXCEPTION 'required greenfield notification table is missing: %',
                populated_table;
        END IF;
        EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I)', populated_table)
            INTO STRICT has_rows;
        IF has_rows THEN
            RAISE EXCEPTION
                'notification provider schema requires empty greenfield table: %',
                populated_table;
        END IF;
    END LOOP;
END
$$;

CREATE TABLE notification_provider_packages (
    package_id UUID PRIMARY KEY,
    provider_key TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    package_version TEXT NOT NULL,
    display_name TEXT NOT NULL,
    config_schema_json JSONB NOT NULL,
    channels TEXT[] NOT NULL,
    templates TEXT[] NOT NULL,
    error_codes TEXT[] NOT NULL,
    built_in BOOLEAN NOT NULL DEFAULT FALSE,
    installed_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT notification_provider_packages_key_chk
        CHECK (provider_key ~ '^[a-z][a-z0-9._-]{2,127}$'),
    CONSTRAINT notification_provider_packages_contract_chk
        CHECK (contract_version = 'NotificationProvider/v1'),
    CONSTRAINT notification_provider_packages_version_chk
        CHECK (package_version ~ '^[0-9]+\.[0-9]+\.[0-9]+([-+][a-zA-Z0-9.-]+)?$'),
    CONSTRAINT notification_provider_packages_display_name_chk
        CHECK (char_length(trim(display_name)) BETWEEN 2 AND 120),
    CONSTRAINT notification_provider_packages_config_shape_chk
        CHECK (jsonb_typeof(config_schema_json) = 'object'),
    CONSTRAINT notification_provider_packages_capabilities_chk
        CHECK (cardinality(channels) > 0 AND cardinality(templates) > 0),
    UNIQUE (provider_key, package_version),
    UNIQUE (package_id, provider_key)
);

CREATE TABLE notification_provider_instances (
    instance_id UUID PRIMARY KEY,
    package_id UUID NOT NULL,
    provider_key TEXT NOT NULL,
    scope TEXT NOT NULL,
    organization_id UUID NULL REFERENCES identity_organizations(organization_id)
        ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    secret_ref TEXT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    health_status TEXT NULL,
    health_error_code TEXT NULL,
    health_checked_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT notification_provider_instances_package_fk
        FOREIGN KEY (package_id, provider_key)
        REFERENCES notification_provider_packages(package_id, provider_key)
        ON DELETE RESTRICT,
    CONSTRAINT notification_provider_instances_scope_chk CHECK (
        (scope = 'installation' AND organization_id IS NULL)
        OR (scope = 'organization' AND organization_id IS NOT NULL)
    ),
    CONSTRAINT notification_provider_instances_status_chk
        CHECK (status IN ('active', 'disabled', 'degraded')),
    CONSTRAINT notification_provider_instances_health_chk
        CHECK (health_status IS NULL OR health_status IN ('ready', 'degraded', 'disabled')),
    CONSTRAINT notification_provider_instances_display_name_chk
        CHECK (char_length(trim(display_name)) BETWEEN 2 AND 120),
    CONSTRAINT notification_provider_instances_config_shape_chk
        CHECK (jsonb_typeof(config_json) = 'object'),
    CONSTRAINT notification_provider_instances_no_raw_secrets_chk CHECK (
        config_json::TEXT !~* '"(password|token|secret|credential|cookie|authorization|api[_-]?key|chat[_-]?id)"[[:space:]]*:'
    ),
    CONSTRAINT notification_provider_instances_secret_ref_chk CHECK (
        (
            provider_key = 'telegram_bot_api'
            AND secret_ref = format(
                'openbao://kv/roehub/telegram/providers/%s/%s#bot_token',
                COALESCE(organization_id::TEXT, 'installation'),
                instance_id::TEXT
            )
        )
        OR (
            provider_key <> 'telegram_bot_api'
            AND secret_ref IS NULL
        )
        OR (
            provider_key <> 'telegram_bot_api'
            AND
            secret_ref ~ '^openbao://[a-zA-Z0-9][a-zA-Z0-9._-]*/'
            AND secret_ref !~ '[[:space:]%]'
            AND secret_ref ~ '#[a-zA-Z0-9][a-zA-Z0-9._-]*$'
            AND split_part(secret_ref, '#', 1) = format(
                'openbao://kv/roehub/plugins/%s/%s',
                COALESCE(organization_id::TEXT, 'installation'),
                instance_id::TEXT
            )
        )
    ),
    UNIQUE (instance_id, provider_key)
);

CREATE UNIQUE INDEX idx_notification_provider_instances_installation_name
    ON notification_provider_instances(provider_key, display_name)
    WHERE organization_id IS NULL;
CREATE UNIQUE INDEX idx_notification_provider_instances_org_name
    ON notification_provider_instances(organization_id, provider_key, display_name)
    WHERE organization_id IS NOT NULL;

INSERT INTO notification_provider_packages
  (package_id, provider_key, contract_version, package_version, display_name,
   config_schema_json, channels, templates, error_codes, built_in, installed_at)
VALUES
  ('00000000-0000-4000-8000-000000000101', 'log_only', 'NotificationProvider/v1',
   '1.0.0', 'Log only', '{"type":"object","additionalProperties":false}'::jsonb,
   ARRAY['telegram','email','webhook','push','in_app'], ARRAY['plain_text.v1'],
   ARRAY['provider_disabled'], TRUE, '2026-07-13T00:00:00Z'),
  ('00000000-0000-4000-8000-000000000102', 'fake', 'NotificationProvider/v1',
   '1.0.0', 'Controlled fake', '{"type":"object","additionalProperties":false}'::jsonb,
   ARRAY['telegram','email','webhook','push','in_app'], ARRAY['plain_text.v1'],
   ARRAY['provider_disabled'], TRUE, '2026-07-13T00:00:00Z'),
  ('00000000-0000-4000-8000-000000000103', 'telegram_bot_api',
   'NotificationProvider/v1', '1.0.0', 'Telegram Bot',
   '{
      "type":"object",
      "additionalProperties":false,
      "properties":{
        "api_base_url":{"type":"string","format":"uri"},
        "connect_timeout_seconds":{"type":"number","minimum":0.1,"maximum":3},
        "overall_timeout_seconds":{"type":"number","minimum":1,"maximum":10}
      }
    }'::jsonb,
   ARRAY['telegram'], ARRAY['plain_text.v1','telegram_command_response.v1'],
   ARRAY[
      'provider_disabled','provider_scope_mismatch','provider_secret_unavailable',
      'provider_connect_timeout','provider_transport_error',
      'provider_timeout_after_acceptance_possible','provider_rate_limited',
      'provider_http_error','provider_response_invalid','provider_cancelled',
      'provider_shutdown'
   ], TRUE, '2026-07-13T00:00:00Z');

INSERT INTO notification_provider_instances
  (instance_id, package_id, provider_key, scope, organization_id, display_name,
   config_json, secret_ref, status, health_status, health_checked_at, created_at, updated_at)
VALUES
  ('00000000-0000-4000-8000-000000000001',
   '00000000-0000-4000-8000-000000000101', 'log_only', 'installation', NULL,
   'Built-in log only', '{}'::jsonb, NULL, 'active', 'ready',
   '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z'),
  ('00000000-0000-4000-8000-000000000002',
   '00000000-0000-4000-8000-000000000102', 'fake', 'installation', NULL,
   'Built-in controlled fake', '{}'::jsonb, NULL, 'active', 'ready',
   '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z');

CREATE TABLE notification_telegram_update_cursors (
    provider_instance_id UUID PRIMARY KEY
        REFERENCES notification_provider_instances(instance_id) ON DELETE CASCADE,
    organization_id UUID NULL REFERENCES identity_organizations(organization_id)
        ON DELETE CASCADE,
    last_update_id BIGINT NOT NULL DEFAULT -1,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT notification_telegram_cursor_value_chk CHECK (last_update_id >= -1)
);

CREATE TABLE notification_telegram_command_registry (
    provider_instance_id UUID NOT NULL
        REFERENCES notification_provider_instances(instance_id) ON DELETE CASCADE,
    command_name TEXT NOT NULL,
    description TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (provider_instance_id, command_name),
    CONSTRAINT notification_telegram_command_name_chk
        CHECK (command_name ~ '^[a-z][a-z0-9_]{1,31}$'),
    CONSTRAINT notification_telegram_command_description_chk
        CHECK (char_length(trim(description)) BETWEEN 2 AND 160)
);

CREATE OR REPLACE FUNCTION notification_initialize_telegram_instance()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.provider_key = 'telegram_bot_api' THEN
        INSERT INTO notification_telegram_update_cursors
          (provider_instance_id, organization_id, last_update_id, updated_at)
        VALUES (NEW.instance_id, NEW.organization_id, -1, NEW.created_at);
        INSERT INTO notification_telegram_command_registry
          (provider_instance_id, command_name, description, enabled, updated_at)
        VALUES
          (NEW.instance_id, 'start', 'Connect this Telegram recipient', TRUE, NEW.created_at),
          (NEW.instance_id, 'stats', 'Show portfolio statistics', TRUE, NEW.created_at),
          (NEW.instance_id, 'strategy', 'Show strategy statistics', TRUE, NEW.created_at),
          (NEW.instance_id, 'exchange', 'Show exchange statistics', TRUE, NEW.created_at),
          (NEW.instance_id, 'settings', 'Show notification settings', TRUE, NEW.created_at),
          (NEW.instance_id, 'critical_only', 'Enable critical notifications', TRUE, NEW.created_at),
          (NEW.instance_id, 'signals_on', 'Enable signal notifications', TRUE, NEW.created_at),
          (NEW.instance_id, 'signals_off', 'Disable signal notifications', TRUE, NEW.created_at),
          (NEW.instance_id, 'reports', 'Configure scheduled reports', TRUE, NEW.created_at);
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER notification_provider_instances_telegram_initialize
AFTER INSERT ON notification_provider_instances
FOR EACH ROW EXECUTE FUNCTION notification_initialize_telegram_instance();

CREATE TABLE notification_telegram_binding_codes (
    binding_code_id UUID PRIMARY KEY,
    organization_id UUID NOT NULL,
    provider_instance_id UUID NOT NULL
        REFERENCES notification_provider_instances(instance_id) ON DELETE CASCADE,
    owner_user_id UUID NOT NULL,
    code_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ NULL,
    CONSTRAINT notification_telegram_binding_code_hash_chk
        CHECK (code_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT notification_telegram_binding_code_expiry_chk
        CHECK (expires_at > created_at),
    CONSTRAINT notification_telegram_binding_code_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX idx_notification_telegram_binding_codes_active_hash
    ON notification_telegram_binding_codes(
        organization_id, provider_instance_id, code_hash
    ) WHERE consumed_at IS NULL;

CREATE TABLE notification_telegram_recipient_bindings (
    organization_id UUID NOT NULL,
    provider_instance_id UUID NOT NULL
        REFERENCES notification_provider_instances(instance_id) ON DELETE CASCADE,
    owner_user_id UUID NOT NULL,
    chat_id_ref TEXT NOT NULL,
    recipient_secret_ref TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'confirmed',
    confirmed_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, provider_instance_id, owner_user_id),
    CONSTRAINT notification_telegram_recipient_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    CONSTRAINT notification_telegram_recipient_status_chk
        CHECK (status IN ('confirmed', 'revoked', 'requires_rebind')),
    CONSTRAINT notification_telegram_recipient_ref_chk
        CHECK (
            char_length(trim(chat_id_ref)) BETWEEN 8 AND 180
            AND chat_id_ref !~ '^-?[0-9]+$'
        ),
    CONSTRAINT notification_telegram_recipient_secret_ref_chk
        CHECK (
            recipient_secret_ref ~ '^openbao://kv/roehub/telegram/'
            AND recipient_secret_ref !~ '[[:space:]%]'
            AND recipient_secret_ref ~ '#chat_id$'
        ),
    UNIQUE (organization_id, provider_instance_id, chat_id_ref)
);

CREATE OR REPLACE FUNCTION notification_enforce_provider_scope()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    instance_organization_id UUID;
    instance_provider_key TEXT;
BEGIN
    SELECT organization_id, provider_key
      INTO STRICT instance_organization_id, instance_provider_key
    FROM notification_provider_instances
    WHERE instance_id = NEW.provider_instance_id;
    IF instance_organization_id IS NOT NULL
       AND instance_organization_id <> NEW.organization_id
    THEN
        RAISE EXCEPTION 'notification provider instance belongs to another organization'
            USING ERRCODE = '23514';
    END IF;
    IF to_jsonb(NEW) ? 'provider_key'
       AND instance_provider_key <> (to_jsonb(NEW) ->> 'provider_key')
    THEN
        RAISE EXCEPTION 'notification provider key does not match provider instance'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END
$$;

ALTER TABLE notification_events
    ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE notification_events
    DROP CONSTRAINT notification_events_dedupe_unique,
    ADD CONSTRAINT notification_events_org_fk
        FOREIGN KEY (organization_id) REFERENCES identity_organizations(organization_id)
        ON DELETE CASCADE,
    ADD CONSTRAINT notification_events_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    ADD CONSTRAINT notification_events_org_id_unique UNIQUE (organization_id, event_id),
    ADD CONSTRAINT notification_events_org_dedupe_unique
        UNIQUE (organization_id, dedupe_key);

ALTER TABLE notification_routes
    ADD COLUMN organization_id UUID NOT NULL,
    ADD COLUMN provider_instance_id UUID NOT NULL;
ALTER TABLE notification_routes
    DROP CONSTRAINT notification_routes_unique_active_ref,
    DROP CONSTRAINT notification_routes_provider_key_chk,
    ADD CONSTRAINT notification_routes_provider_key_v1_chk
        CHECK (provider_key ~ '^[a-z][a-z0-9._-]{2,127}$'),
    ADD CONSTRAINT notification_routes_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    ADD CONSTRAINT notification_routes_provider_instance_fk
        FOREIGN KEY (provider_instance_id, provider_key)
        REFERENCES notification_provider_instances(instance_id, provider_key)
        ON DELETE RESTRICT,
    ADD CONSTRAINT notification_routes_org_id_unique UNIQUE (organization_id, route_id),
    ADD CONSTRAINT notification_routes_org_provider_unique
        UNIQUE (organization_id, route_id, provider_instance_id, provider_key),
    ADD CONSTRAINT notification_routes_org_active_ref_unique UNIQUE (
        organization_id, recipient_kind, owner_user_id, channel_key,
        provider_instance_id, recipient_address_ref
    );
CREATE TRIGGER notification_routes_provider_scope_guard
BEFORE INSERT OR UPDATE ON notification_routes
FOR EACH ROW EXECUTE FUNCTION notification_enforce_provider_scope();

ALTER TABLE notification_report_runs
    ADD COLUMN organization_id UUID NOT NULL;
ALTER TABLE notification_report_runs
    DROP CONSTRAINT notification_report_runs_dedupe_unique,
    ADD CONSTRAINT notification_report_runs_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    ADD CONSTRAINT notification_report_runs_org_id_unique
        UNIQUE (organization_id, report_run_id),
    ADD CONSTRAINT notification_report_runs_org_dedupe_unique
        UNIQUE (organization_id, dedupe_key);

ALTER TABLE notification_deliveries
    ADD COLUMN organization_id UUID NOT NULL,
    ADD COLUMN provider_instance_id UUID NOT NULL,
    ADD COLUMN replayed_from_delivery_id UUID NULL;
ALTER TABLE notification_deliveries
    DROP CONSTRAINT notification_deliveries_provider_key_chk,
    ADD CONSTRAINT notification_deliveries_provider_key_v1_chk
        CHECK (provider_key ~ '^[a-z][a-z0-9._-]{2,127}$'),
    ADD CONSTRAINT notification_deliveries_org_route_provider_fk
        FOREIGN KEY (organization_id, route_id, provider_instance_id, provider_key)
        REFERENCES notification_routes(
            organization_id, route_id, provider_instance_id, provider_key
        ) ON DELETE RESTRICT,
    ADD CONSTRAINT notification_deliveries_org_event_fk
        FOREIGN KEY (organization_id, event_id)
        REFERENCES notification_events(organization_id, event_id) ON DELETE RESTRICT,
    ADD CONSTRAINT notification_deliveries_org_report_fk
        FOREIGN KEY (organization_id, report_run_id)
        REFERENCES notification_report_runs(organization_id, report_run_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT notification_deliveries_org_id_unique
        UNIQUE (organization_id, delivery_id),
    ADD CONSTRAINT notification_deliveries_org_instance_id_unique
        UNIQUE (organization_id, provider_instance_id, delivery_id),
    ADD CONSTRAINT notification_deliveries_replay_source_fk
        FOREIGN KEY (
            organization_id, provider_instance_id, replayed_from_delivery_id
        ) REFERENCES notification_deliveries(
            organization_id, provider_instance_id, delivery_id
        ) ON DELETE RESTRICT;
CREATE INDEX idx_notification_deliveries_replay_source
    ON notification_deliveries(
        organization_id, provider_instance_id, replayed_from_delivery_id
    ) WHERE replayed_from_delivery_id IS NOT NULL;
CREATE TRIGGER notification_deliveries_provider_scope_guard
BEFORE INSERT OR UPDATE ON notification_deliveries
FOR EACH ROW EXECUTE FUNCTION notification_enforce_provider_scope();

ALTER TABLE notification_delivery_attempts
    ADD COLUMN organization_id UUID NOT NULL,
    ADD COLUMN provider_instance_id UUID NOT NULL;
ALTER TABLE notification_delivery_attempts
    ADD CONSTRAINT notification_delivery_attempts_delivery_fk
        FOREIGN KEY (organization_id, provider_instance_id, delivery_id)
        REFERENCES notification_deliveries(
            organization_id, provider_instance_id, delivery_id
        ) ON DELETE CASCADE;

ALTER TABLE notification_telegram_updates
    ADD COLUMN organization_id UUID NOT NULL,
    ADD COLUMN provider_instance_id UUID NOT NULL;
ALTER TABLE notification_telegram_updates
    DROP CONSTRAINT notification_telegram_updates_pkey,
    DROP CONSTRAINT notification_telegram_updates_idempotency_unique,
    ADD PRIMARY KEY (provider_instance_id, telegram_update_id),
    ADD CONSTRAINT notification_telegram_updates_org_instance_unique
        UNIQUE (organization_id, provider_instance_id, telegram_update_id),
    ADD CONSTRAINT notification_telegram_updates_org_idempotency_unique
        UNIQUE (organization_id, idempotency_key),
    ADD CONSTRAINT notification_telegram_updates_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE CASCADE,
    ADD CONSTRAINT notification_telegram_updates_provider_fk
        FOREIGN KEY (provider_instance_id)
        REFERENCES notification_provider_instances(instance_id) ON DELETE CASCADE;
CREATE TRIGGER notification_telegram_updates_provider_scope_guard
BEFORE INSERT OR UPDATE ON notification_telegram_updates
FOR EACH ROW EXECUTE FUNCTION notification_enforce_provider_scope();

CREATE TRIGGER notification_telegram_cursors_provider_scope_guard
BEFORE INSERT OR UPDATE ON notification_telegram_update_cursors
FOR EACH ROW EXECUTE FUNCTION notification_enforce_provider_scope();
CREATE TRIGGER notification_telegram_binding_codes_provider_scope_guard
BEFORE INSERT OR UPDATE ON notification_telegram_binding_codes
FOR EACH ROW EXECUTE FUNCTION notification_enforce_provider_scope();
CREATE TRIGGER notification_telegram_bindings_provider_scope_guard
BEFORE INSERT OR UPDATE ON notification_telegram_recipient_bindings
FOR EACH ROW EXECUTE FUNCTION notification_enforce_provider_scope();

COMMIT;
