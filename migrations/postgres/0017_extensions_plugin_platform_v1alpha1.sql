BEGIN;

DO $$
BEGIN
    IF to_regclass('public.identity_installations') IS NULL
       OR to_regclass('public.identity_organizations') IS NULL
       OR to_regclass('public.identity_users') IS NULL THEN
        RAISE EXCEPTION 'extensions plugin platform requires Stage 05 identity schema';
    END IF;
END
$$;

CREATE TABLE extensions_publisher_keys (
    installation_id UUID NOT NULL
        REFERENCES identity_installations(installation_id) ON DELETE CASCADE,
    key_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    public_key_b64 TEXT NOT NULL,
    fingerprint_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'trusted',
    added_by_user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NULL,
    PRIMARY KEY (installation_id, key_id),
    CONSTRAINT extensions_publisher_keys_id_chk
        CHECK (key_id ~ '^[a-z][a-z0-9._-]{2,127}$'),
    CONSTRAINT extensions_publisher_keys_algorithm_chk CHECK (algorithm = 'Ed25519'),
    CONSTRAINT extensions_publisher_keys_public_chk
        CHECK (public_key_b64 ~ '^[A-Za-z0-9+/]{42}[AEIMQUYcgkosw048]=$'),
    CONSTRAINT extensions_publisher_keys_fingerprint_chk
        CHECK (fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT extensions_publisher_keys_status_chk
        CHECK (status IN ('trusted', 'revoked')),
    CONSTRAINT extensions_publisher_keys_revocation_chk CHECK (
        (status = 'trusted' AND revoked_at IS NULL)
        OR (status = 'revoked' AND revoked_at IS NOT NULL)
    )
);

CREATE TABLE extensions_plugin_packages (
    package_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL
        REFERENCES identity_installations(installation_id) ON DELETE CASCADE,
    plugin_id TEXT NOT NULL,
    version TEXT NOT NULL,
    package_digest TEXT NOT NULL,
    image_reference TEXT NOT NULL,
    image_digest TEXT NOT NULL,
    publisher_key_id TEXT NULL,
    manifest JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT extensions_plugin_packages_publisher_key_fk
        FOREIGN KEY (installation_id, publisher_key_id)
        REFERENCES extensions_publisher_keys(installation_id, key_id)
        ON DELETE RESTRICT,
    CONSTRAINT extensions_plugin_packages_id_chk
        CHECK (plugin_id ~ '^[a-z][a-z0-9._-]{2,127}$'),
    CONSTRAINT extensions_plugin_packages_version_chk
        CHECK (version ~ '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'),
    CONSTRAINT extensions_plugin_packages_digest_chk
        CHECK (package_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT extensions_plugin_packages_image_digest_chk
        CHECK (image_digest ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT extensions_plugin_packages_reference_chk
        CHECK (char_length(image_reference) BETWEEN 3 AND 512 AND image_reference !~ '[[:space:]]'),
    CONSTRAINT extensions_plugin_packages_manifest_chk CHECK (
        jsonb_typeof(manifest) = 'object'
        AND manifest ->> 'apiVersion' = 'roehub.io/v1alpha1'
        AND manifest ->> 'kind' = 'Plugin'
        AND manifest #>> '{spec,pluginApi}' = 'v1alpha1'
        AND manifest #>> '{spec,rpc,version}' = 'roehub.plugin.rpc/v1alpha1'
        AND manifest #>> '{metadata,id}' = plugin_id
        AND manifest #>> '{metadata,version}' = version
        AND manifest #>> '{spec,image,digest}' = image_digest
    ),
    CONSTRAINT extensions_plugin_packages_signature_mode_chk CHECK (
        (publisher_key_id IS NOT NULL AND manifest ? 'signature')
        OR (
            publisher_key_id IS NULL
            AND manifest #>> '{metadata,developmentMode}' = 'true'
        )
    ),
    UNIQUE (installation_id, plugin_id, version),
    UNIQUE (installation_id, plugin_id, version, package_digest),
    UNIQUE (package_id, installation_id, plugin_id)
);

CREATE OR REPLACE FUNCTION extensions_text_array_is_unique(values_to_check TEXT[])
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
STRICT
AS $$
    SELECT cardinality(values_to_check) = count(DISTINCT value)
    FROM unnest(values_to_check) AS value
$$;

CREATE TABLE extensions_plugin_installations (
    plugin_installation_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL,
    organization_id UUID NOT NULL,
    plugin_id TEXT NOT NULL,
    package_id UUID NOT NULL,
    previous_package_id UUID NULL,
    granted_permissions TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    status TEXT NOT NULL DEFAULT 'enabled',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT extensions_plugin_installations_org_fk
        FOREIGN KEY (installation_id, organization_id)
        REFERENCES identity_organizations(installation_id, organization_id)
        ON DELETE CASCADE,
    CONSTRAINT extensions_plugin_installations_package_fk
        FOREIGN KEY (package_id, installation_id, plugin_id)
        REFERENCES extensions_plugin_packages(package_id, installation_id, plugin_id)
        ON DELETE RESTRICT,
    CONSTRAINT extensions_plugin_installations_previous_package_fk
        FOREIGN KEY (previous_package_id, installation_id, plugin_id)
        REFERENCES extensions_plugin_packages(package_id, installation_id, plugin_id)
        ON DELETE RESTRICT,
    CONSTRAINT extensions_plugin_installations_permissions_chk CHECK (
        granted_permissions <@ ARRAY[
            'app.action', 'data.read', 'notification.send', 'panel.describe'
        ]::TEXT[]
        AND extensions_text_array_is_unique(granted_permissions)
    ),
    CONSTRAINT extensions_plugin_installations_status_chk
        CHECK (status IN ('enabled', 'disabled', 'degraded')),
    CONSTRAINT extensions_plugin_installations_time_chk CHECK (updated_at >= created_at),
    UNIQUE (organization_id, plugin_id),
    UNIQUE (plugin_installation_id, installation_id, organization_id)
);

CREATE TABLE extensions_plugin_instances (
    instance_id UUID PRIMARY KEY,
    plugin_installation_id UUID NOT NULL,
    installation_id UUID NOT NULL,
    organization_id UUID NOT NULL,
    name TEXT NOT NULL,
    config JSONB NOT NULL DEFAULT '{}'::JSONB,
    config_revision INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'enabled',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT extensions_plugin_instances_installation_fk
        FOREIGN KEY (plugin_installation_id, installation_id, organization_id)
        REFERENCES extensions_plugin_installations(
            plugin_installation_id, installation_id, organization_id
        ) ON DELETE CASCADE,
    CONSTRAINT extensions_plugin_instances_name_chk
        CHECK (char_length(trim(name)) BETWEEN 1 AND 120),
    CONSTRAINT extensions_plugin_instances_config_chk CHECK (
        jsonb_typeof(config) = 'object'
        AND config::TEXT !~* '"(password|token|secret|credential|cookie|authorization|api[_-]?key)"[[:space:]]*:'
    ),
    CONSTRAINT extensions_plugin_instances_revision_chk CHECK (config_revision > 0),
    CONSTRAINT extensions_plugin_instances_status_chk
        CHECK (status IN ('enabled', 'disabled', 'degraded')),
    CONSTRAINT extensions_plugin_instances_time_chk CHECK (updated_at >= created_at),
    UNIQUE (organization_id, plugin_installation_id, name),
    UNIQUE (instance_id, installation_id, organization_id)
);

CREATE TABLE extensions_plugin_operations (
    operation_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL,
    organization_id UUID NOT NULL,
    actor_user_id UUID NOT NULL,
    kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    request_payload JSONB NOT NULL,
    status TEXT NOT NULL,
    result JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT extensions_plugin_operations_org_fk
        FOREIGN KEY (installation_id, organization_id)
        REFERENCES identity_organizations(installation_id, organization_id)
        ON DELETE CASCADE,
    CONSTRAINT extensions_plugin_operations_actor_fk
        FOREIGN KEY (organization_id, actor_user_id)
        REFERENCES identity_memberships(organization_id, user_id)
        ON DELETE RESTRICT,
    CONSTRAINT extensions_plugin_operations_kind_chk
        CHECK (kind IN ('install', 'update', 'rollback', 'configure', 'enable', 'disable', 'health')),
    CONSTRAINT extensions_plugin_operations_target_chk
        CHECK (char_length(target_id) BETWEEN 1 AND 160),
    CONSTRAINT extensions_plugin_operations_idempotency_chk
        CHECK (idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$'),
    CONSTRAINT extensions_plugin_operations_hash_chk CHECK (request_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT extensions_plugin_operations_request_chk CHECK (
        jsonb_typeof(request_payload) = 'object'
        AND request_payload ? 'contract'
        AND (
            NOT request_payload ? 'config'
            OR (request_payload -> 'config')::TEXT
                !~* '"(password|token|secret|credential|cookie|authorization|api[_-]?key)"[[:space:]]*:'
        )
    ),
    CONSTRAINT extensions_plugin_operations_status_chk
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'unknown')),
    CONSTRAINT extensions_plugin_operations_result_chk CHECK (jsonb_typeof(result) = 'object'),
    CONSTRAINT extensions_plugin_operations_time_chk CHECK (updated_at >= created_at),
    UNIQUE (organization_id, idempotency_key)
);

CREATE TABLE extensions_plugin_events (
    event_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL,
    organization_id UUID NOT NULL,
    actor_user_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT extensions_plugin_events_org_fk
        FOREIGN KEY (installation_id, organization_id)
        REFERENCES identity_organizations(installation_id, organization_id)
        ON DELETE CASCADE,
    CONSTRAINT extensions_plugin_events_actor_fk
        FOREIGN KEY (organization_id, actor_user_id)
        REFERENCES identity_memberships(organization_id, user_id)
        ON DELETE RESTRICT,
    CONSTRAINT extensions_plugin_events_type_chk
        CHECK (event_type ~ '^plugin\.[a-z0-9_.-]{2,120}$'),
    CONSTRAINT extensions_plugin_events_target_chk
        CHECK (char_length(target_type) BETWEEN 2 AND 80 AND char_length(target_id) BETWEEN 1 AND 160),
    CONSTRAINT extensions_plugin_events_outcome_chk
        CHECK (outcome IN ('succeeded', 'rejected')),
    CONSTRAINT extensions_plugin_events_metadata_chk CHECK (
        jsonb_typeof(metadata) = 'object'
        AND metadata::TEXT !~* '"(password|token|secret|credential|cookie|authorization|api[_-]?key)"[[:space:]]*:'
    )
);

CREATE OR REPLACE FUNCTION extensions_reject_event_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'extensions plugin events are immutable';
END
$$;

CREATE TRIGGER extensions_plugin_events_immutable_update
BEFORE UPDATE ON extensions_plugin_events
FOR EACH ROW EXECUTE FUNCTION extensions_reject_event_mutation();

CREATE TRIGGER extensions_plugin_events_immutable_delete
BEFORE DELETE ON extensions_plugin_events
FOR EACH ROW EXECUTE FUNCTION extensions_reject_event_mutation();

CREATE INDEX idx_extensions_plugin_operations_status
    ON extensions_plugin_operations(status, created_at);
CREATE INDEX idx_extensions_plugin_events_org_created
    ON extensions_plugin_events(organization_id, created_at DESC);

COMMIT;
