BEGIN;

DO $$
BEGIN
    IF to_regclass('public.identity_organizations') IS NULL THEN
        RAISE EXCEPTION 'artifact store requires Stage 05 organization schema';
    END IF;
END
$$;

CREATE TABLE artifact_store_objects (
    digest TEXT PRIMARY KEY,
    size_bytes BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT artifact_store_objects_digest_chk CHECK (digest ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT artifact_store_objects_size_chk CHECK (size_bytes BETWEEN 0 AND 67108864)
);

CREATE TABLE artifact_store_object_locations (
    digest TEXT NOT NULL REFERENCES artifact_store_objects(digest) ON DELETE CASCADE,
    backend TEXT NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (digest, backend),
    CONSTRAINT artifact_store_object_locations_backend_chk
        CHECK (backend IN ('local_cas', 's3_compatible'))
);

CREATE TABLE artifact_store_org_blobs (
    organization_id UUID NOT NULL
        REFERENCES identity_organizations(organization_id) ON DELETE CASCADE,
    digest TEXT NOT NULL,
    backend TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, digest),
    FOREIGN KEY (digest, backend)
        REFERENCES artifact_store_object_locations(digest, backend) ON DELETE RESTRICT
);

CREATE TABLE artifact_store_quotas (
    organization_id UUID PRIMARY KEY
        REFERENCES identity_organizations(organization_id) ON DELETE CASCADE,
    max_bytes BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT artifact_store_quotas_max_chk CHECK (max_bytes BETWEEN 1 AND 1099511627776)
);

CREATE TABLE artifact_store_manifests (
    organization_id UUID NOT NULL
        REFERENCES identity_organizations(organization_id) ON DELETE CASCADE,
    manifest_digest TEXT NOT NULL,
    bundle_id TEXT NOT NULL,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    manifest JSONB NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, manifest_digest),
    CONSTRAINT artifact_store_manifests_digest_chk
        CHECK (manifest_digest ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT artifact_store_manifests_bundle_chk
        CHECK (bundle_id ~ '^[a-z][a-z0-9._-]{2,127}$'),
    CONSTRAINT artifact_store_manifests_version_chk
        CHECK (version ~ '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'),
    CONSTRAINT artifact_store_manifests_payload_chk CHECK (
        jsonb_typeof(manifest) = 'object'
        AND manifest ->> 'schema' = 'ArtifactManifest/v1'
        AND manifest ->> 'bundle_id' = bundle_id
        AND manifest ->> 'version' = version
        AND jsonb_typeof(manifest -> 'entries') = 'array'
        AND jsonb_array_length(manifest -> 'entries') BETWEEN 1 AND 256
        AND manifest #>> '{signature,algorithm}' = 'Ed25519'
    ),
    UNIQUE (organization_id, bundle_id, version)
);

CREATE TABLE artifact_store_manifest_entries (
    organization_id UUID NOT NULL,
    manifest_digest TEXT NOT NULL,
    path TEXT NOT NULL,
    digest TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    media_type TEXT NOT NULL,
    PRIMARY KEY (organization_id, manifest_digest, path),
    CONSTRAINT artifact_store_manifest_entries_manifest_fk
        FOREIGN KEY (organization_id, manifest_digest)
        REFERENCES artifact_store_manifests(organization_id, manifest_digest)
        ON DELETE CASCADE,
    CONSTRAINT artifact_store_manifest_entries_blob_fk
        FOREIGN KEY (organization_id, digest)
        REFERENCES artifact_store_org_blobs(organization_id, digest)
        ON DELETE RESTRICT,
    CONSTRAINT artifact_store_manifest_entries_path_chk CHECK (
        char_length(path) BETWEEN 1 AND 240
        AND path !~ '(^/|\\|(^|/)\.\.?(/|$)|/$)'
    ),
    CONSTRAINT artifact_store_manifest_entries_digest_chk
        CHECK (digest ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT artifact_store_manifest_entries_size_chk
        CHECK (size_bytes BETWEEN 0 AND 67108864)
);

CREATE TABLE artifact_store_pins (
    organization_id UUID NOT NULL,
    digest TEXT NOT NULL,
    pinned_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, digest),
    FOREIGN KEY (organization_id, digest)
        REFERENCES artifact_store_org_blobs(organization_id, digest)
        ON DELETE CASCADE
);

CREATE TABLE artifact_store_leases (
    organization_id UUID NOT NULL,
    lease_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, lease_id),
    CONSTRAINT artifact_store_leases_id_chk
        CHECK (lease_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$'),
    CONSTRAINT artifact_store_leases_expiry_chk CHECK (expires_at > created_at),
    FOREIGN KEY (organization_id, digest)
        REFERENCES artifact_store_org_blobs(organization_id, digest)
        ON DELETE CASCADE
);

CREATE TABLE artifact_store_gc_candidates (
    digest TEXT NOT NULL,
    backend TEXT NOT NULL,
    scheduled_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (digest, backend),
    FOREIGN KEY (digest, backend)
        REFERENCES artifact_store_object_locations(digest, backend) ON DELETE CASCADE,
    CONSTRAINT artifact_store_gc_candidates_backend_chk
        CHECK (backend IN ('local_cas', 's3_compatible'))
);

CREATE INDEX idx_artifact_store_entries_digest
    ON artifact_store_manifest_entries(organization_id, digest);
CREATE INDEX idx_artifact_store_leases_expiry
    ON artifact_store_leases(expires_at, organization_id, digest);

COMMIT;
