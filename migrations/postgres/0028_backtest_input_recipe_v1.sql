-- Additive only: no existing recipe, hash or provenance is inferred/backfilled.
ALTER TABLE backtest_jobs
    ADD COLUMN IF NOT EXISTS input_recipe_json JSONB,
    ADD COLUMN IF NOT EXISTS preparation_provenance_json JSONB;

-- The physical mutex key deliberately excludes generation.
CREATE TABLE IF NOT EXISTS backtest_artifact_slot_ownership (
    exchange TEXT NOT NULL CHECK (exchange ~ '^[a-z0-9_]+$'),
    market_type TEXT NOT NULL CHECK (market_type IN ('spot', 'futures')),
    symbol TEXT NOT NULL CHECK (symbol ~ '^[A-Z0-9_]+$'),
    slot TEXT NOT NULL CHECK (slot IN ('slot_a', 'slot_b')),
    generation BIGINT CHECK (generation > 0),
    manifest_sha256 TEXT CHECK (manifest_sha256 ~ '^[a-f0-9]{64}$'),
    state TEXT NOT NULL DEFAULT 'available'
        CHECK (state IN ('available', 'writing', 'quarantined')),
    owner_token UUID,
    owner_attempt INTEGER CHECK (owner_attempt > 0),
    parent_incarnation UUID,
    writer_epoch BIGINT NOT NULL DEFAULT 0 CHECK (writer_epoch >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (exchange, market_type, symbol, slot),
    CHECK ((generation IS NULL) = (manifest_sha256 IS NULL)),
    CHECK ((state = 'available' AND owner_token IS NULL AND owner_attempt IS NULL
             AND parent_incarnation IS NULL)
        OR (state IN ('writing', 'quarantined') AND owner_token IS NOT NULL
             AND owner_attempt IS NOT NULL AND parent_incarnation IS NOT NULL
             AND writer_epoch > 0))
);

-- Internal ownership records: application repository access must be org scoped.
-- No expiration/heartbeat predicate authorizes deleting a live reservation.
CREATE TABLE IF NOT EXISTS backtest_artifact_slot_readers (
    exchange TEXT NOT NULL,
    market_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    slot TEXT NOT NULL,
    owner_kind TEXT NOT NULL CHECK (owner_kind IN ('job', 'lazy', 'publisher_source')),
    organization_id UUID,
    owner_id UUID NOT NULL,
    owner_token UUID NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt >= 0),
    parent_incarnation UUID,
    expected_generation BIGINT NOT NULL CHECK (expected_generation > 0),
    expected_manifest_sha256 TEXT NOT NULL CHECK (expected_manifest_sha256 ~ '^[a-f0-9]{64}$'),
    ownership_epoch BIGINT NOT NULL CHECK (ownership_epoch >= 0),
    state TEXT NOT NULL CHECK (state IN ('queued', 'active', 'quarantined')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (exchange, market_type, symbol, slot, owner_token),
    FOREIGN KEY (exchange, market_type, symbol, slot)
        REFERENCES backtest_artifact_slot_ownership (exchange, market_type, symbol, slot)
        ON DELETE RESTRICT,
    CHECK ((owner_kind = 'publisher_source' AND organization_id IS NULL)
        OR (owner_kind IN ('job', 'lazy') AND organization_id IS NOT NULL)),
    CHECK ((state = 'queued' AND owner_kind = 'job' AND attempt = 0
             AND parent_incarnation IS NULL)
        OR (state IN ('active', 'quarantined') AND attempt > 0
             AND parent_incarnation IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS backtest_artifact_slot_readers_owner
    ON backtest_artifact_slot_readers (organization_id, owner_kind, owner_id, attempt);
