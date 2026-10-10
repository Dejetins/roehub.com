BEGIN;
-- Additive only. No activation, default reseeding, legacy conversion or recipe rewrite.
CREATE TABLE IF NOT EXISTS backtest_artifact_installation_settings (
    installation_id UUID PRIMARY KEY REFERENCES identity_installations ON DELETE RESTRICT,
    control_version BIGINT NOT NULL CHECK (control_version > 0),
    capability_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    paused BOOLEAN NOT NULL DEFAULT TRUE,
    root_id TEXT NOT NULL,
    root_revision BIGINT NOT NULL CHECK (root_revision > 0),
    schedule JSONB NOT NULL,
    budget_bytes BIGINT CHECK (budget_bytes > 0),
    org_queue_limit INTEGER NOT NULL DEFAULT 200 CHECK (org_queue_limit BETWEEN 1 AND 200),
    global_queue_limit INTEGER NOT NULL DEFAULT 2000 CHECK (global_queue_limit BETWEEN 1 AND 2000),
    legacy_handoff TEXT NOT NULL DEFAULT 'pending' CHECK (legacy_handoff IN ('pending','complete')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS backtest_artifact_profiles (
    organization_id UUID NOT NULL REFERENCES identity_organizations ON DELETE RESTRICT,
    profile_id UUID NOT NULL,
    revision BIGINT NOT NULL CHECK (revision > 0),
    profile JSONB NOT NULL,
    created_by UUID NOT NULL REFERENCES identity_users ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (organization_id, profile_id, revision)
);
CREATE TABLE IF NOT EXISTS backtest_artifact_policies (
    organization_id UUID NOT NULL REFERENCES identity_organizations ON DELETE RESTRICT,
    exchange TEXT NOT NULL, market_type TEXT NOT NULL, symbol TEXT NOT NULL,
    control_version BIGINT NOT NULL CHECK (control_version > 0),
    profile_id UUID NOT NULL, profile_revision BIGINT NOT NULL,
    schedule JSONB NOT NULL, next_due TIMESTAMPTZ, last_success TIMESTAMPTZ,
    PRIMARY KEY (organization_id, exchange, market_type, symbol),
    FOREIGN KEY (organization_id, profile_id, profile_revision)
        REFERENCES backtest_artifact_profiles ON DELETE RESTRICT
);
CREATE TABLE IF NOT EXISTS backtest_artifact_commands (
    command_id UUID PRIMARY KEY,
    organization_id UUID NOT NULL REFERENCES identity_organizations ON DELETE RESTRICT,
    actor_id UUID NOT NULL REFERENCES identity_users ON DELETE RESTRICT,
    idempotency_key TEXT NOT NULL CHECK (idempotency_key ~ '^[A-Za-z0-9._:-]{8,128}$'),
    body_hash TEXT NOT NULL CHECK (body_hash ~ '^[0-9a-f]{64}$'),
    kind TEXT NOT NULL CHECK (kind IN ('update_now','policy')),
    response JSONB NOT NULL, response_status INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (organization_id, idempotency_key), UNIQUE (command_id, organization_id)
);
CREATE TABLE IF NOT EXISTS backtest_artifact_requests (
    request_id UUID PRIMARY KEY,
    organization_id UUID REFERENCES identity_organizations ON DELETE RESTRICT,
    installation_id UUID NOT NULL REFERENCES identity_installations ON DELETE RESTRICT,
    command_id UUID,
    internal_dedupe_key TEXT, organization_dedupe_key TEXT,
    exchange TEXT NOT NULL, market_type TEXT NOT NULL, symbol TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('base','derivatives')),
    trigger TEXT NOT NULL CHECK (trigger IN ('initial','schedule','manual','source_revision','legacy')),
    snapshot JSONB NOT NULL,
    state TEXT NOT NULL CHECK (state IN
        ('waiting_for_source','queued','running','retry_wait','ready','error')),
    reason TEXT, control_version BIGINT NOT NULL DEFAULT 1 CHECK (control_version > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (command_id, organization_id)
        REFERENCES backtest_artifact_commands(command_id, organization_id) ON DELETE RESTRICT,
    FOREIGN KEY (installation_id, organization_id)
        REFERENCES identity_organizations(installation_id, organization_id) ON DELETE RESTRICT,
    CHECK ((organization_id IS NULL AND command_id IS NULL
            AND internal_dedupe_key IS NOT NULL AND organization_dedupe_key IS NULL)
        OR (organization_id IS NOT NULL AND internal_dedupe_key IS NULL
            AND ((command_id IS NOT NULL AND organization_dedupe_key IS NULL)
                OR (command_id IS NULL AND organization_dedupe_key IS NOT NULL)))),
    UNIQUE (command_id, exchange, market_type, symbol, kind)
);
CREATE UNIQUE INDEX IF NOT EXISTS artifact_internal_dedupe
    ON backtest_artifact_requests(installation_id, internal_dedupe_key)
    WHERE internal_dedupe_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS artifact_organization_dedupe
    ON backtest_artifact_requests(organization_id, organization_dedupe_key)
    WHERE organization_dedupe_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS artifact_request_queue
    ON backtest_artifact_requests(installation_id, state, created_at, request_id);
CREATE TABLE IF NOT EXISTS backtest_artifact_builds (
    build_id UUID PRIMARY KEY,
    installation_id UUID NOT NULL REFERENCES identity_installations ON DELETE RESTRICT,
    exchange TEXT NOT NULL, market_type TEXT NOT NULL, symbol TEXT NOT NULL,
    snapshot JSONB NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('running','quarantined','retry_wait','ready','error')),
    owner_token UUID NOT NULL, parent_incarnation UUID NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt BETWEEN 1 AND 8),
    epoch BIGINT NOT NULL CHECK (epoch > 0),
    progress JSONB NOT NULL DEFAULT '{}', next_retry_at TIMESTAMPTZ,
    reservation_bytes BIGINT NOT NULL CHECK (reservation_bytes >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- No organization/installation in this physical key; quarantine never expires by timer.
CREATE UNIQUE INDEX IF NOT EXISTS artifact_coordinate_builder
    ON backtest_artifact_builds(exchange, market_type, symbol)
    WHERE state IN ('running','quarantined');
CREATE TABLE IF NOT EXISTS backtest_artifact_build_requests (
    build_id UUID NOT NULL REFERENCES backtest_artifact_builds ON DELETE RESTRICT,
    request_id UUID NOT NULL REFERENCES backtest_artifact_requests ON DELETE RESTRICT,
    PRIMARY KEY (build_id, request_id)
);
CREATE TABLE IF NOT EXISTS backtest_artifact_attempts (
    build_id UUID NOT NULL REFERENCES backtest_artifact_builds ON DELETE RESTRICT,
    attempt INTEGER NOT NULL, epoch BIGINT NOT NULL,
    owner_token UUID NOT NULL, parent_incarnation UUID NOT NULL,
    state TEXT NOT NULL, progress JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (build_id, attempt)
);
CREATE TABLE IF NOT EXISTS backtest_artifact_events (
    event_id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES backtest_artifact_requests ON DELETE RESTRICT,
    state TEXT NOT NULL, reason TEXT, progress JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS backtest_artifact_source_state (
    exchange TEXT NOT NULL, market_type TEXT NOT NULL, symbol TEXT NOT NULL,
    desired_target JSONB NOT NULL, published JSONB, reconciliation_cursor TEXT,
    PRIMARY KEY (exchange, market_type, symbol)
);
CREATE TABLE IF NOT EXISTS backtest_artifact_legacy_demands (
    installation_id UUID NOT NULL REFERENCES identity_installations ON DELETE RESTRICT,
    exchange TEXT NOT NULL, market_type TEXT NOT NULL, symbol TEXT NOT NULL,
    config_hash TEXT NOT NULL, snapshot JSONB NOT NULL,
    control_version BIGINT NOT NULL DEFAULT 1,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (installation_id, exchange, market_type, symbol)
);
COMMIT;
