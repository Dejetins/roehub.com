BEGIN;

-- Immutable, bounded public metadata projections; candle truth stays in ClickHouse.
CREATE TABLE market_data_catalog_snapshots (
    snapshot_id UUID PRIMARY KEY,
    market_id SMALLINT NOT NULL CHECK (market_id > 0),
    refreshed_at TIMESTAMPTZ NOT NULL,
    items JSONB NOT NULL CHECK (jsonb_typeof(items) = 'array'
        AND jsonb_array_length(items) <= 20000)
);
CREATE INDEX market_data_catalog_snapshot_latest
    ON market_data_catalog_snapshots (market_id, refreshed_at DESC, snapshot_id DESC);

-- Requests are executed by the existing trusted Market Data scheduler.
CREATE TABLE market_data_work_requests (
    job_id UUID PRIMARY KEY,
    organization_id UUID NOT NULL REFERENCES identity_organizations(organization_id),
    actor_user_id UUID NOT NULL,
    idempotency_key UUID NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('catalog_refresh', 'candle_ingestion')),
    market_id SMALLINT NOT NULL CHECK (market_id > 0),
    symbols JSONB NOT NULL CHECK (jsonb_typeof(symbols) = 'array'
        AND jsonb_array_length(symbols) <= 8),
    start_at TIMESTAMPTZ,
    end_at TIMESTAMPTZ,
    timeframe TEXT NOT NULL DEFAULT '1m' CHECK (timeframe = '1m'),
    state TEXT NOT NULL CHECK (state IN
        ('queued', 'running', 'cancel_requested', 'cancelled', 'succeeded', 'failed')),
    attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt BETWEEN 1 AND 5),
    completed_units INTEGER NOT NULL DEFAULT 0 CHECK (completed_units >= 0),
    total_units INTEGER NOT NULL CHECK (total_units BETWEEN 1 AND 10080),
    rows_read BIGINT NOT NULL DEFAULT 0 CHECK (rows_read >= 0),
    rows_written BIGINT NOT NULL DEFAULT 0 CHECK (rows_written >= 0),
    error_code TEXT,
    worker_token UUID,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    attempts JSONB NOT NULL DEFAULT '[]'::jsonb,
    UNIQUE (organization_id, idempotency_key),
    -- Actor is an audit identifier. History must not prevent membership revocation.
    CHECK (completed_units <= total_units),
    CHECK ((kind = 'catalog_refresh' AND start_at IS NULL AND end_at IS NULL)
        OR (kind = 'candle_ingestion' AND start_at IS NOT NULL AND end_at IS NOT NULL
            AND end_at > start_at AND end_at - start_at <= INTERVAL '7 days'
            AND jsonb_array_length(symbols) BETWEEN 1 AND 8)),
    CHECK (error_code IS NULL OR error_code ~ '^[a-z][a-z0-9_]{0,63}$')
);
CREATE UNIQUE INDEX market_data_work_one_active_per_org
    ON market_data_work_requests (organization_id)
    WHERE state IN ('queued', 'running', 'cancel_requested');
CREATE INDEX market_data_work_org_history
    ON market_data_work_requests (organization_id, created_at DESC, job_id DESC);
CREATE INDEX market_data_work_queue
    ON market_data_work_requests (created_at, job_id) WHERE state = 'queued';

COMMIT;
