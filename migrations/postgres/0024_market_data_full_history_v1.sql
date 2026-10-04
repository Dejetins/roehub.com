BEGIN;

-- User-selected history is no longer capped at seven days. Worker windows remain bounded.
ALTER TABLE market_data_work_requests
    DROP CONSTRAINT market_data_work_requests_total_units_check,
    DROP CONSTRAINT market_data_work_requests_check1,
    ADD CONSTRAINT market_data_work_requests_total_units_check CHECK (total_units > 0),
    ADD CONSTRAINT market_data_work_requests_range_check CHECK (
        (kind = 'catalog_refresh' AND start_at IS NULL AND end_at IS NULL)
        OR (kind = 'candle_ingestion' AND start_at IS NOT NULL AND end_at IS NOT NULL
            AND start_at >= TIMESTAMPTZ '2017-01-01 00:00:00+00'
            AND end_at > start_at AND jsonb_array_length(symbols) BETWEEN 1 AND 8));

-- Public exchange metadata cache/inbox. No user selections or candle data are stored here.
CREATE TABLE market_data_history_bounds (
    market_id SMALLINT NOT NULL CHECK (market_id > 0),
    symbol TEXT NOT NULL CHECK (symbol ~ '^[A-Z0-9][A-Z0-9._-]{1,63}$'),
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'ready', 'unavailable')),
    first_open_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL,
    worker_token UUID,
    PRIMARY KEY (market_id, symbol),
    CHECK ((state = 'ready') = (first_open_at IS NOT NULL))
);
CREATE INDEX market_data_history_bounds_queue ON market_data_history_bounds (updated_at)
    WHERE state IN ('queued', 'running');

-- A volatile function gets a fresh READ COMMITTED snapshot after the admission lock.
-- Concurrent reads cannot enqueue an unbounded number of exchange probes.
CREATE FUNCTION request_market_data_history_bounds(
    requested_market SMALLINT, requested_symbol TEXT, requested_at TIMESTAMPTZ
) RETURNS VOID LANGUAGE plpgsql VOLATILE AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(744948345611);
    IF (SELECT count(*) FROM market_data_history_bounds
        WHERE state IN ('queued', 'running')) < 64 THEN
        INSERT INTO market_data_history_bounds (market_id, symbol, state, updated_at)
        VALUES (requested_market, requested_symbol, 'queued', requested_at)
        ON CONFLICT (market_id, symbol) DO UPDATE
            SET state = 'queued', updated_at = requested_at
            WHERE market_data_history_bounds.state = 'unavailable'
              AND market_data_history_bounds.updated_at < requested_at - INTERVAL '30 seconds';
    END IF;
END;
$$;

COMMIT;
