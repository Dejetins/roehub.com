-- Additive metadata journal. No candle payloads, secrets or user-request mutations.
CREATE TABLE IF NOT EXISTS market_data_stream_recovery (
    task_key TEXT PRIMARY KEY,
    token UUID NOT NULL,
    market_id INTEGER NOT NULL CHECK (market_id BETWEEN 1 AND 4),
    symbol TEXT NOT NULL,
    start_at TIMESTAMPTZ NOT NULL,
    end_at TIMESTAMPTZ NOT NULL CHECK (end_at > start_at),
    reason TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'failed')),
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    next_retry_at TIMESTAMPTZ,
    error_code TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS market_data_stream_recovery_pending
    ON market_data_stream_recovery (state, next_retry_at);
