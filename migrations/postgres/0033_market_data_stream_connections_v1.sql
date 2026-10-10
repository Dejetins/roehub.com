-- Ephemeral per-instrument socket observations, fenced to the collector lease session.
CREATE TABLE IF NOT EXISTS market_data_stream_connections (
    market_id SMALLINT NOT NULL CHECK (market_id BETWEEN 1 AND 4),
    symbol TEXT NOT NULL,
    connected BOOLEAN NOT NULL,
    collector_pid INTEGER NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (market_id, symbol)
);
