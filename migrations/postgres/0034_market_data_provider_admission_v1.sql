BEGIN;
-- Never recycle a permit by time alone: its owner may still be performing I/O.
CREATE TABLE IF NOT EXISTS market_data_provider_scopes (
    provider text NOT NULL,
    scope text NOT NULL,
    cooldown_until timestamptz NOT NULL DEFAULT '-infinity',
    PRIMARY KEY (provider, scope)
);
CREATE TABLE IF NOT EXISTS market_data_provider_windows (
    provider text NOT NULL,
    scope text NOT NULL,
    dimension text NOT NULL,
    seconds integer NOT NULL CHECK (seconds > 0),
    bucket bigint NOT NULL,
    spent bigint NOT NULL CHECK (spent >= 0),
    PRIMARY KEY (provider, scope, dimension, seconds)
);
CREATE TABLE IF NOT EXISTS market_data_provider_permits (
    token uuid PRIMARY KEY,
    provider text NOT NULL,
    scope text NOT NULL,
    acquired_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    owner_pid integer NOT NULL,
    owner_host text NOT NULL
);
CREATE INDEX IF NOT EXISTS market_data_provider_permits_scope
    ON market_data_provider_permits (provider, scope);
CREATE TABLE IF NOT EXISTS market_data_coordinator_owners (
    lock_key bigint PRIMARY KEY,
    token uuid NOT NULL,
    acquired_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE market_data_history_bounds
    ADD COLUMN IF NOT EXISTS next_retry_at timestamptz,
    ADD COLUMN IF NOT EXISTS error_code text;
COMMIT;
