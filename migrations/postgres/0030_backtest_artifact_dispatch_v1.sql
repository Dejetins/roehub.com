BEGIN;
CREATE TABLE IF NOT EXISTS backtest_artifact_dispatch_state (
    installation_id UUID PRIMARY KEY REFERENCES identity_installations ON DELETE RESTRICT,
    source_cursor JSONB, schedule_cursor JSONB,
    last_lane TEXT NOT NULL DEFAULT '', retry_turn BOOLEAN NOT NULL DEFAULT FALSE,
    handoff JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS backtest_artifact_cadence (
    installation_id UUID NOT NULL REFERENCES identity_installations ON DELETE RESTRICT,
    exchange TEXT NOT NULL, market_type TEXT NOT NULL, symbol TEXT NOT NULL,
    lane TEXT NOT NULL, schedule_hash TEXT NOT NULL,
    last_occurrence TIMESTAMPTZ, next_due TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (installation_id,exchange,market_type,symbol,lane)
);
-- The override is explicit; old S02 policies already hold explicit schedules.
ALTER TABLE backtest_artifact_policies ADD COLUMN IF NOT EXISTS
    inherits_schedule BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE backtest_artifact_policies ADD COLUMN IF NOT EXISTS
    updated_by UUID REFERENCES identity_users ON DELETE RESTRICT;
COMMIT;
