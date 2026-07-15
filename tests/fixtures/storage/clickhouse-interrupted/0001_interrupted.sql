CREATE DATABASE IF NOT EXISTS market_data;

CREATE TABLE IF NOT EXISTS market_data.interrupted_migration_probe
(
    value UInt8
)
ENGINE = Memory;

SELECT roehub_intentional_interruption_for_recovery_proof();
