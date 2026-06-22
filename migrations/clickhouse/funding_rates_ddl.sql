CREATE DATABASE IF NOT EXISTS market_data;
USE market_data;

CREATE TABLE IF NOT EXISTS market_data.funding_instrument_universe
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    exchange LowCardinality(String),
    market_type LowCardinality(String),
    status LowCardinality(String),
    is_tradable UInt8,
    base_asset LowCardinality(Nullable(String)),
    quote_asset LowCardinality(Nullable(String)),
    funding_interval_minutes Nullable(UInt16),
    funding_interval_source LowCardinality(Nullable(String)),
    funding_cap Nullable(Float64),
    funding_floor Nullable(Float64),
    updated_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (market_id, symbol);

CREATE TABLE IF NOT EXISTS market_data.raw_binance_funding_rates
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    funding_time DateTime64(3, 'UTC'),
    funding_rate Float64,
    funding_interval_minutes UInt16,
    funding_interval_source LowCardinality(String),
    mark_price Nullable(Float64),
    source LowCardinality(String) DEFAULT 'rest',
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3),
    ingest_id Nullable(UUID)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMMDD(funding_time)
ORDER BY (market_id, symbol, funding_time);

CREATE TABLE IF NOT EXISTS market_data.raw_bybit_funding_rates
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    category LowCardinality(String),
    funding_time DateTime64(3, 'UTC'),
    funding_rate Float64,
    funding_interval_minutes UInt16,
    funding_interval_source LowCardinality(String),
    source LowCardinality(String) DEFAULT 'rest',
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3),
    ingest_id Nullable(UUID)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMMDD(funding_time)
ORDER BY (market_id, symbol, funding_time);

CREATE TABLE IF NOT EXISTS market_data.canonical_funding_rates
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    funding_time DateTime64(3, 'UTC'),
    funding_rate Float64,
    funding_interval_minutes UInt16,
    funding_interval_source LowCardinality(String),
    source LowCardinality(String),
    ingested_at DateTime64(3, 'UTC'),
    ingest_id Nullable(UUID)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMMDD(funding_time)
ORDER BY (market_id, symbol, funding_time);
