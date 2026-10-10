-- Successful, fully consumed REST reads may contain legitimate trading gaps.
-- These are availability observations, never synthetic OHLCV candles.
CREATE TABLE IF NOT EXISTS market_data.source_absent_candles_1m
(
    market_id UInt16,
    symbol LowCardinality(String),
    minute_key UInt32,
    observed_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(observed_at)
ORDER BY (market_id, symbol, minute_key);
