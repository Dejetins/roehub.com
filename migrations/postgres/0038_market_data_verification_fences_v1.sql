-- Additive lifecycle checkpoints and generation-bound verification leases.
CREATE TABLE market_data.verification_leases (
 market_id integer NOT NULL, symbol text NOT NULL, generation uuid NOT NULL,
 token uuid NOT NULL, expires_at timestamptz NOT NULL,
 PRIMARY KEY(market_id,symbol),
 FOREIGN KEY(market_id,symbol,generation) REFERENCES market_data.instrument_generations
 (market_id,symbol,generation) ON UPDATE CASCADE
);
CREATE TABLE market_data.coverage_rebuilds (
 market_id integer NOT NULL, symbol text NOT NULL, generation uuid NOT NULL,
 next_day date NOT NULL, end_day date NOT NULL,
 PRIMARY KEY(market_id,symbol), CHECK(next_day<=end_day),
 FOREIGN KEY(market_id,symbol,generation) REFERENCES market_data.instrument_generations
 (market_id,symbol,generation) ON UPDATE CASCADE
);
CREATE INDEX candle_coverage_occupied ON market_data.candle_coverage_days
 (market_id,symbol,day) WHERE actual_count>0;
