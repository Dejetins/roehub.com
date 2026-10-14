-- Additive PG16 market storage. Existing control/identity tables stay in public.
-- Caller owns the transaction; bootstrap commits DDL and its checksum marker together.
CREATE SCHEMA IF NOT EXISTS market_data;
REVOKE ALL ON SCHEMA market_data FROM PUBLIC;

CREATE TABLE IF NOT EXISTS market_data.ref_market (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    exchange_name text NOT NULL,
    market_type text NOT NULL,
    market_code text NOT NULL,
    is_enabled integer NOT NULL DEFAULT 1 CHECK (is_enabled IN (0,1)),
    count_symbols bigint NOT NULL CHECK (count_symbols BETWEEN 0 AND 4294967295) DEFAULT 0,
    updated_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    payload_order bytea NOT NULL DEFAULT ''::bytea,
    PRIMARY KEY (market_id),
    CHECK (market_id > 4 OR (market_id, exchange_name, market_type, market_code) IN (
        (1,'binance','spot','binance:spot'), (2,'binance','futures','binance:futures'),
        (3,'bybit','spot','bybit:spot'), (4,'bybit','futures','bybit:futures')))
);

CREATE TABLE IF NOT EXISTS market_data.ref_instruments (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    status text NOT NULL DEFAULT 'UNKNOWN',
    is_tradable integer NOT NULL DEFAULT 1 CHECK (is_tradable IN (0,1)),
    base_asset text,
    quote_asset text,
    price_step double precision,
    qty_step double precision,
    min_notional double precision,
    updated_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    payload_order bytea NOT NULL DEFAULT ''::bytea,
    PRIMARY KEY (market_id, symbol)
);

CREATE TABLE IF NOT EXISTS market_data.raw_binance_klines_1m (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    instrument_key text NOT NULL,
    open_time timestamptz(3) NOT NULL,
    open double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    close double precision NOT NULL,
    volume double precision NOT NULL,
    close_time timestamptz(3) NOT NULL,
    quote_asset_volume double precision NOT NULL,
    number_of_trades bigint NOT NULL CHECK (number_of_trades BETWEEN 0 AND 4294967295),
    taker_buy_base_asset_volume double precision NOT NULL,
    taker_buy_quote_asset_volume double precision NOT NULL,
    source text NOT NULL DEFAULT 'unknown',
    ingested_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    ingest_id uuid
) PARTITION BY RANGE (open_time);

CREATE TABLE IF NOT EXISTS market_data.raw_bybit_klines_1m (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    instrument_key text NOT NULL,
    interval_min integer NOT NULL CHECK (interval_min BETWEEN 1 AND 65535),
    start_time_ms bigint NOT NULL CHECK (start_time_ms BETWEEN 0 AND 253402300799999),
    start_time_utc timestamptz(3) NOT NULL,
    open double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    close double precision NOT NULL,
    volume double precision NOT NULL,
    turnover double precision NOT NULL,
    source text NOT NULL DEFAULT 'unknown',
    ingested_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    ingest_id uuid,
    CHECK (interval_min = 1),
    CHECK (extract(epoch FROM start_time_utc) * 1000 = start_time_ms)
) PARTITION BY RANGE (start_time_utc);

CREATE TABLE IF NOT EXISTS market_data.canonical_candles_1m (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    instrument_key text NOT NULL,
    ts_open timestamptz(3) NOT NULL,
    ts_close timestamptz(3) NOT NULL,
    open double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    close double precision NOT NULL,
    volume_base double precision NOT NULL,
    volume_quote double precision,
    trades_count bigint CHECK (trades_count BETWEEN 0 AND 4294967295),
    taker_buy_volume_base double precision,
    taker_buy_volume_quote double precision,
    source text NOT NULL,
    ingested_at timestamptz(3) NOT NULL,
    ingest_id uuid,
    payload_order bytea NOT NULL DEFAULT ''::bytea,
    PRIMARY KEY (market_id, symbol, ts_open),
    CHECK (extract(epoch FROM ts_open)::numeric % 60 = 0),
    CHECK (ts_close >= ts_open AND ts_close <= ts_open + interval '1 minute')
) PARTITION BY RANGE (ts_open);

CREATE TABLE IF NOT EXISTS market_data.funding_instrument_universe (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    instrument_key text NOT NULL,
    exchange text NOT NULL,
    market_type text NOT NULL,
    status text NOT NULL,
    is_tradable integer NOT NULL DEFAULT 1 CHECK (is_tradable IN (0,1)),
    base_asset text,
    quote_asset text,
    funding_interval_minutes integer CHECK (funding_interval_minutes BETWEEN 1 AND 65535),
    funding_interval_source text,
    funding_cap double precision,
    funding_floor double precision,
    updated_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    payload_order bytea NOT NULL DEFAULT ''::bytea,
    PRIMARY KEY (market_id, symbol)
);

CREATE TABLE IF NOT EXISTS market_data.raw_binance_funding_rates (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    instrument_key text NOT NULL,
    funding_time timestamptz(3) NOT NULL,
    funding_rate double precision NOT NULL,
    funding_interval_minutes integer NOT NULL CHECK (funding_interval_minutes BETWEEN 1 AND 65535),
    funding_interval_source text NOT NULL,
    mark_price double precision,
    source text NOT NULL DEFAULT 'rest',
    ingested_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    ingest_id uuid,
    payload_order bytea NOT NULL DEFAULT ''::bytea,
    PRIMARY KEY (market_id, symbol, funding_time)
) PARTITION BY RANGE (funding_time);

CREATE TABLE IF NOT EXISTS market_data.raw_bybit_funding_rates (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    instrument_key text NOT NULL,
    category text NOT NULL DEFAULT 'linear',
    funding_time timestamptz(3) NOT NULL,
    funding_rate double precision NOT NULL,
    funding_interval_minutes integer NOT NULL CHECK (funding_interval_minutes BETWEEN 1 AND 65535),
    funding_interval_source text NOT NULL,
    source text NOT NULL DEFAULT 'rest',
    ingested_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    ingest_id uuid,
    payload_order bytea NOT NULL DEFAULT ''::bytea,
    PRIMARY KEY (market_id, symbol, funding_time)
) PARTITION BY RANGE (funding_time);

CREATE TABLE IF NOT EXISTS market_data.canonical_funding_rates (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    instrument_key text NOT NULL,
    funding_time timestamptz(3) NOT NULL,
    funding_rate double precision NOT NULL,
    funding_interval_minutes integer NOT NULL CHECK (funding_interval_minutes BETWEEN 1 AND 65535),
    funding_interval_source text NOT NULL,
    source text NOT NULL,
    ingested_at timestamptz(3) NOT NULL,
    ingest_id uuid,
    payload_order bytea NOT NULL DEFAULT ''::bytea,
    PRIMARY KEY (market_id, symbol, funding_time)
) PARTITION BY RANGE (funding_time);

CREATE INDEX IF NOT EXISTS ref_instruments_search ON market_data.ref_instruments
    (market_id, symbol text_pattern_ops) WHERE is_tradable = 1;
CREATE INDEX IF NOT EXISTS raw_binance_klines_key ON market_data.raw_binance_klines_1m
    (market_id, symbol, open_time);
CREATE INDEX IF NOT EXISTS raw_bybit_klines_key ON market_data.raw_bybit_klines_1m
    (market_id, symbol, start_time_utc);

-- A generation is never inferred from bounds, reused on reset, or changed on restart.
CREATE TABLE IF NOT EXISTS market_data.instrument_generations (
    market_id integer NOT NULL CHECK (market_id BETWEEN 1 AND 65535),
    symbol text NOT NULL CHECK (symbol <> ''),
    generation uuid NOT NULL DEFAULT gen_random_uuid(),
    content_revision bigint NOT NULL DEFAULT 0 CHECK (content_revision >= 0),
    funding_revision bigint NOT NULL DEFAULT 0 CHECK (funding_revision >= 0),
    index_state text NOT NULL DEFAULT 'unknown' CHECK (index_state IN ('unknown','rebuilding','ready')),
    invalidation_reason text NOT NULL DEFAULT 'initial',
    updated_at timestamptz(3) NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (market_id, symbol), UNIQUE (market_id, symbol, generation)
);
CREATE TABLE IF NOT EXISTS market_data.candle_coverage_days (
    market_id integer NOT NULL, symbol text NOT NULL, day date NOT NULL,
    generation uuid NOT NULL,
    actual bit(1440) NOT NULL DEFAULT 0::bit(1440),
    source_absent bit(1440) NOT NULL DEFAULT 0::bit(1440),
    actual_count integer GENERATED ALWAYS AS (bit_count(actual)) STORED,
    absent_count integer GENERATED ALWAYS AS (bit_count(source_absent)) STORED,
    policy_version integer NOT NULL CHECK (policy_version > 0),
    PRIMARY KEY (market_id, symbol, day),
    FOREIGN KEY (market_id,symbol,generation) REFERENCES market_data.instrument_generations
        (market_id,symbol,generation),
    CHECK ((actual & source_absent) = 0::bit(1440))
);
CREATE TABLE IF NOT EXISTS market_data.candle_coverage_summary (
    market_id integer NOT NULL, symbol text NOT NULL, generation uuid NOT NULL,
    actual_count bigint NOT NULL DEFAULT 0 CHECK (actual_count >= 0),
    first_ts timestamptz(3), last_ts timestamptz(3),
    PRIMARY KEY (market_id,symbol),
    FOREIGN KEY (market_id,symbol,generation) REFERENCES market_data.instrument_generations
        (market_id,symbol,generation),
    CHECK ((actual_count = 0 AND first_ts IS NULL AND last_ts IS NULL)
        OR (actual_count > 0 AND first_ts IS NOT NULL AND last_ts IS NOT NULL AND first_ts <= last_ts))
);
CREATE TABLE IF NOT EXISTS market_data.source_absent_candles_1m (
    market_id integer NOT NULL, symbol text NOT NULL, minute_key bigint NOT NULL
        CHECK (minute_key BETWEEN 0 AND 4294967295),
    generation uuid NOT NULL, observed_at timestamptz(3) NOT NULL,
    policy_version integer NOT NULL CHECK (policy_version > 0),
    observation_id uuid NOT NULL, source text NOT NULL,
    range_start timestamptz(3) NOT NULL, range_end timestamptz(3) NOT NULL,
    PRIMARY KEY (market_id,symbol,minute_key),
    FOREIGN KEY (market_id,symbol,generation) REFERENCES market_data.instrument_generations
        (market_id,symbol,generation),
    CHECK (range_start < range_end),
    CHECK (minute_key * 60 >= extract(epoch FROM range_start)
       AND minute_key * 60 < extract(epoch FROM range_end))
);
CREATE TABLE IF NOT EXISTS market_data.candle_history_verification (
    market_id integer NOT NULL, symbol text NOT NULL, generation uuid NOT NULL,
    origin timestamptz(3) NOT NULL, verified_until timestamptz(3) NOT NULL,
    policy_version integer NOT NULL CHECK (policy_version > 0),
    lease_token uuid NOT NULL, verified_at timestamptz(3) NOT NULL,
    PRIMARY KEY (market_id,symbol),
    FOREIGN KEY (market_id,symbol,generation) REFERENCES market_data.instrument_generations
        (market_id,symbol,generation),
    CHECK (verified_until >= origin),
    CHECK (extract(epoch FROM origin)::numeric % 60 = 0),
    CHECK (extract(epoch FROM verified_until)::numeric % 60 = 0)
);
-- Nullable additions preserve old writers until the coordinated S05 serializer switch.
-- NULL means legacy/unverified, never revision zero or current generation.
ALTER TABLE public.backtest_artifact_source_state
    ADD COLUMN IF NOT EXISTS source_generation uuid,
    ADD COLUMN IF NOT EXISTS content_revision bigint CHECK (content_revision >= 0),
    ADD COLUMN IF NOT EXISTS funding_revision bigint CHECK (funding_revision >= 0),
    ADD COLUMN IF NOT EXISTS source_contract_version integer CHECK (source_contract_version = 2);

-- At most twelve months / 72 partitions per call, no caller-controlled identifiers.
-- Invoker must have EXECUTE explicitly granted; owner is the migration principal.
CREATE OR REPLACE FUNCTION market_data.ensure_partitions(first_month date, month_count integer)
RETURNS integer LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, pg_temp SET timezone = 'UTC'
SET statement_timeout = '60s' SET lock_timeout = '2s'
AS $$
DECLARE m date; n date; t text; created integer := 0;
BEGIN
    IF first_month IS NULL OR month_count IS NULL OR month_count NOT BETWEEN 1 AND 12
       OR first_month < date '1970-01-01' OR first_month > date '9998-01-01'
       OR extract(day FROM first_month) <> 1 THEN
        RAISE EXCEPTION 'invalid bounded partition range' USING ERRCODE = '22023';
    END IF;
    PERFORM pg_advisory_xact_lock(56329814722);
    FOR i IN 0..month_count-1 LOOP
        m := (first_month + make_interval(months => i))::date;
        n := (m + interval '1 month')::date;
        FOREACH t IN ARRAY ARRAY['raw_binance_klines_1m','raw_bybit_klines_1m',
            'canonical_candles_1m','raw_binance_funding_rates',
            'raw_bybit_funding_rates','canonical_funding_rates'] LOOP
            EXECUTE format('CREATE TABLE IF NOT EXISTS market_data.%I PARTITION OF market_data.%I FOR VALUES FROM (%L) TO (%L)',
                t || '_' || to_char(m,'YYYYMM'), t, m::timestamptz, n::timestamptz);
            created := created + 1;
        END LOOP;
    END LOOP;
    RETURN created;
END $$;
REVOKE ALL ON FUNCTION market_data.ensure_partitions(date,integer) FROM PUBLIC;

-- Explicit per-instrument administrative invalidation. Data remains available for
-- bounded rebuild; unknown metadata must never be consumed as verified completeness.
CREATE OR REPLACE FUNCTION market_data.invalidate_instrument(
    p_market integer, p_symbol text, expected_generation uuid, reason text)
RETURNS uuid LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, pg_temp SET lock_timeout = '2s'
AS $$
DECLARE previous uuid; replacement uuid := gen_random_uuid();
BEGIN
    IF reason IS NULL OR length(reason) NOT BETWEEN 1 AND 256 THEN
        RAISE EXCEPTION 'invalidation reason required' USING ERRCODE = '22023';
    END IF;
    SELECT generation INTO previous FROM market_data.instrument_generations
        WHERE market_id=p_market AND symbol=p_symbol FOR UPDATE;
    IF previous IS NULL OR previous IS DISTINCT FROM expected_generation THEN
        RAISE EXCEPTION 'stale source generation' USING ERRCODE = '40001';
    END IF;
    DELETE FROM market_data.candle_history_verification WHERE market_id=p_market AND symbol=p_symbol;
    DELETE FROM market_data.candle_coverage_days WHERE market_id=p_market AND symbol=p_symbol;
    DELETE FROM market_data.candle_coverage_summary WHERE market_id=p_market AND symbol=p_symbol;
    DELETE FROM market_data.source_absent_candles_1m WHERE market_id=p_market AND symbol=p_symbol;
    UPDATE market_data.instrument_generations SET generation=replacement,
        content_revision=0, funding_revision=0, index_state='unknown',
        invalidation_reason=reason, updated_at=clock_timestamp()
        WHERE market_id=p_market AND symbol=p_symbol;
    RETURN replacement;
END $$;
REVOKE ALL ON FUNCTION market_data.invalidate_instrument(integer,text,uuid,text) FROM PUBLIC;
