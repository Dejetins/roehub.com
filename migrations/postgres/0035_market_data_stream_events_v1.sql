-- Public market-data lifecycle only: no organizations, users, tokens or provider payloads.
-- Events commit with the existing recovery/connection transaction, not per received candle.
CREATE TABLE IF NOT EXISTS market_data_stream_events (
    event_id BIGSERIAL PRIMARY KEY,
    market_id SMALLINT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason TEXT,
    error_code TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    start_at TIMESTAMPTZ,
    end_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS market_data_stream_events_instrument
    ON market_data_stream_events (market_id, symbol, event_id DESC);
CREATE INDEX IF NOT EXISTS market_data_stream_events_retention ON market_data_stream_events (occurred_at);

CREATE OR REPLACE FUNCTION market_data_record_stream_event() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    kind TEXT;
    item RECORD;
    range_end TIMESTAMPTZ;
BEGIN
    IF TG_TABLE_NAME = 'market_data_stream_connections' THEN
        IF TG_OP = 'UPDATE' AND NEW.connected = OLD.connected
           AND NEW.collector_pid = OLD.collector_pid THEN RETURN NEW; END IF;
        INSERT INTO market_data_stream_events(market_id,symbol,event_type)
        VALUES (NEW.market_id,NEW.symbol,
            CASE WHEN NEW.connected THEN 'connection_opened' ELSE 'connection_lost' END);
        RETURN NEW;
    END IF;
    IF TG_OP = 'DELETE' THEN item := OLD; ELSE item := NEW; END IF;
    -- Durable per-candle receipts are not user events: keep the batched hot path unchanged.
    IF item.reason = 'ws_recovery' THEN RETURN NULL; END IF;
    range_end := item.end_at;
    IF TG_OP = 'INSERT' THEN kind := 'recovery_queued';
    ELSIF TG_OP = 'DELETE' THEN kind := 'recovery_closed';
    ELSIF NEW.state = 'failed' AND OLD.state <> 'failed' THEN kind := 'recovery_failed';
    ELSIF NEW.start_at > OLD.start_at THEN
        kind := 'recovery_progress'; item := OLD; range_end := NEW.start_at;
    ELSIF NEW.error_code IS NOT NULL AND
          (NEW.error_code IS DISTINCT FROM OLD.error_code OR
           (NEW.attempt > OLD.attempt AND (NEW.attempt & (NEW.attempt - 1)) = 0)) THEN
        kind := 'recovery_waiting';
    ELSE RETURN NULL;
    END IF;
    INSERT INTO market_data_stream_events
        (market_id,symbol,event_type,reason,error_code,attempt,start_at,end_at)
    VALUES (item.market_id,item.symbol,kind,item.reason,
            CASE WHEN kind='recovery_closed' THEN NULL ELSE item.error_code END,
            item.attempt,item.start_at,range_end);
    RETURN NULL;
END $$;

DROP TRIGGER IF EXISTS market_data_connection_event ON market_data_stream_connections;
CREATE TRIGGER market_data_connection_event AFTER INSERT OR UPDATE
    ON market_data_stream_connections FOR EACH ROW EXECUTE FUNCTION market_data_record_stream_event();
DROP TRIGGER IF EXISTS market_data_recovery_event ON market_data_stream_recovery;
CREATE TRIGGER market_data_recovery_event AFTER INSERT OR UPDATE OR DELETE
    ON market_data_stream_recovery FOR EACH ROW EXECUTE FUNCTION market_data_record_stream_event();

-- Amortized retention maintenance also runs in continuously running collectors.
-- Reads hide expired rows immediately; physical removal is bounded and asynchronous.
CREATE OR REPLACE FUNCTION market_data_prune_stream_events() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.event_id % 1024 = 0 THEN
        DELETE FROM market_data_stream_events WHERE event_id IN (
            SELECT event_id FROM market_data_stream_events
            WHERE occurred_at < now() - INTERVAL '30 days'
            ORDER BY occurred_at LIMIT 4096);
    END IF;
    RETURN NULL;
END $$;
DROP TRIGGER IF EXISTS market_data_stream_event_retention ON market_data_stream_events;
CREATE TRIGGER market_data_stream_event_retention AFTER INSERT ON market_data_stream_events
    FOR EACH ROW EXECUTE FUNCTION market_data_prune_stream_events();
