BEGIN;

-- Apply with the request consumer stopped. Preserve every existing checkpoint.
DROP INDEX market_data_work_one_active_per_org;
ALTER TABLE market_data_work_requests ADD COLUMN error_phase TEXT
    CHECK (error_phase IS NULL OR error_phase IN
        ('catalog_refresh', 'checkpoint', 'release', 'lease_check', 'fill_window', 'finish'));

CREATE TABLE market_data_work_batches (
    organization_id UUID NOT NULL REFERENCES identity_organizations(organization_id),
    batch_key UUID NOT NULL,
    request JSONB NOT NULL CHECK (jsonb_typeof(request) = 'array'
        AND jsonb_array_length(request) BETWEEN 1 AND 50),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, batch_key)
);
ALTER TABLE market_data_work_requests ADD COLUMN batch_key UUID,
    ADD FOREIGN KEY (organization_id, batch_key)
        REFERENCES market_data_work_batches(organization_id, batch_key);
CREATE INDEX market_data_work_batch_jobs
    ON market_data_work_requests (organization_id, batch_key) WHERE batch_key IS NOT NULL;

CREATE TABLE market_data_work_events (
    event_id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES market_data_work_requests(job_id) ON DELETE CASCADE,
    organization_id UUID NOT NULL REFERENCES identity_organizations(organization_id),
    event_type TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    completed_units INTEGER NOT NULL,
    total_units INTEGER NOT NULL,
    rows_written BIGINT NOT NULL,
    error_code TEXT,
    error_phase TEXT,
    next_retry_at TIMESTAMPTZ
);
CREATE INDEX market_data_work_event_page
    ON market_data_work_events (organization_id, job_id, event_id DESC);
CREATE INDEX market_data_work_event_expiry ON market_data_work_events (occurred_at);

-- A shared storage outage must not make every queued instrument retry in a burst.
CREATE TABLE market_data_storage_cooldown (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    next_retry_at TIMESTAMPTZ NOT NULL,
    error_code TEXT NOT NULL
);

CREATE FUNCTION market_data_work_admission() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.state NOT IN ('queued','running','pause_requested','paused','retry_wait','cancel_requested')
       OR (TG_OP = 'UPDATE' AND OLD.state IN
           ('queued','running','pause_requested','paused','retry_wait','cancel_requested')) THEN
        RETURN NEW;
    END IF;
    -- Serialize admission, including different idempotency keys and overlapping batches.
    PERFORM pg_advisory_xact_lock(hashtextextended(NEW.organization_id::text, 26004));
    IF EXISTS (
        SELECT 1 FROM market_data_work_requests j
        WHERE j.organization_id = NEW.organization_id AND j.job_id <> NEW.job_id
          AND j.state IN ('queued','running','pause_requested','paused','retry_wait','cancel_requested')
          AND j.market_id = NEW.market_id AND j.kind = NEW.kind
          AND (NEW.kind = 'catalog_refresh' OR
               (j.start_at < NEW.end_at AND j.end_at > NEW.start_at
                AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(NEW.symbols) s
                            WHERE j.symbols ? s)))
    ) THEN
        RAISE EXCEPTION 'overlapping unfinished work exists'
            USING ERRCODE = '23505', CONSTRAINT = 'market_data_work_duplicate';
    END IF;
    IF (SELECT count(*) FROM market_data_work_requests j
        WHERE j.organization_id = NEW.organization_id AND j.job_id <> NEW.job_id
          AND j.state IN ('queued','running','pause_requested','paused','retry_wait','cancel_requested'))
        >= 200 THEN
        RAISE EXCEPTION 'organization queue is full'
            USING ERRCODE = '23505', CONSTRAINT = 'market_data_work_queue_full';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER market_data_work_admission BEFORE INSERT OR UPDATE OF state
    ON market_data_work_requests FOR EACH ROW EXECUTE FUNCTION market_data_work_admission();

CREATE FUNCTION market_data_work_record_event() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE event TEXT;
BEGIN
    IF TG_OP = 'INSERT' THEN event := 'queued';
    ELSIF NEW.attempt > OLD.attempt THEN event := 'continued';
    ELSIF NEW.state = 'running' AND OLD.state = 'retry_wait' THEN event := 'recovered';
    ELSIF NEW.state = 'running' AND OLD.started_at IS NULL THEN event := 'started';
    ELSIF OLD.state = 'paused' AND NEW.state IN ('queued','retry_wait') THEN event := 'resumed';
    ELSIF NEW.state = 'retry_wait' AND OLD.state <> 'retry_wait' THEN event := 'retry_scheduled';
    ELSIF NEW.state IN ('pause_requested','paused','cancel_requested','cancelled','succeeded','failed')
          AND OLD.state <> NEW.state THEN event := NEW.state;
    ELSE RETURN NEW;
    END IF;
    INSERT INTO market_data_work_events
        (job_id,organization_id,event_type,occurred_at,state,attempt,completed_units,
         total_units,rows_written,error_code,error_phase,next_retry_at)
    VALUES (NEW.job_id,NEW.organization_id,event,NEW.updated_at,NEW.state,NEW.attempt,
            NEW.completed_units,NEW.total_units,NEW.rows_written,NEW.error_code,
            NEW.error_phase,NEW.next_retry_at);
    -- Bounded per job. Scheduled worker retention also expires inactive histories.
    DELETE FROM market_data_work_events WHERE job_id = NEW.job_id AND event_id < (
        SELECT event_id FROM market_data_work_events WHERE job_id = NEW.job_id
        ORDER BY event_id DESC OFFSET 511 LIMIT 1);
    RETURN NEW;
END $$;
CREATE TRIGGER market_data_work_event AFTER INSERT OR UPDATE
    ON market_data_work_requests FOR EACH ROW EXECUTE FUNCTION market_data_work_record_event();

-- Explicit baseline, never an invented historical chronology.
INSERT INTO market_data_work_events
    (job_id,organization_id,event_type,occurred_at,state,attempt,completed_units,
     total_units,rows_written,error_code,error_phase,next_retry_at)
SELECT job_id,organization_id,'history_started',now(),state,attempt,completed_units,
       total_units,rows_written,error_code,error_phase,next_retry_at
FROM market_data_work_requests;


-- One atomic group admission and durable read reconciliation after an unknown response.
CREATE FUNCTION market_data_submit_batch(org UUID, actor UUID, key UUID,
                                        commands JSONB, submitted_at TIMESTAMPTZ)
RETURNS SETOF market_data_work_requests LANGUAGE plpgsql AS $$
DECLARE previous JSONB; command JSONB;
BEGIN
    PERFORM 1 FROM identity_memberships m
        JOIN identity_organizations o ON o.organization_id=m.organization_id
        WHERE m.organization_id=org AND m.user_id=actor
          AND m.status='active' AND o.status='active' FOR SHARE OF m,o;
    IF NOT FOUND THEN RAISE EXCEPTION 'active membership required' USING ERRCODE='42501'; END IF;
    PERFORM pg_advisory_xact_lock(hashtextextended(org::text,26004));
    SELECT request INTO previous FROM market_data_work_batches
        WHERE organization_id=org AND batch_key=key;
    IF FOUND THEN
        IF previous <> commands THEN
            RAISE EXCEPTION 'batch key belongs to different work' USING ERRCODE='22023';
        END IF;
    ELSE
        INSERT INTO market_data_work_batches VALUES (org,key,commands,submitted_at);
        FOR command IN SELECT value FROM jsonb_array_elements(commands) LOOP
            INSERT INTO market_data_work_requests
                (job_id,organization_id,actor_user_id,idempotency_key,batch_key,kind,
                 market_id,symbols,start_at,end_at,total_units,state,created_at,updated_at)
            VALUES ((command->>'id')::uuid,org,actor,(command->>'id')::uuid,key,
                    'candle_ingestion',(command->>'market_id')::smallint,
                    jsonb_build_array(command->>'symbol'),(command->>'start_at')::timestamptz,
                    (command->>'end_at')::timestamptz,(command->>'total_units')::integer,
                    'queued',submitted_at,submitted_at);
        END LOOP;
    END IF;
    RETURN QUERY SELECT * FROM market_data_work_requests
        WHERE organization_id=org AND batch_key=key ORDER BY symbols::text,job_id;
END $$;

COMMIT;
