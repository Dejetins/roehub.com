BEGIN;

DO $$
BEGIN
    IF to_regclass('public.identity_organizations') IS NULL
       OR to_regclass('public.artifact_store_manifests') IS NULL THEN
        RAISE EXCEPTION 'isolated jobs require organization and artifact store schemas';
    END IF;
END
$$;

CREATE TABLE job_runtime_jobs (
    organization_id UUID NOT NULL
        REFERENCES identity_organizations(organization_id) ON DELETE CASCADE,
    job_id UUID NOT NULL,
    semantic_job_key TEXT NOT NULL,
    semantic_spec_digest TEXT NOT NULL,
    capability TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    cancel_requested_at TIMESTAMPTZ,
    result_artifact_manifest_digest TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (organization_id, job_id),
    UNIQUE (organization_id, semantic_job_key),
    CONSTRAINT job_runtime_jobs_semantic_key_chk CHECK (
        semantic_job_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$'
    ),
    CONSTRAINT job_runtime_jobs_spec_digest_chk CHECK (
        semantic_spec_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    CONSTRAINT job_runtime_jobs_capability_chk CHECK (capability IN (
        'backtest', 'optimize', 'history_import', 'report', 'artifact_transform',
        'ml_training', 'ml_inference', 'rl_training', 'rl_inference',
        'custom_strategy'
    )),
    CONSTRAINT job_runtime_jobs_status_chk CHECK (status IN (
        'queued', 'running', 'recovering', 'succeeded', 'failed', 'crashed', 'canceled',
        'timed_out', 'resource_exhausted'
    )),
    CONSTRAINT job_runtime_jobs_attempt_count_chk CHECK (
        attempt_count BETWEEN 1 AND 10000
    ),
    CONSTRAINT job_runtime_jobs_result_state_chk CHECK (
        (status = 'succeeded' AND result_artifact_manifest_digest IS NOT NULL)
        OR (status <> 'succeeded' AND result_artifact_manifest_digest IS NULL)
    ),
    CONSTRAINT job_runtime_jobs_result_manifest_fk
        FOREIGN KEY (organization_id, result_artifact_manifest_digest)
        REFERENCES artifact_store_manifests(organization_id, manifest_digest)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE job_runtime_attempts (
    organization_id UUID NOT NULL,
    job_id UUID NOT NULL,
    attempt_id UUID NOT NULL,
    attempt_number INTEGER NOT NULL,
    envelope_digest TEXT NOT NULL,
    image_digest TEXT NOT NULL,
    envelope JSONB NOT NULL,
    status TEXT NOT NULL,
    worker_id TEXT,
    claimed_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    recovery_owner_id TEXT,
    recovery_claimed_at TIMESTAMPTZ,
    deadline TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    result JSONB,
    exit_code INTEGER,
    error_code TEXT,
    PRIMARY KEY (organization_id, attempt_id),
    UNIQUE (organization_id, job_id, attempt_number),
    CONSTRAINT job_runtime_attempts_job_fk
        FOREIGN KEY (organization_id, job_id)
        REFERENCES job_runtime_jobs(organization_id, job_id) ON DELETE CASCADE,
    CONSTRAINT job_runtime_attempts_number_chk CHECK (
        attempt_number BETWEEN 1 AND 10000
    ),
    CONSTRAINT job_runtime_attempts_envelope_digest_chk CHECK (
        envelope_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    CONSTRAINT job_runtime_attempts_image_digest_chk CHECK (
        image_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    CONSTRAINT job_runtime_attempts_status_chk CHECK (status IN (
        'queued', 'running', 'recovering', 'succeeded', 'failed', 'crashed', 'canceled',
        'timed_out', 'resource_exhausted'
    )),
    CONSTRAINT job_runtime_attempts_worker_chk CHECK (
        worker_id IS NULL OR worker_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$'
    ),
    CONSTRAINT job_runtime_attempts_recovery_owner_chk CHECK (
        recovery_owner_id IS NULL
        OR recovery_owner_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$'
    ),
    CONSTRAINT job_runtime_attempts_exit_code_chk CHECK (
        exit_code IS NULL OR exit_code BETWEEN 0 AND 255
    ),
    CONSTRAINT job_runtime_attempts_error_code_chk CHECK (
        error_code IS NULL OR error_code ~ '^[a-z][a-z0-9._-]{2,127}$'
    ),
    CONSTRAINT job_runtime_attempts_envelope_chk CHECK (
        jsonb_typeof(envelope) = 'object'
        AND envelope ->> 'schema' = 'JobEnvelope/v1'
        AND (envelope ->> 'organization_id')::UUID = organization_id
        AND (envelope ->> 'job_id')::UUID = job_id
        AND (envelope ->> 'attempt_id')::UUID = attempt_id
        AND (envelope ->> 'attempt_number')::INTEGER = attempt_number
        AND envelope ->> 'image_digest' = image_digest
        AND envelope ->> 'network' = 'none'
    ),
    CONSTRAINT job_runtime_attempts_running_state_chk CHECK (
        (status = 'queued' AND worker_id IS NULL AND claimed_at IS NULL)
        OR (status IN ('running', 'recovering') AND worker_id IS NOT NULL
            AND claimed_at IS NOT NULL AND heartbeat_at IS NOT NULL)
        OR (status IN ('succeeded', 'failed', 'crashed', 'canceled',
                      'timed_out', 'resource_exhausted')
            AND finished_at IS NOT NULL)
    ),
    CONSTRAINT job_runtime_attempts_recovery_state_chk CHECK (
        status <> 'recovering'
        OR (recovery_owner_id IS NOT NULL AND recovery_claimed_at IS NOT NULL)
    )
);

CREATE FUNCTION job_runtime_jobs_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.organization_id IS DISTINCT FROM OLD.organization_id
       OR NEW.job_id IS DISTINCT FROM OLD.job_id
       OR NEW.semantic_job_key IS DISTINCT FROM OLD.semantic_job_key
       OR NEW.semantic_spec_digest IS DISTINCT FROM OLD.semantic_spec_digest
       OR NEW.capability IS DISTINCT FROM OLD.capability
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'job identity and semantic specification are immutable';
    END IF;
    IF OLD.cancel_requested_at IS NOT NULL
       AND NEW.cancel_requested_at IS DISTINCT FROM OLD.cancel_requested_at THEN
        RAISE EXCEPTION 'job cancellation request is immutable';
    END IF;
    IF OLD.result_artifact_manifest_digest IS NOT NULL
       AND NEW.result_artifact_manifest_digest
           IS DISTINCT FROM OLD.result_artifact_manifest_digest THEN
        RAISE EXCEPTION 'job result artifact is immutable';
    END IF;
    IF OLD.status IN ('succeeded', 'canceled') AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'terminal job is immutable';
    END IF;
    IF OLD.status IN ('failed', 'crashed', 'timed_out', 'resource_exhausted')
       AND NEW.status = 'queued' THEN
        IF NEW.attempt_count <> OLD.attempt_count + 1
           OR NEW.cancel_requested_at IS NOT NULL
           OR NEW.result_artifact_manifest_digest IS NOT NULL THEN
            RAISE EXCEPTION 'invalid retry transition';
        END IF;
    ELSIF OLD.status IN ('failed', 'crashed', 'timed_out', 'resource_exhausted')
          AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'terminal job is immutable outside retry';
    ELSIF NEW.attempt_count <> OLD.attempt_count THEN
        RAISE EXCEPTION 'attempt count may change only during retry';
    END IF;
    IF NOT (
        NEW.status = OLD.status
        OR (OLD.status = 'queued' AND NEW.status IN ('running', 'canceled', 'timed_out'))
        OR (OLD.status = 'running' AND NEW.status IN (
            'recovering', 'succeeded', 'failed', 'crashed', 'canceled',
            'timed_out', 'resource_exhausted'
        ))
        OR (OLD.status = 'recovering' AND NEW.status IN ('crashed', 'canceled', 'timed_out'))
        OR (OLD.status IN ('failed', 'crashed', 'timed_out', 'resource_exhausted')
            AND NEW.status = 'queued')
    ) THEN
        RAISE EXCEPTION 'invalid job state transition';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER trg_job_runtime_jobs_guard
BEFORE UPDATE ON job_runtime_jobs
FOR EACH ROW EXECUTE FUNCTION job_runtime_jobs_guard();

CREATE FUNCTION job_runtime_attempts_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.organization_id IS DISTINCT FROM OLD.organization_id
       OR NEW.job_id IS DISTINCT FROM OLD.job_id
       OR NEW.attempt_id IS DISTINCT FROM OLD.attempt_id
       OR NEW.attempt_number IS DISTINCT FROM OLD.attempt_number
       OR NEW.envelope_digest IS DISTINCT FROM OLD.envelope_digest
       OR NEW.image_digest IS DISTINCT FROM OLD.image_digest
       OR NEW.envelope IS DISTINCT FROM OLD.envelope
       OR NEW.deadline IS DISTINCT FROM OLD.deadline THEN
        RAISE EXCEPTION 'job attempt envelope and identity are immutable';
    END IF;
    IF OLD.worker_id IS NOT NULL AND NEW.worker_id IS DISTINCT FROM OLD.worker_id THEN
        RAISE EXCEPTION 'job attempt worker identity is immutable';
    END IF;
    IF OLD.claimed_at IS NOT NULL AND NEW.claimed_at IS DISTINCT FROM OLD.claimed_at THEN
        RAISE EXCEPTION 'job attempt claim timestamp is immutable';
    END IF;
    IF OLD.result IS NOT NULL AND NEW.result IS DISTINCT FROM OLD.result THEN
        RAISE EXCEPTION 'job attempt result is immutable';
    END IF;
    IF OLD.status IN ('succeeded', 'failed', 'crashed', 'canceled', 'timed_out',
                      'resource_exhausted') AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'terminal job attempt is immutable';
    END IF;
    IF NOT (
        NEW.status = OLD.status
        OR (OLD.status = 'queued' AND NEW.status IN ('running', 'canceled', 'timed_out'))
        OR (OLD.status = 'running' AND NEW.status IN (
            'recovering', 'succeeded', 'failed', 'crashed', 'canceled',
            'timed_out', 'resource_exhausted'
        ))
        OR (OLD.status = 'recovering' AND NEW.status IN ('crashed', 'canceled', 'timed_out'))
    ) THEN
        RAISE EXCEPTION 'invalid job attempt state transition';
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER trg_job_runtime_attempts_guard
BEFORE UPDATE ON job_runtime_attempts
FOR EACH ROW EXECUTE FUNCTION job_runtime_attempts_guard();

CREATE INDEX idx_job_runtime_attempts_claim
    ON job_runtime_attempts (status, deadline, organization_id, job_id, attempt_number);
CREATE INDEX idx_job_runtime_attempts_recovery
    ON job_runtime_attempts (status, heartbeat_at, deadline)
    WHERE status = 'running';

COMMIT;
