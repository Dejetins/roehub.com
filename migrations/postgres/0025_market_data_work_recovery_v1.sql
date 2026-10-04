BEGIN;

ALTER TABLE market_data_work_requests
    DROP CONSTRAINT market_data_work_requests_state_check,
    ADD CONSTRAINT market_data_work_requests_state_check CHECK (state IN (
        'queued', 'running', 'pause_requested', 'paused', 'retry_wait',
        'cancel_requested', 'cancelled', 'succeeded', 'failed')),
    ADD COLUMN control_version BIGINT NOT NULL DEFAULT 0 CHECK (control_version >= 0),
    ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    ADD COLUMN next_retry_at TIMESTAMPTZ,
    ADD COLUMN progress_epoch BIGINT NOT NULL DEFAULT 0 CHECK (progress_epoch >= 0);

-- Paused and delayed requests still own their organization's single unfinished slot.
DROP INDEX market_data_work_one_active_per_org;
CREATE UNIQUE INDEX market_data_work_one_active_per_org
    ON market_data_work_requests (organization_id) WHERE state IN (
        'queued', 'running', 'pause_requested', 'paused', 'retry_wait', 'cancel_requested');
CREATE INDEX market_data_work_retry_due
    ON market_data_work_requests (next_retry_at, job_id) WHERE state = 'retry_wait';

COMMIT;
