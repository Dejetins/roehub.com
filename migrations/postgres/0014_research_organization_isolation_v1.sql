BEGIN;

-- Stage 09 is greenfield-only. Research ownership is never inferred from
-- current users or historical rows, so every mutable research table must be
-- empty before organization columns and composite foreign keys are installed.
DO $$
DECLARE
    populated_table TEXT;
    has_rows BOOLEAN;
BEGIN
    FOREACH populated_table IN ARRAY ARRAY[
        'backtest_jobs',
        'backtest_job_top_variants',
        'backtest_job_stage_a_shortlist',
        'backtest_lazy_trades_materializations'
    ]
    LOOP
        IF to_regclass('public.' || populated_table) IS NULL THEN
            RAISE EXCEPTION 'required greenfield research table is missing: %',
                populated_table;
        END IF;
        EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I)', populated_table)
            INTO STRICT has_rows;
        IF has_rows THEN
            RAISE EXCEPTION
                'research organization schema requires empty greenfield table: %',
                populated_table;
        END IF;
    END LOOP;
END
$$;

ALTER TABLE backtest_jobs
    ADD CONSTRAINT backtest_jobs_org_user_id_unique
        UNIQUE (organization_id, user_id, job_id);

ALTER TABLE backtest_job_top_variants
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT backtest_top_variants_org_job_fk
        FOREIGN KEY (organization_id, job_id)
        REFERENCES backtest_jobs(organization_id, job_id) ON DELETE CASCADE;

ALTER TABLE backtest_job_stage_a_shortlist
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT backtest_stage_a_shortlist_org_job_fk
        FOREIGN KEY (organization_id, job_id)
        REFERENCES backtest_jobs(organization_id, job_id) ON DELETE CASCADE;

ALTER TABLE backtest_lazy_trades_materializations
    ADD COLUMN organization_id UUID NOT NULL,
    ADD CONSTRAINT backtest_lazy_materializations_org_member_fk
        FOREIGN KEY (organization_id, owner_user_id)
        REFERENCES identity_memberships(organization_id, user_id) ON DELETE RESTRICT,
    ADD CONSTRAINT backtest_lazy_materializations_org_job_fk
        FOREIGN KEY (organization_id, owner_user_id, job_id)
        REFERENCES backtest_jobs(organization_id, user_id, job_id) ON DELETE CASCADE,
    DROP CONSTRAINT backtest_lazy_trades_materializations_identity_unique,
    ADD CONSTRAINT backtest_lazy_materializations_org_identity_unique
        UNIQUE (
            organization_id,
            owner_user_id,
            job_id,
            public_variant_key,
            cache_key
        );

CREATE INDEX idx_backtest_jobs_org_user_state_created_desc
    ON backtest_jobs (organization_id, user_id, state, created_at DESC, job_id DESC);

CREATE INDEX idx_backtest_jobs_org_user_created_desc
    ON backtest_jobs (organization_id, user_id, created_at DESC, job_id DESC);

CREATE INDEX idx_backtest_jobs_org_user_execution_created_desc
    ON backtest_jobs (
        organization_id,
        user_id,
        execution_mode,
        created_at DESC,
        job_id DESC
    );

CREATE INDEX idx_backtest_jobs_org_user_idempotency_created_desc
    ON backtest_jobs (
        organization_id,
        user_id,
        (request_json #>> '{idempotency,key_hash}'),
        created_at DESC,
        job_id DESC
    )
    WHERE request_json #>> '{idempotency,key_hash}' IS NOT NULL;

CREATE INDEX idx_backtest_lazy_materializations_org_owner_created
    ON backtest_lazy_trades_materializations (
        organization_id,
        owner_user_id,
        created_at DESC,
        task_id DESC
    );

COMMIT;
