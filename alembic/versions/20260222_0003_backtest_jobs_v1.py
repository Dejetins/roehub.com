"""Create Backtest Jobs v1 storage tables and deterministic indexes."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260222_0003"
down_revision = "20260216_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply Backtest Jobs v1 storage schema.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Migration is additive and safe for fresh and already-migrated Strategy v1 schemas.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Creates Backtest jobs tables, constraints, and deterministic indexes.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_jobs (
            job_id UUID PRIMARY KEY,
            user_id UUID NOT NULL,
            mode TEXT NOT NULL,
            state TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            started_at TIMESTAMPTZ NULL,
            finished_at TIMESTAMPTZ NULL,
            cancel_requested_at TIMESTAMPTZ NULL,
            request_json JSONB NOT NULL,
            request_hash TEXT NOT NULL,
            spec_hash TEXT NULL,
            spec_payload_json JSONB NULL,
            engine_params_hash TEXT NOT NULL,
            backtest_runtime_config_hash TEXT NOT NULL,
            stage TEXT NOT NULL,
            processed_units INTEGER NOT NULL DEFAULT 0,
            total_units INTEGER NOT NULL DEFAULT 0,
            progress_updated_at TIMESTAMPTZ NULL,
            locked_by TEXT NULL,
            locked_at TIMESTAMPTZ NULL,
            lease_expires_at TIMESTAMPTZ NULL,
            heartbeat_at TIMESTAMPTZ NULL,
            attempt INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NULL,
            last_error_json JSONB NULL,
            CONSTRAINT backtest_jobs_mode_chk
                CHECK (mode IN ('saved', 'template')),
            CONSTRAINT backtest_jobs_state_chk
                CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
            CONSTRAINT backtest_jobs_stage_chk
                CHECK (stage IN ('stage_a', 'stage_b', 'finalizing')),
            CONSTRAINT backtest_jobs_attempt_chk
                CHECK (attempt >= 0),
            CONSTRAINT backtest_jobs_processed_units_chk
                CHECK (processed_units >= 0),
            CONSTRAINT backtest_jobs_total_units_chk
                CHECK (total_units >= 0),
            CONSTRAINT backtest_jobs_processed_lte_total_chk
                CHECK (total_units = 0 OR processed_units <= total_units),
            CONSTRAINT backtest_jobs_request_json_shape_chk
                CHECK (jsonb_typeof(request_json) = 'object'),
            CONSTRAINT backtest_jobs_spec_payload_shape_chk
                CHECK (
                    spec_payload_json IS NULL
                    OR jsonb_typeof(spec_payload_json) = 'object'
                ),
            CONSTRAINT backtest_jobs_last_error_json_shape_chk
                CHECK (
                    last_error_json IS NULL
                    OR jsonb_typeof(last_error_json) = 'object'
                ),
            CONSTRAINT backtest_jobs_saved_mode_snapshot_chk
                CHECK (
                    (mode = 'saved' AND spec_hash IS NOT NULL AND spec_payload_json IS NOT NULL)
                    OR (mode = 'template' AND spec_hash IS NULL AND spec_payload_json IS NULL)
                ),
            CONSTRAINT backtest_jobs_hashes_sha256_chk
                CHECK (
                    request_hash ~ '^[0-9a-f]{64}$'
                    AND engine_params_hash ~ '^[0-9a-f]{64}$'
                    AND backtest_runtime_config_hash ~ '^[0-9a-f]{64}$'
                    AND (spec_hash IS NULL OR spec_hash ~ '^[0-9a-f]{64}$')
                ),
            CONSTRAINT backtest_jobs_state_timestamps_chk
                CHECK (
                    (state = 'queued' AND started_at IS NULL AND finished_at IS NULL)
                    OR (state = 'running' AND started_at IS NOT NULL AND finished_at IS NULL)
                    OR (state IN ('succeeded', 'failed', 'cancelled') AND finished_at IS NOT NULL)
                ),
            CONSTRAINT backtest_jobs_running_lease_chk
                CHECK (
                    (state = 'running'
                        AND locked_by IS NOT NULL
                        AND locked_at IS NOT NULL
                        AND lease_expires_at IS NOT NULL
                        AND heartbeat_at IS NOT NULL)
                    OR (state <> 'running'
                        AND locked_by IS NULL
                        AND locked_at IS NULL
                        AND lease_expires_at IS NULL
                        AND heartbeat_at IS NULL)
                ),
            CONSTRAINT backtest_jobs_failed_error_payload_chk
                CHECK (
                    (state = 'failed' AND last_error IS NOT NULL AND last_error_json IS NOT NULL)
                    OR (state <> 'failed' AND last_error IS NULL AND last_error_json IS NULL)
                )
        )
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_user_state_created_desc
            ON backtest_jobs (user_id, state, created_at DESC, job_id DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_claim_fifo
            ON backtest_jobs (state, created_at ASC, job_id ASC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_reclaim
            ON backtest_jobs (state, lease_expires_at ASC, created_at ASC, job_id ASC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_user_active
            ON backtest_jobs (user_id)
            WHERE state IN ('queued', 'running')
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_job_top_variants (
            job_id UUID NOT NULL REFERENCES backtest_jobs (job_id) ON DELETE CASCADE,
            rank INTEGER NOT NULL,
            variant_key TEXT NOT NULL,
            indicator_variant_key TEXT NOT NULL,
            variant_index INTEGER NOT NULL,
            total_return_pct DOUBLE PRECISION NOT NULL,
            payload_json JSONB NOT NULL,
            report_table_md TEXT NULL,
            trades_json JSONB NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (job_id, rank),
            CONSTRAINT backtest_job_top_variants_unique_variant
                UNIQUE (job_id, variant_key),
            CONSTRAINT backtest_job_top_variants_rank_chk
                CHECK (rank > 0),
            CONSTRAINT backtest_job_top_variants_variant_index_chk
                CHECK (variant_index >= 0),
            CONSTRAINT backtest_job_top_variants_payload_shape_chk
                CHECK (jsonb_typeof(payload_json) = 'object'),
            CONSTRAINT backtest_job_top_variants_trades_shape_chk
                CHECK (
                    trades_json IS NULL
                    OR jsonb_typeof(trades_json) = 'array'
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_job_top_variants_job_rank
            ON backtest_job_top_variants (job_id, rank)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_job_stage_a_shortlist (
            job_id UUID PRIMARY KEY REFERENCES backtest_jobs (job_id) ON DELETE CASCADE,
            stage_a_indexes_json JSONB NOT NULL,
            stage_a_variants_total INTEGER NOT NULL,
            risk_total INTEGER NOT NULL,
            preselect_used INTEGER NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT backtest_job_stage_a_shortlist_indexes_shape_chk
                CHECK (jsonb_typeof(stage_a_indexes_json) = 'array'),
            CONSTRAINT backtest_job_stage_a_shortlist_variants_total_chk
                CHECK (stage_a_variants_total > 0),
            CONSTRAINT backtest_job_stage_a_shortlist_risk_total_chk
                CHECK (risk_total > 0),
            CONSTRAINT backtest_job_stage_a_shortlist_preselect_used_chk
                CHECK (preselect_used > 0)
        )
        """
    )


def downgrade() -> None:
    """
    Revert Backtest Jobs v1 storage schema.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade drops dependent tables in reverse dependency order.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Drops Backtest jobs result/shortlist/core tables.
    """
    op.execute("DROP TABLE IF EXISTS backtest_job_stage_a_shortlist")
    op.execute("DROP TABLE IF EXISTS backtest_job_top_variants")
    op.execute("DROP TABLE IF EXISTS backtest_jobs")
