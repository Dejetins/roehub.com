"""Add Backtest AI configurator durable queue and audit storage."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260511_0011"
down_revision = "20260511_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create additive Backtest AI configurator storage tables.

    Existing `backtest_jobs` rows and request-hash semantics are intentionally untouched.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_ai_config_jobs (
            job_id UUID PRIMARY KEY,
            idempotency_key TEXT NULL,
            owner_user_id UUID NOT NULL,
            mode TEXT NOT NULL,
            locale TEXT NOT NULL,
            state TEXT NOT NULL,
            source_page TEXT NOT NULL DEFAULT 'backtests',
            user_prompt_text TEXT NOT NULL,
            user_prompt_hash TEXT NOT NULL,
            current_config_hash TEXT NULL,
            current_config_json JSONB NULL,
            validated_config_json JSONB NULL,
            assistant_message TEXT NULL,
            suggestions_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            validation_errors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            model_id TEXT NULL,
            model_path_hash TEXT NULL,
            system_prompt_version TEXT NOT NULL,
            catalog_snapshot_hash TEXT NOT NULL,
            runtime_defaults_hash TEXT NOT NULL,
            queued_at TIMESTAMPTZ NOT NULL,
            started_at TIMESTAMPTZ NULL,
            finished_at TIMESTAMPTZ NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            locked_by TEXT NULL,
            locked_at TIMESTAMPTZ NULL,
            lease_expires_at TIMESTAMPTZ NULL,
            heartbeat_at TIMESTAMPTZ NULL,
            attempt INTEGER NOT NULL DEFAULT 0,
            quota_charged BOOLEAN NOT NULL DEFAULT false,
            applied_at TIMESTAMPTZ NULL,
            user_feedback_json JSONB NULL,
            last_error TEXT NULL,
            last_error_json JSONB NULL,
            CONSTRAINT backtest_ai_config_jobs_mode_chk
                CHECK (mode IN ('create', 'edit', 'explain', 'repair', 'suggest_safer')),
            CONSTRAINT backtest_ai_config_jobs_locale_chk
                CHECK (locale IN ('ru', 'en')),
            CONSTRAINT backtest_ai_config_jobs_state_chk
                CHECK (
                    state IN (
                        'queued',
                        'running',
                        'repairing',
                        'ready',
                        'needs_clarification',
                        'blocked_by_policy',
                        'input_too_large',
                        'security_review',
                        'failed',
                        'cancelled'
                    )
                ),
            CONSTRAINT backtest_ai_config_jobs_source_page_chk
                CHECK (btrim(source_page) <> ''),
            CONSTRAINT backtest_ai_config_jobs_idempotency_key_chk
                CHECK (
                    idempotency_key IS NULL
                    OR (btrim(idempotency_key) <> '' AND length(idempotency_key) <= 128)
                ),
            CONSTRAINT backtest_ai_config_jobs_prompt_chk
                CHECK (btrim(user_prompt_text) <> ''),
            CONSTRAINT backtest_ai_config_jobs_attempt_chk
                CHECK (attempt >= 0),
            CONSTRAINT backtest_ai_config_jobs_json_shapes_chk
                CHECK (
                    (current_config_json IS NULL OR jsonb_typeof(current_config_json) = 'object')
                    AND (
                        validated_config_json IS NULL
                        OR jsonb_typeof(validated_config_json) = 'object'
                    )
                    AND jsonb_typeof(suggestions_json) = 'array'
                    AND jsonb_typeof(validation_errors_json) = 'array'
                    AND (
                        user_feedback_json IS NULL
                        OR jsonb_typeof(user_feedback_json) = 'object'
                    )
                    AND (
                        last_error_json IS NULL
                        OR jsonb_typeof(last_error_json) = 'object'
                    )
                ),
            CONSTRAINT backtest_ai_config_jobs_hashes_chk
                CHECK (
                    user_prompt_hash ~ '^[0-9a-f]{64}$'
                    AND catalog_snapshot_hash ~ '^[0-9a-f]{64}$'
                    AND runtime_defaults_hash ~ '^[0-9a-f]{64}$'
                    AND (current_config_hash IS NULL OR current_config_hash ~ '^[0-9a-f]{64}$')
                    AND (model_path_hash IS NULL OR model_path_hash ~ '^[0-9a-f]{64}$')
                ),
            CONSTRAINT backtest_ai_config_jobs_terminal_ts_chk
                CHECK (
                    (
                        state IN (
                            'ready',
                            'needs_clarification',
                            'blocked_by_policy',
                            'input_too_large',
                            'security_review',
                            'failed',
                            'cancelled'
                        )
                        AND finished_at IS NOT NULL
                    )
                    OR (
                        state IN ('queued', 'running', 'repairing')
                        AND finished_at IS NULL
                    )
                ),
            CONSTRAINT backtest_ai_config_jobs_running_lease_chk
                CHECK (
                    (
                        state IN ('running', 'repairing')
                        AND locked_by IS NOT NULL
                        AND locked_at IS NOT NULL
                        AND lease_expires_at IS NOT NULL
                        AND heartbeat_at IS NOT NULL
                    )
                    OR (
                        state NOT IN ('running', 'repairing')
                        AND locked_by IS NULL
                        AND locked_at IS NULL
                        AND lease_expires_at IS NULL
                        AND heartbeat_at IS NULL
                    )
                )
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_backtest_ai_config_jobs_owner_idempotency_key
            ON backtest_ai_config_jobs (owner_user_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_jobs_state_queued
            ON backtest_ai_config_jobs (state, queued_at ASC, job_id ASC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_jobs_owner_queued_desc
            ON backtest_ai_config_jobs (owner_user_id, queued_at DESC, job_id DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_jobs_lease
            ON backtest_ai_config_jobs (lease_expires_at ASC, queued_at ASC, job_id ASC)
            WHERE state IN ('queued', 'running', 'repairing')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_jobs_finished
            ON backtest_ai_config_jobs (finished_at DESC, job_id DESC)
            WHERE state IN (
                'ready',
                'needs_clarification',
                'blocked_by_policy',
                'failed'
            )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_ai_config_events (
            event_id UUID PRIMARY KEY,
            event_seq BIGINT GENERATED BY DEFAULT AS IDENTITY,
            job_id UUID NOT NULL REFERENCES backtest_ai_config_jobs (job_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            event_name TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT backtest_ai_config_events_name_chk
                CHECK (
                    event_name IN (
                        'queued',
                        'preparing_catalog',
                        'assembling_prompt',
                        'generating',
                        'validating_json',
                        'validating_business',
                        'repairing',
                        'ready',
                        'needs_clarification',
                        'blocked_by_policy',
                        'input_too_large',
                        'security_review',
                        'failed',
                        'heartbeat'
                    )
                ),
            CONSTRAINT backtest_ai_config_events_payload_shape_chk
                CHECK (jsonb_typeof(payload_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_events_job_seq
            ON backtest_ai_config_events (job_id, event_seq ASC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_events_owner_created
            ON backtest_ai_config_events (owner_user_id, created_at DESC, event_seq DESC)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_ai_config_llm_attempts (
            attempt_id UUID PRIMARY KEY,
            job_id UUID NOT NULL REFERENCES backtest_ai_config_jobs (job_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            attempt_no INTEGER NOT NULL,
            attempt_kind TEXT NOT NULL,
            prompt_profile TEXT NOT NULL,
            user_prompt_text TEXT NOT NULL,
            catalog_subset_json JSONB NOT NULL,
            raw_model_response TEXT NULL,
            parsed_json_draft JSONB NULL,
            validation_errors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            input_tokens_estimate INTEGER NULL,
            output_tokens_estimate INTEGER NULL,
            latency_ms INTEGER NULL,
            finish_reason TEXT NULL,
            success BOOLEAN NOT NULL,
            failure_reason TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT backtest_ai_config_llm_attempts_kind_chk
                CHECK (attempt_kind IN ('generate', 'repair')),
            CONSTRAINT backtest_ai_config_llm_attempts_attempt_no_chk
                CHECK (attempt_no > 0),
            CONSTRAINT backtest_ai_config_llm_attempts_catalog_shape_chk
                CHECK (jsonb_typeof(catalog_subset_json) = 'object'),
            CONSTRAINT backtest_ai_config_llm_attempts_draft_shape_chk
                CHECK (
                    parsed_json_draft IS NULL
                    OR jsonb_typeof(parsed_json_draft) = 'object'
                ),
            CONSTRAINT backtest_ai_config_llm_attempts_errors_shape_chk
                CHECK (jsonb_typeof(validation_errors_json) = 'array'),
            CONSTRAINT backtest_ai_config_llm_attempts_token_counts_chk
                CHECK (
                    (input_tokens_estimate IS NULL OR input_tokens_estimate >= 0)
                    AND (output_tokens_estimate IS NULL OR output_tokens_estimate >= 0)
                    AND (latency_ms IS NULL OR latency_ms >= 0)
                )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_llm_attempts_job_attempt
            ON backtest_ai_config_llm_attempts (job_id, attempt_no ASC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_config_llm_attempts_owner_created
            ON backtest_ai_config_llm_attempts (owner_user_id, created_at DESC)
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_ai_quota_events (
            quota_event_id UUID PRIMARY KEY,
            job_id UUID NULL REFERENCES backtest_ai_config_jobs (job_id)
                ON DELETE SET NULL,
            owner_user_id UUID NOT NULL,
            paid_level TEXT NOT NULL,
            quota_action TEXT NOT NULL,
            units INTEGER NOT NULL DEFAULT 1,
            idempotency_key TEXT NULL,
            reason TEXT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            occurred_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT backtest_ai_quota_events_paid_level_chk
                CHECK (paid_level IN ('base', 'free', 'pro', 'ultra')),
            CONSTRAINT backtest_ai_quota_events_action_chk
                CHECK (
                    quota_action IN (
                        'request_charged',
                        'quota_rejected',
                        'capacity_rejected'
                    )
                ),
            CONSTRAINT backtest_ai_quota_events_units_chk
                CHECK (units >= 0),
            CONSTRAINT backtest_ai_quota_events_metadata_shape_chk
                CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_quota_events_owner_occurred
            ON backtest_ai_quota_events (owner_user_id, occurred_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_ai_quota_events_job
            ON backtest_ai_quota_events (job_id)
            WHERE job_id IS NOT NULL
        """
    )


def downgrade() -> None:
    """
    Drop only additive Backtest AI configurator storage.
    """
    op.execute("DROP TABLE IF EXISTS backtest_ai_quota_events")
    op.execute("DROP TABLE IF EXISTS backtest_ai_config_llm_attempts")
    op.execute("DROP TABLE IF EXISTS backtest_ai_config_events")
    op.execute("DROP TABLE IF EXISTS backtest_ai_config_jobs")
