"""Add lazy trades materialization queue for production-safe result reads."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260511_0010"
down_revision = "20260418_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create the owner-scoped lazy trades materialization queue.

    The table is additive and lets the API persist/cache-miss work without running
    expensive detail recompute in the request process. Rollback is straightforward:
    drop only this table and its indexes.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_lazy_trades_materializations (
            task_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            job_id UUID NOT NULL REFERENCES backtest_jobs (job_id) ON DELETE CASCADE,
            public_variant_key TEXT NOT NULL,
            variant_hash TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            engine_params_hash TEXT NOT NULL,
            artifact_manifest_hash TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            status TEXT NOT NULL,
            priority_class TEXT NOT NULL DEFAULT 'interactive',
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            started_at TIMESTAMPTZ NULL,
            finished_at TIMESTAMPTZ NULL,
            locked_by TEXT NULL,
            locked_at TIMESTAMPTZ NULL,
            lease_expires_at TIMESTAMPTZ NULL,
            heartbeat_at TIMESTAMPTZ NULL,
            attempt INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NULL,
            last_error_json JSONB NULL,
            cache_status TEXT NOT NULL,
            cache_path TEXT NULL,
            ttl_seconds INTEGER NOT NULL,
            CONSTRAINT backtest_lazy_trades_materializations_status_chk
                CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
            CONSTRAINT backtest_lazy_trades_materializations_attempt_chk
                CHECK (attempt >= 0),
            CONSTRAINT backtest_lazy_trades_materializations_ttl_chk
                CHECK (ttl_seconds > 0),
            CONSTRAINT backtest_lazy_trades_materializations_public_key_chk
                CHECK (btrim(public_variant_key) <> '' AND length(public_variant_key) <= 256),
            CONSTRAINT backtest_lazy_trades_materializations_priority_chk
                CHECK (btrim(priority_class) <> ''),
            CONSTRAINT backtest_lazy_trades_materializations_hashes_chk
                CHECK (
                    variant_hash ~ '^[0-9a-f]{64}$'
                    AND request_hash ~ '^[0-9a-f]{64}$'
                    AND engine_params_hash ~ '^[0-9a-f]{64}$'
                    AND artifact_manifest_hash ~ '^[0-9a-f]{64}$'
                    AND cache_key ~ '^[0-9a-f]{64}$'
                ),
            CONSTRAINT backtest_lazy_trades_materializations_error_json_chk
                CHECK (
                    last_error_json IS NULL
                    OR jsonb_typeof(last_error_json) = 'object'
                ),
            CONSTRAINT backtest_lazy_trades_materializations_terminal_ts_chk
                CHECK (
                    (status IN ('completed', 'failed', 'cancelled') AND finished_at IS NOT NULL)
                    OR (status IN ('queued', 'running') AND finished_at IS NULL)
                ),
            CONSTRAINT backtest_lazy_trades_materializations_running_lease_chk
                CHECK (
                    (
                        status = 'running'
                        AND started_at IS NOT NULL
                        AND locked_by IS NOT NULL
                        AND locked_at IS NOT NULL
                        AND lease_expires_at IS NOT NULL
                        AND heartbeat_at IS NOT NULL
                    )
                    OR (
                        status <> 'running'
                        AND locked_by IS NULL
                        AND locked_at IS NULL
                        AND lease_expires_at IS NULL
                        AND heartbeat_at IS NULL
                    )
                ),
            CONSTRAINT backtest_lazy_trades_materializations_identity_unique
                UNIQUE (owner_user_id, job_id, public_variant_key, cache_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_lazy_trades_materializations_pickup
            ON backtest_lazy_trades_materializations
                (status, priority_class, created_at ASC, task_id ASC)
            WHERE status IN ('queued', 'running')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_lazy_trades_materializations_owner_created
            ON backtest_lazy_trades_materializations
                (owner_user_id, created_at DESC, task_id DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_lazy_trades_materializations_job_variant
            ON backtest_lazy_trades_materializations
                (job_id, public_variant_key, status)
        """
    )


def downgrade() -> None:
    """
    Drop only the additive lazy trades materialization queue.
    """
    op.execute("DROP TABLE IF EXISTS backtest_lazy_trades_materializations")
