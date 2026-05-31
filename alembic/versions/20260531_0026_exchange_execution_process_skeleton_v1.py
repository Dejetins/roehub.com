"""Add exchange-execution process heartbeat and observation tables."""

from __future__ import annotations

from alembic import op

revision = "20260531_0026"
down_revision = "20260531_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_execution_process_heartbeats (
            service_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            status_reason TEXT NOT NULL,
            adapter_mode TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            heartbeat_at TIMESTAMPTZ NOT NULL,
            request_stream TEXT NOT NULL,
            consumer_group TEXT NOT NULL,
            consumer_name TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT exchange_execution_process_heartbeats_status_chk
                CHECK (status IN ('ready', 'degraded', 'not_ready')),
            CONSTRAINT exchange_execution_process_heartbeats_adapter_mode_chk
                CHECK (adapter_mode IN ('disabled'))
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_execution_request_observations (
            observation_id UUID PRIMARY KEY,
            service_id TEXT NOT NULL,
            intent_id UUID NULL,
            stream_name TEXT NOT NULL,
            redis_message_id TEXT NOT NULL,
            status TEXT NOT NULL,
            status_reason TEXT NOT NULL,
            adapter_mode TEXT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            CONSTRAINT exchange_execution_request_observations_status_chk
                CHECK (status IN ('adapter_disabled', 'quarantined', 'skipped')),
            CONSTRAINT exchange_execution_request_observations_adapter_mode_chk
                CHECK (adapter_mode IN ('disabled'))
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_exchange_execution_observation_message_status
            ON exchange_execution_request_observations (stream_name, redis_message_id, status)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_exchange_execution_observations_intent
            ON exchange_execution_request_observations (intent_id, observed_at)
            WHERE intent_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_exchange_execution_observations_status
            ON exchange_execution_request_observations (status, observed_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_exchange_execution_observations_status")
    op.execute("DROP INDEX IF EXISTS idx_exchange_execution_observations_intent")
    op.execute("DROP INDEX IF EXISTS ux_exchange_execution_observation_message_status")
    op.execute("DROP TABLE IF EXISTS exchange_execution_request_observations")
    op.execute("DROP TABLE IF EXISTS exchange_execution_process_heartbeats")
