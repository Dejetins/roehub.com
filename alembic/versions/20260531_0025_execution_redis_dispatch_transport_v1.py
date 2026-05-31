"""Add execution Redis dispatch state."""

from __future__ import annotations

from alembic import op

revision = "20260531_0025"
down_revision = "20260531_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE execution_intents DROP CONSTRAINT IF EXISTS execution_intents_status_chk"
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD CONSTRAINT execution_intents_status_chk
            CHECK (
                status IN (
                    'recorded',
                    'accepted',
                    'rejected',
                    'dispatching',
                    'dispatched',
                    'retry',
                    'quarantined'
                )
            )
        """
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD COLUMN IF NOT EXISTS dispatch_attempt_count INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS dispatch_stream_name TEXT NULL,
            ADD COLUMN IF NOT EXISTS dispatch_redis_message_id TEXT NULL,
            ADD COLUMN IF NOT EXISTS dispatch_last_error TEXT NULL,
            ADD COLUMN IF NOT EXISTS dispatch_updated_at TIMESTAMPTZ NULL
        """
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            DROP CONSTRAINT IF EXISTS execution_intents_dispatch_attempt_count_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD CONSTRAINT execution_intents_dispatch_attempt_count_chk
            CHECK (dispatch_attempt_count >= 0)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_intents_dispatch_retry
            ON execution_intents (status, dispatch_updated_at)
            WHERE status IN ('accepted', 'retry', 'dispatching')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_execution_intents_dispatch_message
            ON execution_intents (dispatch_stream_name, dispatch_redis_message_id)
            WHERE dispatch_redis_message_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_execution_intents_dispatch_message")
    op.execute("DROP INDEX IF EXISTS idx_execution_intents_dispatch_retry")
    op.execute(
        """
        ALTER TABLE execution_intents
            DROP CONSTRAINT IF EXISTS execution_intents_dispatch_attempt_count_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            DROP COLUMN IF EXISTS dispatch_updated_at,
            DROP COLUMN IF EXISTS dispatch_last_error,
            DROP COLUMN IF EXISTS dispatch_redis_message_id,
            DROP COLUMN IF EXISTS dispatch_stream_name,
            DROP COLUMN IF EXISTS dispatch_attempt_count
        """
    )
    op.execute(
        "ALTER TABLE execution_intents DROP CONSTRAINT IF EXISTS execution_intents_status_chk"
    )
    op.execute(
        """
        ALTER TABLE execution_intents
            ADD CONSTRAINT execution_intents_status_chk
            CHECK (status IN ('recorded', 'accepted', 'rejected'))
        """
    )
