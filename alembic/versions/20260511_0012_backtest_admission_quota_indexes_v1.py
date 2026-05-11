"""Add quota-read indexes for backtest admission control."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260511_0012"
down_revision = "20260511_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add deterministic indexes for v1 admission quota reads.
    """
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_user_created_desc
            ON backtest_jobs (user_id, created_at DESC, job_id DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_lazy_trades_materializations_owner_active
            ON backtest_lazy_trades_materializations
                (owner_user_id, status, created_at DESC, task_id DESC)
            WHERE status IN ('queued', 'running')
        """
    )


def downgrade() -> None:
    """
    Drop only the additive quota-read indexes.
    """
    op.execute(
        "DROP INDEX IF EXISTS idx_backtest_lazy_trades_materializations_owner_active"
    )
    op.execute("DROP INDEX IF EXISTS idx_backtest_jobs_user_created_desc")
