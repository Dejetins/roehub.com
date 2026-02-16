"""Add metadata_json to strategy_runs for deterministic run warmup traceability."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260216_0002"
down_revision = "20260215_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add immutable JSON metadata payload column for strategy runs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Existing rows should receive deterministic empty-object default metadata.
    Raises:
        Exception: Database execution errors from Alembic runtime.
    Side Effects:
        Alters `strategy_runs` table schema.
    """
    op.execute(
        """
        ALTER TABLE strategy_runs
        ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
        """
    )


def downgrade() -> None:
    """
    Remove run metadata column from strategy_runs table.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade is used only for local/test rollback flows.
    Raises:
        Exception: Database execution errors from Alembic runtime.
    Side Effects:
        Alters `strategy_runs` table schema.
    """
    op.execute("ALTER TABLE strategy_runs DROP COLUMN IF EXISTS metadata_json")
