"""Add manual source-event link to paper orders."""

from __future__ import annotations

from alembic import op

revision = "20260618_0036"
down_revision = "20260617_0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE paper_orders
            ADD COLUMN IF NOT EXISTS source_event_id UUID NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_orders_source_event_unique
            ON paper_orders (source_event_id)
            WHERE source_event_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_paper_orders_source_event_unique")
    op.execute(
        """
        ALTER TABLE paper_orders
            DROP COLUMN IF EXISTS source_event_id
        """
    )
