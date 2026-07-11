"""Index user notification delivery counters."""

from __future__ import annotations

from alembic import op

revision = "20260711_0043"
down_revision = "20260703_0042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notification_deliveries_sent_route
            ON notification_deliveries (route_id, sent_at DESC)
            WHERE status = 'sent'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_notification_deliveries_sent_route")
