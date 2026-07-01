"""Add Stage 13 execution notification event types."""

from __future__ import annotations

from alembic import op

revision = "20260702_0039"
down_revision = "20260629_0038"
branch_labels = None
depends_on = None

_STAGE13_EVENT_TYPES = (
    "producer_rejected",
    "producer_signal_rejected",
    "producer_order_rejected",
    "producer_fill",
    "producer_manual_exit",
    "producer_unknown",
    "producer_reconciliation_pending",
    "producer_kill_switch",
    "producer_terminal",
    "producer_strategy_stopped",
    "producer_strategy_restarted",
    "producer_soak_failed",
    "producer_soak_succeeded",
    "producer_resource_threshold_breached",
)

_STAGE16_EVENT_TYPES = (
    "producer_rejected",
    "producer_fill",
    "producer_unknown",
    "producer_kill_switch",
    "producer_terminal",
)


def upgrade() -> None:
    _replace_event_type_constraint(event_types=_STAGE13_EVENT_TYPES)


def downgrade() -> None:
    _replace_event_type_constraint(event_types=_STAGE16_EVENT_TYPES)


def _replace_event_type_constraint(*, event_types: tuple[str, ...]) -> None:
    values = ",\n                    ".join(f"'{event_type}'" for event_type in event_types)
    op.execute(
        """
        ALTER TABLE execution_notification_outbox
        DROP CONSTRAINT IF EXISTS execution_notification_event_type_chk
        """
    )
    op.execute(
        f"""
        ALTER TABLE execution_notification_outbox
        ADD CONSTRAINT execution_notification_event_type_chk CHECK (
            event_type IN (
                    {values}
            )
        )
        """
    )
