"""Allow durable mainnet hard-block order guard rows."""

from __future__ import annotations

from alembic import op

revision = "20260531_0028"
down_revision = "20260531_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE execution_orders
        DROP CONSTRAINT IF EXISTS execution_orders_environment_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_orders
        ADD CONSTRAINT execution_orders_environment_chk
            CHECK (
                environment = 'testnet'
                OR (
                    environment = 'mainnet'
                    AND status = 'guard_rejected'
                    AND status_reason = 'mainnet_hard_block'
                    AND exchange_order_id IS NULL
                    AND submitted_at IS NULL
                    AND cancel_requested_at IS NULL
                    AND cancelled_at IS NULL
                )
            )
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE execution_orders
        DROP CONSTRAINT IF EXISTS execution_orders_environment_chk
        """
    )
    op.execute(
        """
        ALTER TABLE execution_orders
        ADD CONSTRAINT execution_orders_environment_chk
            CHECK (environment IN ('testnet'))
        """
    )
