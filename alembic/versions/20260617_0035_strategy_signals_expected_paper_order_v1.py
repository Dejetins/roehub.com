"""Allow bounded expected paper order payloads on strategy signals."""

from __future__ import annotations

from alembic import op

revision = "20260617_0035"
down_revision = "20260617_0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE strategy_signals
            DROP CONSTRAINT IF EXISTS strategy_signals_expected_order_stage05_chk
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_signals
            ADD CONSTRAINT strategy_signals_expected_order_stage07_chk
            CHECK (
                expected_order_json = '{}'::jsonb
                OR (
                    jsonb_typeof(expected_order_json) = 'object'
                    AND expected_order_json ->> 'schema'
                        = 'strategy_signal_expected_order_v1'
                    AND expected_order_json ->> 'mode' = 'paper'
                    AND expected_order_json ->> 'paper_no_exchange_submit' = 'true'
                    AND expected_order_json ? 'quote_notional'
                    AND (
                        expected_order_json ->> 'quote_notional'
                    ) ~ '^([1-9][0-9]*(\\.[0-9]+)?|0\\.[0-9]*[1-9][0-9]*)$'
                    AND (
                        NOT expected_order_json ? 'exchange_connection_id'
                        OR (
                            expected_order_json ->> 'exchange_connection_id'
                        ) ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                    )
                    AND (
                        expected_order_json
                        - 'schema'
                        - 'mode'
                        - 'quote_notional'
                        - 'paper_no_exchange_submit'
                        - 'exchange_connection_id'
                    ) = '{}'::jsonb
                )
            )
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE strategy_signals
            DROP CONSTRAINT IF EXISTS strategy_signals_expected_order_stage07_chk
        """
    )
    op.execute(
        """
        UPDATE strategy_signals
        SET expected_order_json = '{}'::jsonb
        WHERE expected_order_json <> '{}'::jsonb
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_signals
            ADD CONSTRAINT strategy_signals_expected_order_stage05_chk
            CHECK (expected_order_json = '{}'::jsonb)
        """
    )
