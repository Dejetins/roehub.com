"""Add strategy position ownership lock."""

from __future__ import annotations

from alembic import op

revision = "20260531_0021"
down_revision = "20260531_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_position_ownership (
            ownership_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            exchange_connection_id UUID NOT NULL,
            strategy_id UUID NOT NULL,
            live_profile_id UUID NULL,
            strategy_run_id UUID NOT NULL,
            market_type TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            position_mode TEXT NOT NULL DEFAULT 'net',
            state TEXT NOT NULL,
            acquired_at TIMESTAMPTZ NOT NULL,
            released_at TIMESTAMPTZ NULL,
            expires_at TIMESTAMPTZ NULL,
            reason TEXT NOT NULL,
            CONSTRAINT strategy_position_ownership_market_type_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT strategy_position_ownership_position_mode_chk
                CHECK (position_mode IN ('net')),
            CONSTRAINT strategy_position_ownership_state_chk
                CHECK (
                    state IN (
                        'reserved',
                        'active',
                        'releasing',
                        'released',
                        'stale_requires_repair'
                    )
                ),
            CONSTRAINT strategy_position_ownership_instrument_chk
                CHECK (char_length(trim(instrument_key)) > 0),
            CONSTRAINT strategy_position_ownership_reason_chk
                CHECK (char_length(trim(reason)) > 0),
            CONSTRAINT strategy_position_ownership_released_at_chk
                CHECK (
                    (state = 'released' AND released_at IS NOT NULL)
                    OR (state <> 'released')
                )
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS strategy_position_ownership_one_blocking
            ON strategy_position_ownership (
                owner_user_id,
                exchange_connection_id,
                market_type,
                instrument_key
            )
            WHERE state IN (
                'reserved',
                'active',
                'releasing',
                'stale_requires_repair'
            )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_position_ownership_run
            ON strategy_position_ownership (owner_user_id, strategy_run_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_position_ownership_stale
            ON strategy_position_ownership (state, acquired_at)
            WHERE state = 'stale_requires_repair'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_strategy_position_ownership_stale")
    op.execute("DROP INDEX IF EXISTS idx_strategy_position_ownership_run")
    op.execute("DROP INDEX IF EXISTS strategy_position_ownership_one_blocking")
    op.execute("DROP TABLE IF EXISTS strategy_position_ownership")
