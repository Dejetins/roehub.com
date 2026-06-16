"""Allow explicit testnet strategy profile mode."""

from __future__ import annotations

from alembic import op

revision = "20260617_0031"
down_revision = "20260603_0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Expand Strategy mode constraints for paper/testnet launch UI.
    """
    op.execute(
        """
        ALTER TABLE strategy_live_profiles
            DROP CONSTRAINT IF EXISTS strategy_live_profiles_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_live_profiles
            ADD CONSTRAINT strategy_live_profiles_mode_chk
            CHECK (mode IN ('monitor_only', 'paper', 'live', 'testnet'))
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_signals
            DROP CONSTRAINT IF EXISTS strategy_signals_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_signals
            ADD CONSTRAINT strategy_signals_mode_chk
            CHECK (mode IN ('monitor_only', 'paper', 'live', 'testnet'))
        """
    )


def downgrade() -> None:
    """
    Restore the pre-Stage-02 Strategy mode constraints.
    """
    op.execute(
        """
        ALTER TABLE strategy_live_profiles
            DROP CONSTRAINT IF EXISTS strategy_live_profiles_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_live_profiles
            ADD CONSTRAINT strategy_live_profiles_mode_chk
            CHECK (mode IN ('monitor_only', 'paper', 'live'))
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_signals
            DROP CONSTRAINT IF EXISTS strategy_signals_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE strategy_signals
            ADD CONSTRAINT strategy_signals_mode_chk
            CHECK (mode IN ('monitor_only', 'paper', 'live'))
        """
    )
