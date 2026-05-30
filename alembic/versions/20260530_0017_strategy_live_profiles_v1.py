"""Add live strategy profiles for safe execution readiness."""

from __future__ import annotations

from alembic import op

revision = "20260530_0017"
down_revision = "20260530_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply additive LiveStrategyProfile storage for Stage 03.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_live_profiles (
            profile_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL REFERENCES strategy_strategies(strategy_id)
                ON DELETE CASCADE,
            mode TEXT NOT NULL,
            exchange_connection_id UUID NULL,
            sizing_method TEXT NOT NULL,
            sizing_value NUMERIC(28, 10) NOT NULL DEFAULT 0,
            max_position_notional NUMERIC(28, 10) NULL,
            max_orders_per_run INTEGER NOT NULL DEFAULT 0,
            max_notional_per_run NUMERIC(28, 10) NOT NULL DEFAULT 0,
            readiness_status TEXT NOT NULL,
            readiness_reason TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT strategy_live_profiles_owner_strategy_unique
                UNIQUE (owner_user_id, strategy_id),
            CONSTRAINT strategy_live_profiles_mode_chk
                CHECK (mode IN ('monitor_only', 'paper', 'live')),
            CONSTRAINT strategy_live_profiles_sizing_method_chk
                CHECK (sizing_method IN ('fixed_quote', 'fixed_equity_pct')),
            CONSTRAINT strategy_live_profiles_non_negative_limits_chk
                CHECK (
                    sizing_value >= 0
                    AND (max_position_notional IS NULL OR max_position_notional >= 0)
                    AND max_orders_per_run >= 0
                    AND max_notional_per_run >= 0
                ),
            CONSTRAINT strategy_live_profiles_readiness_status_chk
                CHECK (readiness_status IN ('ready', 'blocked')),
            CONSTRAINT strategy_live_profiles_readiness_reason_chk
                CHECK (char_length(trim(readiness_reason)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_live_profiles_owner_mode
            ON strategy_live_profiles (owner_user_id, mode, updated_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_live_profiles_exchange_connection
            ON strategy_live_profiles (owner_user_id, exchange_connection_id)
            WHERE exchange_connection_id IS NOT NULL
        """
    )


def downgrade() -> None:
    """
    Remove additive LiveStrategyProfile storage.
    """
    op.execute("DROP TABLE IF EXISTS strategy_live_profiles")
