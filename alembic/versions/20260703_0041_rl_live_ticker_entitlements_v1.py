"""Add RL live ticker entitlement storage."""

from __future__ import annotations

from alembic import op

revision = "20260703_0041"
down_revision = "20260702_0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply additive RL live ticker entitlement tables.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rl_live_ticker_entitlement_overrides (
            owner_user_id UUID PRIMARY KEY,
            live_slots_allowed INTEGER NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            reason TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_live_ticker_entitlement_overrides_slots_chk
                CHECK (live_slots_allowed >= 0),
            CONSTRAINT rl_live_ticker_entitlement_overrides_reason_chk
                CHECK (char_length(trim(reason)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rl_live_ticker_activations (
            activation_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL REFERENCES strategy_strategies(strategy_id)
                ON DELETE CASCADE,
            live_profile_id UUID NOT NULL REFERENCES strategy_live_profiles(profile_id)
                ON DELETE CASCADE,
            exchange_name TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            activated_at TIMESTAMPTZ NOT NULL,
            deactivated_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_live_ticker_activations_exchange_chk
                CHECK (char_length(trim(exchange_name)) > 0),
            CONSTRAINT rl_live_ticker_activations_market_type_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT rl_live_ticker_activations_symbol_chk
                CHECK (char_length(trim(symbol)) > 0),
            CONSTRAINT rl_live_ticker_activations_mode_chk
                CHECK (mode = 'live'),
            CONSTRAINT rl_live_ticker_activations_deactivation_chk
                CHECK (
                    (active = TRUE AND deactivated_at IS NULL)
                    OR (active = FALSE AND deactivated_at IS NOT NULL)
                )
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            uq_rl_live_ticker_activations_active_owner_ticker
            ON rl_live_ticker_activations (
                owner_user_id,
                exchange_name,
                market_type,
                symbol
            )
            WHERE active
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rl_live_ticker_activations_owner_active
            ON rl_live_ticker_activations (owner_user_id, active, updated_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rl_live_ticker_activations_profile_active
            ON rl_live_ticker_activations (owner_user_id, strategy_id, active)
        """
    )


def downgrade() -> None:
    """
    Remove additive RL live ticker entitlement tables.
    """
    op.execute("DROP TABLE IF EXISTS rl_live_ticker_activations")
    op.execute("DROP TABLE IF EXISTS rl_live_ticker_entitlement_overrides")
