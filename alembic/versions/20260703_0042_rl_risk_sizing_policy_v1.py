"""Add RL risk sizing policy storage."""

from __future__ import annotations

from alembic import op

revision = "20260703_0042"
down_revision = "20260703_0041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply additive owner/ticker/market scoped RL risk policy and audit tables.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rl_risk_sizing_policies (
            policy_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL REFERENCES strategy_strategies(strategy_id)
                ON DELETE CASCADE,
            exchange_name TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            sizing_method TEXT NOT NULL,
            base_quote_notional NUMERIC(36, 18) NOT NULL,
            max_position_notional NUMERIC(36, 18) NOT NULL,
            max_daily_loss_notional NUMERIC(36, 18) NOT NULL,
            max_drawdown_pct NUMERIC(18, 12) NOT NULL,
            max_turnover_notional NUMERIC(36, 18) NOT NULL,
            max_exposure_notional NUMERIC(36, 18) NOT NULL,
            min_expected_pnl_pct NUMERIC(18, 12) NOT NULL,
            min_confidence NUMERIC(18, 12) NULL,
            take_profit_pct NUMERIC(18, 12) NULL,
            stop_loss_pct NUMERIC(18, 12) NULL,
            trailing_stop_pct NUMERIC(18, 12) NULL,
            validation_status TEXT NOT NULL,
            validation_reasons TEXT[] NOT NULL,
            synthetic_exit_rules_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_risk_sizing_policies_exchange_chk
                CHECK (char_length(trim(exchange_name)) > 0),
            CONSTRAINT rl_risk_sizing_policies_market_type_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT rl_risk_sizing_policies_symbol_chk
                CHECK (char_length(trim(symbol)) > 0),
            CONSTRAINT rl_risk_sizing_policies_sizing_method_chk
                CHECK (sizing_method IN ('fixed_quote', 'fixed_equity_pct')),
            CONSTRAINT rl_risk_sizing_policies_non_negative_amounts_chk
                CHECK (
                    base_quote_notional >= 0
                    AND max_position_notional >= 0
                    AND max_daily_loss_notional >= 0
                    AND max_turnover_notional >= 0
                    AND max_exposure_notional >= 0
                ),
            CONSTRAINT rl_risk_sizing_policies_ratio_bounds_chk
                CHECK (
                    max_drawdown_pct >= 0
                    AND max_drawdown_pct <= 1
                    AND min_expected_pnl_pct >= 0
                    AND min_expected_pnl_pct <= 1
                    AND (min_confidence IS NULL OR (min_confidence >= 0 AND min_confidence <= 1))
                    AND (take_profit_pct IS NULL OR (take_profit_pct >= 0 AND take_profit_pct <= 1))
                    AND (stop_loss_pct IS NULL OR (stop_loss_pct >= 0 AND stop_loss_pct <= 1))
                    AND (
                        trailing_stop_pct IS NULL
                        OR (trailing_stop_pct >= 0 AND trailing_stop_pct <= 1)
                    )
                ),
            CONSTRAINT rl_risk_sizing_policies_validation_status_chk
                CHECK (validation_status IN ('ready', 'blocked')),
            CONSTRAINT rl_risk_sizing_policies_exit_rules_array_chk
                CHECK (jsonb_typeof(synthetic_exit_rules_json) = 'array')
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_rl_risk_sizing_policies_scope
            ON rl_risk_sizing_policies (
                owner_user_id,
                strategy_id,
                exchange_name,
                market_type,
                symbol
            )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rl_risk_sizing_policies_owner_updated
            ON rl_risk_sizing_policies (owner_user_id, updated_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rl_risk_sizing_policy_audit_events (
            event_id UUID PRIMARY KEY,
            policy_id UUID NOT NULL REFERENCES rl_risk_sizing_policies(policy_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL,
            exchange_name TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            validation_reasons TEXT[] NOT NULL,
            changes_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT rl_risk_sizing_policy_audit_event_type_chk
                CHECK (event_type IN ('upsert')),
            CONSTRAINT rl_risk_sizing_policy_audit_validation_status_chk
                CHECK (validation_status IN ('ready', 'blocked')),
            CONSTRAINT rl_risk_sizing_policy_audit_changes_object_chk
                CHECK (jsonb_typeof(changes_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rl_risk_sizing_policy_audit_policy_created
            ON rl_risk_sizing_policy_audit_events (policy_id, created_at DESC)
        """
    )


def downgrade() -> None:
    """
    Remove additive RL risk policy tables.
    """
    op.execute("DROP TABLE IF EXISTS rl_risk_sizing_policy_audit_events")
    op.execute("DROP TABLE IF EXISTS rl_risk_sizing_policies")
