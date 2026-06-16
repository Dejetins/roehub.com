"""Add Strategy variant scenario matrix rows for Stage 03."""

from __future__ import annotations

from alembic import op

revision = "20260617_0032"
down_revision = "20260617_0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_variant_scenario_matrix_rows (
            scenario_matrix_row_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            source_job_id UUID NOT NULL REFERENCES backtest_jobs(job_id)
                ON DELETE CASCADE,
            source_variant_key TEXT NOT NULL,
            variant_hash TEXT NOT NULL,
            strategy_spec_hash TEXT NOT NULL,
            scenario_key TEXT NOT NULL,
            mode TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_sizing TEXT NOT NULL,
            risk_mode TEXT NOT NULL,
            direction TEXT NOT NULL,
            backtest_risk_mode TEXT NOT NULL,
            backtest_direction_mode TEXT NOT NULL,
            scenario_state TEXT NOT NULL,
            scenario_reason_codes_json JSONB NOT NULL,
            order_capability TEXT NOT NULL,
            order_capability_reason_codes_json JSONB NOT NULL,
            compatibility_check_id UUID NULL
                REFERENCES strategy_variant_compatibility_checks(compatibility_check_id)
                ON DELETE SET NULL,
            market_data_requirement_id UUID NULL
                REFERENCES market_data_subscription_requirements(market_data_requirement_id)
                ON DELETE SET NULL,
            compatibility_state TEXT NOT NULL,
            compatibility_reason_codes_json JSONB NOT NULL,
            market_data_state TEXT NOT NULL,
            market_data_reason_codes_json JSONB NOT NULL,
            checked_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT strategy_variant_scenario_matrix_variant_hash_chk
                CHECK (variant_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT strategy_variant_scenario_matrix_spec_hash_chk
                CHECK (strategy_spec_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT strategy_variant_scenario_matrix_key_chk
                CHECK (scenario_key ~ '^[0-9a-f]{64}$'),
            CONSTRAINT strategy_variant_scenario_matrix_mode_chk
                CHECK (mode IN ('paper', 'testnet')),
            CONSTRAINT strategy_variant_scenario_matrix_market_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT strategy_variant_scenario_matrix_symbol_chk
                CHECK (symbol = 'BTCUSDT'),
            CONSTRAINT strategy_variant_scenario_matrix_sizing_chk
                CHECK (entry_sizing IN ('fixed_quote', 'fixed_equity_pct')),
            CONSTRAINT strategy_variant_scenario_matrix_risk_chk
                CHECK (risk_mode IN ('single_position_cap')),
            CONSTRAINT strategy_variant_scenario_matrix_direction_chk
                CHECK (direction IN ('long', 'short')),
            CONSTRAINT strategy_variant_scenario_matrix_state_chk
                CHECK (scenario_state IN ('launchable', 'degraded', 'blocked')),
            CONSTRAINT strategy_variant_scenario_matrix_capability_chk
                CHECK (order_capability IN ('paper_only', 'real_order_capable', 'unsupported')),
            CONSTRAINT strategy_variant_scenario_matrix_reason_shape_chk
                CHECK (jsonb_typeof(scenario_reason_codes_json) = 'array'),
            CONSTRAINT strategy_variant_scenario_matrix_capability_reason_shape_chk
                CHECK (jsonb_typeof(order_capability_reason_codes_json) = 'array'),
            CONSTRAINT strategy_variant_scenario_matrix_compat_reason_shape_chk
                CHECK (jsonb_typeof(compatibility_reason_codes_json) = 'array'),
            CONSTRAINT strategy_variant_scenario_matrix_market_reason_shape_chk
                CHECK (jsonb_typeof(market_data_reason_codes_json) = 'array'),
            CONSTRAINT strategy_variant_scenario_matrix_unique_row
                UNIQUE (owner_user_id, source_job_id, source_variant_key, scenario_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_variant_scenario_matrix_owner_checked
            ON strategy_variant_scenario_matrix_rows (owner_user_id, checked_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_variant_scenario_matrix_variant
            ON strategy_variant_scenario_matrix_rows
            (owner_user_id, source_job_id, source_variant_key, mode, market_type, direction)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS strategy_variant_scenario_matrix_rows")
