"""Add paper scenario coverage results for Stage 07."""

from __future__ import annotations

from alembic import op

revision = "20260617_0033"
down_revision = "20260617_0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_paper_scenario_coverage_results (
            coverage_result_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            scenario_matrix_row_id UUID NOT NULL,
            scenario_key TEXT NOT NULL,
            source_job_id UUID NOT NULL,
            source_variant_key TEXT NOT NULL,
            mode TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_sizing TEXT NOT NULL,
            risk_mode TEXT NOT NULL,
            direction TEXT NOT NULL,
            coverage_state TEXT NOT NULL,
            coverage_reason TEXT NOT NULL,
            strategy_id UUID NULL,
            live_profile_id UUID NULL,
            strategy_run_id UUID NULL,
            strategy_signal_id UUID NULL,
            source_event_id UUID NULL,
            intent_id UUID NULL,
            paper_order_id UUID NULL,
            paper_fill_id UUID NULL,
            accounting_id UUID NULL,
            fee_model TEXT NULL,
            funding_model TEXT NULL,
            pnl_complete BOOLEAN NOT NULL DEFAULT FALSE,
            no_exchange_dispatch BOOLEAN NOT NULL DEFAULT TRUE,
            checked_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT strategy_paper_coverage_scenario_key_chk
                CHECK (scenario_key ~ '^[0-9a-f]{64}$'),
            CONSTRAINT strategy_paper_coverage_mode_chk
                CHECK (mode = 'paper'),
            CONSTRAINT strategy_paper_coverage_market_chk
                CHECK (market_type IN ('spot', 'futures')),
            CONSTRAINT strategy_paper_coverage_symbol_chk
                CHECK (symbol = 'BTCUSDT'),
            CONSTRAINT strategy_paper_coverage_sizing_chk
                CHECK (entry_sizing IN ('fixed_quote', 'fixed_equity_pct')),
            CONSTRAINT strategy_paper_coverage_risk_chk
                CHECK (risk_mode = 'single_position_cap'),
            CONSTRAINT strategy_paper_coverage_direction_chk
                CHECK (direction IN ('long', 'short')),
            CONSTRAINT strategy_paper_coverage_state_chk
                CHECK (coverage_state IN ('covered', 'blocked')),
            CONSTRAINT strategy_paper_coverage_reason_chk
                CHECK (char_length(trim(coverage_reason)) > 0),
            CONSTRAINT strategy_paper_coverage_no_dispatch_chk
                CHECK (no_exchange_dispatch IS TRUE),
            CONSTRAINT strategy_paper_coverage_covered_ids_chk
                CHECK (
                    coverage_state <> 'covered'
                    OR (
                        strategy_id IS NOT NULL
                        AND strategy_run_id IS NOT NULL
                        AND strategy_signal_id IS NOT NULL
                        AND source_event_id IS NOT NULL
                        AND intent_id IS NOT NULL
                        AND paper_order_id IS NOT NULL
                        AND paper_fill_id IS NOT NULL
                        AND accounting_id IS NOT NULL
                    )
                ),
            CONSTRAINT strategy_paper_coverage_unique_scenario
                UNIQUE (owner_user_id, scenario_key)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_paper_coverage_owner_checked
            ON strategy_paper_scenario_coverage_results (owner_user_id, checked_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_paper_coverage_run
            ON strategy_paper_scenario_coverage_results
            (owner_user_id, strategy_id, strategy_run_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_strategy_paper_coverage_run")
    op.execute("DROP INDEX IF EXISTS idx_strategy_paper_coverage_owner_checked")
    op.execute("DROP TABLE IF EXISTS strategy_paper_scenario_coverage_results")
