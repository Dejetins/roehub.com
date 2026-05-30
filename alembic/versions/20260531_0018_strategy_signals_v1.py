"""Add StrategySignal journal for live evaluator Stage 05."""

from __future__ import annotations

from alembic import op

revision = "20260531_0018"
down_revision = "20260530_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply additive StrategySignal journal storage.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_signals (
            signal_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NOT NULL REFERENCES strategy_strategies(strategy_id)
                ON DELETE CASCADE,
            strategy_run_id UUID NOT NULL REFERENCES strategy_runs(run_id)
                ON DELETE CASCADE,
            live_profile_id UUID NULL REFERENCES strategy_live_profiles(profile_id)
                ON DELETE SET NULL,
            mode TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            bar_ts_open TIMESTAMPTZ NOT NULL,
            bar_ts_close TIMESTAMPTZ NOT NULL,
            signal_action TEXT NOT NULL,
            side TEXT NULL,
            outcome TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            reference_price NUMERIC(28, 10) NOT NULL,
            confidence NUMERIC(8, 7) NULL,
            expected_order_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_message_id TEXT NOT NULL,
            evaluator_version TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT strategy_signals_mode_chk
                CHECK (mode IN ('monitor_only', 'paper', 'live')),
            CONSTRAINT strategy_signals_action_chk
                CHECK (signal_action IN ('none', 'open', 'close', 'reduce', 'reverse')),
            CONSTRAINT strategy_signals_side_chk
                CHECK (side IS NULL OR side IN ('buy', 'sell')),
            CONSTRAINT strategy_signals_outcome_chk
                CHECK (outcome IN ('warmup', 'no_signal', 'signal', 'blocked')),
            CONSTRAINT strategy_signals_action_side_chk
                CHECK (
                    (signal_action = 'none' AND side IS NULL)
                    OR (signal_action <> 'none' AND side IS NOT NULL)
                ),
            CONSTRAINT strategy_signals_reason_nonempty_chk
                CHECK (char_length(trim(reason_code)) > 0),
            CONSTRAINT strategy_signals_source_message_nonempty_chk
                CHECK (char_length(trim(source_message_id)) > 0),
            CONSTRAINT strategy_signals_evaluator_nonempty_chk
                CHECK (char_length(trim(evaluator_version)) > 0),
            CONSTRAINT strategy_signals_bar_window_chk
                CHECK (bar_ts_open < bar_ts_close),
            CONSTRAINT strategy_signals_reference_price_chk
                CHECK (reference_price >= 0),
            CONSTRAINT strategy_signals_confidence_chk
                CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
            CONSTRAINT strategy_signals_expected_order_stage05_chk
                CHECK (expected_order_json = '{}'::jsonb)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_signals_owner_strategy_created
            ON strategy_signals (owner_user_id, strategy_id, created_at DESC, signal_id DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_signals_run_bar
            ON strategy_signals (strategy_run_id, bar_ts_open, signal_id)
        """
    )


def downgrade() -> None:
    """
    Remove additive StrategySignal journal storage.
    """
    op.execute("DROP TABLE IF EXISTS strategy_signals")
