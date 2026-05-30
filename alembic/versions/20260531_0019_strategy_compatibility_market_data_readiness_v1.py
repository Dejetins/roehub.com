"""Add Strategy compatibility and market-data readiness evidence for Stage 06."""

from __future__ import annotations

from alembic import op

revision = "20260531_0019"
down_revision = "20260531_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_variant_compatibility_checks (
            compatibility_check_id UUID PRIMARY KEY,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NULL REFERENCES strategy_strategies(strategy_id)
                ON DELETE CASCADE,
            source_job_id UUID NULL REFERENCES backtest_jobs(job_id)
                ON DELETE SET NULL,
            source_variant_key TEXT NULL,
            strategy_spec_hash TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            compatibility_state TEXT NOT NULL,
            reason_codes_json JSONB NOT NULL,
            checked_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT strategy_variant_compatibility_state_chk
                CHECK (compatibility_state IN ('launchable', 'not_launchable', 'degraded')),
            CONSTRAINT strategy_variant_compatibility_hash_chk
                CHECK (strategy_spec_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT strategy_variant_compatibility_reason_shape_chk
                CHECK (jsonb_typeof(reason_codes_json) = 'array'),
            CONSTRAINT strategy_variant_compatibility_instrument_chk
                CHECK (char_length(trim(instrument_key)) > 0),
            CONSTRAINT strategy_variant_compatibility_timeframe_chk
                CHECK (char_length(trim(timeframe)) > 0)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data_subscription_requirements (
            market_data_requirement_id UUID PRIMARY KEY,
            compatibility_check_id UUID NOT NULL
                REFERENCES strategy_variant_compatibility_checks(compatibility_check_id)
                ON DELETE CASCADE,
            owner_user_id UUID NOT NULL,
            strategy_id UUID NULL REFERENCES strategy_strategies(strategy_id)
                ON DELETE CASCADE,
            source_job_id UUID NULL REFERENCES backtest_jobs(job_id)
                ON DELETE SET NULL,
            source_variant_key TEXT NULL,
            instrument_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            readiness_state TEXT NOT NULL,
            reason_codes_json JSONB NOT NULL,
            stream_name TEXT NOT NULL,
            stream_length BIGINT NULL,
            last_message_id TEXT NULL,
            last_observed_at TIMESTAMPTZ NULL,
            age_seconds BIGINT NULL,
            checked_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT market_data_subscription_requirements_state_chk
                CHECK (readiness_state IN ('ready', 'missing', 'stale', 'pending')),
            CONSTRAINT market_data_subscription_requirements_reason_shape_chk
                CHECK (jsonb_typeof(reason_codes_json) = 'array'),
            CONSTRAINT market_data_subscription_requirements_stream_chk
                CHECK (char_length(trim(stream_name)) > 0),
            CONSTRAINT market_data_subscription_requirements_age_chk
                CHECK (age_seconds IS NULL OR age_seconds >= 0),
            CONSTRAINT market_data_subscription_requirements_length_chk
                CHECK (stream_length IS NULL OR stream_length >= 0)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_variant_compatibility_owner_checked
            ON strategy_variant_compatibility_checks (owner_user_id, checked_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_data_subscription_requirements_owner_checked
            ON market_data_subscription_requirements (owner_user_id, checked_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_data_subscription_requirements_stream_state
            ON market_data_subscription_requirements (stream_name, readiness_state, checked_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS market_data_subscription_requirements")
    op.execute("DROP TABLE IF EXISTS strategy_variant_compatibility_checks")
