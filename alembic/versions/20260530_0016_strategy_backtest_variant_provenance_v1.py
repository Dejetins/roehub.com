"""Add strategy provenance for backtest variant creation."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260530_0016"
down_revision = "20260519_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply additive provenance storage for Stage 02 backtest variant strategy creation.
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_backtest_variant_provenance (
            strategy_id UUID PRIMARY KEY REFERENCES strategy_strategies(strategy_id)
                ON DELETE CASCADE,
            user_id UUID NOT NULL,
            source_job_id UUID NOT NULL,
            source_variant_key TEXT NOT NULL,
            source_variant_hash TEXT NOT NULL,
            source_indicator_variant_hash TEXT NULL,
            backtest_request_hash TEXT NOT NULL,
            backtest_result_config_hash TEXT NOT NULL,
            strategy_spec_hash TEXT NOT NULL,
            launch_request_hash TEXT NOT NULL,
            idempotency_key_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            metadata_json JSONB NOT NULL,
            CONSTRAINT strategy_backtest_variant_provenance_variant_key_chk
                CHECK (char_length(trim(source_variant_key)) > 0),
            CONSTRAINT strategy_backtest_variant_provenance_variant_hash_chk
                CHECK (char_length(trim(source_variant_hash)) > 0),
            CONSTRAINT strategy_backtest_variant_provenance_spec_hash_chk
                CHECK (char_length(trim(strategy_spec_hash)) > 0),
            CONSTRAINT strategy_backtest_variant_provenance_idempotency_chk
                CHECK (char_length(trim(idempotency_key_hash)) > 0),
            CONSTRAINT strategy_backtest_variant_provenance_metadata_shape_chk
                CHECK (jsonb_typeof(metadata_json) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_backtest_variant_provenance_idempotency
            ON strategy_backtest_variant_provenance (user_id, idempotency_key_hash)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_backtest_variant_provenance_source
            ON strategy_backtest_variant_provenance
                (user_id, source_job_id, source_variant_key, strategy_spec_hash)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_strategy_backtest_variant_provenance_created
            ON strategy_backtest_variant_provenance (user_id, created_at, strategy_id)
        """
    )


def downgrade() -> None:
    """
    Remove additive provenance storage.
    """
    op.execute("DROP TABLE IF EXISTS strategy_backtest_variant_provenance")
