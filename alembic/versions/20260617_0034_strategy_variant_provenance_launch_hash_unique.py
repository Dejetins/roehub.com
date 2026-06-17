"""Include launch request hash in strategy variant provenance source uniqueness."""

from __future__ import annotations

from alembic import op

revision = "20260617_0034"
down_revision = "20260617_0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_strategy_backtest_variant_provenance_source")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_backtest_variant_provenance_source
            ON strategy_backtest_variant_provenance
                (
                    user_id,
                    source_job_id,
                    source_variant_key,
                    strategy_spec_hash,
                    launch_request_hash
                )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_strategy_backtest_variant_provenance_source")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_backtest_variant_provenance_source
            ON strategy_backtest_variant_provenance
                (user_id, source_job_id, source_variant_key, strategy_spec_hash)
        """
    )
