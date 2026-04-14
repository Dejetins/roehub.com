"""Add additive compact no-risk exact rows column to `backtest_job_stage_a_shortlist`."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260414_0007"
down_revision = "20260411_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply the additive shortlist exact-row storage migration for worker no-risk reuse.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Existing shortlist rows remain readable with `NULL no_risk_exact_rows_json`, while new
        no-risk rows may persist compact exact payloads additively without widening old columns.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Alters `backtest_job_stage_a_shortlist` by adding one nullable JSONB column and shape
        constraint.
    """
    op.execute(
        """
        ALTER TABLE backtest_job_stage_a_shortlist
            ADD COLUMN IF NOT EXISTS no_risk_exact_rows_json JSONB NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_job_stage_a_shortlist_no_risk_exact_rows_shape_chk'
            ) THEN
                ALTER TABLE backtest_job_stage_a_shortlist
                ADD CONSTRAINT backtest_job_stage_a_shortlist_no_risk_exact_rows_shape_chk
                CHECK (
                    no_risk_exact_rows_json IS NULL
                    OR jsonb_typeof(no_risk_exact_rows_json) = 'array'
                );
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """
    Revert the additive shortlist exact-row storage column.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade removes only the additive C4 contract column and its shape constraint.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Alters `backtest_job_stage_a_shortlist` by dropping one constraint and one column.
    """
    op.execute(
        """
        ALTER TABLE backtest_job_stage_a_shortlist
            DROP CONSTRAINT IF EXISTS
                backtest_job_stage_a_shortlist_no_risk_exact_rows_shape_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_job_stage_a_shortlist
            DROP COLUMN IF EXISTS no_risk_exact_rows_json
        """
    )
