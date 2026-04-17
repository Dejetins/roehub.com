"""Add additive parity runtime state column to `backtest_job_stage_a_shortlist`."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260418_0008"
down_revision = "20260414_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply the additive shortlist parity-runtime-state storage migration for D5 worker resume.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Existing shortlist rows remain readable with `NULL parity_runtime_state_json`, while new
        parity-first no-risk rows may persist compact runtime-shape evidence additively.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Alters `backtest_job_stage_a_shortlist` by adding one nullable JSONB column and shape
        constraint.
    """
    op.execute(
        """
        ALTER TABLE backtest_job_stage_a_shortlist
            ADD COLUMN IF NOT EXISTS parity_runtime_state_json JSONB NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_job_stage_a_shortlist_parity_runtime_state_shape_chk'
            ) THEN
                ALTER TABLE backtest_job_stage_a_shortlist
                ADD CONSTRAINT backtest_job_stage_a_shortlist_parity_runtime_state_shape_chk
                CHECK (
                    parity_runtime_state_json IS NULL
                    OR jsonb_typeof(parity_runtime_state_json) = 'object'
                );
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """
    Revert the additive shortlist parity-runtime-state storage column.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade removes only the D5 additive contract column and its shape constraint.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Alters `backtest_job_stage_a_shortlist` by dropping one constraint and one column.
    """
    op.execute(
        """
        ALTER TABLE backtest_job_stage_a_shortlist
            DROP CONSTRAINT IF EXISTS
                backtest_job_stage_a_shortlist_parity_runtime_state_shape_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_job_stage_a_shortlist
            DROP COLUMN IF EXISTS parity_runtime_state_json
        """
    )
