"""Add additive execution-profile metadata columns to `backtest_jobs`."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260411_0006"
down_revision = "20260329_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Apply the additive migration for persisted execution-profile metadata split.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        This additive migration keeps existing rows readable without backfill and moves new
        read-model execution-profile metadata into dedicated nullable columns.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Alters `backtest_jobs` by adding nullable metadata columns and compatibility constraints.
    """
    op.execute(
        """
        ALTER TABLE backtest_jobs
            ADD COLUMN IF NOT EXISTS execution_profile_mode_hint TEXT NULL,
            ADD COLUMN IF NOT EXISTS effective_execution_profile_mode TEXT NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_jobs_execution_profile_mode_hint_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_execution_profile_mode_hint_chk
                CHECK (
                    execution_profile_mode_hint IS NULL
                    OR execution_profile_mode_hint IN (
                        'exact_small',
                        'exact_parallel',
                        'hybrid_conservative',
                        'hybrid_family'
                    )
                );
            END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_jobs_effective_execution_profile_mode_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_effective_execution_profile_mode_chk
                CHECK (
                    effective_execution_profile_mode IS NULL
                    OR effective_execution_profile_mode IN (
                        'exact_small',
                        'exact_parallel',
                        'hybrid_conservative',
                        'hybrid_family'
                    )
                );
            END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_jobs_execution_profile_metadata_requires_run_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_execution_profile_metadata_requires_run_chk
                CHECK (
                    (
                        execution_profile_mode_hint IS NULL
                        AND effective_execution_profile_mode IS NULL
                    )
                    OR execution_mode IS NOT NULL
                );
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """
    Revert the additive execution-profile metadata split from `backtest_jobs`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade removes only the additive metadata columns introduced by this migration.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Drops compatibility constraints and the nullable execution-profile metadata columns.
    """
    op.execute(
        """
        ALTER TABLE backtest_jobs
            DROP CONSTRAINT IF EXISTS
                backtest_jobs_execution_profile_metadata_requires_run_chk,
            DROP CONSTRAINT IF EXISTS
                backtest_jobs_effective_execution_profile_mode_chk,
            DROP CONSTRAINT IF EXISTS
                backtest_jobs_execution_profile_mode_hint_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_jobs
            DROP COLUMN IF EXISTS effective_execution_profile_mode,
            DROP COLUMN IF EXISTS execution_profile_mode_hint
        """
    )
