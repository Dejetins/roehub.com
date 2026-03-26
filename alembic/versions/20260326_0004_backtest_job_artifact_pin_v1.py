"""Add additive artifact pin columns to `backtest_jobs` for R2-02 publish safety."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260326_0004"
down_revision = "20260222_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add strict artifact pin metadata columns and indexes to `backtest_jobs`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Migration is additive and keeps existing jobs rows/API contracts backward compatible.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Alters `backtest_jobs` and creates one partial helper index for pin-guard reads.
    """
    op.execute(
        """
        ALTER TABLE backtest_jobs
            ADD COLUMN IF NOT EXISTS artifact_slot TEXT NULL,
            ADD COLUMN IF NOT EXISTS artifact_slot_generation INTEGER NULL,
            ADD COLUMN IF NOT EXISTS artifact_manifest_hash TEXT NULL,
            ADD COLUMN IF NOT EXISTS artifact_asof_date TEXT NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_jobs_artifact_pin_all_or_none_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_artifact_pin_all_or_none_chk
                CHECK (
                    (
                        artifact_slot IS NULL
                        AND artifact_slot_generation IS NULL
                        AND artifact_manifest_hash IS NULL
                        AND artifact_asof_date IS NULL
                    )
                    OR (
                        artifact_slot IS NOT NULL
                        AND artifact_slot_generation IS NOT NULL
                        AND artifact_manifest_hash IS NOT NULL
                        AND artifact_asof_date IS NOT NULL
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
                WHERE conname = 'backtest_jobs_artifact_slot_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_artifact_slot_chk
                CHECK (
                    artifact_slot IS NULL
                    OR artifact_slot IN ('slot_a', 'slot_b')
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
                WHERE conname = 'backtest_jobs_artifact_slot_generation_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_artifact_slot_generation_chk
                CHECK (
                    artifact_slot_generation IS NULL
                    OR artifact_slot_generation > 0
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
                WHERE conname = 'backtest_jobs_artifact_manifest_hash_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_artifact_manifest_hash_chk
                CHECK (
                    artifact_manifest_hash IS NULL
                    OR artifact_manifest_hash ~ '^[0-9a-f]{64}$'
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
                WHERE conname = 'backtest_jobs_artifact_asof_date_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_artifact_asof_date_chk
                CHECK (
                    artifact_asof_date IS NULL
                    OR artifact_asof_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                );
            END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_active_artifact_pin
            ON backtest_jobs (artifact_slot, artifact_manifest_hash)
            WHERE state IN ('queued', 'running')
              AND artifact_slot IS NOT NULL
              AND artifact_manifest_hash IS NOT NULL
        """
    )


def downgrade() -> None:
    """
    Remove additive artifact pin columns and helper index from `backtest_jobs`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade removes only the additive R2-02 schema extensions.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Drops helper index, constraints, and nullable artifact pin columns.
    """
    op.execute("DROP INDEX IF EXISTS idx_backtest_jobs_active_artifact_pin")
    op.execute(
        """
        ALTER TABLE backtest_jobs
            DROP CONSTRAINT IF EXISTS backtest_jobs_artifact_asof_date_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_artifact_manifest_hash_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_artifact_slot_generation_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_artifact_slot_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_artifact_pin_all_or_none_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_jobs
            DROP COLUMN IF EXISTS artifact_asof_date,
            DROP COLUMN IF EXISTS artifact_manifest_hash,
            DROP COLUMN IF EXISTS artifact_slot_generation,
            DROP COLUMN IF EXISTS artifact_slot
        """
    )
