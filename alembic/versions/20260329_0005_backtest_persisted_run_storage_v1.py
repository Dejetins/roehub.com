"""Generalize Backtest jobs tables into persisted-run storage for R7-01."""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260329_0005"
down_revision = "20260326_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add persisted-run metadata and summary-only top-row columns for R7-01 storage unification.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Migration is additive, keeps existing job/top rows readable, and allows transitional
        nullable legacy rows without destructive backfill.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Alters `backtest_jobs` and `backtest_job_top_variants`, adds constraints, and creates
        helper indexes for future persisted-run history reads.
    """
    op.execute(
        """
        ALTER TABLE backtest_jobs
            ADD COLUMN IF NOT EXISTS execution_mode TEXT NULL,
            ADD COLUMN IF NOT EXISTS market_id INTEGER NULL,
            ADD COLUMN IF NOT EXISTS symbol TEXT NULL,
            ADD COLUMN IF NOT EXISTS timeframe TEXT NULL,
            ADD COLUMN IF NOT EXISTS requested_top_n INTEGER NULL,
            ADD COLUMN IF NOT EXISTS ranking_primary_metric TEXT NULL,
            ADD COLUMN IF NOT EXISTS ranking_secondary_metric TEXT NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_jobs_execution_mode_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_execution_mode_chk
                CHECK (
                    execution_mode IS NULL
                    OR execution_mode IN (
                        'sync_inline',
                        'background_auto',
                        'background_manual_legacy'
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
                WHERE conname = 'backtest_jobs_market_id_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_market_id_chk
                CHECK (
                    market_id IS NULL
                    OR market_id > 0
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
                WHERE conname = 'backtest_jobs_symbol_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_symbol_chk
                CHECK (
                    symbol IS NULL
                    OR btrim(symbol) <> ''
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
                WHERE conname = 'backtest_jobs_timeframe_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_timeframe_chk
                CHECK (
                    timeframe IS NULL
                    OR btrim(timeframe) <> ''
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
                WHERE conname = 'backtest_jobs_requested_top_n_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_requested_top_n_chk
                CHECK (
                    requested_top_n IS NULL
                    OR requested_top_n > 0
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
                WHERE conname = 'backtest_jobs_ranking_primary_metric_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_ranking_primary_metric_chk
                CHECK (
                    ranking_primary_metric IS NULL
                    OR ranking_primary_metric IN (
                        'total_return_pct',
                        'max_drawdown_pct',
                        'return_over_max_drawdown',
                        'profit_factor',
                        'sharpe_trades',
                        'win_rate_pct'
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
                WHERE conname = 'backtest_jobs_ranking_secondary_metric_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_ranking_secondary_metric_chk
                CHECK (
                    ranking_secondary_metric IS NULL
                    OR ranking_secondary_metric IN (
                        'total_return_pct',
                        'max_drawdown_pct',
                        'return_over_max_drawdown',
                        'profit_factor',
                        'sharpe_trades',
                        'win_rate_pct'
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
                WHERE conname = 'backtest_jobs_ranking_metrics_distinct_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_ranking_metrics_distinct_chk
                CHECK (
                    ranking_primary_metric IS NULL
                    OR ranking_secondary_metric IS NULL
                    OR ranking_primary_metric <> ranking_secondary_metric
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
                WHERE conname = 'backtest_jobs_persisted_run_metadata_chk'
            ) THEN
                ALTER TABLE backtest_jobs
                ADD CONSTRAINT backtest_jobs_persisted_run_metadata_chk
                CHECK (
                    (
                        execution_mode IS NULL
                        AND market_id IS NULL
                        AND symbol IS NULL
                        AND timeframe IS NULL
                        AND requested_top_n IS NULL
                        AND ranking_primary_metric IS NULL
                        AND ranking_secondary_metric IS NULL
                    )
                    OR (
                        execution_mode IS NOT NULL
                        AND market_id IS NOT NULL
                        AND symbol IS NOT NULL
                        AND timeframe IS NOT NULL
                        AND requested_top_n IS NOT NULL
                        AND ranking_primary_metric IS NOT NULL
                    )
                );
            END IF;
        END
        $$;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_user_execution_created_desc
            ON backtest_jobs (user_id, execution_mode, created_at DESC, job_id DESC)
            WHERE execution_mode IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_user_instrument_created_desc
            ON backtest_jobs (
                user_id,
                market_id,
                symbol,
                timeframe,
                created_at DESC,
                job_id DESC
            )
            WHERE market_id IS NOT NULL
              AND symbol IS NOT NULL
              AND timeframe IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_user_ranking_created_desc
            ON backtest_jobs (
                user_id,
                ranking_primary_metric,
                ranking_secondary_metric,
                created_at DESC,
                job_id DESC
            )
            WHERE ranking_primary_metric IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_backtest_jobs_active_pin_instrument
            ON backtest_jobs (
                artifact_slot,
                artifact_manifest_hash,
                market_id,
                symbol
            )
            WHERE state IN ('queued', 'running')
              AND artifact_slot IS NOT NULL
              AND artifact_manifest_hash IS NOT NULL
              AND market_id IS NOT NULL
              AND symbol IS NOT NULL
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_job_top_variants
            ADD COLUMN IF NOT EXISTS summary_metrics_json JSONB NULL,
            ADD COLUMN IF NOT EXISTS best_tp_pct DOUBLE PRECISION NULL,
            ADD COLUMN IF NOT EXISTS best_sl_pct DOUBLE PRECISION NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_job_top_variants_summary_metrics_shape_chk'
            ) THEN
                ALTER TABLE backtest_job_top_variants
                ADD CONSTRAINT backtest_job_top_variants_summary_metrics_shape_chk
                CHECK (
                    summary_metrics_json IS NULL
                    OR jsonb_typeof(summary_metrics_json) = 'object'
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
                WHERE conname = 'backtest_job_top_variants_best_tp_pct_chk'
            ) THEN
                ALTER TABLE backtest_job_top_variants
                ADD CONSTRAINT backtest_job_top_variants_best_tp_pct_chk
                CHECK (
                    best_tp_pct IS NULL
                    OR best_tp_pct >= 0
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
                WHERE conname = 'backtest_job_top_variants_best_sl_pct_chk'
            ) THEN
                ALTER TABLE backtest_job_top_variants
                ADD CONSTRAINT backtest_job_top_variants_best_sl_pct_chk
                CHECK (
                    best_sl_pct IS NULL
                    OR best_sl_pct >= 0
                );
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """
    Remove additive persisted-run metadata and summary-only top-row columns for R7-01.

    Docs:
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_results_repository.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Downgrade removes only additive R7-01 schema extensions.
    Raises:
        Exception: Postgres execution errors from Alembic runtime.
    Side Effects:
        Drops helper indexes, constraints, and nullable persisted-run columns.
    """
    op.execute("DROP INDEX IF EXISTS idx_backtest_jobs_active_pin_instrument")
    op.execute("DROP INDEX IF EXISTS idx_backtest_jobs_user_ranking_created_desc")
    op.execute("DROP INDEX IF EXISTS idx_backtest_jobs_user_instrument_created_desc")
    op.execute("DROP INDEX IF EXISTS idx_backtest_jobs_user_execution_created_desc")
    op.execute(
        """
        ALTER TABLE backtest_job_top_variants
            DROP CONSTRAINT IF EXISTS backtest_job_top_variants_best_sl_pct_chk,
            DROP CONSTRAINT IF EXISTS backtest_job_top_variants_best_tp_pct_chk,
            DROP CONSTRAINT IF EXISTS backtest_job_top_variants_summary_metrics_shape_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_job_top_variants
            DROP COLUMN IF EXISTS best_sl_pct,
            DROP COLUMN IF EXISTS best_tp_pct,
            DROP COLUMN IF EXISTS summary_metrics_json
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_jobs
            DROP CONSTRAINT IF EXISTS backtest_jobs_persisted_run_metadata_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_ranking_metrics_distinct_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_ranking_secondary_metric_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_ranking_primary_metric_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_requested_top_n_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_timeframe_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_symbol_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_market_id_chk,
            DROP CONSTRAINT IF EXISTS backtest_jobs_execution_mode_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_jobs
            DROP COLUMN IF EXISTS ranking_secondary_metric,
            DROP COLUMN IF EXISTS ranking_primary_metric,
            DROP COLUMN IF EXISTS requested_top_n,
            DROP COLUMN IF EXISTS timeframe,
            DROP COLUMN IF EXISTS symbol,
            DROP COLUMN IF EXISTS market_id,
            DROP COLUMN IF EXISTS execution_mode
        """
    )
