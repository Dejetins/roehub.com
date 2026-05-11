"""Add Backtest AI prompt hash audit metadata."""

from __future__ import annotations

from alembic import op

revision = "20260511_0013"
down_revision = "20260511_0012"
branch_labels = None
depends_on = None

_DEFAULT_SYSTEM_PROMPT_VERSION = "backtest-ai-configurator-v1"
_DEFAULT_SYSTEM_PROMPT_HASH = (
    "f7423c43c6d1ab9d6b744eb3e5e5e135d876fb570a6ea0298602d0b754e14719"
)


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE backtest_ai_config_jobs
            ADD COLUMN IF NOT EXISTS system_prompt_hash TEXT NULL
        """
    )
    op.execute(
        f"""
        UPDATE backtest_ai_config_jobs
        SET
            system_prompt_version = '{_DEFAULT_SYSTEM_PROMPT_VERSION}',
            system_prompt_hash = '{_DEFAULT_SYSTEM_PROMPT_HASH}'
        WHERE system_prompt_hash IS NULL
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_ai_config_jobs
            ALTER COLUMN system_prompt_hash SET NOT NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_ai_config_jobs_system_prompt_hash_chk'
            ) THEN
                ALTER TABLE backtest_ai_config_jobs
                    ADD CONSTRAINT backtest_ai_config_jobs_system_prompt_hash_chk
                    CHECK (system_prompt_hash ~ '^[0-9a-f]{64}$');
            END IF;
        END $$;
        """
    )

    op.execute(
        """
        ALTER TABLE backtest_ai_config_llm_attempts
            ADD COLUMN IF NOT EXISTS system_prompt_version TEXT NULL,
            ADD COLUMN IF NOT EXISTS system_prompt_hash TEXT NULL
        """
    )
    op.execute(
        f"""
        UPDATE backtest_ai_config_llm_attempts
        SET
            system_prompt_version = COALESCE(
                system_prompt_version,
                '{_DEFAULT_SYSTEM_PROMPT_VERSION}'
            ),
            system_prompt_hash = COALESCE(
                system_prompt_hash,
                '{_DEFAULT_SYSTEM_PROMPT_HASH}'
            )
        WHERE system_prompt_version IS NULL
           OR system_prompt_hash IS NULL
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_ai_config_llm_attempts
            ALTER COLUMN system_prompt_version SET NOT NULL,
            ALTER COLUMN system_prompt_hash SET NOT NULL
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'backtest_ai_config_llm_attempts_prompt_hash_chk'
            ) THEN
                ALTER TABLE backtest_ai_config_llm_attempts
                    ADD CONSTRAINT backtest_ai_config_llm_attempts_prompt_hash_chk
                    CHECK (
                        btrim(system_prompt_version) <> ''
                        AND system_prompt_hash ~ '^[0-9a-f]{64}$'
                    );
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE backtest_ai_config_llm_attempts
            DROP CONSTRAINT IF EXISTS backtest_ai_config_llm_attempts_prompt_hash_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_ai_config_llm_attempts
            DROP COLUMN IF EXISTS system_prompt_hash,
            DROP COLUMN IF EXISTS system_prompt_version
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_ai_config_jobs
            DROP CONSTRAINT IF EXISTS backtest_ai_config_jobs_system_prompt_hash_chk
        """
    )
    op.execute(
        """
        ALTER TABLE backtest_ai_config_jobs
            DROP COLUMN IF EXISTS system_prompt_hash
        """
    )
