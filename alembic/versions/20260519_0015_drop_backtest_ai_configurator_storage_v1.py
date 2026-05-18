"""Drop retired Backtest AI configurator storage."""

from __future__ import annotations

from alembic import op

revision = "20260519_0015"
down_revision = "20260518_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS backtest_ai_conversation_runs")
    op.execute("DROP TABLE IF EXISTS backtest_ai_conversation_messages")
    op.execute("DROP TABLE IF EXISTS backtest_ai_conversations")
    op.execute("DROP TABLE IF EXISTS backtest_ai_quota_events")
    op.execute("DROP TABLE IF EXISTS backtest_ai_config_llm_attempts")
    op.execute("DROP TABLE IF EXISTS backtest_ai_config_events")
    op.execute("DROP TABLE IF EXISTS backtest_ai_config_jobs")


def downgrade() -> None:
    # The retired AI configurator schema is intentionally not recreated.
    # Restoring it requires reverting the cleanup and applying the original
    # storage migrations in order.
    return None
