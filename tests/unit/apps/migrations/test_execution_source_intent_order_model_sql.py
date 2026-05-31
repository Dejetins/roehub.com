from pathlib import Path


def test_execution_source_intent_order_model_migration_is_additive_and_scoped() -> None:
    migration = Path("alembic/versions/20260531_0023_execution_source_intent_order_model_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS execution_source_events" in text
    assert "CREATE TABLE IF NOT EXISTS execution_intents" in text
    assert "strategy_signal" in text
    assert "manual_request" in text
    assert "ml_agent_decision" in text
    assert "ops_test" in text
    assert "order_type IN ('market', 'limit')" in text
    assert "execution_intents_idempotency_unique" in text
    assert "DROP TABLE IF EXISTS execution_intents" in text
