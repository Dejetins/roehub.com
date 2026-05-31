from pathlib import Path


def test_exchange_execution_process_skeleton_migration_is_additive_and_scoped() -> None:
    migration = Path(
        "alembic/versions/20260531_0026_exchange_execution_process_skeleton_v1.py"
    )
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS exchange_execution_process_heartbeats" in text
    assert "CREATE TABLE IF NOT EXISTS exchange_execution_request_observations" in text
    assert "adapter_mode IN ('disabled')" in text
    assert "status IN ('ready', 'degraded', 'not_ready')" in text
    assert "status IN ('adapter_disabled', 'quarantined', 'skipped')" in text
    assert "ux_exchange_execution_observation_message_status" in text
    assert "DROP TABLE IF EXISTS execution_intents" not in text
