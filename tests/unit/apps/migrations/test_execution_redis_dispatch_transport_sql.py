from pathlib import Path


def test_execution_redis_dispatch_transport_migration_is_additive_and_scoped() -> None:
    migration = Path(
        "alembic/versions/20260531_0025_execution_redis_dispatch_transport_v1.py"
    )
    text = migration.read_text(encoding="utf-8")

    assert "dispatch_attempt_count INTEGER NOT NULL DEFAULT 0" in text
    assert "dispatch_stream_name TEXT NULL" in text
    assert "dispatch_redis_message_id TEXT NULL" in text
    assert "dispatch_last_error TEXT NULL" in text
    assert "dispatch_updated_at TIMESTAMPTZ NULL" in text
    assert "'dispatching'" in text
    assert "'dispatched'" in text
    assert "'retry'" in text
    assert "'quarantined'" in text
    assert "idx_execution_intents_dispatch_retry" in text
    assert "DROP TABLE IF EXISTS execution_intents" not in text
