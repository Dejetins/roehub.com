from pathlib import Path


def test_execution_notifications_migration_is_additive_and_redacted() -> None:
    migration = Path("alembic/versions/20260603_0030_execution_notifications_producers_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS execution_notification_outbox" in text
    assert "producer_rejected" in text
    assert "producer_fill" in text
    assert "producer_unknown" in text
    assert "producer_kill_switch" in text
    assert "producer_terminal" in text
    assert "risk_rejected" in text
    assert "reconciliation_required" in text
    assert "execution_notification_outbox_dedupe" in text
    assert "jsonb_typeof(labels_json) = 'object'" in text
    assert "DROP TABLE IF EXISTS execution_intents" not in text
    assert "DROP TABLE IF EXISTS execution_orders" not in text
