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


def test_stage13_notification_event_type_migration_widens_check_constraint() -> None:
    migration = Path(
        "alembic/versions/20260702_0039_execution_notification_stage13_event_types.py"
    )
    text = migration.read_text(encoding="utf-8")

    assert "DROP CONSTRAINT IF EXISTS execution_notification_event_type_chk" in text
    assert "ADD CONSTRAINT execution_notification_event_type_chk" in text
    assert "producer_signal_rejected" in text
    assert "producer_order_rejected" in text
    assert "producer_manual_exit" in text
    assert "producer_reconciliation_pending" in text
    assert "producer_strategy_stopped" in text
    assert "producer_strategy_restarted" in text
    assert "producer_soak_failed" in text
    assert "producer_soak_succeeded" in text
    assert "producer_resource_threshold_breached" in text
    assert "DROP TABLE" not in text
