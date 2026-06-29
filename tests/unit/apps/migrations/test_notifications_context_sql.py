from pathlib import Path


def _migration_text() -> str:
    return Path("alembic/versions/20260629_0037_notifications_context_v1.py").read_text(
        encoding="utf-8"
    )


def test_notifications_context_migration_adds_provider_neutral_tables() -> None:
    text = _migration_text()

    assert 'down_revision = "20260618_0036"' in text
    assert "CREATE TABLE IF NOT EXISTS notification_events" in text
    assert "CREATE TABLE IF NOT EXISTS notification_routes" in text
    assert "CREATE TABLE IF NOT EXISTS notification_deliveries" in text
    assert "CREATE TABLE IF NOT EXISTS notification_delivery_attempts" in text
    assert "CREATE TABLE IF NOT EXISTS notification_telegram_updates" in text
    assert "CREATE TABLE IF NOT EXISTS notification_report_runs" in text
    assert "DROP TABLE IF EXISTS execution_notification_outbox" not in text


def test_notifications_context_migration_enforces_status_and_redaction_constraints() -> None:
    text = _migration_text()

    assert "notification_deliveries_status_chk" in text
    for status in (
        "pending",
        "claimed",
        "sent",
        "failed",
        "retry",
        "dead_letter",
        "suppressed",
        "unknown",
    ):
        assert status in text
    assert "notification_delivery_attempts_request_hash_chk" in text
    assert "notification_delivery_attempts_response_hash_chk" in text
    assert "redacted_request_hash ~ '^[a-f0-9]{64}$'" in text
    assert "recipient_address_ref !~* '(token|secret|password|cookie|authorization)'" in text
    assert "raw_payload" not in text
    assert "provider_payload" not in text
