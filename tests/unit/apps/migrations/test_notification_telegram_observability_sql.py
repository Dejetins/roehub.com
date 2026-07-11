from pathlib import Path


def test_notification_delivery_counter_migration_is_additive_and_indexed() -> None:
    text = Path(
        "alembic/versions/20260711_0043_notification_delivery_counters_v1.py"
    ).read_text(encoding="utf-8")

    assert 'down_revision = "20260703_0042"' in text
    assert "idx_notification_deliveries_sent_route" in text
    assert "WHERE status = 'sent'" in text
    assert "CREATE TABLE" not in text
