from pathlib import Path


def _migration_text() -> str:
    return Path(
        "alembic/versions/20260629_0038_market_data_candle_repair_events_v1.py"
    ).read_text(encoding="utf-8")


def test_market_data_candle_repair_events_migration_is_additive() -> None:
    text = _migration_text()

    assert 'down_revision = "20260629_0037"' in text
    assert "CREATE TABLE IF NOT EXISTS market_data_candle_repair_events" in text
    assert "event_id UUID PRIMARY KEY" in text
    assert "sources_attempted_json JSONB NOT NULL" in text
    assert "restored_ts_opens_json JSONB NOT NULL" in text
    assert "missing_ts_opens_json JSONB NOT NULL" in text
    assert "DROP TABLE IF EXISTS market_data_candle_repair_events" in text


def test_market_data_candle_repair_events_migration_enforces_redacted_status_shape() -> None:
    text = _migration_text()

    for status in (
        "attempted",
        "succeeded",
        "miss",
        "failed",
        "circuit_open",
        "rate_limited",
    ):
        assert status in text
    assert "jsonb_typeof(sources_attempted_json) = 'array'" in text
    assert "error_code ~ '^[a-z0-9][a-z0-9_:-]{0,95}$'" in text
    assert "error_summary !~* (" in text
    assert "'(api[_-]?key|authorization|bearer|cookie|dsn|'" in text
    assert "|| 'password|secret|token)'" in text
    assert "raw_payload" not in text
    assert "provider_payload" not in text
    assert "api_secret" not in text
