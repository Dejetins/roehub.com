from pathlib import Path


def test_strategy_position_ownership_migration_is_additive_and_scoped() -> None:
    migration = Path("alembic/versions/20260531_0021_strategy_position_ownership_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_position_ownership" in text
    assert "strategy_position_ownership_one_blocking" in text
    assert "owner_user_id" in text
    assert "exchange_connection_id" in text
    assert "market_type" in text
    assert "instrument_key" in text
    assert "state IN (" in text
    assert "'reserved'" in text
    assert "'active'" in text
    assert "'releasing'" in text
    assert "'released'" in text
    assert "'stale_requires_repair'" in text
    assert "WHERE state IN (" in text
    assert "api_secret" not in text
    assert "passphrase" not in text
    assert "Authorization" not in text
