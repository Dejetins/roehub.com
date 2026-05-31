from pathlib import Path


def test_exchange_account_projection_migration_is_additive_and_secret_safe() -> None:
    migration = (
        Path("alembic/versions")
        / "20260531_0020_exchange_account_projection_config_guard_v1.py"
    )
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS exchange_account_snapshots" in text
    assert "CREATE TABLE IF NOT EXISTS exchange_balance_snapshots" in text
    assert "CREATE TABLE IF NOT EXISTS exchange_position_snapshots" in text
    assert "CREATE TABLE IF NOT EXISTS exchange_open_order_snapshots" in text
    assert "CREATE TABLE IF NOT EXISTS exchange_instrument_filter_snapshots" in text
    assert "CREATE TABLE IF NOT EXISTS exchange_account_config_guard_results" in text
    assert "source_hash ~ '^[0-9a-f]{64}$'" in text
    assert "CHECK (status IN ('verified', 'mismatch', 'degraded'))" in text
    assert "api_secret" not in text
    assert "passphrase" not in text
    assert "Authorization" not in text
    assert "DROP TABLE IF EXISTS exchange_account_snapshots" in text
