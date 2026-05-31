from pathlib import Path


def test_testnet_order_adapters_migration_is_additive_and_testnet_only() -> None:
    migration = Path("alembic/versions/20260531_0027_testnet_order_adapters_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS execution_orders" in text
    assert "CREATE TABLE IF NOT EXISTS exchange_private_stream_sessions" in text
    assert "adapter_mode IN ('disabled', 'testnet')" in text
    assert "environment IN ('testnet')" in text
    assert "exchange_name IN ('binance', 'bybit')" in text
    assert "DROP TABLE IF EXISTS execution_intents" not in text
