from pathlib import Path


def test_rl_live_ticker_entitlements_migration_is_additive_and_transaction_ready() -> None:
    migration = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "20260703_0041_rl_live_ticker_entitlements_v1.py"
    )

    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS rl_live_ticker_entitlement_overrides" in text
    assert "CREATE TABLE IF NOT EXISTS rl_live_ticker_activations" in text
    assert "uq_rl_live_ticker_activations_active_owner_ticker" in text
    active_unique_columns = (
        "owner_user_id,\n                exchange_name,\n"
        "                market_type,\n                symbol"
    )
    assert active_unique_columns in text
    assert "WHERE active" in text
    assert "CHECK (mode = 'live')" in text
    assert "DROP TABLE IF EXISTS rl_live_ticker_activations" in text
