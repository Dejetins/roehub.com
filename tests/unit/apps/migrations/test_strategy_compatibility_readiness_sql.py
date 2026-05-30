from pathlib import Path


def test_strategy_compatibility_readiness_migration_is_additive_stage06() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "20260531_0019_strategy_compatibility_market_data_readiness_v1.py"
    )
    text = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_variant_compatibility_checks" in text
    assert "CREATE TABLE IF NOT EXISTS market_data_subscription_requirements" in text
    assert "compatibility_state IN ('launchable', 'not_launchable', 'degraded')" in text
    assert "readiness_state IN ('ready', 'missing', 'stale', 'pending')" in text
    assert "strategy_spec_hash ~ '^[0-9a-f]{64}$'" in text
    assert "api_secret" not in text
    assert "DROP TABLE IF EXISTS market_data_subscription_requirements" in text
