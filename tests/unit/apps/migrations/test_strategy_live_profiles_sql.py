from pathlib import Path


def test_strategy_live_profiles_migration_is_additive_and_safe_default_ready() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "20260530_0017_strategy_live_profiles_v1.py"
    )
    text = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_live_profiles" in text
    assert "mode IN ('monitor_only', 'paper', 'live')" in text
    assert "sizing_method IN ('fixed_quote', 'fixed_equity_pct')" in text
    assert "readiness_status IN ('ready', 'blocked')" in text
    assert "owner_user_id, strategy_id" in text
    assert "exchange_connection_id" in text
    assert "api_secret" not in text
    assert "DROP TABLE IF EXISTS strategy_live_profiles" in text
