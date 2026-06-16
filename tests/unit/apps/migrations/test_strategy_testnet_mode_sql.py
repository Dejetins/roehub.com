from pathlib import Path


def test_strategy_testnet_mode_migration_expands_profile_and_signal_modes() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "20260617_0031_strategy_testnet_mode_v1.py"
    )
    text = sql_path.read_text(encoding="utf-8")

    assert "DROP CONSTRAINT IF EXISTS strategy_live_profiles_mode_chk" in text
    assert "DROP CONSTRAINT IF EXISTS strategy_signals_mode_chk" in text
    assert "mode IN ('monitor_only', 'paper', 'live', 'testnet')" in text
    assert "mode IN ('monitor_only', 'paper', 'live')" in text
    assert "api_secret" not in text
