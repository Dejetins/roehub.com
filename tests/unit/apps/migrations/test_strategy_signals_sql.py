from pathlib import Path


def test_strategy_signals_migration_is_additive_stage05_journal() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "20260531_0018_strategy_signals_v1.py"
    )
    text = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_signals" in text
    assert "strategy_run_id UUID NOT NULL REFERENCES strategy_runs" in text
    assert "live_profile_id UUID NULL REFERENCES strategy_live_profiles" in text
    assert "mode IN ('monitor_only', 'paper', 'live')" in text
    assert "signal_action IN ('none', 'open', 'close', 'reduce', 'reverse')" in text
    assert "outcome IN ('warmup', 'no_signal', 'signal', 'blocked')" in text
    assert "expected_order_json = '{}'::jsonb" in text
    assert "api_secret" not in text
    assert "DROP TABLE IF EXISTS strategy_signals" in text
