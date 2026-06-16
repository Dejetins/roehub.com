from pathlib import Path


def test_strategy_variant_scenario_matrix_migration_is_additive_stage03() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "20260617_0032_strategy_variant_scenario_matrix_v1.py"
    )
    text = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_variant_scenario_matrix_rows" in text
    assert "mode IN ('paper', 'testnet')" in text
    assert "market_type IN ('spot', 'futures')" in text
    assert "symbol = 'BTCUSDT'" in text
    assert "entry_sizing IN ('fixed_quote', 'fixed_equity_pct')" in text
    assert "risk_mode IN ('single_position_cap')" in text
    assert "direction IN ('long', 'short')" in text
    assert "order_capability IN ('paper_only', 'real_order_capable', 'unsupported')" in text
    assert "jsonb_typeof(scenario_reason_codes_json) = 'array'" in text
    assert "DROP TABLE IF EXISTS strategy_variant_scenario_matrix_rows" in text
    assert "api_secret" not in text
