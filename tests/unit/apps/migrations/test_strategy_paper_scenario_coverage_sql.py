from pathlib import Path


def test_strategy_paper_scenario_coverage_migration_is_additive_and_no_dispatch() -> None:
    migration = Path(
        "alembic/versions/20260617_0033_strategy_paper_scenario_coverage_v1.py"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_paper_scenario_coverage_results" in migration
    assert "strategy_paper_coverage_unique_scenario" in migration
    assert "CHECK (no_exchange_dispatch IS TRUE)" in migration
    assert "coverage_state IN ('covered', 'blocked')" in migration
    assert "DROP TABLE IF EXISTS strategy_paper_scenario_coverage_results" in migration
    assert "DROP TABLE IF EXISTS strategy_variant_scenario_matrix_rows" not in migration
    assert "DROP TABLE IF EXISTS paper_orders" not in migration
