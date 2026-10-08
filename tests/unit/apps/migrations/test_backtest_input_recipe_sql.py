"""Bounded additive-schema contract. Real execution is a separate disposable proof."""

from pathlib import Path

from apps.migrations.storage import _load_postgres_phases

ROOT = Path(__file__).resolve().parents[4]


def test_recipe_migration_is_ordered_additive_and_uses_physical_mutex_key() -> None:
    phases = _load_postgres_phases(ROOT / "migrations/postgres/manifest.json")
    assert list(phases)[-1] == "backtest-input-recipe-0028"
    sql = (ROOT / "migrations/postgres/0028_backtest_input_recipe_v1.sql").read_text()
    for destructive in ("DROP ", "DELETE FROM ", "UPDATE backtest_jobs", "ALTER COLUMN"):
        assert destructive not in sql
    assert "ADD COLUMN IF NOT EXISTS input_recipe_json JSONB" in sql
    assert "ADD COLUMN IF NOT EXISTS preparation_provenance_json JSONB" in sql
    owner = sql.split("CREATE TABLE IF NOT EXISTS backtest_artifact_slot_ownership (", 1)[1]
    owner = owner.split("CREATE TABLE IF NOT EXISTS backtest_artifact_slot_readers", 1)[0]
    assert "PRIMARY KEY (exchange, market_type, symbol, slot)" in owner
    assert "writer_epoch BIGINT NOT NULL DEFAULT 0" in owner
    assert "generation BIGINT CHECK (generation > 0)" in owner
    assert "owner_token UUID" in owner and "parent_incarnation UUID" in owner
    assert "owner_kind IN ('job', 'lazy') AND organization_id IS NOT NULL" in sql
    assert "ON DELETE RESTRICT" in sql
