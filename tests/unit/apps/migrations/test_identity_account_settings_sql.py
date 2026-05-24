from pathlib import Path


def test_identity_account_settings_migration_repairs_preexisting_stage_tables() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0006_identity_account_settings_v1.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS identity_user_preferences" in sql
    assert "CREATE TABLE IF NOT EXISTS identity_audit_events" in sql
    assert "ADD COLUMN IF NOT EXISTS autorefresh_preset" in sql
    assert "ADD COLUMN IF NOT EXISTS refresh_interval_seconds" in sql
    assert "ADD COLUMN IF NOT EXISTS summary" in sql
