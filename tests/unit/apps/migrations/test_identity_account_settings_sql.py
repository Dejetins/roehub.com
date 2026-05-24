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
    assert "ADD COLUMN IF NOT EXISTS username" in sql
    assert "ADD COLUMN IF NOT EXISTS email" in sql
    assert "ADD COLUMN IF NOT EXISTS telegram_discord" in sql
    assert "display_name" in sql
    assert "ADD COLUMN IF NOT EXISTS integration_key" in sql
    assert "ADD COLUMN IF NOT EXISTS webhook_url_masked" in sql
    assert "PRIMARY KEY (owner_user_id, integration_key)" in sql
    assert "WHEN 'webhook_alerts' THEN 'slack'" in sql
    assert "ADD COLUMN IF NOT EXISTS summary" in sql
