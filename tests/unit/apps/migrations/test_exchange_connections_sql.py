from pathlib import Path


def test_exchange_connections_migration_creates_schema_and_backfill_path() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0008_exchange_connections_v1.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS exchange_connections" in sql
    assert "CREATE TABLE IF NOT EXISTS exchange_credential_versions" in sql
    assert "active_credential_version_id" in sql
    assert "archived_at TIMESTAMPTZ NULL" in sql
    assert "status IN ('active', 'disabled', 'archived')" in sql
    assert "exchange_connections_lifecycle_timestamps_chk" in sql
    assert "status = 'active'" in sql
    assert "disabled_at IS NULL" in sql
    assert "archived_at IS NULL" in sql
    assert "status = 'disabled'" in sql
    assert "disabled_at IS NOT NULL" in sql
    assert "status = 'archived'" in sql
    assert "archived_at IS NOT NULL" in sql
    assert "CHECK (market_type IN ('spot', 'futures'))" in sql
    assert "linear" not in sql
    assert "inverse" not in sql
    assert "FROM identity_exchange_keys" in sql
    assert "key_id AS connection_id" in sql
    assert "key_id AS credential_version_id" in sql
    assert "ON CONFLICT (connection_id) DO NOTHING" in sql
    assert "DELETE FROM exchange_connections" not in sql
    assert "DELETE FROM exchange_credential_versions" not in sql
