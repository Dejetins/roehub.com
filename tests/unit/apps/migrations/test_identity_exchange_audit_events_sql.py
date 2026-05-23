from pathlib import Path


def test_identity_exchange_audit_events_migration_extends_check_constraint() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0007_identity_exchange_audit_events_v1.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    assert "identity_audit_events_type_check" in sql
    for event_type in (
        "exchange_key_created",
        "exchange_key_deleted",
        "exchange_connection_created",
        "exchange_connection_validated",
        "exchange_connection_validation_failed",
        "exchange_credential_rotated",
        "exchange_connection_disabled",
        "exchange_connection_deleted",
    ):
        assert event_type in sql
