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
        "exchange_connection_archived",
        "exchange_connection_deleted",
        "exchange_connection_reclassified",
        "exchange_connection_disconnect_blocked",
        "strategy_exchange_binding_created",
        "strategy_exchange_binding_disabled",
        "strategy_exchange_binding_archived",
    ):
        assert event_type in sql


def test_identity_exchange_reclassification_audit_migration_is_additive() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0009_identity_exchange_reclassification_audit_v1.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    assert "ADD COLUMN IF NOT EXISTS target_id TEXT" in sql
    assert "exchange_connection_reclassified" in sql
    assert "exchange_connection_disconnect_blocked" in sql
    assert "strategy_exchange_binding_created" in sql
    assert "strategy_exchange_binding_disabled" in sql
    assert "strategy_exchange_binding_archived" in sql
    assert "idx_identity_audit_events_target_created" in sql
