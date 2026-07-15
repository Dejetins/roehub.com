from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]


def test_control_operation_audit_migration_is_durable_and_idempotent() -> None:
    sql = (
        _ROOT / "migrations/postgres/0021_control_operation_audit_v1.sql"
    ).read_text(encoding="utf-8")

    for required in (
        "CREATE TABLE IF NOT EXISTS control_operation_audit_events",
        "entry_hash TEXT PRIMARY KEY",
        "sequence BIGINT NOT NULL UNIQUE",
        "operation_id UUID NOT NULL",
        "payload JSONB NOT NULL",
        "CREATE TABLE IF NOT EXISTS control_operation_audit_cursor",
        "REFERENCES control_operation_audit_events(entry_hash)",
        "ON CONFLICT (singleton) DO NOTHING",
    ):
        assert required in sql
