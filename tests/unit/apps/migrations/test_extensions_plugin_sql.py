from __future__ import annotations

from pathlib import Path


def test_extensions_plugin_sql_has_tenant_scope_immutability_and_no_runtime_escape() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    sql = (
        repo_root / "migrations/postgres/0017_extensions_plugin_platform_v1alpha1.sql"
    ).read_text(encoding="utf-8")

    for table in (
        "extensions_publisher_keys",
        "extensions_plugin_packages",
        "extensions_plugin_installations",
        "extensions_plugin_instances",
        "extensions_plugin_operations",
        "extensions_plugin_events",
    ):
        assert f"CREATE TABLE {table}" in sql
    assert "extensions_plugin_events_immutable_update" in sql
    assert "extensions_plugin_events_immutable_delete" in sql
    assert "UNIQUE (organization_id, idempotency_key)" in sql
    assert "request_payload JSONB NOT NULL" in sql
    assert "request_payload -> 'config'" in sql
    assert "roehub.plugin.rpc/v1alpha1" in sql
    assert "docker.sock" not in sql
    assert "CREATE EXTENSION" not in sql
