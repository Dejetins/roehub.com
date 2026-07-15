from pathlib import Path


def test_greenfield_organization_schema_is_fail_closed_and_immutable() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0011_identity_organizations_rbac_audit_v1.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    for table in (
        "identity_installations",
        "identity_installation_owners",
        "identity_organizations",
        "identity_memberships",
        "identity_invitations",
        "identity_plugin_permissions",
        "identity_support_access_grants",
        "identity_administrative_audit_events",
    ):
        assert f"CREATE TABLE {table}" in sql

    assert "organization schema requires empty greenfield table" in sql
    assert "identity_memberships_last_owner_guard" in sql
    assert "identity_installation_owners_last_owner_guard" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "identity_admin_audit_immutable" in sql
    assert "identity_admin_audit_no_sensitive_keys_chk" in sql
    assert "expires_at <= created_at + INTERVAL '24 hours'" in sql


def test_resource_links_use_composite_same_organization_foreign_keys() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0011_identity_organizations_rbac_audit_v1.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    for table in (
        "exchange_connections",
        "strategy_strategies",
        "backtest_jobs",
        "strategy_backtest_variant_provenance",
        "strategy_position_ownership",
        "exchange_account_snapshots",
        "exchange_position_snapshots",
        "strategy_exchange_bindings",
    ):
        assert f"ALTER TABLE {table} ADD COLUMN organization_id UUID NOT NULL" in sql

    for constraint in (
        "strategy_provenance_org_strategy_fk",
        "strategy_provenance_org_job_fk",
        "strategy_position_ownership_org_strategy_fk",
        "strategy_position_ownership_org_connection_fk",
        "exchange_position_snapshots_org_account_fk",
        "exchange_position_snapshots_org_connection_fk",
        "strategy_exchange_bindings_org_strategy_fk",
        "strategy_exchange_bindings_org_connection_fk",
    ):
        assert f"CONSTRAINT {constraint}" in sql

    assert sql.count("REFERENCES identity_memberships(organization_id, user_id)") >= 8
