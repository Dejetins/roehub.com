from pathlib import Path


def test_strategy_exchange_bindings_migration_creates_usage_registry() -> None:
    sql_path = (
        Path(__file__).resolve().parents[4]
        / "migrations"
        / "postgres"
        / "0010_strategy_exchange_bindings_v1.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_exchange_bindings" in sql
    assert "owner_user_id UUID NOT NULL REFERENCES identity_users" in sql
    assert "strategy_id UUID NOT NULL REFERENCES strategy_strategies" in sql
    assert "exchange_connection_id UUID NOT NULL REFERENCES exchange_connections" in sql
    assert "usage_mode IN ('trading')" in sql
    assert "binding_status IN ('active', 'paused', 'disabled', 'archived')" in sql
    assert "idx_strategy_exchange_bindings_active_unique" in sql
    assert "idx_strategy_exchange_bindings_connection_active" in sql
    for event_type in (
        "strategy_exchange_binding_created",
        "strategy_exchange_binding_disabled",
        "strategy_exchange_binding_archived",
        "exchange_connection_disconnect_blocked",
    ):
        assert event_type in sql
    assert "DELETE FROM strategy_exchange_bindings" not in sql
