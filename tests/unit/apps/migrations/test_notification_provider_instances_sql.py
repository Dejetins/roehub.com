from pathlib import Path


def test_notification_provider_schema_is_greenfield_scoped_and_secret_safe() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    sql = (
        repo_root
        / "migrations/postgres/0016_notification_provider_instances_v1.sql"
    ).read_text(encoding="utf-8")

    assert "notification provider schema requires empty greenfield table" in sql
    assert "NotificationProvider/v1" in sql
    assert "CREATE TABLE notification_provider_packages" in sql
    assert "CREATE TABLE notification_provider_instances" in sql
    assert "CREATE TABLE notification_telegram_update_cursors" in sql
    assert "CREATE TABLE notification_telegram_command_registry" in sql
    assert "CREATE TABLE notification_telegram_binding_codes" in sql
    assert "CREATE TABLE notification_telegram_recipient_bindings" in sql
    assert "notification_provider_instances_no_raw_secrets_chk" in sql
    assert "notification_provider_instances_secret_ref_chk" in sql
    assert "roehub/telegram/providers/%s/%s#bot_token" in sql
    assert "roehub/plugins/%s/%s" in sql
    assert "recipient_secret_ref" in sql
    assert "notification_enforce_provider_scope" in sql
    assert "notification_deliveries_org_route_provider_fk" in sql
    assert "notification_delivery_attempts_delivery_fk" in sql
    assert "DROP TABLE" not in sql.upper()


def test_notification_provider_schema_has_durable_idempotency_and_no_fallback() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    sql = (
        repo_root
        / "migrations/postgres/0016_notification_provider_instances_v1.sql"
    ).read_text(encoding="utf-8")

    assert "PRIMARY KEY (provider_instance_id, telegram_update_id)" in sql
    assert "UNIQUE (organization_id, idempotency_key)" in sql
    assert "UNIQUE (organization_id, route_id, provider_instance_id, provider_key)" in sql
    assert "notification_deliveries_replay_source_fk" in sql
    assert "replayed_from_delivery_id" in sql
    assert "fallback" not in sql.casefold()
