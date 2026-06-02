from pathlib import Path


def test_execution_reconciliation_pitr_migration_is_additive_and_dedupes_facts() -> None:
    migration = Path("alembic/versions/20260602_0029_execution_reconciliation_pitr_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS execution_order_events" in text
    assert "CREATE TABLE IF NOT EXISTS execution_fills" in text
    assert "CREATE TABLE IF NOT EXISTS execution_funding_events" in text
    assert "CREATE TABLE IF NOT EXISTS execution_reconciliation_runs" in text
    assert "CREATE TABLE IF NOT EXISTS execution_ledger_retention_policies" in text
    assert "CREATE TABLE IF NOT EXISTS execution_ledger_pitr_drills" in text
    assert "UNIQUE (order_id, provider_trade_id)" in text
    assert "UNIQUE (order_id, provider_event_id)" in text
    assert "execution_orders_status_chk" in text
    assert "'reconciled'" in text
    assert "DROP TABLE IF EXISTS execution_orders" not in text
    assert "DROP TABLE IF EXISTS execution_intents" not in text


def test_execution_reconciliation_pitr_migration_seeds_money_ledger_retention() -> None:
    migration = Path("alembic/versions/20260602_0029_execution_reconciliation_pitr_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "execution_orders_money_ledger_v1" in text
    assert "execution_fills_money_ledger_v1" in text
    assert "execution_funding_events_money_ledger_v1" in text
    assert "execution_reconciliation_runs_money_ledger_v1" in text
    assert "2555" in text
    assert "archive_before_purge" in text
    assert "pitr_required" in text
