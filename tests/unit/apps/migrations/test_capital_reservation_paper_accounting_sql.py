from pathlib import Path


def test_capital_reservation_paper_accounting_migration_is_additive_and_scoped() -> None:
    migration = Path("alembic/versions/20260531_0022_capital_reservation_paper_accounting_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_capital_reservations" in text
    assert "CREATE TABLE IF NOT EXISTS paper_orders" in text
    assert "CREATE TABLE IF NOT EXISTS paper_fills" in text
    assert "CREATE TABLE IF NOT EXISTS strategy_paper_accounting" in text
    assert "paper_orders_signal_unique" in text
    assert "strategy_paper_accounting_fill_unique" in text
    assert "DROP TABLE IF EXISTS strategy_capital_reservations" in text
