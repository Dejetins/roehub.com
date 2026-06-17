from pathlib import Path


def test_manual_paper_orders_source_event_migration_is_additive() -> None:
    text = Path(
        "alembic/versions/20260618_0036_manual_paper_orders_source_event_v1.py"
    ).read_text()

    assert "ADD COLUMN IF NOT EXISTS source_event_id UUID NULL" in text
    assert "idx_paper_orders_source_event_unique" in text
    assert "DROP TABLE IF EXISTS paper_orders" not in text
