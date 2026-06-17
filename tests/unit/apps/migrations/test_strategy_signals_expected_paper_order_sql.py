from pathlib import Path


def test_strategy_signals_expected_paper_order_constraint_is_bounded() -> None:
    migration = Path(
        "alembic/versions/20260617_0035_strategy_signals_expected_paper_order_v1.py"
    ).read_text()

    assert "DROP CONSTRAINT IF EXISTS strategy_signals_expected_order_stage05_chk" in migration
    assert "strategy_signals_expected_order_stage07_chk" in migration
    assert "strategy_signal_expected_order_v1" in migration
    assert "paper_no_exchange_submit" in migration
    assert "quote_notional" in migration
    assert "- 'exchange_connection_id'" in migration
    assert "DROP TABLE" not in migration
