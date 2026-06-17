from pathlib import Path


def test_strategy_variant_provenance_source_uniqueness_includes_launch_hash() -> None:
    migration = Path(
        "alembic/versions/20260617_0034_strategy_variant_provenance_launch_hash_unique.py"
    ).read_text(encoding="utf-8")

    assert "DROP INDEX IF EXISTS idx_strategy_backtest_variant_provenance_source" in migration
    assert "launch_request_hash" in migration
    assert "(user_id, source_job_id, source_variant_key, strategy_spec_hash)" in migration
    assert "DROP TABLE" not in migration
