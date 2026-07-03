from pathlib import Path


def test_rl_risk_sizing_policy_migration_is_additive_scoped_and_audited() -> None:
    migration = (
        Path(__file__).resolve().parents[4]
        / "alembic"
        / "versions"
        / "20260703_0042_rl_risk_sizing_policy_v1.py"
    )

    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS rl_risk_sizing_policies" in text
    assert "CREATE TABLE IF NOT EXISTS rl_risk_sizing_policy_audit_events" in text
    assert "strategy_id UUID NOT NULL REFERENCES strategy_strategies(strategy_id)" in text
    assert "uq_rl_risk_sizing_policies_scope" in text
    assert "owner_user_id,\n                strategy_id,\n                exchange_name" in text
    assert "WHERE active" not in text
    assert "synthetic_exit_rules_json JSONB NOT NULL DEFAULT '[]'::jsonb" in text
    assert "validation_status IN ('ready', 'blocked')" in text
    assert "DROP TABLE IF EXISTS rl_risk_sizing_policy_audit_events" in text
