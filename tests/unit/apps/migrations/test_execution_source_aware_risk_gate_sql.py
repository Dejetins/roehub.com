from pathlib import Path


def test_execution_source_aware_risk_gate_migration_is_additive_and_scoped() -> None:
    migration = Path("alembic/versions/20260531_0024_execution_source_aware_risk_gate_v1.py")
    text = migration.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS execution_risk_audit_events" in text
    assert "risk_gate_accepted" in text
    assert "risk_gate_rejected" in text
    assert "CHECK (status IN ('recorded', 'accepted', 'rejected'))" in text
    assert "CHECK (risk_status IN ('not_evaluated', 'accepted', 'rejected'))" in text
    assert "idx_execution_risk_audit_events_reason" in text
    assert "owner_user_id UUID NOT NULL" in text
    assert "DROP TABLE IF EXISTS execution_intents" not in text
