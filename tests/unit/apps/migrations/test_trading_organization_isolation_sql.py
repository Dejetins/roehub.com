from __future__ import annotations

from pathlib import Path


def test_trading_schema_is_greenfield_and_organization_scoped() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    sql = (
        repo_root / "migrations/postgres/0015_trading_organization_isolation_v1.sql"
    ).read_text(encoding="utf-8")

    assert "trading organization schema requires empty greenfield table" in sql
    assert "strategy_runs_org_strategy_fk" in sql
    assert "strategy_position_ownership_org_run_fk" in sql
    assert "strategy_compatibility_org_job_fk" in sql
    assert "strategy_scenario_matrix_org_job_fk" in sql
    assert "strategy_paper_coverage_org_scenario_fk" in sql
    assert "rl_risk_policies_org_strategy_fk" in sql
    assert "rl_risk_policy_audit_org_policy_fk" in sql
    assert "rl_live_ticker_overrides_org_member_fk" in sql
    assert "rl_live_ticker_activations_org_strategy_fk" in sql
    assert "rl_live_ticker_activations_org_profile_fk" in sql
    assert "exchange_credential_versions_org_connection_fk" in sql
    assert "execution_source_events_org_idempotency_unique" in sql
    assert "execution_source_events_org_signal_fk" in sql
    assert "execution_intents_org_idempotency_unique" in sql
    assert "execution_intents_org_connection_fk" in sql
    assert "execution_intents_org_signal_fk" in sql
    assert "execution_orders_org_intent_fk" in sql
    assert "execution_notification_org_signal_fk" in sql
    assert "exchange_execution_observations_org_intent_fk" in sql
    assert "exchange_private_stream_sessions_org_connection_fk" in sql
    assert "execution_fills_org_order_fk" in sql
    assert "execution_reconciliation_org_order_fk" in sql
    assert "UPDATE " not in sql
    assert "INSERT INTO" not in sql
