from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_execution_gateway_migration_is_persisted_fail_closed_and_audited() -> None:
    sql = (
        _REPO_ROOT / "migrations/postgres/0020_execution_gateway_mainnet_safety_v1.sql"
    ).read_text(encoding="utf-8")

    for required in (
        "execution gateway safety migration requires empty greenfield execution tables",
        "canonical_intent_hash",
        "CREATE TABLE execution_gateway_audit_events",
        "CREATE TABLE execution_provider_allowlist",
        "provider_kind IN ('core', 'verified')",
        "CREATE TABLE execution_account_safety_state",
        "max_order_notional NUMERIC NOT NULL",
        "daily_notional_limit NUMERIC NOT NULL",
        "max_account_exposure_notional NUMERIC NOT NULL",
        "risk_valid_until TIMESTAMPTZ NOT NULL",
        "CREATE TABLE execution_kill_switch_state",
        "CREATE TABLE execution_mainnet_approvals",
        "recent_auth_session_id UUID NOT NULL",
        "approved_at - recent_auth_at <= INTERVAL '10 minutes'",
        "submit_claim_expires_at",
        "submit_guard_audit_event_id",
        "mainnet_approval_id",
        "execution gateway audit events are immutable",
        "execution_provider_allowlist_no_delete",
        "execution_account_safety_no_delete",
        "execution_kill_switch_no_delete",
        "execution_mainnet_approval_no_delete",
        "execution_provider_allowlist_audited_update",
        "execution_account_safety_audited_update",
        "execution_kill_switch_audited_update",
        "mainnet approval is immutable except first revocation",
        "'emulator_submitted'",
    ):
        assert required in sql

    assert "INSERT INTO execution_provider_allowlist" not in sql
    assert "UPDATE execution_intents" not in sql
