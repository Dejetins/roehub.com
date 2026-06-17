from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping
from uuid import UUID

from trading.shared_kernel.primitives import UserId

from .execution_source import ExecutionIntent, ExecutionSourceType

RiskStatus = Literal["accepted", "rejected"]


@dataclass(frozen=True, slots=True)
class ExecutionRiskContext:
    exchange_connection_active: bool = False
    secret_custody_ready: bool = False
    source_authorized: bool = False
    strategy_variant_compatible: bool = False
    market_data_state: str = "missing"
    strategy_binding_active: bool = False
    strategy_live_profile_ready: bool = False
    strategy_run_active: bool = False
    exchange_config_verified: bool = False
    account_state_fresh: bool = False
    position_ownership_active: bool = False
    capital_reservation_active: bool = False
    capital_reservation_sufficient: bool = False
    paper_accounting_ready: bool = False
    paper_no_exchange_submit: bool = False
    manual_recent_auth: bool = False
    ml_agent_policy_active: bool = False
    kill_switch_open: bool = False
    environment_policy_allows: bool = False
    max_order_size_ok: bool = False
    daily_limit_ok: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionRiskDecision:
    status: RiskStatus
    reason: str
    check_name: str

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


@dataclass(frozen=True, slots=True)
class ExecutionRiskAuditEvent:
    event_id: UUID
    intent_id: UUID
    source_event_id: UUID
    owner_user_id: UserId
    source_type: ExecutionSourceType
    event_type: str
    risk_status: RiskStatus
    risk_reason: str
    check_name: str
    metadata_json: Mapping[str, str]
    created_at: datetime


def evaluate_execution_risk(
    *,
    intent: ExecutionIntent,
    context: ExecutionRiskContext | None,
) -> ExecutionRiskDecision:
    if context is None:
        return _reject(check_name="risk_context", reason="risk_state_unavailable")

    common_checks: tuple[tuple[str, bool, str], ...] = (
        (
            "exchange_connection_active",
            context.exchange_connection_active,
            "exchange_connection_inactive",
        ),
        ("secret_custody_ready", context.secret_custody_ready, "secret_custody_unavailable"),
        ("source_authorized", context.source_authorized, "source_unauthorized"),
        ("kill_switch_open", context.kill_switch_open, "kill_switch_closed"),
        (
            "environment_policy",
            context.environment_policy_allows,
            "mainnet_canary_not_approved",
        ),
        ("max_order_size", context.max_order_size_ok, "max_order_size_exceeded"),
        ("daily_limit", context.daily_limit_ok, "daily_notional_limit_exceeded"),
    )
    common_decision = _first_rejection(common_checks)
    if common_decision is not None:
        return common_decision

    if intent.source_type == "strategy_signal":
        strategy_decision = _evaluate_strategy_signal_context(context=context)
        if strategy_decision is not None:
            return strategy_decision
    elif intent.source_type == "manual_request":
        if context.paper_no_exchange_submit:
            manual_paper_decision = _evaluate_manual_paper_context(context=context)
            if manual_paper_decision is not None:
                return manual_paper_decision
            return _reject(
                check_name="paper_no_exchange_submit",
                reason="paper_no_exchange_submit",
            )
        manual_decision = _evaluate_account_context(context=context)
        if manual_decision is not None:
            return manual_decision
        if not context.manual_recent_auth:
            return _reject(check_name="manual_recent_auth", reason="manual_recent_auth_required")
    elif intent.source_type == "ml_agent_decision":
        ml_decision = _evaluate_account_context(context=context)
        if ml_decision is not None:
            return ml_decision
        if not context.ml_agent_policy_active:
            return _reject(check_name="ml_agent_policy", reason="ml_agent_policy_missing")
    elif intent.source_type == "ops_test":
        ops_decision = _evaluate_account_context(context=context)
        if ops_decision is not None:
            return ops_decision

    return ExecutionRiskDecision(status="accepted", reason="risk_gate_accepted", check_name="all")


def _evaluate_strategy_signal_context(
    *, context: ExecutionRiskContext
) -> ExecutionRiskDecision | None:
    checks: tuple[tuple[str, bool, str], ...] = (
        (
            "strategy_variant_compatible",
            context.strategy_variant_compatible,
            "strategy_variant_incompatible",
        ),
        ("strategy_binding_active", context.strategy_binding_active, "strategy_binding_missing"),
        (
            "strategy_live_profile_ready",
            context.strategy_live_profile_ready,
            "strategy_live_profile_blocked",
        ),
        ("strategy_run_active", context.strategy_run_active, "strategy_run_inactive"),
        (
            "position_ownership_active",
            context.position_ownership_active,
            "position_ownership_conflict",
        ),
        (
            "capital_reservation_active",
            context.capital_reservation_active,
            "capital_reservation_missing",
        ),
        (
            "capital_reservation_sufficient",
            context.capital_reservation_sufficient,
            "capital_reservation_insufficient",
        ),
        ("paper_accounting_ready", context.paper_accounting_ready, "paper_accounting_unavailable"),
    )
    if context.market_data_state != "ready":
        return _reject(
            check_name="market_data_ready",
            reason=f"market_data_{context.market_data_state or 'missing'}",
        )
    if context.paper_no_exchange_submit:
        strategy_decision = _first_rejection(checks)
        if strategy_decision is not None:
            return strategy_decision
        return _reject(
            check_name="paper_no_exchange_submit",
            reason="paper_no_exchange_submit",
        )
    account_decision = _evaluate_account_context(context=context)
    if account_decision is not None:
        return account_decision
    return _first_rejection(checks)


def _evaluate_manual_paper_context(
    *, context: ExecutionRiskContext
) -> ExecutionRiskDecision | None:
    checks: tuple[tuple[str, bool, str], ...] = (
        (
            "strategy_live_profile_ready",
            context.strategy_live_profile_ready,
            "strategy_live_profile_blocked",
        ),
        ("strategy_run_active", context.strategy_run_active, "strategy_run_inactive"),
        (
            "position_ownership_active",
            context.position_ownership_active,
            "position_ownership_conflict",
        ),
        (
            "capital_reservation_active",
            context.capital_reservation_active,
            "capital_reservation_missing",
        ),
        (
            "capital_reservation_sufficient",
            context.capital_reservation_sufficient,
            "capital_reservation_insufficient",
        ),
        ("paper_accounting_ready", context.paper_accounting_ready, "paper_accounting_unavailable"),
        ("manual_recent_auth", context.manual_recent_auth, "manual_recent_auth_required"),
    )
    if context.market_data_state != "ready":
        return _reject(
            check_name="market_data_ready",
            reason=f"market_data_{context.market_data_state or 'missing'}",
        )
    return _first_rejection(checks)


def _evaluate_account_context(*, context: ExecutionRiskContext) -> ExecutionRiskDecision | None:
    checks: tuple[tuple[str, bool, str], ...] = (
        ("exchange_config_verified", context.exchange_config_verified, "exchange_config_mismatch"),
        ("account_state_fresh", context.account_state_fresh, "account_projection_stale"),
    )
    return _first_rejection(checks)


def _first_rejection(
    checks: tuple[tuple[str, bool, str], ...],
) -> ExecutionRiskDecision | None:
    for check_name, passed, reason in checks:
        if not passed:
            return _reject(check_name=check_name, reason=reason)
    return None


def _reject(*, check_name: str, reason: str) -> ExecutionRiskDecision:
    return ExecutionRiskDecision(status="rejected", reason=reason, check_name=check_name)
