from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

from trading.contexts.live_execution.application.ports import ExecutionGatewayPolicyRepository
from trading.contexts.live_execution.domain import (
    ExecutionAccountSafetyState,
    ExecutionCurrentRiskSnapshot,
    ExecutionGatewayAuditEvent,
    ExecutionKillSwitchState,
    ExecutionMainnetApproval,
    ExecutionPolicyPrincipal,
    ExecutionProviderRegistration,
    ExecutionSubmitGuardDecision,
    ExecutionSubmitGuardQuery,
    evaluate_execution_submit_guard,
)


class InMemoryExecutionGatewayPolicyRepository(ExecutionGatewayPolicyRepository):
    def __init__(self) -> None:
        self.providers: dict[str, ExecutionProviderRegistration] = {}
        self.safety_states: dict[tuple[object, object], ExecutionAccountSafetyState] = {}
        self.kill_switches: dict[
            tuple[str, object | None, object | None], ExecutionKillSwitchState
        ] = {}
        self.approvals: dict[object, ExecutionMainnetApproval] = {}
        self.audit_events: list[ExecutionGatewayAuditEvent] = []

    def register_provider(
        self,
        *,
        provider: ExecutionProviderRegistration,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionProviderRegistration:
        _ = principal
        self.providers[provider.provider_id] = provider
        self.audit_events.append(audit)
        return provider

    def set_account_safety_state(
        self,
        *,
        state: ExecutionAccountSafetyState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionAccountSafetyState:
        _ = principal
        self.safety_states[(state.organization_id, state.exchange_connection_id)] = state
        self.audit_events.append(audit)
        return state

    def set_kill_switch(
        self,
        *,
        state: ExecutionKillSwitchState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionKillSwitchState:
        _ = principal
        key = (state.scope_type, state.organization_id, state.exchange_connection_id)
        self.kill_switches[key] = state
        self.audit_events.append(audit)
        return state

    def approve_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval:
        _ = principal
        self.approvals[approval.approval_id] = approval
        self.audit_events.append(audit)
        return approval

    def revoke_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval:
        _ = principal
        existing = self.approvals.get(approval.approval_id)
        if existing is None:
            raise ValueError("mainnet approval not found")
        updated = replace(
            existing,
            revoked_at=approval.revoked_at,
            revocation_reason=approval.revocation_reason,
        )
        self.approvals[approval.approval_id] = updated
        self.audit_events.append(audit)
        return updated

    def evaluate_and_record(
        self, *, query: ExecutionSubmitGuardQuery
    ) -> ExecutionSubmitGuardDecision:
        safety = self.safety_states.get(
            (query.intent.organization_id, query.intent.exchange_connection_id)
        )
        provider = self.providers.get(query.adapter.provider_id)
        approval = self._latest_approval(query=query)
        decision = evaluate_execution_submit_guard(
            query=query,
            safety=safety,
            provider=provider,
            kill_switch_active=self._kill_switch_active(query=query),
            approval=approval,
            risk_snapshot=ExecutionCurrentRiskSnapshot(
                account_snapshot_fresh=safety is not None,
                config_guard_verified=safety is not None,
                daily_notional_used=Decimal("0"),
                account_exposure_notional=Decimal("0"),
                account_exposure_complete=True,
                observed_at=safety.updated_at if safety is not None else None,
            ),
        )
        audit_event_id = uuid4()
        decision = replace(decision, audit_event_id=audit_event_id)
        self.audit_events.append(
            ExecutionGatewayAuditEvent(
                event_id=audit_event_id,
                organization_id=query.intent.organization_id,
                owner_user_id=query.intent.owner_user_id,
                exchange_connection_id=query.intent.exchange_connection_id,
                intent_id=query.intent.intent_id,
                approval_id=decision.approval_id,
                event_type=f"execution_submit_guard_{query.phase}",
                decision=decision.status,
                reason=decision.reason,
                actor_user_id=None,
                created_at=query.evaluated_at,
                metadata=decision.metadata,
            )
        )
        return decision

    def _latest_approval(
        self, *, query: ExecutionSubmitGuardQuery
    ) -> ExecutionMainnetApproval | None:
        matches = [
            item
            for item in self.approvals.values()
            if item.organization_id == query.intent.organization_id
            and item.exchange_connection_id == query.intent.exchange_connection_id
            and item.exchange_name == query.connection.exchange_name
            and item.market_type == query.intent.market_type
            and item.provider_id == query.adapter.provider_id
        ]
        return max(matches, key=lambda item: item.approved_at) if matches else None

    def _kill_switch_active(self, *, query: ExecutionSubmitGuardQuery) -> bool:
        for state in self.kill_switches.values():
            if not state.active:
                continue
            if state.scope_type == "installation":
                return True
            if (
                state.scope_type == "organization"
                and state.organization_id == query.intent.organization_id
            ):
                return True
            if (
                state.scope_type == "account"
                and state.organization_id == query.intent.organization_id
                and state.exchange_connection_id == query.intent.exchange_connection_id
            ):
                return True
        return False
