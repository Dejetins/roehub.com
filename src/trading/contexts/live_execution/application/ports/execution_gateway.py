from __future__ import annotations

from typing import Protocol

from trading.contexts.live_execution.domain import (
    ExecutionAccountSafetyState,
    ExecutionGatewayAuditEvent,
    ExecutionKillSwitchState,
    ExecutionMainnetApproval,
    ExecutionPolicyPrincipal,
    ExecutionProviderRegistration,
    ExecutionSubmitGuardDecision,
    ExecutionSubmitGuardQuery,
)


class ExecutionGatewayPolicyRepository(Protocol):
    def register_provider(
        self,
        *,
        provider: ExecutionProviderRegistration,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionProviderRegistration: ...

    def set_account_safety_state(
        self,
        *,
        state: ExecutionAccountSafetyState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionAccountSafetyState: ...

    def set_kill_switch(
        self,
        *,
        state: ExecutionKillSwitchState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionKillSwitchState: ...

    def approve_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval: ...

    def revoke_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval: ...

    def evaluate_and_record(
        self, *, query: ExecutionSubmitGuardQuery
    ) -> ExecutionSubmitGuardDecision: ...


class FailClosedExecutionGatewayPolicyRepository(ExecutionGatewayPolicyRepository):
    def register_provider(
        self,
        *,
        provider: ExecutionProviderRegistration,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionProviderRegistration:
        raise RuntimeError("execution gateway policy repository unavailable")

    def set_account_safety_state(
        self,
        *,
        state: ExecutionAccountSafetyState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionAccountSafetyState:
        raise RuntimeError("execution gateway policy repository unavailable")

    def set_kill_switch(
        self,
        *,
        state: ExecutionKillSwitchState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionKillSwitchState:
        raise RuntimeError("execution gateway policy repository unavailable")

    def approve_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval:
        raise RuntimeError("execution gateway policy repository unavailable")

    def revoke_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval:
        raise RuntimeError("execution gateway policy repository unavailable")

    def evaluate_and_record(
        self, *, query: ExecutionSubmitGuardQuery
    ) -> ExecutionSubmitGuardDecision:
        return ExecutionSubmitGuardDecision(
            status="rejected",
            reason="execution_gateway_policy_unavailable",
            check_name="gateway_policy",
            phase=query.phase,
            evaluated_at=query.evaluated_at,
        )
