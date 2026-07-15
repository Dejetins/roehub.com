from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import ExecutionGatewayPolicyRepository
from trading.contexts.live_execution.domain import (
    ExecutionAccountSafetyState,
    ExecutionGatewayAuditEvent,
    ExecutionGatewayPolicyError,
    ExecutionKillSwitchState,
    ExecutionMainnetApproval,
    ExecutionPolicyPrincipal,
    ExecutionProviderRegistration,
    validate_mainnet_approval,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class ExecutionGatewayPolicyService:
    def __init__(self, *, repository: ExecutionGatewayPolicyRepository) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionGatewayPolicyService requires repository")
        self._repository = repository

    def register_provider(
        self,
        *,
        provider: ExecutionProviderRegistration,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionProviderRegistration:
        if provider.approved_by_user_id != principal.user_id:
            raise ExecutionGatewayPolicyError(reason="execution_policy_principal_mismatch")
        if provider.provider_kind not in {"core", "verified"}:
            raise ExecutionGatewayPolicyError(reason="execution_plugin_submit_forbidden")
        return self._repository.register_provider(
            provider=provider,
            principal=principal,
            audit=_audit(
                event_type="execution_provider_registered",
                decision="accepted",
                reason="execution_provider_allowlisted",
                actor_user_id=provider.approved_by_user_id,
                created_at=provider.updated_at,
                metadata={
                    "provider_id": provider.provider_id,
                    "provider_kind": provider.provider_kind,
                    "exchange": provider.exchange_name,
                },
            ),
        )

    def set_account_safety_state(
        self, *, state: ExecutionAccountSafetyState, principal: ExecutionPolicyPrincipal
    ) -> ExecutionAccountSafetyState:
        if state.updated_by_user_id != principal.user_id:
            raise ExecutionGatewayPolicyError(reason="execution_policy_principal_mismatch")
        return self._repository.set_account_safety_state(
            state=state,
            principal=principal,
            audit=_audit(
                event_type="execution_account_safety_changed",
                decision="accepted",
                reason="execution_account_safety_persisted",
                actor_user_id=state.updated_by_user_id,
                created_at=state.updated_at,
                organization_id=state.organization_id,
                owner_user_id=state.owner_user_id,
                exchange_connection_id=state.exchange_connection_id,
                metadata={"mode": state.mode},
            ),
        )

    def set_kill_switch(
        self, *, state: ExecutionKillSwitchState, principal: ExecutionPolicyPrincipal
    ) -> ExecutionKillSwitchState:
        if state.updated_by_user_id != principal.user_id:
            raise ExecutionGatewayPolicyError(reason="execution_policy_principal_mismatch")
        return self._repository.set_kill_switch(
            state=state,
            principal=principal,
            audit=_audit(
                event_type="execution_kill_switch_changed",
                decision="accepted",
                reason=state.reason,
                actor_user_id=state.updated_by_user_id,
                created_at=state.updated_at,
                organization_id=state.organization_id,
                exchange_connection_id=state.exchange_connection_id,
                metadata={
                    "active": "true" if state.active else "false",
                    "scope_type": state.scope_type,
                },
            ),
        )

    def approve_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        principal: ExecutionPolicyPrincipal,
        now: datetime,
    ) -> ExecutionMainnetApproval:
        validate_mainnet_approval(approval=approval, principal=principal, now=now)
        return self._repository.approve_mainnet(
            approval=approval,
            principal=principal,
            audit=_audit(
                event_id=approval.audit_event_id,
                event_type="mainnet_approval_created",
                decision="accepted",
                reason="mainnet_owner_approved",
                actor_user_id=approval.owner_user_id,
                created_at=now,
                organization_id=approval.organization_id,
                owner_user_id=approval.owner_user_id,
                exchange_connection_id=approval.exchange_connection_id,
                approval_id=approval.approval_id,
                metadata={
                    "exchange": approval.exchange_name,
                    "market_type": approval.market_type,
                    "provider_id": approval.provider_id,
                },
            ),
        )

    def revoke_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        principal: ExecutionPolicyPrincipal,
        revoked_at: datetime,
        reason: str,
    ) -> ExecutionMainnetApproval:
        if approval.owner_user_id != principal.user_id:
            raise ExecutionGatewayPolicyError(reason="mainnet_owner_approval_required")
        if approval.revoked_at is not None:
            return approval
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ExecutionGatewayPolicyError(reason="mainnet_revocation_reason_required")
        revoked = replace(
            approval,
            revoked_at=revoked_at,
            revocation_reason=normalized_reason,
        )
        return self._repository.revoke_mainnet(
            approval=revoked,
            principal=principal,
            audit=_audit(
                event_type="mainnet_approval_revoked",
                decision="accepted",
                reason=normalized_reason,
                actor_user_id=principal.user_id,
                created_at=revoked_at,
                organization_id=approval.organization_id,
                owner_user_id=approval.owner_user_id,
                exchange_connection_id=approval.exchange_connection_id,
                approval_id=approval.approval_id,
            ),
        )


def _audit(
    *,
    event_type: str,
    decision: str,
    reason: str,
    actor_user_id: UserId | None,
    created_at: datetime,
    event_id: UUID | None = None,
    organization_id: OrganizationId | None = None,
    owner_user_id: UserId | None = None,
    exchange_connection_id: UUID | None = None,
    intent_id: UUID | None = None,
    approval_id: UUID | None = None,
    metadata: dict[str, str] | None = None,
) -> ExecutionGatewayAuditEvent:
    return ExecutionGatewayAuditEvent(
        event_id=event_id or uuid4(),
        organization_id=organization_id,
        owner_user_id=owner_user_id,
        exchange_connection_id=exchange_connection_id,
        intent_id=intent_id,
        approval_id=approval_id,
        event_type=event_type,
        decision=decision,
        reason=reason,
        actor_user_id=actor_user_id,
        created_at=created_at,
        metadata=metadata or {},
    )
