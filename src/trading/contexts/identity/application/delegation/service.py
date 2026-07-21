from __future__ import annotations

from trading.contexts.identity.application.authorization import CapabilityAuthorizationService
from trading.contexts.identity.application.authorization.models import (
    AuthorizationDecision,
    AuthorizationDenyReason,
    AuthorizationRequest,
    CapabilityId,
)
from trading.contexts.identity.application.delegation.models import (
    DELEGABLE_CAPABILITIES,
    DELEGABLE_CAPABILITY_GRANTEE_ROLES,
    DelegatedCapabilityDecision,
    DelegatedCapabilityScope,
    DelegationAuthoritySource,
    DelegationCommandResult,
    DelegationDenyReason,
    DelegationEvaluationRequest,
    DelegationResourceScope,
    GrantDelegatedCapabilityCommand,
    RevokeDelegatedCapabilityCommand,
)
from trading.contexts.identity.application.ports.delegation_repository import (
    DelegationRepository,
    DelegationRepositoryConflictError,
)
from trading.contexts.identity.application.ports.organization_repository import (
    OrganizationRepository,
)

_ELIGIBLE_GRANTEE_ROLES = frozenset({"admin", "trader"})
_DELEGATION_CANDIDATE_BASE_DENIALS = frozenset(
    {
        AuthorizationDenyReason.ROLE_CAPABILITY_DENIED,
        AuthorizationDenyReason.RESOURCE_CONTEXT_REQUIRED,
        AuthorizationDenyReason.OWNERSHIP_REQUIRED,
    }
)


class DelegatedCapabilityService:
    """Compose persisted exact delegations with the accepted default-deny capability kernel."""

    def __init__(
        self,
        *,
        capability_authorization_service: CapabilityAuthorizationService,
        organization_repository: OrganizationRepository,
        delegation_repository: DelegationRepository,
    ) -> None:
        if capability_authorization_service is None:  # type: ignore[truthy-bool]
            raise ValueError("DelegatedCapabilityService requires capability_authorization_service")
        if organization_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("DelegatedCapabilityService requires organization_repository")
        if delegation_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("DelegatedCapabilityService requires delegation_repository")
        self._capability_authorization_service = capability_authorization_service
        self._organization_repository = organization_repository
        self._delegation_repository = delegation_repository

    def grant(self, *, command: GrantDelegatedCapabilityCommand) -> DelegationCommandResult:
        """Grant one exact active capability only from an active organization owner."""
        capability = self._parse_capability(command.capability)
        if capability is None:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.INVALID_CAPABILITY,
            )
        if capability not in DELEGABLE_CAPABILITIES:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.NON_DELEGABLE_CAPABILITY,
            )
        resource_scope = self._parse_resource_scope(command.resource_scope)
        if resource_scope is None:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.INVALID_RESOURCE_SCOPE,
            )
        if command.expires_at <= command.requested_at:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.INVALID_EXPIRY,
            )
        if not command.recent_authentication_verified:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.RECENT_AUTHENTICATION_REQUIRED,
            )
        if command.actor.user_id == command.grantee_user_id:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.SELF_GRANT_FORBIDDEN,
            )

        grantor_membership = self._organization_repository.get_membership(
            organization_id=command.organization_id,
            user_id=command.actor.user_id,
        )
        if grantor_membership is None or grantor_membership.status != "active":
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.ACTIVE_OWNER_REQUIRED,
            )
        if grantor_membership.role != "owner":
            return DelegationCommandResult(
                allowed=False,
                deny_reason=(
                    DelegationDenyReason.REDELEGATION_FORBIDDEN
                    if grantor_membership.role in _ELIGIBLE_GRANTEE_ROLES
                    else DelegationDenyReason.ACTIVE_OWNER_REQUIRED
                ),
            )

        grantor_decision = self._capability_authorization_service.decide(
            request=AuthorizationRequest(
                actor=command.actor,
                capability=capability,
                selected_organization_id=command.organization_id,
            )
        )
        if not grantor_decision.allowed:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.GRANTOR_CAPABILITY_DENIED,
            )

        grantee_membership = self._organization_repository.get_membership(
            organization_id=command.organization_id,
            user_id=command.grantee_user_id,
        )
        if grantee_membership is None or grantee_membership.status != "active":
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.ACTIVE_GRANTEE_MEMBERSHIP_REQUIRED,
            )
        if grantee_membership.role not in DELEGABLE_CAPABILITY_GRANTEE_ROLES[capability]:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.GRANTEE_ROLE_NOT_ELIGIBLE,
            )

        try:
            grant, created = self._delegation_repository.create_or_get_active_grant(
                organization_id=command.organization_id,
                grantee_user_id=command.grantee_user_id,
                capability=capability,
                resource_scope=resource_scope,
                granted_by_owner_user_id=command.actor.user_id,
                granted_at=command.requested_at,
                expires_at=command.expires_at,
            )
        except DelegationRepositoryConflictError:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.GRANT_CONFLICT,
            )
        return DelegationCommandResult(allowed=True, grant=grant, idempotent=not created)

    def revoke(self, *, command: RevokeDelegatedCapabilityCommand) -> DelegationCommandResult:
        """Revoke one exact grant immediately; repeated revoke is idempotent."""
        if not command.recent_authentication_verified:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.RECENT_AUTHENTICATION_REQUIRED,
            )
        grant = self._delegation_repository.get_grant(delegation_id=command.delegation_id)
        if grant is None:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.GRANT_NOT_FOUND,
            )
        if grant.organization_id != command.organization_id:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.GRANT_ORGANIZATION_MISMATCH,
            )
        grantor_membership = self._organization_repository.get_membership(
            organization_id=command.organization_id,
            user_id=command.actor.user_id,
        )
        if grantor_membership is None or grantor_membership.status != "active":
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.ACTIVE_OWNER_REQUIRED,
            )
        if grantor_membership.role != "owner":
            return DelegationCommandResult(
                allowed=False,
                deny_reason=(
                    DelegationDenyReason.REDELEGATION_FORBIDDEN
                    if grantor_membership.role in _ELIGIBLE_GRANTEE_ROLES
                    else DelegationDenyReason.ACTIVE_OWNER_REQUIRED
                ),
            )
        revoked, changed = self._delegation_repository.revoke_grant(
            delegation_id=command.delegation_id,
            organization_id=command.organization_id,
            revoked_by_owner_user_id=command.actor.user_id,
            revoked_at=command.requested_at,
        )
        if revoked is None:
            return DelegationCommandResult(
                allowed=False,
                deny_reason=DelegationDenyReason.GRANT_NOT_FOUND,
            )
        return DelegationCommandResult(allowed=True, grant=revoked, idempotent=not changed)

    def decide(self, *, request: DelegationEvaluationRequest) -> DelegatedCapabilityDecision:
        """Return a direct kernel decision or an exact active delegated-organization decision."""
        base_decision = self._capability_authorization_service.decide(
            request=AuthorizationRequest(
                actor=request.actor,
                capability=request.capability,
                selected_organization_id=request.selected_organization_id,
                resource=request.resource,
            )
        )
        if base_decision.allowed:
            return DelegatedCapabilityDecision(
                allowed=True,
                capability=base_decision.capability,
                scope=base_decision.scope,
                authority_source=DelegationAuthoritySource.CAPABILITY_KERNEL,
                base_decision=base_decision,
            )

        capability = base_decision.capability
        if capability is None or capability not in DELEGABLE_CAPABILITIES:
            return self._deny_from_base(
                base_decision=base_decision,
                reason=DelegationDenyReason.NON_DELEGABLE_CAPABILITY,
            )
        if base_decision.deny_reason not in _DELEGATION_CANDIDATE_BASE_DENIALS:
            return self._deny_from_base(
                base_decision=base_decision,
                reason=DelegationDenyReason.DELEGATION_NOT_ACTIVE,
            )
        if request.selected_organization_id is None:
            return self._deny_from_base(
                base_decision=base_decision,
                reason=DelegationDenyReason.DELEGATION_NOT_ACTIVE,
            )
        resource_scope = self._parse_resource_scope(request.resource_scope)
        if resource_scope is None:
            return self._deny_from_base(
                base_decision=base_decision,
                reason=DelegationDenyReason.INVALID_RESOURCE_SCOPE,
            )
        grantee_membership = self._organization_repository.get_membership(
            organization_id=request.selected_organization_id,
            user_id=request.actor.user_id,
        )
        if (
            grantee_membership is None
            or grantee_membership.status != "active"
            or grantee_membership.role not in DELEGABLE_CAPABILITY_GRANTEE_ROLES[capability]
        ):
            return self._deny_from_base(
                base_decision=base_decision,
                reason=DelegationDenyReason.GRANTEE_ROLE_NOT_ELIGIBLE,
            )
        grant = self._delegation_repository.find_active_grant(
            organization_id=request.selected_organization_id,
            grantee_user_id=request.actor.user_id,
            capability=capability,
            resource_scope=resource_scope,
            at=request.evaluated_at,
        )
        if grant is None:
            return self._deny_from_base(
                base_decision=base_decision,
                reason=DelegationDenyReason.DELEGATION_NOT_ACTIVE,
            )
        return DelegatedCapabilityDecision(
            allowed=True,
            capability=capability,
            scope=DelegatedCapabilityScope.DELEGATED_ORGANIZATION,
            authority_source=DelegationAuthoritySource.DELEGATION,
            base_decision=base_decision,
            delegation=grant,
        )

    @staticmethod
    def _parse_capability(value: CapabilityId | str) -> CapabilityId | None:
        try:
            return CapabilityId(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_resource_scope(
        value: DelegationResourceScope | str,
    ) -> DelegationResourceScope | None:
        try:
            return DelegationResourceScope(value)
        except ValueError:
            return None

    @staticmethod
    def _deny_from_base(
        *,
        base_decision: AuthorizationDecision,
        reason: DelegationDenyReason,
    ) -> DelegatedCapabilityDecision:
        return DelegatedCapabilityDecision(
            allowed=False,
            capability=base_decision.capability,
            scope=None,
            authority_source=None,
            base_decision=base_decision,
            delegation_deny_reason=reason,
            base_deny_reason=base_decision.deny_reason,
        )
