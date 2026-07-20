from __future__ import annotations

from trading.contexts.identity.application.authorization.models import (
    AuthorizationDecision,
    AuthorizationDenyReason,
    AuthorizationRequest,
    AuthorizationScope,
    CapabilityId,
)
from trading.contexts.identity.application.authorization.policy import CAPABILITY_POLICIES
from trading.contexts.identity.application.ports.organization_repository import (
    OrganizationRepository,
)


class CapabilityAuthorizationService:
    """Server-only capability boundary with deterministic default-deny decisions."""

    def __init__(self, *, organization_repository: OrganizationRepository) -> None:
        if organization_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CapabilityAuthorizationService requires organization_repository")
        self._organization_repository = organization_repository

    def decide(self, *, request: AuthorizationRequest) -> AuthorizationDecision:
        """Decide one requested capability from server-resolved identity and resource facts."""
        capability = self._parse_capability(request.capability)
        if capability is None:
            return AuthorizationDecision.deny(
                capability=None,
                reason=AuthorizationDenyReason.UNKNOWN_CAPABILITY,
            )
        if request.client_supplied_role is not None:
            return AuthorizationDecision.deny(
                capability=capability,
                reason=AuthorizationDenyReason.CLIENT_ROLE_SUPPLIED,
            )
        if capability is CapabilityId.CONNECTIONS_SECRET_REVEAL:
            return AuthorizationDecision.deny(
                capability=capability,
                reason=AuthorizationDenyReason.STORED_SECRET_REVEAL_FORBIDDEN,
            )

        policy = CAPABILITY_POLICIES[capability]
        if policy.installation_owner_required:
            if not self._organization_repository.is_installation_owner(
                user_id=request.actor.user_id
            ):
                return AuthorizationDecision.deny(
                    capability=capability,
                    reason=AuthorizationDenyReason.INSTALLATION_OWNER_REQUIRED,
                )
            return AuthorizationDecision.allow(
                capability=capability,
                scope=AuthorizationScope.INSTALLATION,
            )

        if not policy.requires_organization_context:
            return AuthorizationDecision.allow(
                capability=capability,
                scope=AuthorizationScope.PERSONAL,
            )

        organization_id = request.selected_organization_id
        if organization_id is None:
            return AuthorizationDecision.deny(
                capability=capability,
                reason=AuthorizationDenyReason.ORGANIZATION_CONTEXT_REQUIRED,
            )
        if request.resource is not None and request.resource.organization_id != organization_id:
            return AuthorizationDecision.deny(
                capability=capability,
                reason=AuthorizationDenyReason.RESOURCE_ORGANIZATION_MISMATCH,
            )

        membership = self._organization_repository.get_membership(
            organization_id=organization_id,
            user_id=request.actor.user_id,
        )
        if membership is None or membership.status != "active":
            return AuthorizationDecision.deny(
                capability=capability,
                reason=AuthorizationDenyReason.INACTIVE_OR_MISSING_MEMBERSHIP,
            )

        scope = policy.role_scopes.get(membership.role)
        if scope is None:
            return AuthorizationDecision.deny(
                capability=capability,
                reason=AuthorizationDenyReason.ROLE_CAPABILITY_DENIED,
            )
        if scope is AuthorizationScope.OWN:
            if request.resource is None:
                return AuthorizationDecision.deny(
                    capability=capability,
                    reason=AuthorizationDenyReason.RESOURCE_CONTEXT_REQUIRED,
                )
            if request.resource.owner_user_id != request.actor.user_id:
                return AuthorizationDecision.deny(
                    capability=capability,
                    reason=AuthorizationDenyReason.OWNERSHIP_REQUIRED,
                )
        return AuthorizationDecision.allow(capability=capability, scope=scope)

    @staticmethod
    def _parse_capability(value: CapabilityId | str) -> CapabilityId | None:
        try:
            return CapabilityId(value)
        except ValueError:
            return None
