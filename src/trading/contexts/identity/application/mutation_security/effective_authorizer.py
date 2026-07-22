from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from trading.contexts.identity.application.authorization import (
    AuthorizationRequest,
    AuthorizationResource,
    CapabilityAuthorizationService,
    CapabilityId,
)
from trading.contexts.identity.application.delegation.models import (
    DelegatedCapabilityScope,
    DelegationAuthoritySource,
    DelegationEvaluationRequest,
    DelegationResourceScope,
)
from trading.contexts.identity.application.delegation.service import (
    DelegatedCapabilityService,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import OrganizationId

_INVALID_DECISION = "effective_authorization_invalid"


class EffectiveAuthoritySource(StrEnum):
    CAPABILITY_KERNEL = "capability_kernel"
    DELEGATION = "delegation"


@dataclass(frozen=True, slots=True)
class EffectiveAuthorizationRequest:
    actor: CurrentUserPrincipal
    capability: CapabilityId | str
    selected_organization_id: OrganizationId | None
    resource: AuthorizationResource | None
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class EffectiveAuthorizationDecision:
    """Grant-free authorization projection safe for mutation decisions and audit."""

    allowed: bool
    capability: CapabilityId | None
    scope: str | None
    authority_source: EffectiveAuthoritySource | None
    delegated_organization_id: OrganizationId | None
    deny_reason: str | None = None


class EffectiveAuthorizer(Protocol):
    def decide(
        self, *, request: EffectiveAuthorizationRequest
    ) -> EffectiveAuthorizationDecision: ...


class CapabilityAuthorizationAdapter:
    """Normalize a direct capability-kernel decision for the mutation boundary."""

    def __init__(self, *, service: CapabilityAuthorizationService) -> None:
        if service is None:  # type: ignore[truthy-bool]
            raise ValueError("CapabilityAuthorizationAdapter requires service")
        self._service = service

    def decide(self, *, request: EffectiveAuthorizationRequest) -> EffectiveAuthorizationDecision:
        decision = self._service.decide(
            request=AuthorizationRequest(
                actor=request.actor,
                capability=request.capability,
                selected_organization_id=request.selected_organization_id,
                resource=request.resource,
            )
        )
        if not decision.allowed:
            return EffectiveAuthorizationDecision(
                allowed=False,
                capability=decision.capability,
                scope=None,
                authority_source=None,
                delegated_organization_id=None,
                deny_reason=(None if decision.deny_reason is None else decision.deny_reason.value),
            )
        if decision.capability is None or decision.scope is None:
            return _invalid_decision(capability=decision.capability)
        return EffectiveAuthorizationDecision(
            allowed=True,
            capability=decision.capability,
            scope=decision.scope.value,
            authority_source=EffectiveAuthoritySource.CAPABILITY_KERNEL,
            delegated_organization_id=None,
        )


class DelegatedCapabilityAuthorizationAdapter:
    """Adapt the delegation core without exposing its persisted grant."""

    def __init__(self, *, service: DelegatedCapabilityService) -> None:
        if service is None:  # type: ignore[truthy-bool]
            raise ValueError("DelegatedCapabilityAuthorizationAdapter requires service")
        self._service = service

    def decide(self, *, request: EffectiveAuthorizationRequest) -> EffectiveAuthorizationDecision:
        decision = self._service.decide(
            request=DelegationEvaluationRequest(
                actor=request.actor,
                capability=request.capability,
                selected_organization_id=request.selected_organization_id,
                resource=request.resource,
                resource_scope=DelegationResourceScope.ORGANIZATION,
                evaluated_at=request.evaluated_at,
            )
        )
        if not decision.allowed:
            deny_reason = decision.delegation_deny_reason or decision.base_deny_reason
            return EffectiveAuthorizationDecision(
                allowed=False,
                capability=decision.capability,
                scope=None,
                authority_source=None,
                delegated_organization_id=None,
                deny_reason=None if deny_reason is None else deny_reason.value,
            )
        if (
            decision.capability is None
            or decision.scope is None
            or decision.authority_source is None
        ):
            return _invalid_decision(capability=decision.capability)
        try:
            authority_source = EffectiveAuthoritySource(decision.authority_source.value)
        except ValueError:
            return _invalid_decision(capability=decision.capability)

        if authority_source is EffectiveAuthoritySource.CAPABILITY_KERNEL:
            if decision.delegation is not None:
                return _invalid_decision(capability=decision.capability)
            return EffectiveAuthorizationDecision(
                allowed=True,
                capability=decision.capability,
                scope=decision.scope.value,
                authority_source=authority_source,
                delegated_organization_id=None,
            )

        grant = decision.delegation
        if (
            decision.authority_source is not DelegationAuthoritySource.DELEGATION
            or decision.scope is not DelegatedCapabilityScope.DELEGATED_ORGANIZATION
            or grant is None
            or request.selected_organization_id is None
            or grant.organization_id != request.selected_organization_id
            or grant.grantee_user_id != request.actor.user_id
            or grant.capability is not decision.capability
            or grant.resource_scope is not DelegationResourceScope.ORGANIZATION
            or not grant.is_active_at(at=request.evaluated_at)
        ):
            return _invalid_decision(capability=decision.capability)
        return EffectiveAuthorizationDecision(
            allowed=True,
            capability=decision.capability,
            scope=decision.scope.value,
            authority_source=authority_source,
            delegated_organization_id=grant.organization_id,
        )


def _invalid_decision(*, capability: CapabilityId | None) -> EffectiveAuthorizationDecision:
    return EffectiveAuthorizationDecision(
        allowed=False,
        capability=capability,
        scope=None,
        authority_source=None,
        delegated_organization_id=None,
        deny_reason=_INVALID_DECISION,
    )
