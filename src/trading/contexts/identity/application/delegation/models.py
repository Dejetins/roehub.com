from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping
from uuid import UUID

from trading.contexts.identity.application.authorization.models import (
    AuthorizationDecision,
    AuthorizationDenyReason,
    AuthorizationResource,
    AuthorizationScope,
    CapabilityId,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.identity.domain.entities import OrganizationRole
from trading.shared_kernel.primitives import OrganizationId, UserId

DELEGABLE_CAPABILITIES = frozenset(
    {
        CapabilityId.DATA_SELECTION_MANAGE,
        CapabilityId.STRATEGIES_MANAGE,
        CapabilityId.STRATEGIES_RUN,
        CapabilityId.STRATEGIES_SAFE_STOP,
        CapabilityId.STRATEGIES_MANUAL_TRADE,
        CapabilityId.MODELS_MANAGE,
        CapabilityId.MODELS_PROMOTE_OR_ROLLBACK,
        CapabilityId.BACKTESTS_MANAGE_OWN,
        CapabilityId.BACKTESTS_QUEUE_OPERATE,
        CapabilityId.BACKTESTS_PROMOTE_OWN,
        CapabilityId.LIVE_RECONCILE,
        CapabilityId.MONITORING_SAFE_ACTION,
        CapabilityId.CONNECTIONS_BIND_OWN,
    }
)

DELEGABLE_CAPABILITY_GRANTEE_ROLES: Final[Mapping[CapabilityId, frozenset[OrganizationRole]]] = (
    MappingProxyType(
        {
            CapabilityId.DATA_SELECTION_MANAGE: frozenset({"trader"}),
            CapabilityId.STRATEGIES_MANAGE: frozenset({"admin"}),
            CapabilityId.STRATEGIES_RUN: frozenset({"admin"}),
            CapabilityId.STRATEGIES_SAFE_STOP: frozenset({"admin"}),
            CapabilityId.STRATEGIES_MANUAL_TRADE: frozenset({"admin"}),
            CapabilityId.MODELS_MANAGE: frozenset({"admin"}),
            CapabilityId.MODELS_PROMOTE_OR_ROLLBACK: frozenset({"admin"}),
            CapabilityId.BACKTESTS_MANAGE_OWN: frozenset({"admin"}),
            CapabilityId.BACKTESTS_QUEUE_OPERATE: frozenset({"admin"}),
            CapabilityId.BACKTESTS_PROMOTE_OWN: frozenset({"admin"}),
            CapabilityId.LIVE_RECONCILE: frozenset({"admin"}),
            CapabilityId.MONITORING_SAFE_ACTION: frozenset({"admin"}),
            CapabilityId.CONNECTIONS_BIND_OWN: frozenset({"admin"}),
        }
    )
)


class DelegationResourceScope(StrEnum):
    """The accepted contract allows one server-owned scope: the selected organization."""

    ORGANIZATION = "organization"


class DelegatedCapabilityScope(StrEnum):
    """Authorization scope returned only after a persisted exact delegation matches."""

    DELEGATED_ORGANIZATION = "delegated_organization"


class DelegationAuthoritySource(StrEnum):
    """Distinguish an existing role decision from a persisted delegation."""

    CAPABILITY_KERNEL = "capability_kernel"
    DELEGATION = "delegation"


class DelegationDenyReason(StrEnum):
    """Stable reasons produced by the delegated-capability application boundary."""

    RECENT_AUTHENTICATION_REQUIRED = "recent_authentication_required"
    ACTIVE_OWNER_REQUIRED = "active_owner_required"
    ACTIVE_GRANTEE_MEMBERSHIP_REQUIRED = "active_grantee_membership_required"
    GRANTEE_ROLE_NOT_ELIGIBLE = "grantee_role_not_eligible"
    SELF_GRANT_FORBIDDEN = "self_grant_forbidden"
    REDELEGATION_FORBIDDEN = "redelegation_forbidden"
    NON_DELEGABLE_CAPABILITY = "non_delegable_capability"
    INVALID_CAPABILITY = "invalid_capability"
    INVALID_EXPIRY = "invalid_expiry"
    INVALID_RESOURCE_SCOPE = "invalid_resource_scope"
    GRANTOR_CAPABILITY_DENIED = "grantor_capability_denied"
    GRANT_CONFLICT = "grant_conflict"
    GRANT_NOT_FOUND = "grant_not_found"
    GRANT_ORGANIZATION_MISMATCH = "grant_organization_mismatch"
    DELEGATION_NOT_ACTIVE = "delegation_not_active"
    DELEGATION_SCOPE_MISMATCH = "delegation_scope_mismatch"


@dataclass(frozen=True, slots=True)
class DelegatedCapabilityGrant:
    """One exact organization capability grant; it never represents a static role permission."""

    delegation_id: UUID
    organization_id: OrganizationId
    grantee_user_id: UserId
    capability: CapabilityId
    resource_scope: DelegationResourceScope
    granted_by_owner_user_id: UserId
    granted_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    revoked_by_owner_user_id: UserId | None = None

    def is_active_at(self, *, at: datetime) -> bool:
        return self.revoked_at is None and self.expires_at > at


@dataclass(frozen=True, slots=True)
class DelegationAuditEvent:
    """Redacted projection of one immutable administrative audit record."""

    event_id: UUID
    organization_id: OrganizationId
    actor_user_id: UserId
    action: str
    target_id: str
    metadata: dict[str, str]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class GrantDelegatedCapabilityCommand:
    """Server-side command whose recent-auth input is verified upstream, never client-supplied."""

    actor: CurrentUserPrincipal
    organization_id: OrganizationId
    grantee_user_id: UserId
    capability: CapabilityId | str
    resource_scope: DelegationResourceScope | str
    expires_at: datetime
    requested_at: datetime
    recent_authentication_verified: bool


@dataclass(frozen=True, slots=True)
class RevokeDelegatedCapabilityCommand:
    """Server-side revoke command for one known delegation in one organization."""

    actor: CurrentUserPrincipal
    organization_id: OrganizationId
    delegation_id: UUID
    requested_at: datetime
    recent_authentication_verified: bool


@dataclass(frozen=True, slots=True)
class DelegationEvaluationRequest:
    """Server-resolved request evaluated by the capability kernel and then an exact grant lookup."""

    actor: CurrentUserPrincipal
    capability: CapabilityId | str
    selected_organization_id: OrganizationId | None
    resource: AuthorizationResource | None
    resource_scope: DelegationResourceScope | str
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class DelegationCommandResult:
    """Deterministic result for grant/revoke without an HTTP or browser contract."""

    allowed: bool
    grant: DelegatedCapabilityGrant | None = None
    deny_reason: DelegationDenyReason | None = None
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class DelegatedCapabilityDecision:
    """Effective decision composed from the accepted kernel and an optional persisted grant."""

    allowed: bool
    capability: CapabilityId | None
    scope: AuthorizationScope | DelegatedCapabilityScope | None
    authority_source: DelegationAuthoritySource | None
    base_decision: AuthorizationDecision
    delegation: DelegatedCapabilityGrant | None = None
    delegation_deny_reason: DelegationDenyReason | None = None
    base_deny_reason: AuthorizationDenyReason | None = None
