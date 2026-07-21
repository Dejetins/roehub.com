from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.contexts.identity.application.authorization.models import CapabilityId
from trading.contexts.identity.application.delegation.models import (
    DelegatedCapabilityGrant,
    DelegationAuditEvent,
    DelegationResourceScope,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class DelegationRepositoryConflictError(ValueError):
    """Raised when an unrevoked exact delegation requires explicit revocation before replacement."""

    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class DelegationRepository(Protocol):
    """Persistence port for exact, organization-scoped delegated capabilities."""

    def create_or_get_active_grant(
        self,
        *,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
        granted_by_owner_user_id: UserId,
        granted_at: datetime,
        expires_at: datetime,
    ) -> tuple[DelegatedCapabilityGrant, bool]:
        """Create or idempotently return an exact unrevoked grant; revoke before replacement."""
        ...

    def get_grant(self, *, delegation_id: UUID) -> DelegatedCapabilityGrant | None:
        """Return one grant regardless of expiry/revocation for revoke authorization."""
        ...

    def revoke_grant(
        self,
        *,
        delegation_id: UUID,
        organization_id: OrganizationId,
        revoked_by_owner_user_id: UserId,
        revoked_at: datetime,
    ) -> tuple[DelegatedCapabilityGrant | None, bool]:
        """Persist one revocation and report whether this call changed active state."""
        ...

    def find_active_grant(
        self,
        *,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
        at: datetime,
    ) -> DelegatedCapabilityGrant | None:
        """Return an exact unrevoked, unexpired grant only."""
        ...

    def list_audit_events(
        self, *, organization_id: OrganizationId
    ) -> tuple[DelegationAuditEvent, ...]:
        """Return redacted delegation audit events for focused verification."""
        ...
