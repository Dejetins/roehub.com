from __future__ import annotations

from datetime import datetime
from threading import RLock
from uuid import UUID, uuid4

from trading.contexts.identity.application.authorization.models import CapabilityId
from trading.contexts.identity.application.delegation.models import (
    DelegatedCapabilityGrant,
    DelegationAuditEvent,
    DelegationResourceScope,
)
from trading.contexts.identity.application.ports.delegation_repository import (
    DelegationRepository,
    DelegationRepositoryConflictError,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class InMemoryDelegationRepository(DelegationRepository):
    """Lock-protected in-memory parity adapter for exact delegated-capability lifecycle tests."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._grants: dict[UUID, DelegatedCapabilityGrant] = {}
        self._audit_events: list[DelegationAuditEvent] = []

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
        with self._lock:
            existing = self._find_unrevoked(
                organization_id=organization_id,
                grantee_user_id=grantee_user_id,
                capability=capability,
                resource_scope=resource_scope,
            )
            if existing is not None:
                if self._same_grant_request(
                    grant=existing,
                    granted_by_owner_user_id=granted_by_owner_user_id,
                    granted_at=granted_at,
                    expires_at=expires_at,
                ):
                    return existing, False
                raise DelegationRepositoryConflictError(code="active_delegation_conflict")
            grant = DelegatedCapabilityGrant(
                delegation_id=uuid4(),
                organization_id=organization_id,
                grantee_user_id=grantee_user_id,
                capability=capability,
                resource_scope=resource_scope,
                granted_by_owner_user_id=granted_by_owner_user_id,
                granted_at=granted_at,
                expires_at=expires_at,
            )
            self._grants[grant.delegation_id] = grant
            self._append_audit(
                grant=grant,
                actor_user_id=granted_by_owner_user_id,
                action="delegation.granted",
                at=granted_at,
            )
            return grant, True

    def get_grant(self, *, delegation_id: UUID) -> DelegatedCapabilityGrant | None:
        with self._lock:
            return self._grants.get(delegation_id)

    def revoke_grant(
        self,
        *,
        delegation_id: UUID,
        organization_id: OrganizationId,
        revoked_by_owner_user_id: UserId,
        revoked_at: datetime,
    ) -> tuple[DelegatedCapabilityGrant | None, bool]:
        with self._lock:
            grant = self._grants.get(delegation_id)
            if grant is None or grant.organization_id != organization_id:
                return None, False
            if grant.revoked_at is not None:
                return grant, False
            revoked = DelegatedCapabilityGrant(
                delegation_id=grant.delegation_id,
                organization_id=grant.organization_id,
                grantee_user_id=grant.grantee_user_id,
                capability=grant.capability,
                resource_scope=grant.resource_scope,
                granted_by_owner_user_id=grant.granted_by_owner_user_id,
                granted_at=grant.granted_at,
                expires_at=grant.expires_at,
                revoked_at=revoked_at,
                revoked_by_owner_user_id=revoked_by_owner_user_id,
            )
            self._grants[delegation_id] = revoked
            self._append_audit(
                grant=revoked,
                actor_user_id=revoked_by_owner_user_id,
                action="delegation.revoked",
                at=revoked_at,
            )
            return revoked, True

    def find_active_grant(
        self,
        *,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
        at: datetime,
    ) -> DelegatedCapabilityGrant | None:
        with self._lock:
            grant = self._find_unrevoked(
                organization_id=organization_id,
                grantee_user_id=grantee_user_id,
                capability=capability,
                resource_scope=resource_scope,
            )
            return grant if grant is not None and grant.is_active_at(at=at) else None

    def list_audit_events(
        self, *, organization_id: OrganizationId
    ) -> tuple[DelegationAuditEvent, ...]:
        with self._lock:
            return tuple(
                event
                for event in reversed(self._audit_events)
                if event.organization_id == organization_id
            )

    def _find_unrevoked(
        self,
        *,
        organization_id: OrganizationId,
        grantee_user_id: UserId,
        capability: CapabilityId,
        resource_scope: DelegationResourceScope,
    ) -> DelegatedCapabilityGrant | None:
        return next(
            (
                grant
                for grant in self._grants.values()
                if grant.organization_id == organization_id
                and grant.grantee_user_id == grantee_user_id
                and grant.capability is capability
                and grant.resource_scope == resource_scope
                and grant.revoked_at is None
            ),
            None,
        )

    @staticmethod
    def _same_grant_request(
        *,
        grant: DelegatedCapabilityGrant,
        granted_by_owner_user_id: UserId,
        granted_at: datetime,
        expires_at: datetime,
    ) -> bool:
        return (
            grant.granted_by_owner_user_id == granted_by_owner_user_id
            and grant.granted_at == granted_at
            and grant.expires_at == expires_at
        )

    def _append_audit(
        self,
        *,
        grant: DelegatedCapabilityGrant,
        actor_user_id: UserId,
        action: str,
        at: datetime,
    ) -> None:
        self._audit_events.append(
            DelegationAuditEvent(
                event_id=uuid4(),
                organization_id=grant.organization_id,
                actor_user_id=actor_user_id,
                action=action,
                target_id=str(grant.delegation_id),
                metadata={
                    "capability_id": str(grant.capability),
                    "grantee_user_id": str(grant.grantee_user_id),
                    "resource_scope": grant.resource_scope,
                },
                created_at=at,
            )
        )
