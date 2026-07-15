from __future__ import annotations

from datetime import datetime
from typing import Protocol

from trading.contexts.identity.domain.entities import (
    AdministrativeAuditEvent,
    Installation,
    Organization,
    OrganizationAccess,
    OrganizationInvitation,
    OrganizationMembership,
    OrganizationRole,
    PluginPermission,
    PluginPermissionGrant,
    SupportAccessGrant,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class OrganizationRepositoryInvariantError(RuntimeError):
    """Raised when persistence rejects an organization consistency invariant."""

    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class OrganizationRepository(Protocol):
    """Persistence port for the installation, organization and RBAC aggregate."""

    def get_installation(self) -> Installation | None: ...

    def is_installation_owner(self, *, user_id: UserId) -> bool: ...

    def bootstrap_installation(
        self,
        *,
        owner_user_id: UserId,
        installation_name: str,
        organization_slug: str,
        organization_name: str,
        created_at: datetime,
    ) -> tuple[Installation, Organization]: ...

    def create_organization(
        self,
        *,
        actor_user_id: UserId,
        slug: str,
        display_name: str,
        created_at: datetime,
    ) -> Organization: ...

    def list_accesses_for_user(self, *, user_id: UserId) -> tuple[OrganizationAccess, ...]: ...

    def get_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
    ) -> OrganizationMembership | None: ...

    def list_memberships(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[OrganizationMembership, ...]: ...

    def add_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        actor_user_id: UserId,
        created_at: datetime,
    ) -> OrganizationMembership: ...

    def set_membership_role(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        actor_user_id: UserId,
        changed_at: datetime,
    ) -> OrganizationMembership: ...

    def remove_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        actor_user_id: UserId,
        removed_at: datetime,
    ) -> None: ...

    def create_invitation(
        self,
        *,
        organization_id: OrganizationId,
        recipient_email_sha256: str,
        role: OrganizationRole,
        actor_user_id: UserId,
        expires_at: datetime,
        created_at: datetime,
    ) -> OrganizationInvitation: ...

    def set_plugin_permission(
        self,
        *,
        organization_id: OrganizationId,
        plugin_id: str,
        user_id: UserId,
        permission: PluginPermission,
        actor_user_id: UserId,
        updated_at: datetime,
    ) -> PluginPermissionGrant: ...

    def grant_support_access(
        self,
        *,
        support_user_id: UserId,
        actor_user_id: UserId,
        reason: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> SupportAccessGrant: ...

    def list_audit_events(
        self,
        *,
        organization_id: OrganizationId,
        limit: int,
    ) -> tuple[AdministrativeAuditEvent, ...]: ...

    def record_rejected_event(
        self,
        *,
        organization_id: OrganizationId | None,
        actor_user_id: UserId,
        action: str,
        target_type: str,
        target_id: str,
        reason_code: str,
        created_at: datetime,
    ) -> None: ...
