from __future__ import annotations

from datetime import datetime
from threading import RLock
from typing import Any, cast
from uuid import uuid4

from trading.contexts.identity.application.ports.organization_repository import (
    OrganizationRepository,
    OrganizationRepositoryInvariantError,
)
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
    permissions_for_role,
)
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId


class InMemoryOrganizationRepository(OrganizationRepository):
    """Deterministic in-memory adapter for organization use-case and API tests."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._installation: Installation | None = None
        self._installation_owners: set[UserId] = set()
        self._organizations: dict[OrganizationId, Organization] = {}
        self._memberships: dict[tuple[OrganizationId, UserId], OrganizationMembership] = {}
        self._invitations: dict[object, OrganizationInvitation] = {}
        self._invitation_recipient_hashes: dict[object, str] = {}
        self._accepted_invitations: set[object] = set()
        self._plugin_permissions: dict[
            tuple[OrganizationId, str, UserId], PluginPermissionGrant
        ] = {}
        self._support_grants: dict[object, SupportAccessGrant] = {}
        self._audit_events: list[AdministrativeAuditEvent] = []

    def get_installation(self) -> Installation | None:
        return self._installation

    def is_installation_owner(self, *, user_id: UserId) -> bool:
        return user_id in self._installation_owners

    def bootstrap_installation(
        self,
        *,
        owner_user_id: UserId,
        installation_name: str,
        organization_slug: str,
        organization_name: str,
        created_at: datetime,
    ) -> tuple[Installation, Organization]:
        with self._lock:
            if self._installation is not None:
                raise OrganizationRepositoryInvariantError(
                    code="installation_already_initialized"
                )
            installation = Installation(
                installation_id=InstallationId(uuid4()),
                display_name=installation_name,
                created_at=created_at,
            )
            organization = Organization(
                organization_id=OrganizationId(uuid4()),
                installation_id=installation.installation_id,
                slug=organization_slug,
                display_name=organization_name,
                created_at=created_at,
            )
            membership = OrganizationMembership(
                organization_id=organization.organization_id,
                user_id=owner_user_id,
                role="owner",
                created_at=created_at,
            )
            self._installation = installation
            self._installation_owners.add(owner_user_id)
            self._organizations[organization.organization_id] = organization
            self._memberships[(organization.organization_id, owner_user_id)] = membership
            self._append_audit(
                organization_id=organization.organization_id,
                actor_user_id=owner_user_id,
                action="installation.bootstrap",
                target_type="installation",
                target_id=str(installation.installation_id),
                metadata={"organization_id": str(organization.organization_id)},
                created_at=created_at,
            )
            return installation, organization

    def create_organization(
        self,
        *,
        actor_user_id: UserId,
        slug: str,
        display_name: str,
        created_at: datetime,
    ) -> Organization:
        with self._lock:
            installation = self._require_installation()
            if any(organization.slug == slug for organization in self._organizations.values()):
                raise OrganizationRepositoryInvariantError(code="organization_slug_conflict")
            organization = Organization(
                organization_id=OrganizationId(uuid4()),
                installation_id=installation.installation_id,
                slug=slug,
                display_name=display_name,
                created_at=created_at,
            )
            self._organizations[organization.organization_id] = organization
            self._memberships[(organization.organization_id, actor_user_id)] = (
                OrganizationMembership(
                    organization_id=organization.organization_id,
                    user_id=actor_user_id,
                    role="owner",
                    created_at=created_at,
                )
            )
            self._append_audit(
                organization_id=organization.organization_id,
                actor_user_id=actor_user_id,
                action="organization.created",
                target_type="organization",
                target_id=str(organization.organization_id),
                metadata={"slug": slug},
                created_at=created_at,
            )
            return organization

    def list_accesses_for_user(self, *, user_id: UserId) -> tuple[OrganizationAccess, ...]:
        accesses: list[OrganizationAccess] = []
        for organization in sorted(self._organizations.values(), key=lambda item: item.slug):
            membership = self._memberships.get((organization.organization_id, user_id))
            if membership is None or membership.status != "active":
                continue
            role = membership.role
            accesses.append(
                OrganizationAccess(
                    organization=organization,
                    role=role,
                    permissions=permissions_for_role(role),
                )
            )
        return tuple(accesses)

    def get_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
    ) -> OrganizationMembership | None:
        return self._memberships.get((organization_id, user_id))

    def list_memberships(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[OrganizationMembership, ...]:
        return tuple(
            sorted(
                (
                    membership
                    for membership in self._memberships.values()
                    if membership.organization_id == organization_id
                ),
                key=lambda membership: (membership.role, str(membership.user_id)),
            )
        )

    def add_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        actor_user_id: UserId,
        created_at: datetime,
    ) -> OrganizationMembership:
        with self._lock:
            self._require_organization(organization_id)
            key = (organization_id, user_id)
            if key in self._memberships:
                raise OrganizationRepositoryInvariantError(code="membership_conflict")
            membership = OrganizationMembership(
                organization_id=organization_id,
                user_id=user_id,
                role=role,
                created_at=created_at,
            )
            self._memberships[key] = membership
            self._append_audit(
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action="membership.created",
                target_type="membership",
                target_id=str(user_id),
                metadata={"role": role},
                created_at=created_at,
            )
            return membership

    def set_membership_role(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        actor_user_id: UserId,
        changed_at: datetime,
    ) -> OrganizationMembership:
        with self._lock:
            key = (organization_id, user_id)
            current = self._memberships.get(key)
            if current is None:
                raise OrganizationRepositoryInvariantError(code="membership_not_found")
            if (
                current.role == "owner"
                and role != "owner"
                and self._owner_count(organization_id) == 1
            ):
                raise OrganizationRepositoryInvariantError(code="last_owner")
            membership = OrganizationMembership(
                organization_id=organization_id,
                user_id=user_id,
                role=role,
                status=current.status,
                created_at=current.created_at,
            )
            self._memberships[key] = membership
            self._append_audit(
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action="membership.role_changed",
                target_type="membership",
                target_id=str(user_id),
                metadata={"from_role": current.role, "to_role": role},
                created_at=changed_at,
            )
            return membership

    def remove_membership(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        actor_user_id: UserId,
        removed_at: datetime,
    ) -> None:
        with self._lock:
            key = (organization_id, user_id)
            current = self._memberships.get(key)
            if current is None:
                raise OrganizationRepositoryInvariantError(code="membership_not_found")
            if current.role == "owner" and self._owner_count(organization_id) == 1:
                raise OrganizationRepositoryInvariantError(code="last_owner")
            del self._memberships[key]
            self._append_audit(
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action="membership.removed",
                target_type="membership",
                target_id=str(user_id),
                metadata={"previous_role": current.role},
                created_at=removed_at,
            )

    def create_invitation(
        self,
        *,
        organization_id: OrganizationId,
        recipient_email_sha256: str,
        role: OrganizationRole,
        actor_user_id: UserId,
        expires_at: datetime,
        created_at: datetime,
    ) -> OrganizationInvitation:
        with self._lock:
            self._require_organization(organization_id)
            invitation = OrganizationInvitation(
                invitation_id=uuid4(),
                organization_id=organization_id,
                role=role,
                expires_at=expires_at,
                created_at=created_at,
            )
            self._invitations[invitation.invitation_id] = invitation
            self._invitation_recipient_hashes[invitation.invitation_id] = (
                recipient_email_sha256
            )
            self._append_audit(
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action="invitation.created",
                target_type="invitation",
                target_id=str(invitation.invitation_id),
                metadata={"role": role, "recipient_hash": recipient_email_sha256},
                created_at=created_at,
            )
            return invitation

    def accept_pending_invitations(
        self,
        *,
        user_id: UserId,
        recipient_email_sha256: str,
        accepted_at: datetime,
    ) -> int:
        """Accept exact verified-email invitations and create their memberships."""
        with self._lock:
            matches = [
                invitation
                for invitation_id, invitation in self._invitations.items()
                if invitation_id not in self._accepted_invitations
                and self._invitation_recipient_hashes.get(invitation_id)
                == recipient_email_sha256
                and invitation.expires_at > accepted_at
            ]
            for invitation in matches:
                key = (invitation.organization_id, user_id)
                if key in self._memberships:
                    raise OrganizationRepositoryInvariantError(code="membership_conflict")
                self._memberships[key] = OrganizationMembership(
                    organization_id=invitation.organization_id,
                    user_id=user_id,
                    role=invitation.role,
                    created_at=accepted_at,
                )
                self._accepted_invitations.add(invitation.invitation_id)
                self._append_audit(
                    organization_id=invitation.organization_id,
                    actor_user_id=user_id,
                    action="invitation.accepted",
                    target_type="invitation",
                    target_id=str(invitation.invitation_id),
                    metadata={"role": invitation.role},
                    created_at=accepted_at,
                )
            return len(matches)

    def set_plugin_permission(
        self,
        *,
        organization_id: OrganizationId,
        plugin_id: str,
        user_id: UserId,
        permission: PluginPermission,
        actor_user_id: UserId,
        updated_at: datetime,
    ) -> PluginPermissionGrant:
        with self._lock:
            if (organization_id, user_id) not in self._memberships:
                raise OrganizationRepositoryInvariantError(code="membership_not_found")
            grant = PluginPermissionGrant(
                organization_id=organization_id,
                plugin_id=plugin_id,
                user_id=user_id,
                permission=permission,
                updated_at=updated_at,
            )
            self._plugin_permissions[(organization_id, plugin_id, user_id)] = grant
            self._append_audit(
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action="plugin.permission_set",
                target_type="plugin_permission",
                target_id=f"{plugin_id}:{user_id}",
                metadata={"permission": permission},
                created_at=updated_at,
            )
            return grant

    def grant_support_access(
        self,
        *,
        support_user_id: UserId,
        actor_user_id: UserId,
        reason: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> SupportAccessGrant:
        with self._lock:
            installation = self._require_installation()
            grant = SupportAccessGrant(
                grant_id=uuid4(),
                installation_id=installation.installation_id,
                support_user_id=support_user_id,
                expires_at=expires_at,
                created_at=created_at,
            )
            self._support_grants[grant.grant_id] = grant
            self._append_audit(
                organization_id=None,
                actor_user_id=actor_user_id,
                action="support_access.granted",
                target_type="support_access",
                target_id=str(grant.grant_id),
                metadata={
                    "support_user_id": str(support_user_id),
                    "expires_at": expires_at.isoformat(),
                },
                created_at=created_at,
            )
            return grant

    def list_audit_events(
        self,
        *,
        organization_id: OrganizationId,
        limit: int,
    ) -> tuple[AdministrativeAuditEvent, ...]:
        events = [event for event in self._audit_events if event.organization_id == organization_id]
        events.sort(key=lambda event: (event.created_at, str(event.event_id)), reverse=True)
        return tuple(events[:limit])

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
    ) -> None:
        self._append_audit(
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata={"reason_code": reason_code},
            created_at=created_at,
            outcome="rejected",
        )

    def _append_audit(
        self,
        *,
        organization_id: OrganizationId | None,
        actor_user_id: UserId,
        action: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, str],
        created_at: datetime,
        outcome: str = "succeeded",
    ) -> None:
        installation = self._require_installation()
        self._audit_events.append(
            AdministrativeAuditEvent(
                event_id=uuid4(),
                installation_id=installation.installation_id,
                organization_id=organization_id,
                actor_user_id=actor_user_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                outcome=cast(Any, outcome),
                metadata=dict(metadata),
                created_at=created_at,
            )
        )

    def _owner_count(self, organization_id: OrganizationId) -> int:
        return sum(
            membership.organization_id == organization_id
            and membership.role == "owner"
            and membership.status == "active"
            for membership in self._memberships.values()
        )

    def _require_installation(self) -> Installation:
        if self._installation is None:
            raise OrganizationRepositoryInvariantError(code="installation_not_initialized")
        return self._installation

    def _require_organization(self, organization_id: OrganizationId) -> Organization:
        organization = self._organizations.get(organization_id)
        if organization is None:
            raise OrganizationRepositoryInvariantError(code="organization_not_found")
        return organization
