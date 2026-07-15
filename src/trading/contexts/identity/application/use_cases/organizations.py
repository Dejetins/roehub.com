from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Callable

from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
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
    OrganizationPermission,
    OrganizationRole,
    PluginPermission,
    PluginPermissionGrant,
    SupportAccessGrant,
    permissions_for_role,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_RECENT_AUTH_WINDOW = timedelta(minutes=10)
_MAX_SUPPORT_ACCESS = timedelta(hours=24)
_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}[a-z0-9]$")


class OrganizationAccessError(ValueError):
    """Stable authorization, validation and consistency error for organization APIs."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OrganizationAccessService:
    """Application boundary for installation bootstrap and organization RBAC."""

    def __init__(self, *, repository: OrganizationRepository) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("OrganizationAccessService requires repository")
        self._repository = repository

    def bootstrap_installation(
        self,
        *,
        principal: CurrentUserPrincipal,
        installation_name: str,
        organization_slug: str,
        organization_name: str,
        now: datetime,
    ) -> tuple[Installation, Organization]:
        self._require_recent_auth(principal=principal, now=now)
        if self._repository.get_installation() is not None:
            raise OrganizationAccessError(
                code="installation_already_initialized",
                message="Installation is already initialized",
            )
        return self._translate_invariant(
            lambda: self._repository.bootstrap_installation(
                owner_user_id=principal.user_id,
                installation_name=_display_name(installation_name),
                organization_slug=_slug(organization_slug),
                organization_name=_display_name(organization_name),
                created_at=_utc(now),
            )
        )

    def list_organizations(
        self,
        *,
        principal: CurrentUserPrincipal,
    ) -> tuple[OrganizationAccess, ...]:
        return self._repository.list_accesses_for_user(user_id=principal.user_id)

    def get_access(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> OrganizationAccess:
        for access in self.list_organizations(principal=principal):
            if access.organization.organization_id == organization_id:
                return access
        raise OrganizationAccessError(
            code="organization_forbidden",
            message="Organization access is not granted",
        )

    def list_members(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> tuple[OrganizationMembership, ...]:
        self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="members.read",
        )
        return self._repository.list_memberships(organization_id=organization_id)

    def require_operation_execute(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        now: datetime,
    ) -> None:
        self._require_recent_auth(principal=principal, now=now)
        self.require_operation_read(
            principal=principal,
            organization_id=organization_id,
        )

    def require_operation_read(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> None:
        self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="operations.execute",
        )

    def is_installation_owner(self, *, principal: CurrentUserPrincipal) -> bool:
        return self._repository.is_installation_owner(user_id=principal.user_id)

    def require_installation_control(
        self,
        *,
        principal: CurrentUserPrincipal,
    ) -> None:
        self._require_installation_owner(user_id=principal.user_id)

    def create_organization(
        self,
        *,
        principal: CurrentUserPrincipal,
        slug: str,
        display_name: str,
        now: datetime,
    ) -> Organization:
        self._require_recent_auth(principal=principal, now=now)
        self._require_installation_owner(user_id=principal.user_id)
        return self._translate_invariant(
            lambda: self._repository.create_organization(
                actor_user_id=principal.user_id,
                slug=_slug(slug),
                display_name=_display_name(display_name),
                created_at=_utc(now),
            )
        )

    def add_member(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        now: datetime,
    ) -> OrganizationMembership:
        self._require_recent_auth(principal=principal, now=now)
        actor_role = self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="members.manage",
        )
        self._prevent_admin_owner_assignment(actor_role=actor_role, requested_role=role)
        return self._translate_invariant(
            lambda: self._repository.add_membership(
                organization_id=organization_id,
                user_id=user_id,
                role=role,
                actor_user_id=principal.user_id,
                created_at=_utc(now),
            )
        )

    def change_member_role(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        user_id: UserId,
        role: OrganizationRole,
        now: datetime,
    ) -> OrganizationMembership:
        self._require_recent_auth(principal=principal, now=now)
        actor_role = self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="roles.manage",
        )
        self._prevent_admin_owner_assignment(actor_role=actor_role, requested_role=role)
        target = self._repository.get_membership(
            organization_id=organization_id,
            user_id=user_id,
        )
        if target is None:
            raise OrganizationAccessError(
                code="membership_not_found",
                message="Organization membership is not found",
            )
        if actor_role == "admin" and target.role == "owner":
            raise OrganizationAccessError(
                code="owner_role_required",
                message="Administrator cannot change an owner membership",
            )
        return self._translate_invariant(
            lambda: self._repository.set_membership_role(
                organization_id=organization_id,
                user_id=user_id,
                role=role,
                actor_user_id=principal.user_id,
                changed_at=_utc(now),
            )
        )

    def remove_member(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        user_id: UserId,
        now: datetime,
    ) -> None:
        self._require_recent_auth(principal=principal, now=now)
        actor_role = self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="members.manage",
        )
        target = self._repository.get_membership(
            organization_id=organization_id,
            user_id=user_id,
        )
        if target is None:
            raise OrganizationAccessError(
                code="membership_not_found",
                message="Organization membership is not found",
            )
        if actor_role == "admin" and target.role == "owner":
            raise OrganizationAccessError(
                code="owner_role_required",
                message="Administrator cannot remove an owner membership",
            )
        self._translate_invariant(
            lambda: self._repository.remove_membership(
                organization_id=organization_id,
                user_id=user_id,
                actor_user_id=principal.user_id,
                removed_at=_utc(now),
            )
        )

    def create_invitation(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        recipient_email: str,
        role: OrganizationRole,
        expires_at: datetime,
        now: datetime,
    ) -> OrganizationInvitation:
        self._require_recent_auth(principal=principal, now=now)
        actor_role = self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="members.manage",
        )
        self._prevent_admin_owner_assignment(actor_role=actor_role, requested_role=role)
        normalized_email = recipient_email.strip().lower()
        if "@" not in normalized_email or len(normalized_email) > 320:
            raise OrganizationAccessError(
                code="invalid_invitation_email",
                message="Invitation email is invalid",
            )
        normalized_now = _utc(now)
        normalized_expiry = _utc(expires_at)
        if not normalized_now < normalized_expiry <= normalized_now + timedelta(days=7):
            raise OrganizationAccessError(
                code="invalid_invitation_expiry",
                message="Invitation expiry must be within seven days",
            )
        digest = hashlib.sha256(normalized_email.encode()).hexdigest()
        return self._translate_invariant(
            lambda: self._repository.create_invitation(
                organization_id=organization_id,
                recipient_email_sha256=digest,
                role=role,
                actor_user_id=principal.user_id,
                expires_at=normalized_expiry,
                created_at=normalized_now,
            )
        )

    def set_plugin_permission(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        plugin_id: str,
        user_id: UserId,
        permission: PluginPermission,
        now: datetime,
    ) -> PluginPermissionGrant:
        self._require_recent_auth(principal=principal, now=now)
        self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="plugins.manage",
        )
        normalized_plugin_id = plugin_id.strip()
        if not re.fullmatch(r"[a-z][a-z0-9._-]{2,127}", normalized_plugin_id):
            raise OrganizationAccessError(
                code="invalid_plugin_id",
                message="Plugin id is invalid",
            )
        target = self._repository.get_membership(
            organization_id=organization_id,
            user_id=user_id,
        )
        if target is None or target.status != "active":
            raise OrganizationAccessError(
                code="membership_not_found",
                message="Plugin grantee must be an active organization member",
            )
        return self._translate_invariant(
            lambda: self._repository.set_plugin_permission(
                organization_id=organization_id,
                plugin_id=normalized_plugin_id,
                user_id=user_id,
                permission=permission,
                actor_user_id=principal.user_id,
                updated_at=_utc(now),
            )
        )

    def grant_support_access(
        self,
        *,
        principal: CurrentUserPrincipal,
        support_user_id: UserId,
        reason: str,
        expires_at: datetime,
        now: datetime,
    ) -> SupportAccessGrant:
        self._require_recent_auth(principal=principal, now=now)
        self._require_installation_owner(user_id=principal.user_id)
        normalized_now = _utc(now)
        normalized_expiry = _utc(expires_at)
        if not normalized_now < normalized_expiry <= normalized_now + _MAX_SUPPORT_ACCESS:
            raise OrganizationAccessError(
                code="invalid_support_access_expiry",
                message="Support access must expire within 24 hours",
            )
        normalized_reason = reason.strip()
        if not 8 <= len(normalized_reason) <= 240:
            raise OrganizationAccessError(
                code="invalid_support_access_reason",
                message="Support access reason must contain 8 to 240 characters",
            )
        return self._translate_invariant(
            lambda: self._repository.grant_support_access(
                support_user_id=support_user_id,
                actor_user_id=principal.user_id,
                reason=normalized_reason,
                expires_at=normalized_expiry,
                created_at=normalized_now,
            )
        )

    def list_audit_events(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        limit: int,
    ) -> tuple[AdministrativeAuditEvent, ...]:
        self._require_permission(
            principal=principal,
            organization_id=organization_id,
            permission="audit.read",
        )
        if not 1 <= limit <= 200:
            raise OrganizationAccessError(
                code="invalid_audit_limit",
                message="Audit limit must be between 1 and 200",
            )
        return self._repository.list_audit_events(
            organization_id=organization_id,
            limit=limit,
        )

    def record_rejected_operation(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId | None,
        action: str,
        target_type: str,
        target_id: str,
        reason_code: str,
        now: datetime,
    ) -> None:
        """Persist an API-visible rejected administrative attempt without request payload."""

        try:
            self._repository.record_rejected_event(
                organization_id=organization_id,
                actor_user_id=principal.user_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                reason_code=reason_code,
                created_at=_utc(now),
            )
        except OrganizationRepositoryInvariantError as error:
            if error.code != "installation_not_initialized":
                raise

    def _require_installation_owner(self, *, user_id: UserId) -> None:
        if not self._repository.is_installation_owner(user_id=user_id):
            raise OrganizationAccessError(
                code="installation_owner_required",
                message="Installation owner permission is required",
            )

    def _require_permission(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
        permission: OrganizationPermission,
    ) -> OrganizationRole:
        membership = self._repository.get_membership(
            organization_id=organization_id,
            user_id=principal.user_id,
        )
        if membership is None or membership.status != "active":
            raise OrganizationAccessError(
                code="organization_forbidden",
                message="Organization access is not granted",
            )
        if permission not in permissions_for_role(membership.role):
            raise OrganizationAccessError(
                code="organization_permission_denied",
                message="Organization permission is not granted",
            )
        return membership.role

    @staticmethod
    def _prevent_admin_owner_assignment(
        *, actor_role: OrganizationRole, requested_role: OrganizationRole
    ) -> None:
        if actor_role == "admin" and requested_role == "owner":
            raise OrganizationAccessError(
                code="owner_role_required",
                message="Only an owner may grant the owner role",
            )

    @staticmethod
    def _require_recent_auth(*, principal: CurrentUserPrincipal, now: datetime) -> None:
        normalized_now = _utc(now)
        authenticated_at = principal.session_created_at
        if authenticated_at is None:
            raise OrganizationAccessError(
                code="recent_auth_required",
                message="Recent authentication is required",
            )
        normalized_auth = _utc(authenticated_at)
        if (
            normalized_auth > normalized_now
            or normalized_now - normalized_auth > _RECENT_AUTH_WINDOW
        ):
            raise OrganizationAccessError(
                code="recent_auth_required",
                message="Recent authentication is required",
            )

    @staticmethod
    def _translate_invariant[T](operation: Callable[[], T]) -> T:
        try:
            return operation()
        except OrganizationRepositoryInvariantError as error:
            messages = {
                "installation_already_initialized": "Installation is already initialized",
                "organization_slug_conflict": "Organization slug already exists",
                "membership_conflict": "Organization membership already exists",
                "membership_not_found": "Organization membership is not found",
                "last_owner": "The last organization owner cannot be removed or demoted",
                "user_not_found": "Identity user is not found",
            }
            raise OrganizationAccessError(
                code=error.code,
                message=messages.get(error.code, "Organization operation was rejected"),
            ) from error


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise OrganizationAccessError(code="invalid_datetime", message="Datetime must be UTC")
    return value.astimezone(timezone.utc)


def _display_name(value: str) -> str:
    normalized = " ".join(value.split())
    if not 2 <= len(normalized) <= 120:
        raise OrganizationAccessError(
            code="invalid_display_name",
            message="Display name must contain 2 to 120 characters",
        )
    return normalized


def _slug(value: str) -> str:
    normalized = value.strip().lower()
    if _SLUG_RE.fullmatch(normalized) is None:
        raise OrganizationAccessError(
            code="invalid_organization_slug",
            message="Organization slug is invalid",
        )
    return normalized
