from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal
from uuid import UUID

from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId

OrganizationRole = Literal["owner", "admin", "operator", "trader", "viewer"]
OrganizationPermission = Literal[
    "organization.read",
    "organization.update",
    "members.read",
    "members.manage",
    "roles.manage",
    "plugins.read",
    "plugins.manage",
    "operations.execute",
    "trading.execute",
    "mainnet.approve",
    "audit.read",
]
PluginPermission = Literal["read", "configure", "operate"]

_ROLE_PERMISSIONS: Final[dict[OrganizationRole, frozenset[OrganizationPermission]]] = {
    "owner": frozenset(
        {
            "organization.read",
            "organization.update",
            "members.read",
            "members.manage",
            "roles.manage",
            "plugins.read",
            "plugins.manage",
            "operations.execute",
            "trading.execute",
            "mainnet.approve",
            "audit.read",
        }
    ),
    "admin": frozenset(
        {
            "organization.read",
            "organization.update",
            "members.read",
            "members.manage",
            "roles.manage",
            "plugins.read",
            "plugins.manage",
            "operations.execute",
            "audit.read",
        }
    ),
    "operator": frozenset(
        {
            "organization.read",
            "members.read",
            "plugins.read",
            "operations.execute",
            "audit.read",
        }
    ),
    "trader": frozenset(
        {
            "organization.read",
            "members.read",
            "plugins.read",
            "trading.execute",
        }
    ),
    "viewer": frozenset({"organization.read", "members.read", "plugins.read"}),
}


def permissions_for_role(role: OrganizationRole) -> frozenset[OrganizationPermission]:
    """Return the immutable permission set for one canonical organization role."""

    return _ROLE_PERMISSIONS[role]


@dataclass(frozen=True, slots=True)
class Installation:
    installation_id: InstallationId
    display_name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Organization:
    organization_id: OrganizationId
    installation_id: InstallationId
    slug: str
    display_name: str
    created_at: datetime
    status: Literal["active", "archived"] = "active"


@dataclass(frozen=True, slots=True)
class OrganizationMembership:
    organization_id: OrganizationId
    user_id: UserId
    role: OrganizationRole
    created_at: datetime
    status: Literal["active", "suspended"] = "active"


@dataclass(frozen=True, slots=True)
class OrganizationAccess:
    organization: Organization
    role: OrganizationRole
    permissions: frozenset[OrganizationPermission]


@dataclass(frozen=True, slots=True)
class OrganizationInvitation:
    invitation_id: UUID
    organization_id: OrganizationId
    role: OrganizationRole
    expires_at: datetime
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PluginPermissionGrant:
    organization_id: OrganizationId
    plugin_id: str
    user_id: UserId
    permission: PluginPermission
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SupportAccessGrant:
    grant_id: UUID
    installation_id: InstallationId
    support_user_id: UserId
    expires_at: datetime
    created_at: datetime
    revoked_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AdministrativeAuditEvent:
    event_id: UUID
    installation_id: InstallationId
    organization_id: OrganizationId | None
    actor_user_id: UserId
    action: str
    target_type: str
    target_id: str
    outcome: Literal["succeeded", "rejected"]
    metadata: dict[str, str]
    created_at: datetime
