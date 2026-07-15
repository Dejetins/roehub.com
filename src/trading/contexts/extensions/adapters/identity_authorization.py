from __future__ import annotations

from trading.contexts.extensions.application.ports import (
    DataSourceAuthorization,
    DataSourceAuthorizationError,
    PluginAuthorization,
)
from trading.contexts.identity.application.ports import (
    CurrentUserPrincipal,
    OrganizationRepository,
)
from trading.contexts.identity.domain.entities import permissions_for_role
from trading.shared_kernel.primitives import InstallationId, OrganizationId


class IdentityPluginAuthorization(PluginAuthorization, DataSourceAuthorization):
    """ACL from extensions to the canonical identity organization boundary."""

    def __init__(self, *, repository: OrganizationRepository) -> None:
        self._repository = repository

    def require_manage(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> InstallationId:
        membership = self._repository.get_membership(
            organization_id=organization_id,
            user_id=principal.user_id,
        )
        if (
            membership is None
            or membership.status != "active"
            or "plugins.manage" not in permissions_for_role(membership.role)
        ):
            raise PermissionError("organization plugin management permission is required")
        installation = self._repository.get_installation()
        if installation is None:
            raise PermissionError("installation is not initialized")
        return installation.installation_id

    def require_read(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> InstallationId:
        membership = self._repository.get_membership(
            organization_id=organization_id,
            user_id=principal.user_id,
        )
        if (
            membership is None
            or membership.status != "active"
            or "plugins.read" not in permissions_for_role(membership.role)
        ):
            raise PermissionError("organization plugin read permission is required")
        installation = self._repository.get_installation()
        if installation is None:
            raise PermissionError("installation is not initialized")
        return installation.installation_id

    def resolve_read_scope(
        self,
        *,
        principal: CurrentUserPrincipal,
    ) -> tuple[InstallationId, OrganizationId]:
        accesses = tuple(
            access
            for access in self._repository.list_accesses_for_user(
                user_id=principal.user_id
            )
            if "plugins.read" in access.permissions
        )
        if not accesses:
            raise DataSourceAuthorizationError(
                code="data_source.organization_scope_forbidden"
            )
        if len(accesses) != 1:
            raise DataSourceAuthorizationError(
                code="data_source.organization_scope_ambiguous"
            )
        installation = self._repository.get_installation()
        if installation is None:
            raise DataSourceAuthorizationError(
                code="data_source.organization_scope_unavailable"
            )
        return installation.installation_id, accesses[0].organization.organization_id
