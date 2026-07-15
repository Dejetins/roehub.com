from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.contexts.extensions.domain import (
    PluginEvent,
    PluginInstallation,
    PluginInstance,
    PluginOperation,
    PluginPackage,
)
from trading.contexts.identity.application.ports import CurrentUserPrincipal
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId


class PluginRepositoryInvariantError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class PluginAuthorization(Protocol):
    def require_read(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> InstallationId: ...

    def require_manage(
        self,
        *,
        principal: CurrentUserPrincipal,
        organization_id: OrganizationId,
    ) -> InstallationId: ...


class DataSourceAuthorizationError(PermissionError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class DataSourceAuthorization(Protocol):
    def resolve_read_scope(
        self,
        *,
        principal: CurrentUserPrincipal,
    ) -> tuple[InstallationId, OrganizationId]: ...


class DataSourceGatewayError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


class DataSourceInvoker(Protocol):
    async def query(
        self,
        *,
        organization_id: OrganizationId,
        instance_id: UUID,
        payload: Mapping[str, object],
        timeout_seconds: float,
        response_byte_limit: int,
    ) -> Mapping[str, Any]: ...


class PluginRepository(Protocol):
    def get_operation_by_idempotency(
        self, *, organization_id: OrganizationId, idempotency_key: str
    ) -> PluginOperation | None: ...

    def get_operation(self, *, operation_id: UUID) -> PluginOperation | None: ...

    def create_operation(self, *, operation: PluginOperation) -> PluginOperation: ...

    def claim_pending_operation(
        self,
        *,
        operation_id: UUID,
        updated_at: datetime,
    ) -> PluginOperation: ...

    def set_operation_status(
        self,
        *,
        operation_id: UUID,
        status: str,
        result: Mapping[str, object],
        updated_at: datetime,
    ) -> PluginOperation: ...

    def register_package(
        self,
        *,
        package: PluginPackage,
        actor_user_id: UserId,
    ) -> PluginPackage: ...

    def get_package(self, *, package_id: UUID) -> PluginPackage | None: ...

    def is_publisher_key_trusted(
        self,
        *,
        installation_id: InstallationId,
        key_id: str,
        fingerprint_sha256: str,
    ) -> bool: ...

    def get_plugin_installation(
        self, *, organization_id: OrganizationId, plugin_id: str
    ) -> PluginInstallation | None: ...

    def list_plugin_installations(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[PluginInstallation, ...]: ...

    def get_plugin_installation_by_id(
        self, *, plugin_installation_id: UUID
    ) -> PluginInstallation | None: ...

    def get_instance(self, *, instance_id: UUID) -> PluginInstance | None: ...

    def list_instances_for_organization(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[PluginInstance, ...]: ...

    def list_operations(
        self,
        *,
        organization_id: OrganizationId,
        limit: int,
    ) -> tuple[PluginOperation, ...]: ...

    def install_package(
        self,
        *,
        plugin_installation: PluginInstallation,
        instance: PluginInstance,
    ) -> tuple[PluginInstallation, PluginInstance]: ...

    def rollback_installation(
        self,
        *,
        plugin_installation_id: UUID,
        expected_current_package_id: UUID,
        target_package_id: UUID,
        updated_at: datetime,
    ) -> PluginInstallation: ...

    def record_event(self, *, event: PluginEvent) -> None: ...

    def list_events(
        self, *, organization_id: OrganizationId, limit: int
    ) -> tuple[PluginEvent, ...]: ...
