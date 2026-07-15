from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from threading import RLock
from typing import Mapping
from uuid import UUID

from trading.contexts.extensions.application.ports import PluginRepositoryInvariantError
from trading.contexts.extensions.domain import (
    PluginEvent,
    PluginInstallation,
    PluginInstance,
    PluginOperation,
    PluginPackage,
)
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId


class InMemoryPluginRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._operations: dict[UUID, PluginOperation] = {}
        self._idempotency: dict[tuple[OrganizationId, str], UUID] = {}
        self._packages: dict[UUID, PluginPackage] = {}
        self._publisher_keys: dict[tuple[InstallationId, str], tuple[str, str]] = {}
        self._package_identity: dict[tuple[object, str, str, str], UUID] = {}
        self._installations: dict[UUID, PluginInstallation] = {}
        self._installation_plugin: dict[UUID, str] = {}
        self._organization_plugin: dict[tuple[OrganizationId, str], UUID] = {}
        self._instances: dict[UUID, PluginInstance] = {}
        self._events: list[PluginEvent] = []

    def get_operation_by_idempotency(
        self, *, organization_id: OrganizationId, idempotency_key: str
    ) -> PluginOperation | None:
        operation_id = self._idempotency.get((organization_id, idempotency_key))
        return self._operations.get(operation_id) if operation_id else None

    def get_operation(self, *, operation_id: UUID) -> PluginOperation | None:
        return self._operations.get(operation_id)

    def create_operation(self, *, operation: PluginOperation) -> PluginOperation:
        with self._lock:
            identity = (operation.organization_id, operation.idempotency_key)
            existing_id = self._idempotency.get(identity)
            if existing_id is not None:
                return self._operations[existing_id]
            self._operations[operation.operation_id] = operation
            self._idempotency[identity] = operation.operation_id
            return operation

    def claim_pending_operation(
        self,
        *,
        operation_id: UUID,
        updated_at: datetime,
    ) -> PluginOperation:
        with self._lock:
            current = self._operations.get(operation_id)
            if current is None:
                raise PluginRepositoryInvariantError(code="plugin.operation_not_found")
            if current.status != "pending":
                raise PluginRepositoryInvariantError(code="plugin.operation_not_pending")
            claimed = replace(current, status="running", updated_at=updated_at)
            self._operations[operation_id] = claimed
            return claimed

    def set_operation_status(
        self,
        *,
        operation_id: UUID,
        status: str,
        result: Mapping[str, object],
        updated_at: datetime,
    ) -> PluginOperation:
        with self._lock:
            current = self._operations.get(operation_id)
            if current is None:
                raise PluginRepositoryInvariantError(code="plugin.operation_not_found")
            updated = replace(
                current,
                status=status,  # type: ignore[arg-type]
                result=dict(result),
                updated_at=updated_at,
            )
            self._operations[operation_id] = updated
            return updated

    def register_package(
        self,
        *,
        package: PluginPackage,
        actor_user_id: UserId,
    ) -> PluginPackage:
        with self._lock:
            _ = actor_user_id
            if package.publisher_key_id is not None:
                public_key = package.publisher_public_key_b64
                fingerprint = package.publisher_key_fingerprint_sha256
                if public_key is None or fingerprint is None:
                    raise PluginRepositoryInvariantError(code="plugin.publisher_untrusted")
                key_identity = (package.installation_id, package.publisher_key_id)
                existing_key = self._publisher_keys.get(key_identity)
                if existing_key is not None and existing_key != (fingerprint, "trusted"):
                    raise PluginRepositoryInvariantError(code="plugin.publisher_untrusted")
                self._publisher_keys[key_identity] = (fingerprint, "trusted")
            identity = (
                package.installation_id,
                package.plugin_id,
                package.version,
                package.package_digest,
            )
            existing_id = self._package_identity.get(identity)
            if existing_id is not None:
                return self._packages[existing_id]
            for existing in self._packages.values():
                if (
                    existing.installation_id == package.installation_id
                    and existing.plugin_id == package.plugin_id
                    and existing.version == package.version
                ):
                    raise PluginRepositoryInvariantError(code="plugin.package_version_conflict")
            self._packages[package.package_id] = package
            self._package_identity[identity] = package.package_id
            return package

    def get_package(self, *, package_id: UUID) -> PluginPackage | None:
        return self._packages.get(package_id)

    def is_publisher_key_trusted(
        self,
        *,
        installation_id: InstallationId,
        key_id: str,
        fingerprint_sha256: str,
    ) -> bool:
        return self._publisher_keys.get((installation_id, key_id)) == (
            fingerprint_sha256,
            "trusted",
        )

    def revoke_publisher_key(
        self,
        *,
        installation_id: InstallationId,
        key_id: str,
    ) -> None:
        with self._lock:
            current = self._publisher_keys.get((installation_id, key_id))
            if current is not None:
                self._publisher_keys[(installation_id, key_id)] = (
                    current[0],
                    "revoked",
                )

    def get_plugin_installation(
        self, *, organization_id: OrganizationId, plugin_id: str
    ) -> PluginInstallation | None:
        installation_id = self._organization_plugin.get((organization_id, plugin_id))
        return self._installations.get(installation_id) if installation_id else None

    def list_plugin_installations(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[PluginInstallation, ...]:
        return tuple(
            sorted(
                (
                    installation
                    for installation in self._installations.values()
                    if installation.organization_id == organization_id
                ),
                key=lambda installation: installation.plugin_id,
            )
        )

    def get_plugin_installation_by_id(
        self, *, plugin_installation_id: UUID
    ) -> PluginInstallation | None:
        return self._installations.get(plugin_installation_id)

    def get_instance(self, *, instance_id: UUID) -> PluginInstance | None:
        return self._instances.get(instance_id)

    def list_instances_for_organization(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[PluginInstance, ...]:
        return tuple(
            sorted(
                (
                    instance
                    for instance in self._instances.values()
                    if instance.organization_id == organization_id
                ),
                key=lambda instance: (instance.name, str(instance.instance_id)),
            )
        )

    def list_operations(
        self,
        *,
        organization_id: OrganizationId,
        limit: int,
    ) -> tuple[PluginOperation, ...]:
        operations = [
            operation
            for operation in self._operations.values()
            if operation.organization_id == organization_id
        ]
        operations.sort(
            key=lambda operation: (operation.created_at, str(operation.operation_id)),
            reverse=True,
        )
        return tuple(operations[:limit])

    def install_package(
        self,
        *,
        plugin_installation: PluginInstallation,
        instance: PluginInstance,
    ) -> tuple[PluginInstallation, PluginInstance]:
        with self._lock:
            package = self._packages.get(plugin_installation.package_id)
            if package is None:
                raise PluginRepositoryInvariantError(code="plugin.package_not_found")
            if package.plugin_id != plugin_installation.plugin_id:
                raise PluginRepositoryInvariantError(code="plugin.package_identity_mismatch")
            key = (plugin_installation.organization_id, plugin_installation.plugin_id)
            existing_id = self._organization_plugin.get(key)
            if (
                existing_id is not None
                and existing_id != plugin_installation.plugin_installation_id
            ):
                raise PluginRepositoryInvariantError(code="plugin.installation_conflict")
            self._installations[plugin_installation.plugin_installation_id] = plugin_installation
            self._installation_plugin[plugin_installation.plugin_installation_id] = (
                plugin_installation.plugin_id
            )
            self._organization_plugin[key] = plugin_installation.plugin_installation_id
            existing_instance = next(
                (
                    candidate
                    for candidate in self._instances.values()
                    if candidate.organization_id == instance.organization_id
                    and candidate.plugin_installation_id == instance.plugin_installation_id
                    and candidate.name == instance.name
                ),
                None,
            )
            if existing_instance is None:
                persisted_instance = instance
            else:
                persisted_instance = replace(
                    existing_instance,
                    config=instance.config,
                    config_revision=existing_instance.config_revision + 1,
                    status=instance.status,
                    updated_at=instance.updated_at,
                )
            self._instances[persisted_instance.instance_id] = persisted_instance
            return plugin_installation, persisted_instance

    def rollback_installation(
        self,
        *,
        plugin_installation_id: UUID,
        expected_current_package_id: UUID,
        target_package_id: UUID,
        updated_at: datetime,
    ) -> PluginInstallation:
        with self._lock:
            current = self._installations.get(plugin_installation_id)
            if (
                current is None
                or current.package_id != expected_current_package_id
                or current.previous_package_id != target_package_id
            ):
                raise PluginRepositoryInvariantError(code="plugin.rollback_unavailable")
            updated = replace(
                current,
                package_id=target_package_id,
                previous_package_id=expected_current_package_id,
                updated_at=updated_at,
            )
            self._installations[plugin_installation_id] = updated
            return updated

    def record_event(self, *, event: PluginEvent) -> None:
        with self._lock:
            self._events.append(event)

    def list_events(
        self, *, organization_id: OrganizationId, limit: int
    ) -> tuple[PluginEvent, ...]:
        return tuple(
            event
            for event in reversed(self._events)
            if event.organization_id == organization_id
        )[:limit]
