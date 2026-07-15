from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from psycopg.types.json import Jsonb

from trading.contexts.notifications.domain import (
    NotificationProviderDescriptor,
    NotificationProviderHealth,
    NotificationProviderInstance,
    NotificationProviderPackage,
    TelegramCommandDescriptor,
    TelegramUpdateCursor,
)
from trading.shared_kernel.primitives import OrganizationId

from .gateway import NotificationPostgresGateway


class PostgresNotificationProviderRepository:
    def __init__(self, *, gateway: NotificationPostgresGateway) -> None:
        self._gateway = gateway

    def install_package(
        self, *, package: NotificationProviderPackage
    ) -> NotificationProviderPackage:
        descriptor = package.descriptor
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_provider_packages
              (package_id, provider_key, contract_version, package_version, display_name,
               config_schema_json, channels, templates, error_codes, built_in, installed_at)
            VALUES
              (%(package_id)s, %(provider_key)s, %(contract_version)s, %(package_version)s,
               %(display_name)s, %(config_schema_json)s, %(channels)s, %(templates)s,
               %(error_codes)s, %(built_in)s, %(installed_at)s)
            ON CONFLICT (provider_key, package_version) DO UPDATE SET
              package_id = notification_provider_packages.package_id
            RETURNING package_id, provider_key, contract_version, package_version,
                      display_name, config_schema_json, channels, templates, error_codes,
                      built_in, installed_at
            """,
            parameters={
                "package_id": str(package.package_id),
                "provider_key": descriptor.provider_key,
                "contract_version": descriptor.contract_version,
                "package_version": descriptor.package_version,
                "display_name": descriptor.display_name,
                "config_schema_json": Jsonb(dict(descriptor.config_schema)),
                "channels": list(descriptor.channels),
                "templates": list(descriptor.templates),
                "error_codes": list(descriptor.error_codes),
                "built_in": package.built_in,
                "installed_at": package.installed_at,
            },
        )
        return _map_package(_require_row(row, "provider package insert"))

    def get_package(self, *, package_id: UUID) -> NotificationProviderPackage | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT package_id, provider_key, contract_version, package_version,
                   display_name, config_schema_json, channels, templates, error_codes,
                   built_in, installed_at
            FROM notification_provider_packages
            WHERE package_id = %(package_id)s
            """,
            parameters={"package_id": str(package_id)},
        )
        return None if row is None else _map_package(row)

    def add_instance(
        self, *, instance: NotificationProviderInstance
    ) -> NotificationProviderInstance:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_provider_instances
              (instance_id, package_id, provider_key, scope, organization_id, display_name,
               config_json, secret_ref, status, created_at, updated_at)
            VALUES
              (%(instance_id)s, %(package_id)s, %(provider_key)s, %(scope)s,
               %(organization_id)s, %(display_name)s, %(config_json)s, %(secret_ref)s,
               %(status)s, %(created_at)s, %(updated_at)s)
            RETURNING instance_id, package_id, provider_key, scope, organization_id,
                      display_name, config_json, secret_ref, status, created_at, updated_at
            """,
            parameters=_instance_parameters(instance),
        )
        return _map_instance(_require_row(row, "provider instance insert"))

    def get_instance(self, *, instance_id: UUID) -> NotificationProviderInstance | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT instance_id, package_id, provider_key, scope, organization_id,
                   display_name, config_json, secret_ref, status, created_at, updated_at
            FROM notification_provider_instances
            WHERE instance_id = %(instance_id)s
            """,
            parameters={"instance_id": str(instance_id)},
        )
        return None if row is None else _map_instance(row)

    def list_active_instances(self) -> tuple[NotificationProviderInstance, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT instance_id, package_id, provider_key, scope, organization_id,
                   display_name, config_json, secret_ref, status, created_at, updated_at
            FROM notification_provider_instances
            WHERE status IN ('active', 'degraded')
            ORDER BY organization_id NULLS FIRST, instance_id
            """,
            parameters={},
        )
        return tuple(_map_instance(row) for row in rows)

    def list_instances_for_organization(
        self, *, organization_id: OrganizationId
    ) -> tuple[NotificationProviderInstance, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT instance_id, package_id, provider_key, scope, organization_id,
                   display_name, config_json, secret_ref, status, created_at, updated_at
            FROM notification_provider_instances
            WHERE status IN ('active', 'degraded')
              AND (organization_id IS NULL OR organization_id = %(organization_id)s)
            ORDER BY organization_id NULLS FIRST, instance_id
            """,
            parameters={"organization_id": str(organization_id)},
        )
        return tuple(_map_instance(row) for row in rows)

    def record_health(self, *, health: NotificationProviderHealth) -> None:
        self._gateway.execute(
            query="""
            UPDATE notification_provider_instances SET
              health_status = %(health_status)s,
              health_error_code = %(health_error_code)s,
              health_checked_at = %(health_checked_at)s,
              status = CASE
                WHEN status = 'disabled' THEN status
                WHEN %(health_status)s = 'ready' THEN 'active'
                ELSE 'degraded'
              END
            WHERE instance_id = %(instance_id)s
            """,
            parameters={
                "instance_id": str(health.instance_id),
                "health_status": health.status,
                "health_error_code": health.error_code,
                "health_checked_at": health.checked_at,
            },
        )

    def get_cursor(self, *, provider_instance_id: UUID) -> TelegramUpdateCursor | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT provider_instance_id, organization_id, last_update_id, updated_at
            FROM notification_telegram_update_cursors
            WHERE provider_instance_id = %(provider_instance_id)s
            """,
            parameters={"provider_instance_id": str(provider_instance_id)},
        )
        return None if row is None else _map_cursor(row)

    def advance_cursor(
        self,
        *,
        provider_instance_id: UUID,
        expected_last_update_id: int,
        next_update_id: int,
        updated_at: datetime,
    ) -> TelegramUpdateCursor:
        if next_update_id <= expected_last_update_id:
            raise ValueError("Telegram cursor must advance monotonically")
        row = self._gateway.fetch_one(
            query="""
            UPDATE notification_telegram_update_cursors SET
              last_update_id = %(next_update_id)s,
              updated_at = %(updated_at)s
            WHERE provider_instance_id = %(provider_instance_id)s
              AND last_update_id = %(expected_last_update_id)s
            RETURNING provider_instance_id, organization_id, last_update_id, updated_at
            """,
            parameters={
                "provider_instance_id": str(provider_instance_id),
                "expected_last_update_id": expected_last_update_id,
                "next_update_id": next_update_id,
                "updated_at": updated_at,
            },
        )
        if row is None:
            raise ValueError("Telegram cursor advance conflict")
        return _map_cursor(row)

    def replace_command_registry(
        self,
        *,
        provider_instance_id: UUID,
        commands: tuple[TelegramCommandDescriptor, ...],
    ) -> tuple[TelegramCommandDescriptor, ...]:
        if any(item.provider_instance_id != provider_instance_id for item in commands):
            raise ValueError("Telegram command registry instance mismatch")
        payload = [
            {
                "command_name": item.command_name,
                "description": item.description,
                "enabled": item.enabled,
                "updated_at": item.updated_at.isoformat(),
            }
            for item in commands
        ]
        rows = self._gateway.fetch_all(
            query="""
            WITH deleted AS (
              DELETE FROM notification_telegram_command_registry
              WHERE provider_instance_id = %(provider_instance_id)s
            )
            INSERT INTO notification_telegram_command_registry
              (provider_instance_id, command_name, description, enabled, updated_at)
            SELECT %(provider_instance_id)s, item.command_name, item.description,
                   item.enabled, item.updated_at
            FROM jsonb_to_recordset(%(commands)s::jsonb)
              AS item(command_name text, description text, enabled boolean, updated_at timestamptz)
            RETURNING provider_instance_id, command_name, description, enabled, updated_at
            """,
            parameters={
                "provider_instance_id": str(provider_instance_id),
                "commands": Jsonb(payload),
            },
        )
        return tuple(_map_command(row) for row in rows)


def _instance_parameters(instance: NotificationProviderInstance) -> dict[str, object]:
    return {
        "instance_id": str(instance.instance_id),
        "package_id": str(instance.package_id),
        "provider_key": instance.provider_key,
        "scope": instance.scope,
        "organization_id": (
            str(instance.organization_id) if instance.organization_id is not None else None
        ),
        "display_name": instance.display_name,
        "config_json": Jsonb(dict(instance.config_json)),
        "secret_ref": instance.secret_ref,
        "status": instance.status,
        "created_at": instance.created_at,
        "updated_at": instance.updated_at,
    }


def _map_package(row: Mapping[str, Any]) -> NotificationProviderPackage:
    descriptor = NotificationProviderDescriptor(
        provider_key=str(row["provider_key"]),
        display_name=str(row["display_name"]),
        package_version=str(row["package_version"]),
        config_schema=_mapping(row["config_schema_json"]),
        channels=_strings(row["channels"]),
        templates=_strings(row["templates"]),
        error_codes=_strings(row["error_codes"]),
        contract_version=str(row["contract_version"]),
    )
    return NotificationProviderPackage(
        package_id=UUID(str(row["package_id"])),
        descriptor=descriptor,
        built_in=bool(row["built_in"]),
        installed_at=row["installed_at"],
    )


def _map_instance(row: Mapping[str, Any]) -> NotificationProviderInstance:
    raw_organization_id = row["organization_id"]
    return NotificationProviderInstance(
        instance_id=UUID(str(row["instance_id"])),
        package_id=UUID(str(row["package_id"])),
        provider_key=str(row["provider_key"]),
        scope=row["scope"],
        organization_id=(
            None
            if raw_organization_id is None
            else OrganizationId.from_string(str(raw_organization_id))
        ),
        display_name=str(row["display_name"]),
        config_json=_mapping(row["config_json"]),
        secret_ref=None if row["secret_ref"] is None else str(row["secret_ref"]),
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _map_cursor(row: Mapping[str, Any]) -> TelegramUpdateCursor:
    raw_organization_id = row["organization_id"]
    return TelegramUpdateCursor(
        provider_instance_id=UUID(str(row["provider_instance_id"])),
        organization_id=(
            None
            if raw_organization_id is None
            else OrganizationId.from_string(str(raw_organization_id))
        ),
        last_update_id=int(row["last_update_id"]),
        updated_at=row["updated_at"],
    )


def _map_command(row: Mapping[str, Any]) -> TelegramCommandDescriptor:
    return TelegramCommandDescriptor(
        provider_instance_id=UUID(str(row["provider_instance_id"])),
        command_name=str(row["command_name"]),
        description=str(row["description"]),
        enabled=bool(row["enabled"]),
        updated_at=row["updated_at"],
    )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("provider JSON value must be an object")
    return dict(value)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("provider array value is invalid")
    return tuple(str(item) for item in value)


def _require_row(row: Mapping[str, Any] | None, operation: str) -> Mapping[str, Any]:
    if row is None:
        raise ValueError(f"{operation} returned no row")
    return row
