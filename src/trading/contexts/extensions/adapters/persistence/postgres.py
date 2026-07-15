"""PostgreSQL adapter for the Stage 12 plugin lifecycle."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, cast
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from trading.contexts.extensions.application.ports import PluginRepositoryInvariantError
from trading.contexts.extensions.domain import (
    PluginEvent,
    PluginInstallation,
    PluginInstance,
    PluginOperation,
    PluginPackage,
)
from trading.shared_kernel.primitives import InstallationId, OrganizationId, UserId


class PostgresPluginRepository:
    def __init__(self, *, dsn: str) -> None:
        if not dsn.strip():
            raise ValueError("PostgresPluginRepository requires dsn")
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._dsn, row_factory=cast(Any, dict_row))

    def get_operation_by_idempotency(
        self, *, organization_id: OrganizationId, idempotency_key: str
    ) -> PluginOperation | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT * FROM extensions_plugin_operations
                   WHERE organization_id = %s AND idempotency_key = %s""",
                (organization_id.value, idempotency_key),
            )
            return _operation(cursor.fetchone())

    def get_operation(self, *, operation_id: UUID) -> PluginOperation | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM extensions_plugin_operations WHERE operation_id = %s",
                (operation_id,),
            )
            return _operation(cursor.fetchone())

    def create_operation(self, *, operation: PluginOperation) -> PluginOperation:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO extensions_plugin_operations (
                           operation_id, installation_id, organization_id, actor_user_id,
                           kind, target_id, idempotency_key, request_hash, request_payload,
                           status, result,
                           created_at, updated_at
                       ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (organization_id, idempotency_key) DO UPDATE SET
                           idempotency_key = EXCLUDED.idempotency_key
                       RETURNING *""",
                    (
                        operation.operation_id,
                        operation.installation_id.value,
                        operation.organization_id.value,
                        operation.actor_user_id.value,
                        operation.kind,
                        operation.target_id,
                        operation.idempotency_key,
                        operation.request_hash,
                        Jsonb(dict(operation.request)),
                        operation.status,
                        Jsonb(dict(operation.result)),
                        operation.created_at,
                        operation.updated_at,
                    ),
                )
                persisted = _operation(cursor.fetchone())
            assert persisted is not None
            return persisted
        except psycopg.errors.UniqueViolation as error:
            raise PluginRepositoryInvariantError(code="plugin.idempotency_conflict") from error

    def claim_pending_operation(
        self,
        *,
        operation_id: UUID,
        updated_at: datetime,
    ) -> PluginOperation:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE extensions_plugin_operations
                   SET status = 'running', updated_at = %s
                   WHERE operation_id = %s AND status = 'pending'
                   RETURNING *""",
                (updated_at, operation_id),
            )
            claimed = _operation(cursor.fetchone())
            if claimed is not None:
                return claimed
            cursor.execute(
                "SELECT status FROM extensions_plugin_operations WHERE operation_id = %s",
                (operation_id,),
            )
            existing = cursor.fetchone()
        code = "plugin.operation_not_found" if existing is None else "plugin.operation_not_pending"
        raise PluginRepositoryInvariantError(code=code)

    def set_operation_status(
        self,
        *,
        operation_id: UUID,
        status: str,
        result: Mapping[str, object],
        updated_at: datetime,
    ) -> PluginOperation:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE extensions_plugin_operations
                   SET status = %s, result = %s, updated_at = %s
                   WHERE operation_id = %s RETURNING *""",
                (status, Jsonb(dict(result)), updated_at, operation_id),
            )
            operation = _operation(cursor.fetchone())
        if operation is None:
            raise PluginRepositoryInvariantError(code="plugin.operation_not_found")
        return operation

    def register_package(
        self,
        *,
        package: PluginPackage,
        actor_user_id: UserId,
    ) -> PluginPackage:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                if package.publisher_key_id is not None:
                    public_key = package.publisher_public_key_b64
                    fingerprint = package.publisher_key_fingerprint_sha256
                    if public_key is None or fingerprint is None:
                        raise PluginRepositoryInvariantError(
                            code="plugin.publisher_untrusted"
                        )
                    cursor.execute(
                        """INSERT INTO extensions_publisher_keys (
                               installation_id, key_id, algorithm, public_key_b64,
                               fingerprint_sha256, status, added_by_user_id, created_at
                           ) VALUES (%s,%s,'Ed25519',%s,%s,'trusted',%s,%s)
                           ON CONFLICT (installation_id, key_id) DO NOTHING""",
                        (
                            package.installation_id.value,
                            package.publisher_key_id,
                            public_key,
                            fingerprint,
                            actor_user_id.value,
                            package.created_at,
                        ),
                    )
                    cursor.execute(
                        """SELECT public_key_b64, fingerprint_sha256, status
                           FROM extensions_publisher_keys
                           WHERE installation_id = %s AND key_id = %s
                           FOR SHARE""",
                        (package.installation_id.value, package.publisher_key_id),
                    )
                    publisher_key = cursor.fetchone()
                    if (
                        publisher_key is None
                        or publisher_key["public_key_b64"] != public_key
                        or publisher_key["fingerprint_sha256"] != fingerprint
                        or publisher_key["status"] != "trusted"
                    ):
                        raise PluginRepositoryInvariantError(
                            code="plugin.publisher_untrusted"
                        )
                cursor.execute(
                    """INSERT INTO extensions_plugin_packages (
                           package_id, installation_id, plugin_id, version, package_digest,
                           image_reference,
                           image_digest, publisher_key_id, manifest, created_at
                       ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (installation_id, plugin_id, version, package_digest) DO UPDATE
                       SET plugin_id = EXCLUDED.plugin_id
                       RETURNING *""",
                    (
                        package.package_id,
                        package.installation_id.value,
                        package.plugin_id,
                        package.version,
                        package.package_digest,
                        package.image_reference,
                        package.image_digest,
                        package.publisher_key_id,
                        Jsonb(dict(package.manifest)),
                        package.created_at,
                    ),
                )
                row = cast(Mapping[str, Any], cursor.fetchone())
            return _package(row)
        except PluginRepositoryInvariantError:
            raise
        except psycopg.errors.UniqueViolation as error:
            raise PluginRepositoryInvariantError(code="plugin.package_version_conflict") from error

    def get_package(self, *, package_id: UUID) -> PluginPackage | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT package.*,
                          publisher.public_key_b64 AS publisher_public_key_b64,
                          publisher.fingerprint_sha256 AS publisher_key_fingerprint_sha256
                   FROM extensions_plugin_packages AS package
                   LEFT JOIN extensions_publisher_keys AS publisher
                     ON publisher.installation_id = package.installation_id
                    AND publisher.key_id = package.publisher_key_id
                   WHERE package.package_id = %s""",
                (package_id,),
            )
            row = cursor.fetchone()
        return _package(row) if row is not None else None

    def is_publisher_key_trusted(
        self,
        *,
        installation_id: InstallationId,
        key_id: str,
        fingerprint_sha256: str,
    ) -> bool:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT 1 FROM extensions_publisher_keys
                   WHERE installation_id = %s AND key_id = %s
                     AND fingerprint_sha256 = %s AND status = 'trusted'""",
                (installation_id.value, key_id, fingerprint_sha256),
            )
            return cursor.fetchone() is not None

    def get_plugin_installation(
        self, *, organization_id: OrganizationId, plugin_id: str
    ) -> PluginInstallation | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT installation.* FROM extensions_plugin_installations AS installation
                   WHERE installation.organization_id = %s AND installation.plugin_id = %s""",
                (organization_id.value, plugin_id),
            )
            return _plugin_installation(cursor.fetchone())

    def list_plugin_installations(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[PluginInstallation, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT * FROM extensions_plugin_installations
                   WHERE organization_id = %s
                   ORDER BY plugin_id, plugin_installation_id""",
                (organization_id.value,),
            )
            rows = cursor.fetchall()
        return tuple(
            installation
            for row in rows
            if (installation := _plugin_installation(row)) is not None
        )

    def get_plugin_installation_by_id(
        self, *, plugin_installation_id: UUID
    ) -> PluginInstallation | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM extensions_plugin_installations WHERE plugin_installation_id = %s",
                (plugin_installation_id,),
            )
            return _plugin_installation(cursor.fetchone())

    def get_instance(self, *, instance_id: UUID) -> PluginInstance | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM extensions_plugin_instances WHERE instance_id = %s",
                (instance_id,),
            )
            return _instance(cursor.fetchone())

    def list_instances_for_organization(
        self,
        *,
        organization_id: OrganizationId,
    ) -> tuple[PluginInstance, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT * FROM extensions_plugin_instances
                   WHERE organization_id = %s
                   ORDER BY name, instance_id""",
                (organization_id.value,),
            )
            rows = cursor.fetchall()
        return tuple(
            instance for row in rows if (instance := _instance(row)) is not None
        )

    def list_operations(
        self,
        *,
        organization_id: OrganizationId,
        limit: int,
    ) -> tuple[PluginOperation, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT * FROM extensions_plugin_operations
                   WHERE organization_id = %s
                   ORDER BY created_at DESC, operation_id DESC
                   LIMIT %s""",
                (organization_id.value, limit),
            )
            rows = cursor.fetchall()
        return tuple(
            operation for row in rows if (operation := _operation(row)) is not None
        )

    def install_package(
        self,
        *,
        plugin_installation: PluginInstallation,
        instance: PluginInstance,
    ) -> tuple[PluginInstallation, PluginInstance]:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO extensions_plugin_installations (
                           plugin_installation_id, installation_id, organization_id, plugin_id,
                           package_id,
                           previous_package_id, granted_permissions, status, created_at, updated_at
                       ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (plugin_installation_id) DO UPDATE SET
                           package_id = EXCLUDED.package_id,
                           previous_package_id = EXCLUDED.previous_package_id,
                           granted_permissions = EXCLUDED.granted_permissions,
                           status = EXCLUDED.status,
                           updated_at = EXCLUDED.updated_at
                       RETURNING *""",
                    (
                        plugin_installation.plugin_installation_id,
                        plugin_installation.installation_id.value,
                        plugin_installation.organization_id.value,
                        plugin_installation.plugin_id,
                        plugin_installation.package_id,
                        plugin_installation.previous_package_id,
                        list(plugin_installation.granted_permissions),
                        plugin_installation.status,
                        plugin_installation.created_at,
                        plugin_installation.updated_at,
                    ),
                )
                installed = _plugin_installation(cursor.fetchone())
                cursor.execute(
                    """INSERT INTO extensions_plugin_instances (
                           instance_id, plugin_installation_id, installation_id, organization_id,
                           name, config, config_revision, status, created_at, updated_at
                       ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (organization_id, plugin_installation_id, name)
                       DO UPDATE SET
                           config = EXCLUDED.config,
                           config_revision = extensions_plugin_instances.config_revision + 1,
                           status = EXCLUDED.status,
                           updated_at = EXCLUDED.updated_at
                       RETURNING *""",
                    (
                        instance.instance_id,
                        instance.plugin_installation_id,
                        instance.installation_id.value,
                        instance.organization_id.value,
                        instance.name,
                        Jsonb(dict(instance.config)),
                        instance.config_revision,
                        instance.status,
                        instance.created_at,
                        instance.updated_at,
                    ),
                )
                created_instance = _instance(cursor.fetchone())
            assert installed is not None and created_instance is not None
            return installed, created_instance
        except psycopg.Error as error:
            raise PluginRepositoryInvariantError(code="plugin.installation_conflict") from error

    def rollback_installation(
        self,
        *,
        plugin_installation_id: UUID,
        expected_current_package_id: UUID,
        target_package_id: UUID,
        updated_at: datetime,
    ) -> PluginInstallation:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE extensions_plugin_installations SET
                       package_id = previous_package_id,
                       previous_package_id = package_id,
                       updated_at = %s
                   WHERE plugin_installation_id = %s
                     AND package_id = %s
                     AND previous_package_id = %s
                   RETURNING *""",
                (
                    updated_at,
                    plugin_installation_id,
                    expected_current_package_id,
                    target_package_id,
                ),
            )
            result = _plugin_installation(cursor.fetchone())
        if result is None:
            raise PluginRepositoryInvariantError(code="plugin.rollback_unavailable")
        return result

    def record_event(self, *, event: PluginEvent) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO extensions_plugin_events (
                       event_id, installation_id, organization_id, actor_user_id, event_type,
                       target_type, target_id, outcome, metadata, created_at
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    event.event_id,
                    event.installation_id.value,
                    event.organization_id.value,
                    event.actor_user_id.value,
                    event.event_type,
                    event.target_type,
                    event.target_id,
                    event.outcome,
                    Jsonb(dict(event.metadata)),
                    event.created_at,
                ),
            )

    def list_events(
        self, *, organization_id: OrganizationId, limit: int
    ) -> tuple[PluginEvent, ...]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT * FROM extensions_plugin_events
                   WHERE organization_id = %s ORDER BY created_at DESC LIMIT %s""",
                (organization_id.value, limit),
            )
            return tuple(_event(row) for row in cursor.fetchall())


def _operation(row: Mapping[str, Any] | None) -> PluginOperation | None:
    if row is None:
        return None
    return PluginOperation(
        operation_id=row["operation_id"],
        installation_id=InstallationId(row["installation_id"]),
        organization_id=OrganizationId(row["organization_id"]),
        actor_user_id=UserId(row["actor_user_id"]),
        kind=row["kind"],
        target_id=row["target_id"],
        idempotency_key=row["idempotency_key"],
        request_hash=row["request_hash"],
        request=dict(row["request_payload"]),
        status=row["status"],
        result=dict(row["result"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _package(row: Mapping[str, Any]) -> PluginPackage:
    return PluginPackage(
        package_id=row["package_id"],
        installation_id=InstallationId(row["installation_id"]),
        plugin_id=row["plugin_id"],
        version=row["version"],
        package_digest=row["package_digest"],
        image_reference=row["image_reference"],
        image_digest=row["image_digest"],
        publisher_key_id=row["publisher_key_id"],
        publisher_public_key_b64=row.get("publisher_public_key_b64"),
        publisher_key_fingerprint_sha256=row.get(
            "publisher_key_fingerprint_sha256"
        ),
        manifest=dict(row["manifest"]),
        created_at=row["created_at"],
    )


def _plugin_installation(row: Mapping[str, Any] | None) -> PluginInstallation | None:
    if row is None:
        return None
    return PluginInstallation(
        plugin_installation_id=row["plugin_installation_id"],
        installation_id=InstallationId(row["installation_id"]),
        organization_id=OrganizationId(row["organization_id"]),
        plugin_id=row["plugin_id"],
        package_id=row["package_id"],
        previous_package_id=row["previous_package_id"],
        granted_permissions=tuple(row["granted_permissions"]),
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _instance(row: Mapping[str, Any] | None) -> PluginInstance | None:
    if row is None:
        return None
    return PluginInstance(
        instance_id=row["instance_id"],
        plugin_installation_id=row["plugin_installation_id"],
        installation_id=InstallationId(row["installation_id"]),
        organization_id=OrganizationId(row["organization_id"]),
        name=row["name"],
        config=dict(row["config"]),
        config_revision=row["config_revision"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _event(row: Mapping[str, Any]) -> PluginEvent:
    return PluginEvent(
        event_id=row["event_id"],
        installation_id=InstallationId(row["installation_id"]),
        organization_id=OrganizationId(row["organization_id"]),
        actor_user_id=UserId(row["actor_user_id"]),
        event_type=row["event_type"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        outcome=row["outcome"],
        metadata=dict(row["metadata"]),
        created_at=row["created_at"],
    )
