from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, cast

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from trading.contexts.backtest_artifacts.domain import (
    ArtifactCatalogState,
    ArtifactGarbageCandidate,
    ArtifactStoreError,
)
from trading.integration import (
    ArtifactBackupBlob,
    ArtifactBackupCatalog,
    ArtifactBlobDescriptor,
    ArtifactManifest,
)
from trading.shared_kernel.primitives import OrganizationId

_DEFAULT_QUOTA_BYTES = 10 * 1024 * 1024 * 1024


class PostgresArtifactCatalogRepository:
    def __init__(self, *, dsn: str) -> None:
        if not dsn.strip():
            raise ValueError("PostgresArtifactCatalogRepository requires dsn")
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self._dsn, row_factory=cast(Any, dict_row))

    def register_blob(
        self,
        *,
        descriptor: ArtifactBlobDescriptor,
        backend: str,
        registered_at: datetime,
    ) -> None:
        if backend not in {"local_cas", "s3_compatible"}:
            raise ArtifactStoreError(code="artifact.backend_invalid")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 1))",
                (descriptor.digest,),
            )
            cursor.execute(
                """INSERT INTO artifact_store_objects (digest, size_bytes, created_at)
                   VALUES (%s,%s,%s) ON CONFLICT (digest) DO NOTHING""",
                (descriptor.digest, descriptor.size_bytes, registered_at),
            )
            cursor.execute(
                "SELECT size_bytes FROM artifact_store_objects WHERE digest = %s",
                (descriptor.digest,),
            )
            existing = cursor.fetchone()
            if existing is None or existing["size_bytes"] != descriptor.size_bytes:
                raise ArtifactStoreError(code="artifact.digest_metadata_conflict")
            cursor.execute(
                """INSERT INTO artifact_store_object_locations
                       (digest, backend, registered_at)
                   VALUES (%s,%s,%s) ON CONFLICT DO NOTHING""",
                (descriptor.digest, backend, registered_at),
            )

    def set_quota(self, *, organization_id: OrganizationId, max_bytes: int) -> None:
        if not 1 <= max_bytes <= 1_099_511_627_776:
            raise ArtifactStoreError(code="artifact.quota_invalid")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (str(organization_id.value),),
            )
            current = self._usage_bytes(cursor, organization_id)
            if current > max_bytes:
                raise ArtifactStoreError(code="artifact.quota_below_usage")
            cursor.execute(
                """INSERT INTO artifact_store_quotas (organization_id, max_bytes, updated_at)
                   VALUES (%s, %s, now())
                   ON CONFLICT (organization_id) DO UPDATE SET
                     max_bytes = EXCLUDED.max_bytes, updated_at = EXCLUDED.updated_at""",
                (organization_id.value, max_bytes),
            )

    def usage_bytes(self, *, organization_id: OrganizationId) -> int:
        with self._connect() as connection, connection.cursor() as cursor:
            return self._usage_bytes(cursor, organization_id)

    def publish_manifest(
        self,
        *,
        organization_id: OrganizationId,
        manifest: ArtifactManifest,
        backend: str,
        published_at: datetime,
    ) -> None:
        if backend not in {"local_cas", "s3_compatible"}:
            raise ArtifactStoreError(code="artifact.backend_invalid")
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (str(organization_id.value),),
                )
                cursor.execute(
                    """INSERT INTO artifact_store_quotas (organization_id, max_bytes, updated_at)
                       VALUES (%s, %s, %s) ON CONFLICT (organization_id) DO NOTHING""",
                    (organization_id.value, _DEFAULT_QUOTA_BYTES, published_at),
                )
                for entry in manifest.entries:
                    blob = entry.blob
                    cursor.execute(
                        """SELECT object.size_bytes
                           FROM artifact_store_objects AS object
                           JOIN artifact_store_object_locations AS location
                             ON location.digest = object.digest
                           WHERE object.digest = %s AND location.backend = %s
                           FOR SHARE OF object, location""",
                        (blob.digest, backend),
                    )
                    existing = cursor.fetchone()
                    if existing is None or existing["size_bytes"] != blob.size_bytes:
                        raise ArtifactStoreError(code="artifact.digest_metadata_conflict")
                    cursor.execute(
                        """DELETE FROM artifact_store_gc_candidates
                           WHERE digest = %s AND backend = %s""",
                        (blob.digest, backend),
                    )
                    cursor.execute(
                        """INSERT INTO artifact_store_org_blobs
                               (organization_id, digest, backend, acquired_at)
                           VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (organization_id.value, blob.digest, backend, published_at),
                    )
                cursor.execute(
                    "SELECT max_bytes FROM artifact_store_quotas WHERE organization_id = %s",
                    (organization_id.value,),
                )
                quota = cursor.fetchone()
                assert quota is not None
                if self._usage_bytes(cursor, organization_id) > quota["max_bytes"]:
                    raise ArtifactStoreError(code="artifact.quota_exceeded")
                payload = manifest.model_dump(mode="json", by_alias=True)
                cursor.execute(
                    """INSERT INTO artifact_store_manifests
                           (organization_id, manifest_digest, bundle_id, name, version,
                            manifest, published_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (organization_id, manifest_digest) DO UPDATE
                       SET manifest_digest = EXCLUDED.manifest_digest
                       RETURNING manifest""",
                    (
                        organization_id.value,
                        manifest.manifest_digest,
                        manifest.bundle_id,
                        manifest.name,
                        manifest.version,
                        Jsonb(payload),
                        published_at,
                    ),
                )
                persisted = cursor.fetchone()
                if persisted is None or persisted["manifest"] != payload:
                    raise ArtifactStoreError(code="artifact.manifest_conflict")
                for entry in manifest.entries:
                    cursor.execute(
                        """INSERT INTO artifact_store_manifest_entries
                               (organization_id, manifest_digest, path, digest,
                                size_bytes, media_type)
                           VALUES (%s,%s,%s,%s,%s,%s)
                           ON CONFLICT DO NOTHING""",
                        (
                            organization_id.value,
                            manifest.manifest_digest,
                            entry.path,
                            entry.blob.digest,
                            entry.blob.size_bytes,
                            entry.blob.media_type,
                        ),
                    )
        except ArtifactStoreError:
            raise
        except psycopg.errors.UniqueViolation as error:
            raise ArtifactStoreError(code="artifact.manifest_version_conflict") from error
        except psycopg.Error as error:
            raise ArtifactStoreError(code="artifact.catalog_unavailable") from error

    def get_manifest(
        self, *, organization_id: OrganizationId, manifest_digest: str
    ) -> ArtifactManifest | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT manifest FROM artifact_store_manifests
                   WHERE organization_id = %s AND manifest_digest = %s""",
                (organization_id.value, manifest_digest),
            )
            row = cursor.fetchone()
        return None if row is None else ArtifactManifest.model_validate(row["manifest"])

    def retire_manifest(self, *, organization_id: OrganizationId, manifest_digest: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """DELETE FROM artifact_store_manifests
                   WHERE organization_id = %s AND manifest_digest = %s""",
                (organization_id.value, manifest_digest),
            )
            if cursor.rowcount != 1:
                raise ArtifactStoreError(code="artifact.manifest_not_found")

    def pin(self, *, organization_id: OrganizationId, digest: str, pinned_at: datetime) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO artifact_store_pins (organization_id, digest, pinned_at)
                       VALUES (%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (organization_id.value, digest, pinned_at),
                )
                if cursor.rowcount == 0:
                    cursor.execute(
                        """SELECT 1 FROM artifact_store_org_blobs
                           WHERE organization_id = %s AND digest = %s""",
                        (organization_id.value, digest),
                    )
                    if cursor.fetchone() is None:
                        raise ArtifactStoreError(code="artifact.not_found")
        except psycopg.errors.ForeignKeyViolation as error:
            raise ArtifactStoreError(code="artifact.not_found") from error

    def unpin(self, *, organization_id: OrganizationId, digest: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM artifact_store_pins WHERE organization_id = %s AND digest = %s",
                (organization_id.value, digest),
            )

    def acquire_lease(
        self,
        *,
        organization_id: OrganizationId,
        lease_id: str,
        digest: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO artifact_store_leases
                           (organization_id, lease_id, digest, created_at, expires_at)
                       VALUES (%s,%s,%s,%s,%s)
                       ON CONFLICT (organization_id, lease_id) DO UPDATE SET
                         digest = EXCLUDED.digest, created_at = EXCLUDED.created_at,
                         expires_at = EXCLUDED.expires_at""",
                    (organization_id.value, lease_id, digest, created_at, expires_at),
                )
        except psycopg.errors.ForeignKeyViolation as error:
            raise ArtifactStoreError(code="artifact.not_found") from error
        except psycopg.errors.CheckViolation as error:
            raise ArtifactStoreError(code="artifact.lease_invalid") from error

    def release_lease(self, *, organization_id: OrganizationId, lease_id: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM artifact_store_leases WHERE organization_id = %s AND lease_id = %s",
                (organization_id.value, lease_id),
            )

    def restore_state(
        self,
        *,
        organization_id: OrganizationId,
        backup: ArtifactBackupCatalog,
        backend: str,
        restored_at: datetime,
    ) -> None:
        if backend not in {"local_cas", "s3_compatible"}:
            raise ArtifactStoreError(code="artifact.backend_invalid")
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (str(organization_id.value),),
                )
                cursor.execute(
                    """SELECT EXISTS (
                           SELECT 1 FROM artifact_store_org_blobs WHERE organization_id = %s
                       ) OR EXISTS (
                           SELECT 1 FROM artifact_store_manifests WHERE organization_id = %s
                       ) AS occupied""",
                    (organization_id.value, organization_id.value),
                )
                occupied = cursor.fetchone()
                if occupied is None or occupied["occupied"]:
                    raise ArtifactStoreError(code="artifact.restore_target_not_empty")
                cursor.execute(
                    """INSERT INTO artifact_store_quotas
                           (organization_id, max_bytes, updated_at)
                       VALUES (%s,%s,%s)
                       ON CONFLICT (organization_id) DO UPDATE SET
                         max_bytes = EXCLUDED.max_bytes, updated_at = EXCLUDED.updated_at""",
                    (organization_id.value, backup.quota_bytes, restored_at),
                )
                for blob in backup.blobs:
                    cursor.execute(
                        """SELECT object.size_bytes
                           FROM artifact_store_objects AS object
                           JOIN artifact_store_object_locations AS location
                             ON location.digest = object.digest
                           WHERE object.digest = %s AND location.backend = %s
                           FOR SHARE OF object, location""",
                        (blob.digest, backend),
                    )
                    existing = cursor.fetchone()
                    if existing is None or existing["size_bytes"] != blob.size_bytes:
                        raise ArtifactStoreError(code="artifact.backup_blob_corrupted")
                    cursor.execute(
                        """INSERT INTO artifact_store_org_blobs
                               (organization_id, digest, backend, acquired_at)
                           VALUES (%s,%s,%s,%s)""",
                        (organization_id.value, blob.digest, backend, restored_at),
                    )
                if self._usage_bytes(cursor, organization_id) > backup.quota_bytes:
                    raise ArtifactStoreError(code="artifact.quota_exceeded")
                for manifest in backup.manifests:
                    payload = manifest.model_dump(mode="json", by_alias=True)
                    cursor.execute(
                        """INSERT INTO artifact_store_manifests
                               (organization_id, manifest_digest, bundle_id, name, version,
                                manifest, published_at)
                           VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                        (
                            organization_id.value,
                            manifest.manifest_digest,
                            manifest.bundle_id,
                            manifest.name,
                            manifest.version,
                            Jsonb(payload),
                            restored_at,
                        ),
                    )
                    for entry in manifest.entries:
                        cursor.execute(
                            """INSERT INTO artifact_store_manifest_entries
                                   (organization_id, manifest_digest, path, digest,
                                    size_bytes, media_type)
                               VALUES (%s,%s,%s,%s,%s,%s)""",
                            (
                                organization_id.value,
                                manifest.manifest_digest,
                                entry.path,
                                entry.blob.digest,
                                entry.blob.size_bytes,
                                entry.blob.media_type,
                            ),
                        )
                for digest in backup.pinned_digests:
                    cursor.execute(
                        """INSERT INTO artifact_store_pins
                               (organization_id, digest, pinned_at)
                           VALUES (%s,%s,%s)""",
                        (organization_id.value, digest, restored_at),
                    )
        except ArtifactStoreError:
            raise
        except psycopg.Error as error:
            raise ArtifactStoreError(code="artifact.restore_failed") from error

    def collect_garbage(
        self, *, now: datetime, backend: str
    ) -> tuple[ArtifactGarbageCandidate, ...]:
        if backend not in {"local_cas", "s3_compatible"}:
            raise ArtifactStoreError(code="artifact.backend_invalid")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM artifact_store_leases WHERE expires_at <= %s", (now,))
            cursor.execute(
                """DELETE FROM artifact_store_org_blobs AS owned
                   WHERE NOT EXISTS (
                       SELECT 1 FROM artifact_store_manifest_entries AS entry
                       WHERE entry.organization_id = owned.organization_id
                         AND entry.digest = owned.digest
                   ) AND NOT EXISTS (
                       SELECT 1 FROM artifact_store_pins AS pin
                       WHERE pin.organization_id = owned.organization_id
                         AND pin.digest = owned.digest
                   ) AND NOT EXISTS (
                       SELECT 1 FROM artifact_store_leases AS lease
                       WHERE lease.organization_id = owned.organization_id
                         AND lease.digest = owned.digest AND lease.expires_at > %s
                   )""",
                (now,),
            )
            cursor.execute(
                """INSERT INTO artifact_store_gc_candidates (digest, backend, scheduled_at)
                   SELECT location.digest, location.backend, %s
                   FROM artifact_store_object_locations AS location
                   WHERE location.backend = %s AND NOT EXISTS (
                       SELECT 1 FROM artifact_store_org_blobs AS owned
                       WHERE owned.digest = location.digest
                         AND owned.backend = location.backend
                   ) ON CONFLICT (digest, backend) DO NOTHING""",
                (now, backend),
            )
            cursor.execute(
                """SELECT candidate.digest, candidate.backend
                   FROM artifact_store_gc_candidates AS candidate
                   WHERE candidate.backend = %s ORDER BY candidate.digest""",
                (backend,),
            )
            return tuple(
                ArtifactGarbageCandidate(digest=row["digest"], backend=row["backend"])
                for row in cursor.fetchall()
            )

    def acknowledge_garbage(self, *, digest: str, backend: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT digest FROM artifact_store_object_locations
                   WHERE digest = %s AND backend = %s FOR UPDATE""",
                (digest, backend),
            )
            if cursor.fetchone() is None:
                cursor.execute(
                    """DELETE FROM artifact_store_gc_candidates
                       WHERE digest = %s AND backend = %s""",
                    (digest, backend),
                )
                return
            cursor.execute(
                """SELECT 1 FROM artifact_store_org_blobs
                   WHERE digest = %s AND backend = %s LIMIT 1""",
                (digest, backend),
            )
            if cursor.fetchone() is not None:
                cursor.execute(
                    """DELETE FROM artifact_store_gc_candidates
                       WHERE digest = %s AND backend = %s""",
                    (digest, backend),
                )
                return
            cursor.execute(
                """DELETE FROM artifact_store_object_locations
                   WHERE digest = %s AND backend = %s""",
                (digest, backend),
            )
            cursor.execute(
                """DELETE FROM artifact_store_objects AS object
                   WHERE object.digest = %s AND NOT EXISTS (
                       SELECT 1 FROM artifact_store_object_locations AS location
                       WHERE location.digest = object.digest
                   )""",
                (digest,),
            )

    def export_state(self, *, organization_id: OrganizationId) -> ArtifactCatalogState:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT max_bytes FROM artifact_store_quotas WHERE organization_id = %s",
                (organization_id.value,),
            )
            quota = cursor.fetchone()
            cursor.execute(
                """SELECT manifest FROM artifact_store_manifests
                   WHERE organization_id = %s ORDER BY manifest_digest""",
                (organization_id.value,),
            )
            manifests = tuple(
                ArtifactManifest.model_validate(row["manifest"]) for row in cursor.fetchall()
            )
            cursor.execute(
                """SELECT digest FROM artifact_store_pins
                   WHERE organization_id = %s ORDER BY digest""",
                (organization_id.value,),
            )
            pins = tuple(row["digest"] for row in cursor.fetchall())
            cursor.execute(
                """SELECT object.digest, object.size_bytes
                   FROM artifact_store_org_blobs AS owned
                   JOIN artifact_store_objects AS object ON object.digest = owned.digest
                   WHERE owned.organization_id = %s ORDER BY object.digest""",
                (organization_id.value,),
            )
            blobs = tuple(
                ArtifactBackupBlob(
                    digest=row["digest"],
                    size_bytes=row["size_bytes"],
                )
                for row in cursor.fetchall()
            )
        return ArtifactCatalogState(
            organization_id=organization_id,
            quota_bytes=_DEFAULT_QUOTA_BYTES if quota is None else quota["max_bytes"],
            manifests=manifests,
            pinned_digests=pins,
            blobs=blobs,
        )

    @staticmethod
    def _usage_bytes(cursor: Any, organization_id: OrganizationId) -> int:
        cursor.execute(
            """SELECT COALESCE(sum(object.size_bytes), 0) AS usage_bytes
               FROM artifact_store_org_blobs AS owned
               JOIN artifact_store_objects AS object ON object.digest = owned.digest
               WHERE owned.organization_id = %s""",
            (organization_id.value,),
        )
        row = cast(Mapping[str, Any], cursor.fetchone())
        return int(row["usage_bytes"])


__all__ = ["PostgresArtifactCatalogRepository"]
