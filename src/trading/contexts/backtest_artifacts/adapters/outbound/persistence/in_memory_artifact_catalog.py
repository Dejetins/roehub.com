from __future__ import annotations

from datetime import datetime
from threading import RLock

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


class InMemoryArtifactCatalogRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self._objects: dict[str, int] = {}
        self._locations: set[tuple[str, str]] = set()
        self._owned: dict[OrganizationId, set[str]] = {}
        self._owned_backend: dict[tuple[OrganizationId, str], str] = {}
        self._quotas: dict[OrganizationId, int] = {}
        self._manifests: dict[tuple[OrganizationId, str], ArtifactManifest] = {}
        self._versions: dict[tuple[OrganizationId, str, str], str] = {}
        self._pins: set[tuple[OrganizationId, str]] = set()
        self._leases: dict[tuple[OrganizationId, str], tuple[str, datetime]] = {}
        self._garbage: set[tuple[str, str]] = set()

    def register_blob(
        self,
        *,
        descriptor: ArtifactBlobDescriptor,
        backend: str,
        registered_at: datetime,
    ) -> None:
        del registered_at
        if backend not in {"local_cas", "s3_compatible"}:
            raise ArtifactStoreError(code="artifact.backend_invalid")
        with self._lock:
            existing = self._objects.setdefault(descriptor.digest, descriptor.size_bytes)
            if existing != descriptor.size_bytes:
                raise ArtifactStoreError(code="artifact.digest_metadata_conflict")
            self._locations.add((descriptor.digest, backend))

    def set_quota(self, *, organization_id: OrganizationId, max_bytes: int) -> None:
        if not 1 <= max_bytes <= 1_099_511_627_776:
            raise ArtifactStoreError(code="artifact.quota_invalid")
        with self._lock:
            if self.usage_bytes(organization_id=organization_id) > max_bytes:
                raise ArtifactStoreError(code="artifact.quota_below_usage")
            self._quotas[organization_id] = max_bytes

    def usage_bytes(self, *, organization_id: OrganizationId) -> int:
        with self._lock:
            return sum(self._objects[digest] for digest in self._owned.get(organization_id, set()))

    def publish_manifest(
        self,
        *,
        organization_id: OrganizationId,
        manifest: ArtifactManifest,
        backend: str,
        published_at: datetime,
    ) -> None:
        del published_at
        with self._lock:
            owned = set(self._owned.get(organization_id, set()))
            for entry in manifest.entries:
                if (
                    self._objects.get(entry.blob.digest) != entry.blob.size_bytes
                    or (entry.blob.digest, backend) not in self._locations
                ):
                    raise ArtifactStoreError(code="artifact.digest_metadata_conflict")
                owned.add(entry.blob.digest)
            usage = sum(self._objects[digest] for digest in owned)
            if usage > self._quotas.get(organization_id, _DEFAULT_QUOTA_BYTES):
                raise ArtifactStoreError(code="artifact.quota_exceeded")
            version_key = (organization_id, manifest.bundle_id, manifest.version)
            existing_version = self._versions.get(version_key)
            if existing_version not in {None, manifest.manifest_digest}:
                raise ArtifactStoreError(code="artifact.manifest_version_conflict")
            self._owned[organization_id] = owned
            for entry in manifest.entries:
                self._owned_backend[(organization_id, entry.blob.digest)] = backend
                self._garbage.discard((entry.blob.digest, backend))
            self._manifests[(organization_id, manifest.manifest_digest)] = manifest
            self._versions[version_key] = manifest.manifest_digest

    def get_manifest(
        self, *, organization_id: OrganizationId, manifest_digest: str
    ) -> ArtifactManifest | None:
        with self._lock:
            return self._manifests.get((organization_id, manifest_digest))

    def retire_manifest(self, *, organization_id: OrganizationId, manifest_digest: str) -> None:
        with self._lock:
            manifest = self._manifests.pop((organization_id, manifest_digest), None)
            if manifest is None:
                raise ArtifactStoreError(code="artifact.manifest_not_found")
            self._versions.pop((organization_id, manifest.bundle_id, manifest.version), None)

    def pin(self, *, organization_id: OrganizationId, digest: str, pinned_at: datetime) -> None:
        del pinned_at
        with self._lock:
            if digest not in self._owned.get(organization_id, set()):
                raise ArtifactStoreError(code="artifact.not_found")
            self._pins.add((organization_id, digest))

    def unpin(self, *, organization_id: OrganizationId, digest: str) -> None:
        with self._lock:
            self._pins.discard((organization_id, digest))

    def acquire_lease(
        self,
        *,
        organization_id: OrganizationId,
        lease_id: str,
        digest: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> None:
        if expires_at <= created_at or len(lease_id) < 8:
            raise ArtifactStoreError(code="artifact.lease_invalid")
        with self._lock:
            if digest not in self._owned.get(organization_id, set()):
                raise ArtifactStoreError(code="artifact.not_found")
            self._leases[(organization_id, lease_id)] = (digest, expires_at)

    def release_lease(self, *, organization_id: OrganizationId, lease_id: str) -> None:
        with self._lock:
            self._leases.pop((organization_id, lease_id), None)

    def collect_garbage(
        self, *, now: datetime, backend: str
    ) -> tuple[ArtifactGarbageCandidate, ...]:
        with self._lock:
            self._leases = {key: value for key, value in self._leases.items() if value[1] > now}
            referenced = {
                (organization_id, entry.blob.digest)
                for (organization_id, _), manifest in self._manifests.items()
                for entry in manifest.entries
            }
            leased = {
                (organization_id, digest)
                for (organization_id, _), (digest, _) in self._leases.items()
            }
            for organization_id, digests in self._owned.items():
                retained = {
                    digest
                    for digest in digests
                    if (organization_id, digest) in referenced
                    or (organization_id, digest) in self._pins
                    or (organization_id, digest) in leased
                }
                removed = digests - retained
                self._owned[organization_id] = retained
                for digest in removed:
                    self._owned_backend.pop((organization_id, digest), None)
            globally_owned = {
                (digest, location_backend)
                for (organization_id, digest), location_backend in self._owned_backend.items()
                if digest in self._owned.get(organization_id, set())
            }
            for location in self._locations:
                if location[1] == backend and location not in globally_owned:
                    self._garbage.add(location)
            return tuple(
                ArtifactGarbageCandidate(digest=digest, backend=value)
                for digest, value in sorted(self._garbage)
                if value == backend
            )

    def acknowledge_garbage(self, *, digest: str, backend: str) -> None:
        with self._lock:
            if any(
                digest in values and self._owned_backend.get((organization_id, digest)) == backend
                for organization_id, values in self._owned.items()
            ):
                self._garbage.discard((digest, backend))
                return
            self._locations.discard((digest, backend))
            if not any(location_digest == digest for location_digest, _ in self._locations):
                self._objects.pop(digest, None)
            self._garbage.discard((digest, backend))

    def restore_state(
        self,
        *,
        organization_id: OrganizationId,
        backup: ArtifactBackupCatalog,
        backend: str,
        restored_at: datetime,
    ) -> None:
        del restored_at
        with self._lock:
            if self._owned.get(organization_id) or any(
                owner == organization_id for owner, _ in self._manifests
            ):
                raise ArtifactStoreError(code="artifact.restore_target_not_empty")
            for blob in backup.blobs:
                if (
                    self._objects.get(blob.digest) != blob.size_bytes
                    or (blob.digest, backend) not in self._locations
                ):
                    raise ArtifactStoreError(code="artifact.backup_blob_corrupted")
            if sum(blob.size_bytes for blob in backup.blobs) > backup.quota_bytes:
                raise ArtifactStoreError(code="artifact.quota_exceeded")
            owned = {blob.digest for blob in backup.blobs}
            versions = dict(self._versions)
            for manifest in backup.manifests:
                version_key = (organization_id, manifest.bundle_id, manifest.version)
                existing = versions.get(version_key)
                if existing not in {None, manifest.manifest_digest}:
                    raise ArtifactStoreError(code="artifact.manifest_version_conflict")
                versions[version_key] = manifest.manifest_digest
            self._quotas[organization_id] = backup.quota_bytes
            self._owned[organization_id] = owned
            for digest in owned:
                self._owned_backend[(organization_id, digest)] = backend
                self._garbage.discard((digest, backend))
            self._versions = versions
            for manifest in backup.manifests:
                self._manifests[(organization_id, manifest.manifest_digest)] = manifest
            for digest in backup.pinned_digests:
                self._pins.add((organization_id, digest))

    def export_state(self, *, organization_id: OrganizationId) -> ArtifactCatalogState:
        with self._lock:
            manifests = tuple(
                manifest
                for (owner, _), manifest in sorted(
                    self._manifests.items(), key=lambda item: str(item[0][1])
                )
                if owner == organization_id
            )
            pins = tuple(sorted(digest for owner, digest in self._pins if owner == organization_id))
            blobs = tuple(
                ArtifactBackupBlob(digest=digest, size_bytes=self._objects[digest])
                for digest in sorted(self._owned.get(organization_id, set()))
            )
            return ArtifactCatalogState(
                organization_id=organization_id,
                quota_bytes=self._quotas.get(organization_id, _DEFAULT_QUOTA_BYTES),
                manifests=manifests,
                pinned_digests=pins,
                blobs=blobs,
            )


__all__ = ["InMemoryArtifactCatalogRepository"]
