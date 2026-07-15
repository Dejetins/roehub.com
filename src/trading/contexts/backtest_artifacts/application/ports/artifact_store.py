from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from trading.contexts.backtest_artifacts.domain import (
    ArtifactCatalogState,
    ArtifactGarbageCandidate,
)
from trading.integration import (
    ArtifactBackupCatalog,
    ArtifactBlobDescriptor,
    ArtifactManifest,
    ArtifactStoreDescriptor,
)
from trading.shared_kernel.primitives import OrganizationId


class ArtifactBlobStore(Protocol):
    @property
    def descriptor(self) -> ArtifactStoreDescriptor: ...

    def put_bytes(self, payload: bytes, *, media_type: str) -> ArtifactBlobDescriptor: ...

    def read_bytes(self, *, digest: str) -> bytes: ...

    def exists(self, *, digest: str) -> bool: ...

    def materialize(self, *, digest: str, cache_key: str) -> Path: ...

    def delete(self, *, digest: str) -> None: ...


class ArtifactCatalogRepository(Protocol):
    def register_blob(
        self,
        *,
        descriptor: ArtifactBlobDescriptor,
        backend: str,
        registered_at: datetime,
    ) -> None: ...

    def set_quota(self, *, organization_id: OrganizationId, max_bytes: int) -> None: ...

    def usage_bytes(self, *, organization_id: OrganizationId) -> int: ...

    def publish_manifest(
        self,
        *,
        organization_id: OrganizationId,
        manifest: ArtifactManifest,
        backend: str,
        published_at: datetime,
    ) -> None: ...

    def get_manifest(
        self, *, organization_id: OrganizationId, manifest_digest: str
    ) -> ArtifactManifest | None: ...

    def retire_manifest(self, *, organization_id: OrganizationId, manifest_digest: str) -> None: ...

    def pin(self, *, organization_id: OrganizationId, digest: str, pinned_at: datetime) -> None: ...

    def unpin(self, *, organization_id: OrganizationId, digest: str) -> None: ...

    def acquire_lease(
        self,
        *,
        organization_id: OrganizationId,
        lease_id: str,
        digest: str,
        expires_at: datetime,
        created_at: datetime,
    ) -> None: ...

    def release_lease(self, *, organization_id: OrganizationId, lease_id: str) -> None: ...

    def collect_garbage(
        self, *, now: datetime, backend: str
    ) -> tuple[ArtifactGarbageCandidate, ...]: ...

    def acknowledge_garbage(self, *, digest: str, backend: str) -> None: ...

    def restore_state(
        self,
        *,
        organization_id: OrganizationId,
        backup: ArtifactBackupCatalog,
        backend: str,
        restored_at: datetime,
    ) -> None: ...

    def export_state(self, *, organization_id: OrganizationId) -> ArtifactCatalogState: ...


__all__ = ["ArtifactBlobStore", "ArtifactCatalogRepository"]
