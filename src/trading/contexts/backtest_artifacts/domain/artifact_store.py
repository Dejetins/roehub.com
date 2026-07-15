from __future__ import annotations

from dataclasses import dataclass

from trading.integration import ArtifactBackupBlob, ArtifactManifest
from trading.shared_kernel.primitives import OrganizationId


class ArtifactStoreError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ArtifactGarbageCandidate:
    digest: str
    backend: str


@dataclass(frozen=True, slots=True)
class ArtifactCatalogState:
    organization_id: OrganizationId
    quota_bytes: int
    manifests: tuple[ArtifactManifest, ...]
    pinned_digests: tuple[str, ...]
    blobs: tuple[ArtifactBackupBlob, ...]


__all__ = ["ArtifactCatalogState", "ArtifactGarbageCandidate", "ArtifactStoreError"]
