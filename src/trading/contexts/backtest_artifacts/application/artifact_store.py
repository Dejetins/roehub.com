from __future__ import annotations

import base64
import json
import os
import stat
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from trading.contexts.backtest_artifacts.application.ports import (
    ArtifactBlobStore,
    ArtifactCatalogRepository,
)
from trading.contexts.backtest_artifacts.domain import ArtifactStoreError
from trading.integration import (
    MAX_ARTIFACT_BUNDLE_BYTES,
    ArtifactBackupCatalog,
    ArtifactBlobDescriptor,
    ArtifactManifest,
    sha256_digest,
)
from trading.shared_kernel.primitives import OrganizationId

_MANIFEST_NAME = "artifact.bundle.json"
_MAX_MANIFEST_BYTES = 256 * 1024


class ArtifactStoreService:
    def __init__(
        self,
        *,
        blobs: ArtifactBlobStore,
        catalog: ArtifactCatalogRepository,
        trusted_public_keys: Mapping[str, str],
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._blobs = blobs
        self._catalog = catalog
        self._trusted_public_keys = dict(trusted_public_keys)
        if not self._trusted_public_keys:
            raise ValueError("ArtifactStoreService requires at least one trusted publisher key")
        self._now = now or (lambda: datetime.now(UTC))

    def install_bundle(
        self,
        *,
        organization_id: OrganizationId,
        bundle_root: Path,
    ) -> ArtifactManifest:
        manifest, payloads = load_signed_bundle(
            bundle_root=bundle_root,
            trusted_public_keys=self._trusted_public_keys,
        )
        for entry in manifest.entries:
            self._catalog.register_blob(
                descriptor=entry.blob,
                backend=self._blobs.descriptor.backend,
                registered_at=self._now(),
            )
            stored = self._blobs.put_bytes(
                payloads[entry.path],
                media_type=entry.blob.media_type,
            )
            if stored != entry.blob:
                raise ArtifactStoreError(code="artifact.bundle_descriptor_mismatch")
        try:
            self.publish_manifest(organization_id=organization_id, manifest=manifest)
        except ArtifactStoreError:
            try:
                self.garbage_collect()
            except ArtifactStoreError as cleanup_error:
                raise ArtifactStoreError(code="artifact.install_cleanup_failed") from cleanup_error
            raise
        return manifest

    def publish_manifest(
        self, *, organization_id: OrganizationId, manifest: ArtifactManifest
    ) -> None:
        verify_manifest_signature(
            manifest=manifest,
            trusted_public_keys=self._trusted_public_keys,
        )
        for entry in manifest.entries:
            payload = self._blobs.read_bytes(digest=entry.blob.digest)
            if len(payload) != entry.blob.size_bytes:
                raise ArtifactStoreError(code="artifact.size_mismatch")
            self._catalog.register_blob(
                descriptor=entry.blob,
                backend=self._blobs.descriptor.backend,
                registered_at=self._now(),
            )
        self._catalog.publish_manifest(
            organization_id=organization_id,
            manifest=manifest,
            backend=self._blobs.descriptor.backend,
            published_at=self._now(),
        )

    def read_entry(
        self,
        *,
        organization_id: OrganizationId,
        manifest_digest: str,
        path: str,
    ) -> bytes:
        manifest = self._require_manifest(
            organization_id=organization_id,
            manifest_digest=manifest_digest,
        )
        entry = next((item for item in manifest.entries if item.path == path), None)
        if entry is None:
            raise ArtifactStoreError(code="artifact.entry_not_found")
        payload = self._blobs.read_bytes(digest=entry.blob.digest)
        if len(payload) != entry.blob.size_bytes:
            raise ArtifactStoreError(code="artifact.size_mismatch")
        return payload

    def get_manifest(
        self, *, organization_id: OrganizationId, manifest_digest: str
    ) -> ArtifactManifest:
        """Return one organization-scoped immutable manifest to trusted host code."""

        return self._require_manifest(
            organization_id=organization_id,
            manifest_digest=manifest_digest,
        )

    def materialize_entry(
        self,
        *,
        organization_id: OrganizationId,
        manifest_digest: str,
        path: str,
        cache_key: str,
    ) -> Path:
        manifest = self._require_manifest(
            organization_id=organization_id,
            manifest_digest=manifest_digest,
        )
        entry = next((item for item in manifest.entries if item.path == path), None)
        if entry is None:
            raise ArtifactStoreError(code="artifact.entry_not_found")
        return self._blobs.materialize(digest=entry.blob.digest, cache_key=cache_key)

    def set_quota(self, *, organization_id: OrganizationId, max_bytes: int) -> None:
        self._catalog.set_quota(organization_id=organization_id, max_bytes=max_bytes)

    def pin(self, *, organization_id: OrganizationId, digest: str) -> None:
        self._catalog.pin(organization_id=organization_id, digest=digest, pinned_at=self._now())

    def unpin(self, *, organization_id: OrganizationId, digest: str) -> None:
        self._catalog.unpin(organization_id=organization_id, digest=digest)

    def acquire_lease(
        self,
        *,
        organization_id: OrganizationId,
        lease_id: str,
        digest: str,
        expires_at: datetime,
    ) -> None:
        self._catalog.acquire_lease(
            organization_id=organization_id,
            lease_id=lease_id,
            digest=digest,
            expires_at=expires_at,
            created_at=self._now(),
        )

    def release_lease(self, *, organization_id: OrganizationId, lease_id: str) -> None:
        self._catalog.release_lease(organization_id=organization_id, lease_id=lease_id)

    def retire_manifest(self, *, organization_id: OrganizationId, manifest_digest: str) -> None:
        self._catalog.retire_manifest(
            organization_id=organization_id,
            manifest_digest=manifest_digest,
        )

    def garbage_collect(self) -> tuple[str, ...]:
        candidates = self._catalog.collect_garbage(
            now=self._now(),
            backend=self._blobs.descriptor.backend,
        )
        deleted: list[str] = []
        for candidate in candidates:
            self._blobs.delete(digest=candidate.digest)
            self._catalog.acknowledge_garbage(
                digest=candidate.digest,
                backend=candidate.backend,
            )
            deleted.append(candidate.digest)
        return tuple(deleted)

    def backup(self, *, organization_id: OrganizationId, destination: Path) -> str:
        root = destination.expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise ArtifactStoreError(code="artifact.backup_destination_not_empty")
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        blob_root = root / "blobs" / "sha256"
        blob_root.mkdir(parents=True, mode=0o750)
        state = self._catalog.export_state(organization_id=organization_id)
        for blob in state.blobs:
            payload = self._blobs.read_bytes(digest=blob.digest)
            if len(payload) != blob.size_bytes:
                raise ArtifactStoreError(code="artifact.size_mismatch")
            _atomic_write(blob_root / blob.digest.removeprefix("sha256:"), payload)
        catalog = ArtifactBackupCatalog(
            schema="ArtifactBackup/v1",
            source_organization_id=organization_id.value,
            quota_bytes=state.quota_bytes,
            manifests=state.manifests,
            pinned_digests=state.pinned_digests,
            blobs=state.blobs,
        )
        catalog_bytes = _json_bytes(catalog.model_dump(mode="json", by_alias=True))
        backup_digest = sha256_digest(catalog_bytes)
        _atomic_write(root / "catalog.json", catalog_bytes)
        _atomic_write(root / "catalog.sha256", (backup_digest + "\n").encode())
        return backup_digest

    def restore(
        self,
        *,
        organization_id: OrganizationId,
        source: Path,
        expected_backup_digest: str,
    ) -> tuple[str, ...]:
        root = source.expanduser().resolve()
        catalog_path = root / "catalog.json"
        digest_path = root / "catalog.sha256"
        catalog_bytes = _read_regular_file(catalog_path, max_bytes=4 * 1024 * 1024)
        actual_digest = sha256_digest(catalog_bytes)
        recorded_digest = _read_regular_file(digest_path, max_bytes=128).decode().strip()
        if actual_digest != expected_backup_digest or recorded_digest != expected_backup_digest:
            raise ArtifactStoreError(code="artifact.backup_digest_mismatch")
        try:
            backup = ArtifactBackupCatalog.model_validate_json(catalog_bytes)
        except ValueError as error:
            raise ArtifactStoreError(code="artifact.backup_invalid") from error
        for manifest in backup.manifests:
            verify_manifest_signature(
                manifest=manifest,
                trusted_public_keys=self._trusted_public_keys,
            )
        for blob in backup.blobs:
            blob_path = root / "blobs" / "sha256" / blob.digest.removeprefix("sha256:")
            payload = _read_regular_file(blob_path, max_bytes=blob.size_bytes)
            if sha256_digest(payload) != blob.digest or len(payload) != blob.size_bytes:
                raise ArtifactStoreError(code="artifact.backup_blob_corrupted")
            descriptor = next(
                (
                    entry.blob
                    for manifest in backup.manifests
                    for entry in manifest.entries
                    if entry.blob.digest == blob.digest
                ),
                None,
            )
            media_type = (
                descriptor.media_type if descriptor is not None else "application/octet-stream"
            )
            self._catalog.register_blob(
                descriptor=ArtifactBlobDescriptor(
                    digest=blob.digest,
                    size_bytes=blob.size_bytes,
                    media_type=media_type,
                ),
                backend=self._blobs.descriptor.backend,
                registered_at=self._now(),
            )
            stored = self._blobs.put_bytes(
                payload,
                media_type=media_type,
            )
            if stored.digest != blob.digest or stored.size_bytes != blob.size_bytes:
                raise ArtifactStoreError(code="artifact.backup_blob_corrupted")
        self._catalog.restore_state(
            organization_id=organization_id,
            backup=backup,
            backend=self._blobs.descriptor.backend,
            restored_at=self._now(),
        )
        return tuple(manifest.manifest_digest for manifest in backup.manifests)

    def _require_manifest(
        self, *, organization_id: OrganizationId, manifest_digest: str
    ) -> ArtifactManifest:
        manifest = self._catalog.get_manifest(
            organization_id=organization_id,
            manifest_digest=manifest_digest,
        )
        if manifest is None:
            raise ArtifactStoreError(code="artifact.manifest_not_found")
        return manifest


def load_signed_bundle(
    *, bundle_root: Path, trusted_public_keys: Mapping[str, str]
) -> tuple[ArtifactManifest, dict[str, bytes]]:
    root = bundle_root.expanduser().resolve()
    manifest_payload = _read_regular_file(root / _MANIFEST_NAME, max_bytes=_MAX_MANIFEST_BYTES)
    try:
        manifest = ArtifactManifest.model_validate_json(manifest_payload)
    except ValueError as error:
        raise ArtifactStoreError(code="artifact.bundle_manifest_invalid") from error
    verify_manifest_signature(
        manifest=manifest,
        trusted_public_keys=trusted_public_keys,
    )
    payloads: dict[str, bytes] = {}
    total_size = sum(entry.blob.size_bytes for entry in manifest.entries)
    if total_size > MAX_ARTIFACT_BUNDLE_BYTES:
        raise ArtifactStoreError(code="artifact.bundle_too_large")
    for entry in manifest.entries:
        path = root / "payload" / entry.path
        payload = _read_regular_file(path, max_bytes=entry.blob.size_bytes)
        if len(payload) != entry.blob.size_bytes or sha256_digest(payload) != entry.blob.digest:
            raise ArtifactStoreError(code="artifact.bundle_blob_corrupted")
        payloads[entry.path] = payload
    return manifest, payloads


def verify_manifest_signature(
    *, manifest: ArtifactManifest, trusted_public_keys: Mapping[str, str]
) -> None:
    encoded_key = trusted_public_keys.get(manifest.signature.key_id)
    if encoded_key is None:
        raise ArtifactStoreError(code="artifact.publisher_untrusted")
    try:
        public_key_bytes = base64.b64decode(encoded_key, validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        signature = base64.b64decode(manifest.signature.value_b64, validate=True)
        public_key.verify(signature, manifest.signed_bytes())
    except (ValueError, InvalidSignature) as error:
        raise ArtifactStoreError(code="artifact.signature_invalid") from error


def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ArtifactStoreError(code="artifact.file_unavailable") from error
    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_bytes:
            raise ArtifactStoreError(code="artifact.file_invalid")
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ArtifactStoreError(code="artifact.file_invalid")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.parent / f".{uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


__all__ = ["ArtifactStoreService", "load_signed_bundle", "verify_manifest_signature"]
