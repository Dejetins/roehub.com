from __future__ import annotations

import json
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_cas import LocalCasBlobStore
from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_s3 import (
    S3CompatibleBlobStore,
    S3ConnectionConfig,
    S3ResolvedCredentials,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence import (
    InMemoryArtifactCatalogRepository,
)
from trading.contexts.backtest_artifacts.application import ArtifactStoreService
from trading.contexts.backtest_artifacts.domain import ArtifactStoreError
from trading.integration import ArtifactBlobDescriptor, ArtifactBundleSignature
from trading.shared_kernel.primitives import OrganizationId

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _organization() -> OrganizationId:
    return OrganizationId(uuid4())


def _keys(fixture: Path) -> dict[str, str]:
    return json.loads((fixture / "publisher-keys.json").read_text())


def test_local_cas_concurrent_publish_corruption_and_materialization(tmp_path: Path) -> None:
    store = LocalCasBlobStore(root=tmp_path / "cas")
    payload = b"immutable-payload" * 1024

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = tuple(
            executor.map(
                lambda _: store.put_bytes(payload, media_type="application/octet-stream"),
                range(24),
            )
        )
    assert len({result.digest for result in results}) == 1
    descriptor = results[0]
    assert len(list((tmp_path / "cas/blobs/sha256").rglob(descriptor.digest[7:]))) == 1

    materialized = store.materialize(digest=descriptor.digest, cache_key="numpy:mmap:test")
    assert materialized.read_bytes() == payload
    assert stat.S_IMODE(materialized.stat().st_mode) == 0o440
    restarted = LocalCasBlobStore(root=tmp_path / "cas")
    assert restarted.read_bytes(digest=descriptor.digest) == payload

    blob_path = next((tmp_path / "cas/blobs/sha256").rglob(descriptor.digest[7:]))
    blob_path.chmod(0o640)
    blob_path.write_bytes(b"x" * len(payload))
    with pytest.raises(ArtifactStoreError, match="artifact.digest_mismatch"):
        restarted.read_bytes(digest=descriptor.digest)


def test_bundle_install_scope_quota_pin_lease_gc_and_backup_restore(
    tmp_path: Path,
) -> None:
    fixture = _REPO_ROOT / "tests/fixtures/artifacts/demo_bundle"
    now_value = [datetime(2026, 7, 13, tzinfo=UTC)]
    catalog = InMemoryArtifactCatalogRepository()
    store = LocalCasBlobStore(root=tmp_path / "source-cas")
    public_keys = _keys(fixture)
    service = ArtifactStoreService(
        blobs=store,
        catalog=catalog,
        trusted_public_keys=public_keys,
        now=lambda: now_value[0],
    )
    owner = _organization()
    foreign = _organization()
    service.set_quota(organization_id=owner, max_bytes=4096)
    manifest = service.install_bundle(
        organization_id=owner,
        bundle_root=fixture,
    )
    assert catalog.usage_bytes(organization_id=owner) == 202
    assert service.read_entry(
        organization_id=owner,
        manifest_digest=manifest.manifest_digest,
        path="demo/hello.json",
    ).startswith(b"{")
    with pytest.raises(ArtifactStoreError, match="artifact.manifest_not_found"):
        service.read_entry(
            organization_id=foreign,
            manifest_digest=manifest.manifest_digest,
            path="demo/hello.json",
        )

    tampered_manifest = manifest.model_copy(
        update={
            "signature": ArtifactBundleSignature(
                key_id=manifest.signature.key_id,
                value_b64="A" * 86 + "==",
            )
        }
    )
    with pytest.raises(ArtifactStoreError, match="artifact.signature_invalid"):
        service.publish_manifest(
            organization_id=owner,
            manifest=tampered_manifest,
        )

    first, second = (entry.blob.digest for entry in manifest.entries)
    service.pin(organization_id=owner, digest=first)
    service.acquire_lease(
        organization_id=owner,
        lease_id="lease:test:0001",
        digest=second,
        expires_at=now_value[0] + timedelta(minutes=5),
    )
    service.retire_manifest(
        organization_id=owner,
        manifest_digest=manifest.manifest_digest,
    )
    assert service.garbage_collect() == ()
    service.unpin(organization_id=owner, digest=first)
    assert service.garbage_collect() == (first,)
    now_value[0] += timedelta(minutes=6)
    assert service.garbage_collect() == (second,)
    assert not store.exists(digest=first)
    assert not store.exists(digest=second)

    manifest = service.install_bundle(
        organization_id=owner,
        bundle_root=fixture,
    )
    service.pin(organization_id=owner, digest=manifest.entries[0].blob.digest)
    backup_digest = service.backup(
        organization_id=owner,
        destination=tmp_path / "backup",
    )
    restored_catalog = InMemoryArtifactCatalogRepository()
    restored_store = LocalCasBlobStore(root=tmp_path / "restored-cas")
    restored = ArtifactStoreService(
        blobs=restored_store,
        catalog=restored_catalog,
        trusted_public_keys=public_keys,
    )
    target = _organization()
    restored_manifests = restored.restore(
        organization_id=target,
        source=tmp_path / "backup",
        expected_backup_digest=backup_digest,
    )
    assert restored_manifests == (manifest.manifest_digest,)
    assert restored.read_entry(
        organization_id=target,
        manifest_digest=manifest.manifest_digest,
        path="demo/model-card.txt",
    ).startswith(b"Roehub")
    with pytest.raises(ArtifactStoreError, match="artifact.restore_target_not_empty"):
        restored.restore(
            organization_id=target,
            source=tmp_path / "backup",
            expected_backup_digest=backup_digest,
        )


def test_quota_and_signature_fail_closed(tmp_path: Path) -> None:
    fixture = _REPO_ROOT / "tests/fixtures/artifacts/demo_bundle"
    organization = _organization()
    service = ArtifactStoreService(
        blobs=LocalCasBlobStore(root=tmp_path / "cas"),
        catalog=InMemoryArtifactCatalogRepository(),
        trusted_public_keys=_keys(fixture),
    )
    service.set_quota(organization_id=organization, max_bytes=100)
    with pytest.raises(ArtifactStoreError, match="artifact.quota_exceeded"):
        service.install_bundle(
            organization_id=organization,
            bundle_root=fixture,
        )
    untrusted = ArtifactStoreService(
        blobs=LocalCasBlobStore(root=tmp_path / "untrusted-cas"),
        catalog=InMemoryArtifactCatalogRepository(),
        trusted_public_keys={"different.publisher": "A" * 43 + "="},
    )
    with pytest.raises(ArtifactStoreError, match="artifact.publisher_untrusted"):
        untrusted.install_bundle(
            organization_id=_organization(),
            bundle_root=fixture,
        )


def test_backup_restores_blob_retained_only_by_pin(tmp_path: Path) -> None:
    fixture = _REPO_ROOT / "tests/fixtures/artifacts/demo_bundle"
    public_keys = _keys(fixture)
    owner = _organization()
    source_catalog = InMemoryArtifactCatalogRepository()
    source = ArtifactStoreService(
        blobs=LocalCasBlobStore(root=tmp_path / "source"),
        catalog=source_catalog,
        trusted_public_keys=public_keys,
    )
    manifest = source.install_bundle(organization_id=owner, bundle_root=fixture)
    pinned = manifest.entries[0].blob
    source.pin(organization_id=owner, digest=pinned.digest)
    source.retire_manifest(
        organization_id=owner,
        manifest_digest=manifest.manifest_digest,
    )
    assert source.garbage_collect() == (manifest.entries[1].blob.digest,)
    backup_digest = source.backup(
        organization_id=owner,
        destination=tmp_path / "pinned-backup",
    )

    target = _organization()
    target_catalog = InMemoryArtifactCatalogRepository()
    restored = ArtifactStoreService(
        blobs=LocalCasBlobStore(root=tmp_path / "target"),
        catalog=target_catalog,
        trusted_public_keys=public_keys,
    )
    assert (
        restored.restore(
            organization_id=target,
            source=tmp_path / "pinned-backup",
            expected_backup_digest=backup_digest,
        )
        == ()
    )
    assert target_catalog.usage_bytes(organization_id=target) == pinned.size_bytes
    assert restored.garbage_collect() == ()
    restored.unpin(organization_id=target, digest=pinned.digest)
    assert restored.garbage_collect() == (pinned.digest,)


def test_s3_compatible_adapter_signs_verifies_and_materializes(tmp_path: Path) -> None:
    objects: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        path = request.url.path
        if path == "/roehub-fixture" and request.method == "HEAD":
            return httpx.Response(200)
        if request.method == "PUT":
            objects[path] = request.content
            return httpx.Response(200)
        if request.method == "HEAD":
            return httpx.Response(200 if path in objects else 404)
        if request.method == "GET":
            return (
                httpx.Response(200, content=objects[path])
                if path in objects
                else httpx.Response(404)
            )
        if request.method == "DELETE":
            objects.pop(path, None)
            return httpx.Response(204)
        return httpx.Response(405)

    store = S3CompatibleBlobStore(
        config=S3ConnectionConfig(
            endpoint="https://s3.example.test",
            bucket="roehub-fixture",
            region="us-east-1",
            credentials_ref="openbao://kv/roehub/storage/s3-fixture#credentials",
        ),
        credentials=S3ResolvedCredentials(
            access_key_id="fixture-access",
            secret_access_key="fixture-secret-key",
        ),
        materialization_root=tmp_path / "materialized",
        transport=httpx.MockTransport(handler),
    )
    store.ensure_bucket()
    descriptor = store.put_bytes(b"s3-compatible", media_type="application/octet-stream")
    assert store.exists(digest=descriptor.digest)
    assert store.read_bytes(digest=descriptor.digest) == b"s3-compatible"
    assert (
        store.materialize(digest=descriptor.digest, cache_key="test:s3").read_bytes()
        == b"s3-compatible"
    )
    store.delete(digest=descriptor.digest)
    assert not store.exists(digest=descriptor.digest)
    store.close()


def test_global_content_identity_ignores_media_type_and_backend() -> None:
    catalog = InMemoryArtifactCatalogRepository()
    digest = "sha256:" + "a" * 64
    registered_at = datetime(2026, 7, 13, tzinfo=UTC)
    catalog.register_blob(
        descriptor=ArtifactBlobDescriptor(
            digest=digest,
            size_bytes=42,
            media_type="application/json",
        ),
        backend="local_cas",
        registered_at=registered_at,
    )
    catalog.register_blob(
        descriptor=ArtifactBlobDescriptor(
            digest=digest,
            size_bytes=42,
            media_type="text/plain",
        ),
        backend="s3_compatible",
        registered_at=registered_at,
    )
