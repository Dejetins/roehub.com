from __future__ import annotations

import base64
import contextlib
import io
import json
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apps.cli.commands.artifacts import ArtifactsCli
from apps.migrations.storage import apply_postgres_migrations
from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_cas import LocalCasBlobStore
from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_s3 import (
    S3CompatibleBlobStore,
    S3ConnectionConfig,
    resolve_s3_credentials,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres import (
    PostgresArtifactCatalogRepository,
)
from trading.contexts.backtest_artifacts.application import ArtifactStoreService
from trading.contexts.backtest_artifacts.domain import ArtifactStoreError
from trading.integration import (
    ArtifactBlobDescriptor,
    ArtifactBundleSignature,
    ArtifactManifest,
    ArtifactManifestEntry,
    sha256_digest,
)
from trading.platform.secrets import (
    OpenBaoSecretResolver,
    SecretKind,
    SecretValue,
    SecureCredentialFile,
)
from trading.shared_kernel.primitives import OrganizationId

ROOT = Path(__file__).resolve().parents[3]
POSTGRES_IMAGE = "postgres:16"
MINIO_TAG = "minio/minio:RELEASE.2025-04-22T22-12-26Z"
MINIO_DIGEST = "sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e"
OPENBAO_IMAGE = "ghcr.io/openbao/openbao"
OPENBAO_DIGEST = "sha256:436eaf9778cad75507ff70ea26ace30dcbe15606e619ac3823495663d7f7c115"


def _signed_quota_bundle(root: Path, *, index: int) -> tuple[Path, dict[str, str]]:
    bundle_root = root / f"quota-bundle-{index}"
    payload = (f"quota-proof-{index}:".encode() + bytes([index])) * 32
    relative = f"quota/payload-{index}.bin"
    payload_path = bundle_root / "payload" / relative
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(payload)
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_id = f"stage14.quota.publisher.{index}"
    placeholder = ArtifactBundleSignature(
        key_id=key_id,
        value_b64="A" * 86 + "==",
    )
    unsigned = ArtifactManifest(
        schema="ArtifactManifest/v1",
        bundle_id=f"roehub.stage14.quota.{index}",
        name=f"Stage 14 quota proof {index}",
        version="0.1.0",
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
        entries=(
            ArtifactManifestEntry(
                path=relative,
                blob=ArtifactBlobDescriptor(
                    digest=sha256_digest(payload),
                    size_bytes=len(payload),
                    media_type="application/octet-stream",
                ),
            ),
        ),
        metadata={"purpose": "quota-proof"},
        signature=placeholder,
    )
    signature = ArtifactBundleSignature(
        key_id=key_id,
        value_b64=base64.b64encode(private_key.sign(unsigned.signed_bytes())).decode(),
    )
    manifest = unsigned.model_copy(update={"signature": signature})
    (bundle_root / "artifact.bundle.json").write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2) + "\n"
    )
    return bundle_root, {key_id: base64.b64encode(public_key).decode()}


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def _mapped_port(container: str, port: int) -> int:
    output = _run(["docker", "port", container, f"{port}/tcp"]).stdout.strip()
    return int(output.rsplit(":", 1)[1])


def _wait_postgres(dsn: str) -> None:
    for _ in range(80):
        try:
            with psycopg.connect(dsn, connect_timeout=1):
                return
        except psycopg.Error:
            time.sleep(0.25)
    raise RuntimeError("disposable PostgreSQL did not become ready")


def _wait_http(url: str) -> None:
    for _ in range(80):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("disposable S3 fixture did not become ready")


def _seed_organizations(dsn: str) -> tuple[OrganizationId, OrganizationId, OrganizationId]:
    installation_id = uuid4()
    organizations = (OrganizationId(uuid4()), OrganizationId(uuid4()), OrganizationId(uuid4()))
    now = datetime.now(UTC)
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO identity_installations
                   (installation_id, singleton_key, display_name, created_at)
               VALUES (%s, TRUE, 'Stage 14 fixture', %s)""",
            (installation_id, now),
        )
        for index, organization in enumerate(organizations, start=1):
            cursor.execute(
                """INSERT INTO identity_organizations
                       (organization_id, installation_id, slug, display_name,
                        status, created_at, archived_at)
                   VALUES (%s,%s,%s,%s,'active',%s,NULL)""",
                (
                    organization.value,
                    installation_id,
                    f"stage14-{index}",
                    f"Stage 14 organization {index}",
                    now,
                ),
            )
    return organizations


def _materialization_benchmark(store: LocalCasBlobStore, root: Path) -> dict[str, float | int]:
    payload = bytes(range(256)) * (32 * 1024)
    descriptor = store.put_bytes(payload, media_type="application/octet-stream")
    source = store.materialize(digest=descriptor.digest, cache_key="benchmark:warmup")
    baseline_root = root / "benchmark-copy"
    baseline_root.mkdir()
    for index in range(5):
        store.materialize(digest=descriptor.digest, cache_key=f"benchmark:warmup:{index}")
        shutil.copyfile(source, baseline_root / f"warmup-{index}")
    materialize_samples: list[float] = []
    copy_samples: list[float] = []
    for index in range(30):
        started = time.perf_counter_ns()
        store.materialize(digest=descriptor.digest, cache_key=f"benchmark:sample:{index}")
        materialize_samples.append((time.perf_counter_ns() - started) / 1_000_000)
        started = time.perf_counter_ns()
        shutil.copyfile(source, baseline_root / f"sample-{index}")
        copy_samples.append((time.perf_counter_ns() - started) / 1_000_000)
    materialize_median = statistics.median(materialize_samples)
    copy_median = statistics.median(copy_samples)
    return {
        "payload_bytes": len(payload),
        "samples": 30,
        "warmups": 5,
        "materialize_median_ms": round(materialize_median, 3),
        "copy_median_ms": round(copy_median, 3),
        "materialize_over_copy_ratio": round(materialize_median / copy_median, 3),
    }


def run_proof() -> dict[str, object]:
    suffix = secrets.token_hex(4)
    network = f"roehub-stage14-{suffix}"
    postgres = f"roehub-stage14-pg-{suffix}"
    minio = f"roehub-stage14-s3-{suffix}"
    openbao = f"roehub-stage14-openbao-{suffix}"
    minio_volume = f"roehub-stage14-s3-data-{suffix}"
    postgres_password = secrets.token_urlsafe(24)
    minio_access = "stage14" + secrets.token_hex(8)
    minio_secret = secrets.token_urlsafe(32)
    bao_root = secrets.token_urlsafe(32)
    created = {
        "network": False,
        "postgres": False,
        "minio": False,
        "openbao": False,
        "volume": False,
    }
    cleanup = False
    try:
        image_repo_digests = json.loads(
            _run(
                ["docker", "image", "inspect", MINIO_TAG, "--format", "{{json .RepoDigests}}"]
            ).stdout.strip()
        )
        if not any(item.endswith("@" + MINIO_DIGEST) for item in image_repo_digests):
            raise RuntimeError("MinIO image digest does not match the Stage 14 binding")
        openbao_repo_digests = json.loads(
            _run(
                [
                    "docker",
                    "image",
                    "inspect",
                    f"{OPENBAO_IMAGE}@{OPENBAO_DIGEST}",
                    "--format",
                    "{{json .RepoDigests}}",
                ]
            ).stdout.strip()
        )
        if not any(item.endswith("@" + OPENBAO_DIGEST) for item in openbao_repo_digests):
            raise RuntimeError("OpenBao image digest does not match the Stage 14 binding")
        _run(["docker", "network", "create", network])
        created["network"] = True
        _run(["docker", "volume", "create", minio_volume])
        created["volume"] = True
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                postgres,
                "--network",
                network,
                "-e",
                "POSTGRES_USER=roehub",
                "-e",
                f"POSTGRES_PASSWORD={postgres_password}",
                "-e",
                "POSTGRES_DB=roehub",
                "-p",
                "127.0.0.1::5432",
                POSTGRES_IMAGE,
            ]
        )
        created["postgres"] = True
        postgres_port = _mapped_port(postgres, 5432)
        dsn = f"postgresql://roehub:{postgres_password}@127.0.0.1:{postgres_port}/roehub"
        _wait_postgres(dsn)
        with contextlib.redirect_stdout(io.StringIO()):
            apply_postgres_migrations(
                dsn,
                repo_root=ROOT,
                manifest_path=ROOT / "migrations/postgres/manifest.json",
            )
        owner, restore_owner, quota_owner = _seed_organizations(dsn)

        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                minio,
                "--network",
                network,
                "-e",
                f"MINIO_ROOT_USER={minio_access}",
                "-e",
                f"MINIO_ROOT_PASSWORD={minio_secret}",
                "-p",
                "127.0.0.1::9000",
                "-v",
                f"{minio_volume}:/data",
                f"minio/minio@{MINIO_DIGEST}",
                "server",
                "/data",
                "--address",
                ":9000",
            ]
        )
        created["minio"] = True
        minio_port = _mapped_port(minio, 9000)
        _wait_http(f"http://127.0.0.1:{minio_port}/minio/health/live")

        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                openbao,
                "--network",
                network,
                "-e",
                f"BAO_DEV_ROOT_TOKEN_ID={bao_root}",
                "-e",
                "BAO_DEV_LISTEN_ADDRESS=0.0.0.0:8200",
                "-p",
                "127.0.0.1::8200",
                f"ghcr.io/openbao/openbao@{OPENBAO_DIGEST}",
                "server",
                "-dev",
            ]
        )
        created["openbao"] = True
        openbao_port = _mapped_port(openbao, 8200)
        _wait_http(f"http://127.0.0.1:{openbao_port}/v1/sys/health")
        _run(
            [
                "docker",
                "exec",
                "-e",
                f"BAO_TOKEN={bao_root}",
                "-e",
                "BAO_ADDR=http://127.0.0.1:8200",
                openbao,
                "bao",
                "secrets",
                "enable",
                "-path=kv",
                "kv-v2",
            ]
        )

        fixture = ROOT / "tests/fixtures/artifacts/demo_bundle"
        public_keys = json.loads((fixture / "publisher-keys.json").read_text())
        with tempfile.TemporaryDirectory(prefix="roehub-stage14-") as raw_root:
            temp_root = Path(raw_root)
            quota_bundles: list[Path] = []
            for index in range(3):
                quota_bundle, quota_key = _signed_quota_bundle(temp_root, index=index)
                quota_bundles.append(quota_bundle)
                public_keys.update(quota_key)
            dsn_file = temp_root / "catalog.dsn"
            dsn_file.write_text(dsn + "\n")
            dsn_file.chmod(0o600)
            cas_root = temp_root / "cas"
            cli_output = io.StringIO()
            with contextlib.redirect_stdout(cli_output):
                cli_status = ArtifactsCli().run(
                    [
                        "install",
                        str(fixture),
                        "--organization-id",
                        str(owner.value),
                        "--catalog-dsn-file",
                        str(dsn_file),
                        "--cas-root",
                        str(cas_root),
                        "--publisher-keys",
                        str(fixture / "publisher-keys.json"),
                        "--quota-bytes",
                        "4096",
                    ]
                )
            cli_payload = json.loads(cli_output.getvalue())
            if cli_status != 0 or cli_payload.get("status") != "installed":
                raise RuntimeError("roehubctl artifacts install fixture failed")

            catalog = PostgresArtifactCatalogRepository(dsn=dsn)
            clock = [datetime(2026, 7, 13, 12, tzinfo=UTC)]
            local = LocalCasBlobStore(root=cas_root)
            service = ArtifactStoreService(
                blobs=local,
                catalog=catalog,
                trusted_public_keys=public_keys,
                now=lambda: clock[0],
            )
            manifest_digest = cli_payload["manifest_digest"]
            if not isinstance(manifest_digest, str):
                raise RuntimeError("CLI did not return a manifest digest")

            restart_output = _run(
                [
                    sys.executable,
                    str(ROOT / "tests/fixtures/artifacts/restart_probe.py"),
                    "--dsn-file",
                    str(dsn_file),
                    "--cas-root",
                    str(cas_root),
                    "--publisher-keys",
                    str(fixture / "publisher-keys.json"),
                    "--organization-id",
                    str(owner.value),
                    "--manifest-digest",
                    manifest_digest,
                ]
            )
            if json.loads(restart_output.stdout).get("status") != "passed":
                raise RuntimeError("separate-process restart proof failed")

            with ThreadPoolExecutor(max_workers=8) as executor:
                manifests = tuple(
                    executor.map(
                        lambda _: service.install_bundle(
                            organization_id=owner,
                            bundle_root=fixture,
                        ),
                        range(16),
                    )
                )
            if {manifest.manifest_digest for manifest in manifests} != {manifest_digest}:
                raise RuntimeError("concurrent publish was not idempotent")
            restarted = ArtifactStoreService(
                blobs=LocalCasBlobStore(root=cas_root),
                catalog=PostgresArtifactCatalogRepository(dsn=dsn),
                trusted_public_keys=public_keys,
                now=lambda: clock[0],
            )
            if not restarted.read_entry(
                organization_id=owner,
                manifest_digest=manifest_digest,
                path="demo/hello.json",
            ).startswith(b"{"):
                raise RuntimeError("restart persistence read failed")
            try:
                restarted.read_entry(
                    organization_id=restore_owner,
                    manifest_digest=manifest_digest,
                    path="demo/hello.json",
                )
            except ArtifactStoreError as error:
                if error.code != "artifact.manifest_not_found":
                    raise
            else:
                raise RuntimeError("cross-organization catalog read was allowed")

            restarted.set_quota(organization_id=quota_owner, max_bytes=100)
            for quota_bundle in quota_bundles:
                quota_manifest = ArtifactManifest.model_validate_json(
                    (quota_bundle / "artifact.bundle.json").read_bytes()
                )
                try:
                    restarted.install_bundle(
                        organization_id=quota_owner,
                        bundle_root=quota_bundle,
                    )
                except ArtifactStoreError as error:
                    if error.code != "artifact.quota_exceeded":
                        raise
                else:
                    raise RuntimeError("organization quota was bypassed")
                if local.exists(digest=quota_manifest.entries[0].blob.digest):
                    raise RuntimeError("quota failure left physical blob growth")

            interrupted = ArtifactBlobDescriptor(
                digest=sha256_digest(b"registered-but-never-written"),
                size_bytes=len(b"registered-but-never-written"),
                media_type="application/octet-stream",
            )
            catalog.register_blob(
                descriptor=interrupted,
                backend="local_cas",
                registered_at=clock[0],
            )
            if service.garbage_collect() != (interrupted.digest,):
                raise RuntimeError("interrupted orphan registration was not collected")

            signed_manifest = manifests[0]
            first, second = (entry.blob.digest for entry in signed_manifest.entries)
            service.pin(organization_id=owner, digest=first)
            service.acquire_lease(
                organization_id=owner,
                lease_id="stage14:lease:0001",
                digest=second,
                expires_at=clock[0] + timedelta(minutes=5),
            )
            service.retire_manifest(
                organization_id=owner,
                manifest_digest=manifest_digest,
            )
            if service.garbage_collect():
                raise RuntimeError("GC deleted pinned or leased data")
            service.unpin(organization_id=owner, digest=first)
            if service.garbage_collect() != (first,):
                raise RuntimeError("GC did not delete the unpinned blob")
            clock[0] += timedelta(minutes=6)
            if service.garbage_collect() != (second,):
                raise RuntimeError("GC did not delete the expired-lease blob")

            restored_manifest = service.install_bundle(
                organization_id=owner,
                bundle_root=fixture,
            )
            service.pin(
                organization_id=owner,
                digest=restored_manifest.entries[0].blob.digest,
            )
            backup_root = temp_root / "backup"
            backup_digest = service.backup(
                organization_id=owner,
                destination=backup_root,
            )
            restored_service = ArtifactStoreService(
                blobs=LocalCasBlobStore(root=temp_root / "restored-cas"),
                catalog=PostgresArtifactCatalogRepository(dsn=dsn),
                trusted_public_keys=public_keys,
                now=lambda: clock[0],
            )
            with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
                cursor.execute(
                    """CREATE FUNCTION stage14_restore_fault() RETURNS trigger
                       LANGUAGE plpgsql AS $$
                       BEGIN
                           RAISE EXCEPTION 'stage14 injected restore fault';
                       END
                       $$"""
                )
                cursor.execute(
                    """CREATE TRIGGER stage14_restore_fault_trigger
                       AFTER INSERT ON artifact_store_manifests
                       FOR EACH ROW EXECUTE FUNCTION stage14_restore_fault()"""
                )
            try:
                restored_service.restore(
                    organization_id=restore_owner,
                    source=backup_root,
                    expected_backup_digest=backup_digest,
                )
            except ArtifactStoreError as error:
                if error.code != "artifact.restore_failed":
                    raise
            else:
                raise RuntimeError("injected restore fault did not fail")
            with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
                cursor.execute(
                    """SELECT
                           (SELECT count(*) FROM artifact_store_manifests
                            WHERE organization_id = %s),
                           (SELECT count(*) FROM artifact_store_org_blobs
                            WHERE organization_id = %s)""",
                    (restore_owner.value, restore_owner.value),
                )
                partial = cursor.fetchone()
                if partial != (0, 0):
                    raise RuntimeError("failed restore left partial catalog state")
                cursor.execute(
                    "DROP TRIGGER stage14_restore_fault_trigger ON artifact_store_manifests"
                )
                cursor.execute("DROP FUNCTION stage14_restore_fault()")
            restored_manifests = restored_service.restore(
                organization_id=restore_owner,
                source=backup_root,
                expected_backup_digest=backup_digest,
            )
            if restored_manifests != (manifest_digest,):
                raise RuntimeError("backup restore manifest mismatch")
            if not restored_service.read_entry(
                organization_id=restore_owner,
                manifest_digest=manifest_digest,
                path="demo/model-card.txt",
            ).startswith(b"Roehub"):
                raise RuntimeError("backup restore payload mismatch")

            corruption = LocalCasBlobStore(root=temp_root / "corruption-cas")
            corrupted = corruption.put_bytes(b"safe", media_type="application/octet-stream")
            corrupt_path = next(
                (temp_root / "corruption-cas/blobs/sha256").rglob(corrupted.digest[7:])
            )
            corrupt_path.chmod(0o640)
            corrupt_path.write_bytes(b"tampered")
            try:
                corruption.read_bytes(digest=corrupted.digest)
            except ArtifactStoreError as error:
                if error.code != "artifact.digest_mismatch":
                    raise
            else:
                raise RuntimeError("corrupted local CAS blob was accepted")

            bao_file = temp_root / "openbao.service"
            bao_file.write_text(bao_root + "\n")
            bao_file.chmod(0o600)
            resolver = OpenBaoSecretResolver(
                address=f"http://127.0.0.1:{openbao_port}",
                token_source=SecureCredentialFile(bao_file.resolve()),
            )
            s3_config = S3ConnectionConfig(
                endpoint=f"http://127.0.0.1:{minio_port}",
                bucket=f"stage14-{suffix}",
                region="us-east-1",
                credentials_ref="openbao://kv/roehub/storage/stage14-s3#credentials",
            )
            resolver.store(
                s3_config.credentials_ref,
                value=SecretValue.from_text(
                    json.dumps(
                        dict(
                            [
                                ("access_key_id", minio_access),
                                ("secret_access_key", minio_secret),
                            ]
                        )
                    )
                ),
                expected_kind=SecretKind.STORAGE,
            )
            s3 = S3CompatibleBlobStore(
                config=s3_config,
                credentials=resolve_s3_credentials(
                    config=s3_config,
                    resolver=resolver,
                ),
                materialization_root=temp_root / "s3-materialized",
            )
            try:
                s3.ensure_bucket()
                s3_blob = s3.put_bytes(b"minio-boundary", media_type="application/octet-stream")
                if s3.read_bytes(digest=s3_blob.digest) != b"minio-boundary":
                    raise RuntimeError("S3-compatible read mismatch")
                if (
                    s3.materialize(
                        digest=s3_blob.digest,
                        cache_key="stage14:minio",
                    ).read_bytes()
                    != b"minio-boundary"
                ):
                    raise RuntimeError("S3 materialization mismatch")
                s3.delete(digest=s3_blob.digest)
                if s3.exists(digest=s3_blob.digest):
                    raise RuntimeError("S3 delete did not remove the object")
            finally:
                s3.close()

            benchmark = _materialization_benchmark(local, temp_root)

        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT count(*) FROM artifact_store_manifests
                   WHERE organization_id = %s""",
                (quota_owner.value,),
            )
            quota_row = cursor.fetchone()
            if quota_row is None or quota_row[0] != 0:
                raise RuntimeError("quota failure left catalog state")

        return {
            "schema": "io.roehub.artifact-store-runtime-proof/v1",
            "status": "passed",
            "artifact_store_contract": "passed",
            "atomic_concurrent_publish": "passed",
            "backup_restore": "passed",
            "catalog_postgresql": "passed",
            "cli_bundle_install": "passed",
            "corruption_rejection": "passed",
            "cross_organization_denial": "passed",
            "demo_bundle_signature": "passed",
            "gc_pin_lease": "passed",
            "image_digest_binding": "passed",
            "interrupted_orphan_cleanup": "passed",
            "local_cas_process_restart": "passed",
            "materialization_benchmark": benchmark,
            "openbao_s3_credentials": "passed",
            "quota": "passed",
            "quota_orphan_cleanup": "passed",
            "restore_atomicity": "passed",
            "s3_compatible_minio": "passed",
        }
    finally:
        if created["openbao"]:
            _run(["docker", "rm", "-f", openbao], check=False)
        if created["minio"]:
            _run(["docker", "rm", "-f", minio], check=False)
        if created["postgres"]:
            _run(["docker", "rm", "-f", postgres], check=False)
        if created["volume"]:
            _run(["docker", "volume", "rm", "-f", minio_volume], check=False)
        if created["network"]:
            _run(["docker", "network", "rm", network], check=False)
        remaining = any(
            _run(["docker", "inspect", container], check=False).returncode == 0
            for container in (postgres, minio, openbao)
        )
        volume_remaining = _run(
            ["docker", "volume", "ls", "--filter", f"name={minio_volume}", "-q"],
            check=False,
        ).stdout.strip()
        network_remaining = _run(
            ["docker", "network", "ls", "--filter", f"name={network}", "-q"],
            check=False,
        ).stdout.strip()
        cleanup = not remaining and not volume_remaining and not network_remaining
        if not cleanup:
            raise RuntimeError("Stage 14 disposable runtime cleanup failed")


def main() -> int:
    try:
        result = run_proof()
        result["cleanup"] = "passed"
    except Exception as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "io.roehub.artifact-store-runtime-proof/v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
