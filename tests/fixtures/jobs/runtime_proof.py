from __future__ import annotations

import base64
import contextlib
import io
import json
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import psycopg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apps.control_agent.job_runtime_backend import ControlAgentJobDockerRunner
from apps.migrations.storage import apply_postgres_migrations
from apps.worker.job_runtime import (
    JobAttemptExecutor,
    JobRuntimeAuthorityError,
    OciJobRunner,
    PluginTrustResolution,
    TrustedRuntimeAuthority,
    TrustedRuntimeGrant,
)
from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_cas import (
    LocalCasBlobStore,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres import (
    PostgresArtifactCatalogRepository,
)
from trading.contexts.backtest_artifacts.application import ArtifactStoreService
from trading.contexts.extensions.domain import PluginInstallation, PluginPackage
from trading.integration import (
    JobEnvelope,
    JobResourceLimits,
    JobResultManifest,
)
from trading.integration.job_runtime_postgres import (
    JobRuntimeCatalogError,
    PostgresJobRuntimeCatalog,
)
from trading.shared_kernel.primitives import InstallationId, OrganizationId

ROOT = Path(__file__).resolve().parents[3]
POSTGRES_IMAGE = "postgres:16"
JOB_IMAGE = "alpine:3.22"
JOB_IMAGE_DIGEST = "sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
WORKER_ID = "stage15.worker.001"
SUCCESS_COMMAND = (
    "/bin/sh",
    "-c",
    "test ! -e /var/run/docker.sock; "
    'test "$(id -u)" = 65532; '
    "test -r /job/input/artifacts/0/demo/model-card.txt; "
    "test ! -w /job/input/artifacts/0/demo/model-card.txt; "
    "sha256sum /job/input/artifacts/0/demo/model-card.txt "
    "> /job/output/input.sha256; "
    "printf '{\"value\":42}\\n' > /job/output/result.json",
)
STRATEGY_COMMAND = (
    "/bin/sh",
    "-c",
    'printf \'[{"kind":"intent","instrument_id":"btc.usdt",'
    '"side":"buy","strength_decimal":"0.5",'
    '"observed_at":"2026-07-13T00:00:00Z",'
    '"reason_code":"fixture.signal"}]\\n\' '
    "> /job/output/strategy-decisions.json",
)
CRASH_COMMAND = ("/bin/sh", "-c", "exit 7")
VOLUNTARY_137_COMMAND = ("/bin/sh", "-c", "exit 137")
BACKGROUND_TAMPER_COMMAND = (
    "/bin/sh",
    "-c",
    "printf stable > /job/output/result.txt; "
    "(sleep 1; printf tampered >> /job/output/result.txt) & exit 0",
)
TIMEOUT_COMMAND = ("/bin/sh", "-c", "sleep 5")
CANCEL_COMMAND = ("/bin/sh", "-c", "sleep 10")
RETRY_COMMAND = (
    "/bin/sh",
    "-c",
    "if grep -q '\"attempt_number\":1' /job/input/envelope.json; "
    "then exit 9; else printf retry > /job/output/retry.txt; fi",
)
MEMORY_COMMAND = (
    "/bin/sh",
    "-c",
    "yes 0123456789abcdef | head -c 134217728 | sort >/dev/null",
)
OUTPUT_LIMIT_COMMAND = (
    "/bin/sh",
    "-c",
    "dd if=/dev/zero of=/job/output/too-big.bin bs=1048576 count=2",
)
PID_LIMIT_COMMAND = (
    "/bin/sh",
    "-c",
    "i=0; while [ $i -lt 64 ]; do sleep 2 & i=$((i + 1)); done; wait",
)
INODE_LIMIT_COMMAND = (
    "/bin/sh",
    "-c",
    "i=0; while [ $i -lt 1100 ]; do : > /job/output/file-$i; i=$((i + 1)); done",
)
STALE_COMMAND = ("/bin/sh", "-c", "sleep 30")


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


def _seed_organizations(dsn: str) -> tuple[UUID, UUID]:
    installation_id = uuid4()
    owner, foreign = uuid4(), uuid4()
    now = datetime.now(UTC)
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO identity_installations
                   (installation_id, singleton_key, display_name, created_at)
               VALUES (%s, TRUE, 'Stage 15 fixture', %s)""",
            (installation_id, now),
        )
        for index, organization_id in enumerate((owner, foreign), start=1):
            cursor.execute(
                """INSERT INTO identity_organizations
                       (organization_id, installation_id, slug, display_name,
                        status, created_at, archived_at)
                   VALUES (%s,%s,%s,%s,'active',%s,NULL)""",
                (
                    organization_id,
                    installation_id,
                    f"stage15-{index}",
                    f"Stage 15 organization {index}",
                    now,
                ),
            )
    return owner, foreign


def _limits(
    *,
    wall_time: int = 10,
    memory_bytes: int = 64 * 1024 * 1024,
    pids: int = 32,
    output_bytes: int = 8 * 1024 * 1024,
) -> JobResourceLimits:
    return JobResourceLimits(
        cpu_millis=500,
        memory_bytes=memory_bytes,
        pids=pids,
        wall_time_seconds=wall_time,
        tmpfs_bytes=8 * 1024 * 1024,
        output_bytes=output_bytes,
    )


def _envelope(
    *,
    organization_id: UUID,
    artifact_digest: str,
    semantic_key: str,
    command: tuple[str, ...],
    capability: str = "backtest",
    limits: JobResourceLimits | None = None,
    job_id: UUID | None = None,
    attempt_id: UUID | None = None,
    attempt_number: int = 1,
    deadline: datetime | None = None,
) -> JobEnvelope:
    return JobEnvelope.model_validate(
        {
            "schema": "JobEnvelope/v1",
            "job_id": job_id or uuid4(),
            "attempt_id": attempt_id or uuid4(),
            "attempt_number": attempt_number,
            "organization_id": organization_id,
            "semantic_job_key": semantic_key,
            "capability": capability,
            "image_digest": JOB_IMAGE_DIGEST,
            "runtime": {
                "name": f"roehub.{capability}",
                "version": "1.0.0",
                **({"plugin_package_digest": "1" * 64} if capability == "custom_strategy" else {}),
            },
            "config_snapshot": {
                "seed": 42,
                "mode": "deterministic",
                "source": "synthetic",
            },
            "input_artifact_digests": [artifact_digest],
            "limits": (limits or _limits()).model_dump(mode="json"),
            "deadline": deadline or datetime.now(UTC) + timedelta(minutes=2),
            "command": command,
            "network": "none",
        }
    )


def _job_roots(
    root: Path,
    *,
    envelope: JobEnvelope,
) -> tuple[Path, Path]:
    attempt_root = root / envelope.attempt_id.hex
    input_root = attempt_root / "input"
    output_root = attempt_root / "output"
    output_root.mkdir(parents=True)
    return input_root, output_root


def _execute(
    *,
    envelope: JobEnvelope,
    catalog: PostgresJobRuntimeCatalog,
    executor: JobAttemptExecutor,
    runtime_root: Path,
) -> tuple[JobResultManifest, Path]:
    catalog.submit(envelope=envelope, created_at=datetime.now(UTC))
    claimed = catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC))
    if claimed is None or claimed.envelope.attempt_id != envelope.attempt_id:
        raise RuntimeError("expected attempt was not claimed")
    _input_root, output_root = _job_roots(
        runtime_root,
        envelope=envelope,
    )
    return (
        executor.execute(
            claimed=claimed,
        ),
        output_root,
    )


def _benchmark(
    *, runner: OciJobRunner, root: Path, organization_id: UUID, artifact_digest: str
) -> dict[str, float | int | str]:
    root.mkdir(parents=True)
    command = (
        "/bin/sh",
        "-c",
        "i=0; s=0; while [ $i -lt 12000 ]; do "
        "s=$((s + (i % 97))); i=$((i + 1)); done; "
        "printf '%s\\n' \"$s\" > /job/output/compute.txt",
    )
    warmups = 5
    samples = 20
    baseline_values: list[float] = []
    hardened_values: list[float] = []
    expected = b"575034\n"

    def baseline(index: int) -> float:
        output = root / f"baseline-{index}"
        output.mkdir(mode=0o777)
        started = time.perf_counter_ns()
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--pull",
                "never",
                "--network",
                "none",
                "--user",
                "65532:65532",
                "--mount",
                f"type=bind,source={output},target=/job/output",
                JOB_IMAGE_DIGEST,
                *command,
            ]
        )
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if (output / "compute.txt").read_bytes() != expected:
            raise RuntimeError("baseline compute output changed")
        return elapsed

    def hardened(index: int) -> float:
        envelope = _envelope(
            organization_id=organization_id,
            artifact_digest=artifact_digest,
            semantic_key=f"benchmark:{index:04d}",
            command=command,
        )
        input_root = root / f"hardened-{index}" / "input"
        output_root = root / f"hardened-{index}" / "output"
        started = time.perf_counter_ns()
        result = runner.run(
            envelope=envelope,
            input_root=input_root,
            output_root=output_root,
        )
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if result.outcome != "succeeded" or (output_root / "compute.txt").read_bytes() != expected:
            raise RuntimeError("hardened compute output changed")
        return elapsed

    for index in range(warmups):
        baseline(-(index + 1))
        hardened(-(index + 1))
    for index in range(samples):
        baseline_values.append(baseline(index))
        hardened_values.append(hardened(index))
    baseline_median = statistics.median(baseline_values)
    hardened_median = statistics.median(hardened_values)
    ratio = hardened_median / baseline_median
    if ratio > 4.0:
        raise RuntimeError(
            "hardened job lifecycle exceeded the measured overhead budget: "
            f"baseline={baseline_median:.3f}ms hardened={hardened_median:.3f}ms "
            f"ratio={ratio:.3f}"
        )
    return {
        "baseline_median_ms": round(baseline_median, 3),
        "hardened_median_ms": round(hardened_median, 3),
        "hardened_over_baseline_ratio": round(ratio, 3),
        "samples": samples,
        "warmups": warmups,
        "output_sha256": "sha256:75f206dc943bc3e65701052f1a3a6b3c7bc6f77eeca150cf24f38155e59674fb",
    }


def _successful_catalog_result(*, envelope: JobEnvelope, artifact_digest: str) -> JobResultManifest:
    return JobResultManifest(
        schema="JobResultManifest/v1",
        job_id=envelope.job_id,
        attempt_id=envelope.attempt_id,
        organization_id=envelope.organization_id,
        outcome="succeeded",
        envelope_digest=envelope.envelope_digest,
        output_artifact_manifest_digest=artifact_digest,
        outputs=(),
        strategy_decisions=(),
        completed_at=datetime.now(UTC),
        exit_code=0,
        error_code=None,
    )


def _expect_database_rejection(dsn: str, statement: str, values: tuple[object, ...]) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        try:
            cursor.execute(cast(Any, statement), values)
        except psycopg.Error:
            connection.rollback()
        else:
            raise RuntimeError("database immutability guard accepted a forbidden update")


def _prove_catalog_concurrency(
    *,
    catalog: PostgresJobRuntimeCatalog,
    dsn: str,
    organization_id: UUID,
    artifact_digest: str,
) -> None:
    canceled = _envelope(
        organization_id=organization_id,
        artifact_digest=artifact_digest,
        semantic_key="race:cancel-wins:0001",
        command=SUCCESS_COMMAND,
    )
    catalog.submit(envelope=canceled, created_at=datetime.now(UTC))
    canceled_claim = catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC))
    if canceled_claim is None:
        raise RuntimeError("cancel-wins attempt was not claimed")
    _expect_database_rejection(
        dsn,
        """UPDATE job_runtime_attempts
           SET envelope = jsonb_set(envelope, '{network}', '\"bridge\"'::jsonb)
           WHERE organization_id = %s AND attempt_id = %s""",
        (organization_id, canceled.attempt_id),
    )
    catalog.request_cancel(
        organization_id=organization_id,
        job_id=canceled.job_id,
        requested_at=datetime.now(UTC),
    )
    canceled_result = catalog.finish_attempt(
        envelope=canceled,
        worker_id=WORKER_ID,
        result=_successful_catalog_result(
            envelope=canceled,
            artifact_digest=artifact_digest,
        ),
    )
    if canceled_result.outcome != "canceled":
        raise RuntimeError("accepted cancellation did not win finalization")
    _expect_database_rejection(
        dsn,
        """UPDATE job_runtime_attempts SET status = 'failed'
           WHERE organization_id = %s AND attempt_id = %s""",
        (organization_id, canceled.attempt_id),
    )

    completed = _envelope(
        organization_id=organization_id,
        artifact_digest=artifact_digest,
        semantic_key="race:finish-wins:0001",
        command=SUCCESS_COMMAND,
    )
    catalog.submit(envelope=completed, created_at=datetime.now(UTC))
    completed_claim = catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC))
    if completed_claim is None:
        raise RuntimeError("finish-wins attempt was not claimed")
    catalog.finish_attempt(
        envelope=completed,
        worker_id=WORKER_ID,
        result=_successful_catalog_result(
            envelope=completed,
            artifact_digest=artifact_digest,
        ),
    )
    try:
        catalog.request_cancel(
            organization_id=organization_id,
            job_id=completed.job_id,
            requested_at=datetime.now(UTC),
        )
    except JobRuntimeCatalogError as error:
        if error.code != "job.already_finished":
            raise
    else:
        raise RuntimeError("completed job accepted a late cancellation")

    for index in range(12):
        envelope = _envelope(
            organization_id=organization_id,
            artifact_digest=artifact_digest,
            semantic_key=f"race:concurrent:{index:04d}",
            command=SUCCESS_COMMAND,
        )
        catalog.submit(envelope=envelope, created_at=datetime.now(UTC))
        claim = catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC))
        if claim is None:
            raise RuntimeError("concurrency attempt was not claimed")

        def finish() -> str:
            return catalog.finish_attempt(
                envelope=envelope,
                worker_id=WORKER_ID,
                result=_successful_catalog_result(
                    envelope=envelope,
                    artifact_digest=artifact_digest,
                ),
            ).outcome

        def cancel() -> str:
            try:
                catalog.request_cancel(
                    organization_id=organization_id,
                    job_id=envelope.job_id,
                    requested_at=datetime.now(UTC),
                )
            except JobRuntimeCatalogError as error:
                if error.code != "job.already_finished":
                    raise
                return "already_finished"
            return "accepted"

        with ThreadPoolExecutor(max_workers=2) as pool:
            finish_future = pool.submit(finish)
            cancel_future = pool.submit(cancel)
            finish_outcome = finish_future.result(timeout=5)
            cancel_outcome = cancel_future.result(timeout=5)
        if (finish_outcome, cancel_outcome) not in {
            ("succeeded", "already_finished"),
            ("canceled", "accepted"),
        }:
            raise RuntimeError("finish/cancel race was not linearizable")


def _prove_recovery_linearization(
    *,
    catalog: PostgresJobRuntimeCatalog,
    organization_id: UUID,
    artifact_digest: str,
) -> None:
    lease = _envelope(
        organization_id=organization_id,
        artifact_digest=artifact_digest,
        semantic_key="recovery:lease:0001",
        command=STALE_COMMAND,
    )
    catalog.submit(envelope=lease, created_at=datetime.now(UTC))
    if catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC)) is None:
        raise RuntimeError("recovery lease attempt was not claimed")
    first_now = datetime.now(UTC)
    first_claims = catalog.claim_stale_for_recovery(
        now=first_now,
        worker_heartbeat_before=first_now + timedelta(seconds=1),
        recovery_claimed_before=first_now - timedelta(minutes=1),
        recovery_owner_id="recovery.lease.first",
    )
    if len(first_claims) != 1:
        raise RuntimeError("first recovery owner did not acquire the stale attempt")
    second_claims = catalog.claim_stale_for_recovery(
        now=first_now + timedelta(seconds=1),
        worker_heartbeat_before=first_now + timedelta(minutes=1),
        recovery_claimed_before=first_now - timedelta(seconds=1),
        recovery_owner_id="recovery.lease.second",
    )
    if second_claims:
        raise RuntimeError("active recovery lease was stolen")
    reclaimed_at = first_now + timedelta(minutes=2)
    reclaimed = catalog.claim_stale_for_recovery(
        now=reclaimed_at,
        worker_heartbeat_before=reclaimed_at,
        recovery_claimed_before=first_now + timedelta(minutes=1),
        recovery_owner_id="recovery.lease.reclaimed",
    )
    if len(reclaimed) != 1:
        raise RuntimeError("expired recovery lease was not reclaimable")
    catalog.complete_recovery(claim=reclaimed[0], completed_at=reclaimed_at)

    canceled = _envelope(
        organization_id=organization_id,
        artifact_digest=artifact_digest,
        semantic_key="recovery:cancel:0001",
        command=STALE_COMMAND,
    )
    catalog.submit(envelope=canceled, created_at=datetime.now(UTC))
    if catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC)) is None:
        raise RuntimeError("recovery cancellation attempt was not claimed")
    canceled_at = datetime.now(UTC)
    catalog.request_cancel(
        organization_id=organization_id,
        job_id=canceled.job_id,
        requested_at=canceled_at,
    )
    canceled_claims = catalog.claim_stale_for_recovery(
        now=canceled_at,
        worker_heartbeat_before=canceled_at + timedelta(seconds=1),
        recovery_claimed_before=canceled_at - timedelta(minutes=1),
        recovery_owner_id="recovery.cancel.owner",
    )
    if len(canceled_claims) != 1 or canceled_claims[0].outcome != "canceled":
        raise RuntimeError("accepted cancellation did not win stale recovery")
    canceled_result = catalog.complete_recovery(
        claim=canceled_claims[0],
        completed_at=canceled_at,
    )
    if canceled_result.outcome != "canceled":
        raise RuntimeError("recovery completion lost the cancellation marker")


def main() -> None:
    suffix = secrets.token_hex(4)
    postgres = f"roehub-stage15-postgres-{suffix}"
    postgres_password = secrets.token_urlsafe(24)
    created = False
    cleanup = False
    try:
        image = json.loads(_run(["docker", "image", "inspect", JOB_IMAGE]).stdout)[0]
        if image.get("Id") != JOB_IMAGE_DIGEST:
            raise RuntimeError("job image digest does not match Stage 15 binding")
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                postgres,
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
        created = True
        port = _mapped_port(postgres, 5432)
        dsn = f"postgresql://roehub:{postgres_password}@127.0.0.1:{port}/roehub"
        _wait_postgres(dsn)
        with contextlib.redirect_stdout(io.StringIO()):
            apply_postgres_migrations(
                dsn,
                repo_root=ROOT,
                manifest_path=ROOT / "migrations/postgres/manifest.json",
            )
        organization_id, foreign_organization_id = _seed_organizations(dsn)
        with tempfile.TemporaryDirectory(prefix=".roehub-stage15-", dir=ROOT) as temporary:
            temp_root = Path(temporary)
            demo_root = ROOT / "tests/fixtures/artifacts/demo_bundle"
            trusted_keys = json.loads((demo_root / "publisher-keys.json").read_text())
            signing_key = Ed25519PrivateKey.generate()
            signing_key_id = "stage15.job.publisher"
            public_key = signing_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            trusted_keys[signing_key_id] = base64.b64encode(public_key).decode()
            artifact_service = ArtifactStoreService(
                blobs=LocalCasBlobStore(root=temp_root / "cas"),
                catalog=PostgresArtifactCatalogRepository(dsn=dsn),
                trusted_public_keys=trusted_keys,
            )
            input_manifest = artifact_service.install_bundle(
                organization_id=OrganizationId(organization_id),
                bundle_root=demo_root,
            )
            catalog = PostgresJobRuntimeCatalog(dsn=dsn)
            runner = OciJobRunner(
                utility_image_digest=JOB_IMAGE_DIGEST,
                command_runner=ControlAgentJobDockerRunner(),
            )
            grant_specs = (
                ("backtest", SUCCESS_COMMAND),
                ("custom_strategy", STRATEGY_COMMAND),
                ("backtest", CRASH_COMMAND),
                ("backtest", VOLUNTARY_137_COMMAND),
                ("backtest", BACKGROUND_TAMPER_COMMAND),
                ("backtest", TIMEOUT_COMMAND),
                ("backtest", CANCEL_COMMAND),
                ("backtest", RETRY_COMMAND),
                ("backtest", MEMORY_COMMAND),
                ("backtest", OUTPUT_LIMIT_COMMAND),
                ("backtest", PID_LIMIT_COMMAND),
                ("backtest", INODE_LIMIT_COMMAND),
            )
            grants: list[TrustedRuntimeGrant] = []
            plugin_trust_records: dict[str, PluginTrustResolution] = {}
            for index, (capability, command) in enumerate(grant_specs):
                grant_envelope = _envelope(
                    organization_id=organization_id,
                    artifact_digest=input_manifest.manifest_digest,
                    semantic_key=f"authority:grant:{index:04d}",
                    capability=capability,
                    command=command,
                )
                if capability == "custom_strategy":
                    package = PluginPackage(
                        package_id=uuid4(),
                        installation_id=InstallationId(uuid4()),
                        plugin_id="strategy.fixture",
                        version=grant_envelope.runtime.version,
                        package_digest=cast(str, grant_envelope.runtime.plugin_package_digest),
                        image_reference="alpine@" + JOB_IMAGE_DIGEST,
                        image_digest=JOB_IMAGE_DIGEST,
                        publisher_key_id="stage15.publisher",
                        publisher_public_key_b64=base64.b64encode(public_key).decode(),
                        publisher_key_fingerprint_sha256="1" * 64,
                        manifest={},
                        created_at=datetime.now(UTC),
                    )
                    installation = PluginInstallation(
                        plugin_installation_id=uuid4(),
                        installation_id=package.installation_id,
                        organization_id=OrganizationId(organization_id),
                        plugin_id=package.plugin_id,
                        package_id=package.package_id,
                        previous_package_id=None,
                        granted_permissions=(),
                        status="enabled",
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )
                    trust_root = {
                        cast(str, package.publisher_key_id): cast(
                            str, package.publisher_key_fingerprint_sha256
                        )
                    }
                    plugin_trust_records[package.package_digest] = (
                        package,
                        installation,
                        trust_root,
                    )
                    grants.append(
                        TrustedRuntimeGrant.for_signed_plugin(
                            envelope=grant_envelope,
                            package=package,
                            installation=installation,
                            trusted_publisher_fingerprints=trust_root,
                        )
                    )
                else:
                    grants.append(TrustedRuntimeGrant.for_builtin(grant_envelope))

            def resolve_plugin_trust(envelope: JobEnvelope) -> PluginTrustResolution:
                package_digest = envelope.runtime.plugin_package_digest
                if package_digest is None:
                    raise KeyError("plugin package digest is missing")
                return plugin_trust_records[package_digest]

            authority = TrustedRuntimeAuthority(
                grants=tuple(grants),
                plugin_trust_resolver=resolve_plugin_trust,
            )
            runtime_root = temp_root / "runtime"
            executor = JobAttemptExecutor(
                catalog=catalog,
                runner=runner,
                artifacts=artifact_service,
                result_signing_key=signing_key,
                result_signing_key_id=signing_key_id,
                worker_id=WORKER_ID,
                authority=authority,
                runtime_root=runtime_root,
            )

            first = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="success:deterministic:0001",
                command=SUCCESS_COMMAND,
            )
            first_result, _ = _execute(
                envelope=first,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if getattr(first_result, "outcome") != "succeeded":
                raise RuntimeError(
                    "successful job did not complete: "
                    f"{getattr(first_result, 'outcome', None)}/"
                    f"{getattr(first_result, 'error_code', None)}"
                )
            artifact_service.read_entry(
                organization_id=OrganizationId(organization_id),
                manifest_digest=getattr(first_result, "output_artifact_manifest_digest"),
                path="result.json",
            )
            second = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="success:deterministic:0002",
                command=SUCCESS_COMMAND,
            )
            second_result, _ = _execute(
                envelope=second,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if getattr(first_result, "outputs") != getattr(second_result, "outputs"):
                raise RuntimeError("deterministic replay output changed")

            strategy = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="strategy:intent:0001",
                command=STRATEGY_COMMAND,
                capability="custom_strategy",
            )
            strategy_result, _ = _execute(
                envelope=strategy,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if len(getattr(strategy_result, "strategy_decisions")) != 1:
                raise RuntimeError("strategy decision contract was not enforced")
            strategy_package_digest = cast(str, strategy.runtime.plugin_package_digest)
            strategy_trust_root = cast(
                dict[str, str], plugin_trust_records[strategy_package_digest][2]
            )
            saved_trust_root = dict(strategy_trust_root)
            strategy_trust_root.clear()
            try:
                authority.authorize(strategy)
            except JobRuntimeAuthorityError as error:
                if error.code != "job.plugin_publisher_untrusted":
                    raise
            else:
                raise RuntimeError("revoked plugin trust remained executable")
            finally:
                strategy_trust_root.update(saved_trust_root)

            crash = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="crash:container:0001",
                command=CRASH_COMMAND,
            )
            crash_result, _ = _execute(
                envelope=crash,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if getattr(crash_result, "outcome") != "crashed":
                raise RuntimeError("container crash was not durable")

            voluntary_137 = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="crash:voluntary-137:0001",
                command=VOLUNTARY_137_COMMAND,
            )
            voluntary_137_result, _ = _execute(
                envelope=voluntary_137,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if voluntary_137_result.outcome != "crashed":
                raise RuntimeError("voluntary exit 137 was misclassified as exhaustion")

            background = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="output:background-freeze:0001",
                command=BACKGROUND_TAMPER_COMMAND,
            )
            background_result, background_output = _execute(
                envelope=background,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if background_result.outcome != "succeeded":
                raise RuntimeError("background freeze job did not succeed")
            time.sleep(1.2)
            if (background_output / "result.txt").exists():
                raise RuntimeError("temporary background output was not cleaned")
            artifact_payload = artifact_service.read_entry(
                organization_id=OrganizationId(organization_id),
                manifest_digest=cast(str, background_result.output_artifact_manifest_digest),
                path="result.txt",
            )
            if artifact_payload != b"stable":
                raise RuntimeError("background process changed output after PID 1 exit")

            timeout = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="timeout:container:0001",
                command=TIMEOUT_COMMAND,
                limits=_limits(wall_time=1),
            )
            timeout_result, timeout_output = _execute(
                envelope=timeout,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if getattr(timeout_result, "outcome") != "timed_out" or timeout_output.exists():
                raise RuntimeError("timeout or temporary cleanup failed")

            cancel = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="cancel:container:0001",
                command=CANCEL_COMMAND,
            )
            catalog.submit(envelope=cancel, created_at=datetime.now(UTC))
            cancel_claim = catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC))
            if cancel_claim is None:
                raise RuntimeError("cancel attempt was not claimed")
            _cancel_input, cancel_output = _job_roots(
                runtime_root,
                envelope=cancel,
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    executor.execute,
                    claimed=cancel_claim,
                )
                time.sleep(0.35)
                catalog.request_cancel(
                    organization_id=organization_id,
                    job_id=cancel.job_id,
                    requested_at=datetime.now(UTC),
                )
                cancel_result = future.result(timeout=10)
            if cancel_result.outcome != "canceled" or cancel_output.exists():
                raise RuntimeError("cancel or temporary cleanup failed")

            retry_first = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="retry:same-job:0001",
                command=RETRY_COMMAND,
            )
            retry_first_result, _ = _execute(
                envelope=retry_first,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if getattr(retry_first_result, "outcome") != "crashed":
                raise RuntimeError("retry first attempt did not crash")
            retry_second = retry_first.model_copy(
                update={
                    "attempt_id": uuid4(),
                    "attempt_number": 2,
                    "deadline": datetime.now(UTC) + timedelta(minutes=2),
                }
            )
            catalog.retry(envelope=retry_second, queued_at=datetime.now(UTC))
            retry_claim = catalog.claim_next(worker_id=WORKER_ID, now=datetime.now(UTC))
            if retry_claim is None:
                raise RuntimeError("retry attempt was not claimed")
            retry_second_result = executor.execute(
                claimed=retry_claim,
            )
            if retry_second_result.outcome != "succeeded":
                raise RuntimeError("retry attempt did not succeed")

            exhausted = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="resource:memory:0001",
                command=MEMORY_COMMAND,
                limits=_limits(memory_bytes=16 * 1024 * 1024),
            )
            exhausted_result, _ = _execute(
                envelope=exhausted,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if getattr(exhausted_result, "outcome") != "resource_exhausted":
                raise RuntimeError("memory exhaustion was not classified")

            output_exhausted = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="resource:output:0001",
                command=OUTPUT_LIMIT_COMMAND,
                limits=_limits(output_bytes=1024 * 1024),
            )
            output_result, output_root = _execute(
                envelope=output_exhausted,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if output_result.outcome != "resource_exhausted" or output_root.exists():
                raise RuntimeError("output quota or cleanup was not enforced")

            pid_exhausted = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="resource:pids:0001",
                command=PID_LIMIT_COMMAND,
                limits=_limits(pids=16),
            )
            pid_result, _ = _execute(
                envelope=pid_exhausted,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if pid_result.outcome == "succeeded":
                raise RuntimeError("PID exhaustion was not enforced")

            inode_exhausted = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="resource:inodes:0001",
                command=INODE_LIMIT_COMMAND,
            )
            inode_result, inode_root = _execute(
                envelope=inode_exhausted,
                catalog=catalog,
                executor=executor,
                runtime_root=runtime_root,
            )
            if inode_result.outcome == "succeeded" or inode_root.exists():
                raise RuntimeError("output inode quota or cleanup was not enforced")

            stale = _envelope(
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
                semantic_key="restart:worker:0001",
                command=STALE_COMMAND,
                deadline=datetime.now(UTC) + timedelta(minutes=5),
            )
            catalog.submit(envelope=stale, created_at=datetime.now(UTC))
            dsn_file = temp_root / "postgres.dsn"
            dsn_file.write_text(dsn + "\n", encoding="utf-8")
            dsn_file.chmod(0o600)
            crashed_worker = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.fixtures.jobs.crash_worker",
                    "--dsn-file",
                    str(dsn_file),
                    "--runtime-root",
                    str(runtime_root),
                    "--utility-image-digest",
                    JOB_IMAGE_DIGEST,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stale_container = OciJobRunner.container_name(stale.attempt_id)
            for _ in range(100):
                running = _run(
                    [
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.Running}}",
                        stale_container,
                    ],
                    check=False,
                )
                if running.returncode == 0 and running.stdout.strip() == "true":
                    break
                if crashed_worker.poll() is not None:
                    stdout, stderr = crashed_worker.communicate()
                    raise RuntimeError(
                        f"crash worker exited before OCI start: {stdout[-200:]} {stderr[-200:]}"
                    )
                time.sleep(0.05)
            else:
                crashed_worker.kill()
                crashed_worker.wait(timeout=5)
                raise RuntimeError("crash worker did not start an OCI container")
            stale_volume = OciJobRunner.output_volume_name(stale.attempt_id)
            if _run(["docker", "volume", "inspect", stale_volume], check=False).returncode:
                raise RuntimeError("crash worker did not create its output volume")
            crashed_worker.kill()
            crashed_worker.communicate(timeout=5)
            if crashed_worker.returncode == 0:
                raise RuntimeError("crash worker was not terminated forcibly")
            stale_exporter = OciJobRunner.exporter_name(stale.attempt_id)
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    stale_exporter,
                    "--label",
                    "io.roehub.runtime=JobOutputExporter/v1",
                    "--label",
                    f"io.roehub.attempt={stale.attempt_id}",
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    JOB_IMAGE_DIGEST,
                    "/bin/sleep",
                    "30",
                ]
            )
            restart = _run(
                [
                    sys.executable,
                    "-m",
                    "tests.fixtures.jobs.restart_probe",
                    "--dsn-file",
                    str(dsn_file),
                    "--now",
                    datetime.now(UTC).isoformat(),
                    "--worker-heartbeat-before",
                    (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
                    "--recovery-claimed-before",
                    (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
                    "--runtime-root",
                    str(runtime_root),
                ]
            )
            restart_payload = json.loads(restart.stdout)
            if restart_payload.get("recovered_attempts") != 1:
                raise RuntimeError("new process did not recover stale attempt")
            stale_keeper = OciJobRunner.keeper_name(stale.attempt_id)
            if (
                _run(["docker", "inspect", stale_container], check=False).returncode == 0
                or _run(["docker", "inspect", stale_keeper], check=False).returncode == 0
                or _run(["docker", "inspect", stale_exporter], check=False).returncode == 0
                or _run(["docker", "volume", "inspect", stale_volume], check=False).returncode == 0
                or (runtime_root / stale.attempt_id.hex).exists()
            ):
                raise RuntimeError("restart recovery left OCI or scratch resources behind")

            _prove_catalog_concurrency(
                catalog=catalog,
                dsn=dsn,
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
            )
            _prove_recovery_linearization(
                catalog=catalog,
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
            )

            try:
                catalog.get_state(
                    organization_id=foreign_organization_id,
                    job_id=first.job_id,
                )
            except JobRuntimeCatalogError as error:
                if error.code != "job.not_found":
                    raise
            else:
                raise RuntimeError("cross-organization job lookup was allowed")

            benchmark = _benchmark(
                runner=runner,
                root=temp_root / "benchmark",
                organization_id=organization_id,
                artifact_digest=input_manifest.manifest_digest,
            )
            cleanup = True
            print(
                json.dumps(
                    {
                        "schema": "io.roehub.job-runtime-proof/v1",
                        "status": "passed",
                        "artifact_inputs": "passed",
                        "artifact_result_publication": "passed",
                        "cancel_cleanup": "passed",
                        "cancel_finish_linearizability": "passed",
                        "capability_registry": "passed",
                        "container_crash": "passed",
                        "voluntary_exit_137": "passed",
                        "output_freeze_after_pid1_exit": "passed",
                        "orphan_exporter_cleanup": "passed",
                        "cross_organization_denial": "passed",
                        "deterministic_replay": "passed",
                        "docker_socket_denial": "passed",
                        "durable_postgresql_attempts": "passed",
                        "dynamic_plugin_reauthorization": "passed",
                        "database_immutability_guards": "passed",
                        "image_digest_binding": "passed",
                        "non_root_read_only": "passed",
                        "resource_exhaustion": "passed",
                        "output_byte_and_inode_limits": "passed",
                        "pid_limit": "passed",
                        "restart_recovery": "passed",
                        "recovery_cancel_linearizability": "passed",
                        "recovery_lease_exclusivity": "passed",
                        "retry_same_job": "passed",
                        "strategy_signal_intent_only": "passed",
                        "timeout_cleanup": "passed",
                        "benchmark": benchmark,
                    },
                    sort_keys=True,
                )
            )
    finally:
        if created:
            _run(["docker", "rm", "-f", postgres], check=False)
        leftovers = tuple(
            line
            for line in _run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    "name=roehub-job-",
                    "--format",
                    "{{.Names}}",
                ],
                check=False,
            ).stdout.splitlines()
            if line
        )
        volume_leftovers = tuple(
            line
            for line in _run(
                [
                    "docker",
                    "volume",
                    "ls",
                    "--filter",
                    "name=roehub-job-output-",
                    "--format",
                    "{{.Name}}",
                ],
                check=False,
            ).stdout.splitlines()
            if line
        )
        if leftovers or volume_leftovers:
            raise RuntimeError("job runtime proof left OCI resources behind")
        if not cleanup:
            raise RuntimeError("job runtime proof did not reach acceptance")


if __name__ == "__main__":
    main()
