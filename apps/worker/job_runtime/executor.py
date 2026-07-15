from __future__ import annotations

import base64
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apps.worker.job_runtime.authority import (
    JobRuntimeAuthorityError,
    TrustedRuntimeAuthority,
)
from apps.worker.job_runtime.capabilities import capability_policy
from apps.worker.job_runtime.oci_runner import (
    OciExecutionResult,
    OciJobRunner,
    OciRuntimeError,
)
from trading.contexts.backtest_artifacts.application import ArtifactStoreService
from trading.contexts.backtest_artifacts.domain import ArtifactStoreError
from trading.integration import (
    ArtifactBlobDescriptor,
    ArtifactBundleSignature,
    ArtifactManifest,
    ArtifactManifestEntry,
    JobEnvelope,
    JobResultManifest,
    sha256_job_payload,
)
from trading.integration.job_runtime_postgres import (
    ClaimedJobAttempt,
    PostgresJobRuntimeCatalog,
)
from trading.shared_kernel.primitives import OrganizationId


class JobAttemptExecutor:
    """Host-owned bridge from durable attempt to OCI and signed ArtifactStore result."""

    def __init__(
        self,
        *,
        catalog: PostgresJobRuntimeCatalog,
        runner: OciJobRunner,
        artifacts: ArtifactStoreService,
        result_signing_key: Ed25519PrivateKey,
        result_signing_key_id: str,
        worker_id: str,
        authority: TrustedRuntimeAuthority,
        runtime_root: Path,
    ) -> None:
        self._catalog = catalog
        self._runner = runner
        self._artifacts = artifacts
        self._signing_key = result_signing_key
        self._signing_key_id = result_signing_key_id
        self._worker_id = worker_id
        self._authority = authority
        root_candidate = runtime_root.expanduser()
        if root_candidate.is_symlink():
            raise ValueError("job executor runtime root is unsafe")
        root = root_candidate.resolve()
        if root == Path("/"):
            raise ValueError("job executor runtime root is unsafe")
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        self._runtime_root = root

    def execute(
        self,
        *,
        claimed: ClaimedJobAttempt,
    ) -> JobResultManifest:
        envelope = claimed.envelope
        attempt_root = self._runtime_root / envelope.attempt_id.hex
        input_root = attempt_root / "input"
        output_root = attempt_root / "output"
        try:
            if attempt_root.is_symlink():
                raise OciRuntimeError(code="job.attempt_root_unsafe")
            self._authority.authorize(envelope)
            capability_policy(envelope.capability)
            self._prepare_artifact_inputs(envelope=envelope, input_root=input_root)
            execution = self._runner.run(
                envelope=envelope,
                input_root=input_root,
                output_root=output_root,
                cancellation_requested=lambda: self._catalog.is_cancel_requested(
                    organization_id=envelope.organization_id,
                    job_id=envelope.job_id,
                ),
                heartbeat=lambda: self._catalog.heartbeat(
                    organization_id=envelope.organization_id,
                    attempt_id=envelope.attempt_id,
                    worker_id=self._worker_id,
                    now=datetime.now(UTC),
                ),
            )
        except (
            ArtifactStoreError,
            JobRuntimeAuthorityError,
            OciRuntimeError,
            OSError,
            ValueError,
        ) as error:
            error_code = getattr(error, "code", "job.input_materialization_failed")
            if error_code in {
                "job.cleanup_boundary_failed",
                "job.container_cleanup_failed",
                "job.container_cleanup_identity_mismatch",
                "job.output_volume_cleanup_failed",
                "job.output_volume_cleanup_identity_mismatch",
            }:
                raise
            result = self._failure_result(envelope=envelope, error_code=error_code)
            result = self._catalog.finish_attempt(
                envelope=envelope,
                worker_id=self._worker_id,
                result=result,
            )
            self._cleanup_attempt_root(attempt_root)
            return result
        artifact_digest: str | None = None
        if execution.outcome == "succeeded":
            try:
                artifact_digest = self._publish_outputs(
                    envelope=envelope,
                    execution=execution,
                    output_root=output_root,
                )
            except (ArtifactStoreError, OSError, ValueError):
                result = self._failure_result(
                    envelope=envelope,
                    error_code="job.artifact_publication_failed",
                )
                result = self._catalog.finish_attempt(
                    envelope=envelope,
                    worker_id=self._worker_id,
                    result=result,
                )
                self._cleanup_attempt_root(attempt_root)
                return result
        result = JobResultManifest(
            schema="JobResultManifest/v1",
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            organization_id=envelope.organization_id,
            outcome=execution.outcome,
            envelope_digest=envelope.envelope_digest,
            output_artifact_manifest_digest=artifact_digest,
            outputs=execution.outputs,
            strategy_decisions=execution.strategy_decisions,
            completed_at=execution.completed_at,
            exit_code=execution.exit_code,
            error_code=execution.error_code,
        )
        actual_result = self._catalog.finish_attempt(
            envelope=envelope,
            worker_id=self._worker_id,
            result=result,
        )
        if (
            actual_result.output_artifact_manifest_digest != result.output_artifact_manifest_digest
            and result.output_artifact_manifest_digest is not None
        ):
            organization_id = OrganizationId(envelope.organization_id)
            self._artifacts.retire_manifest(
                organization_id=organization_id,
                manifest_digest=result.output_artifact_manifest_digest,
            )
            self._artifacts.garbage_collect()
        self._cleanup_attempt_root(attempt_root)
        return actual_result

    def _prepare_artifact_inputs(self, *, envelope: JobEnvelope, input_root: Path) -> None:
        root = input_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        if any(root.iterdir()):
            raise ValueError("job input root must be empty before artifact materialization")
        organization_id = OrganizationId(envelope.organization_id)
        for index, manifest_digest in enumerate(envelope.input_artifact_digests):
            manifest = self._artifacts.get_manifest(
                organization_id=organization_id,
                manifest_digest=manifest_digest,
            )
            for entry in manifest.entries:
                source = self._artifacts.materialize_entry(
                    organization_id=organization_id,
                    manifest_digest=manifest_digest,
                    path=entry.path,
                    cache_key=f"job:{envelope.attempt_id}:{index}:{entry.path}",
                )
                target = root / "artifacts" / str(index) / entry.path
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
                target.chmod(0o440)

    @staticmethod
    def _failure_result(*, envelope: JobEnvelope, error_code: str) -> JobResultManifest:
        return JobResultManifest(
            schema="JobResultManifest/v1",
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            organization_id=envelope.organization_id,
            outcome="failed",
            envelope_digest=envelope.envelope_digest,
            output_artifact_manifest_digest=None,
            outputs=(),
            strategy_decisions=(),
            completed_at=datetime.now(UTC),
            exit_code=None,
            error_code=error_code,
        )

    def _publish_outputs(
        self,
        *,
        envelope: JobEnvelope,
        execution: OciExecutionResult,
        output_root: Path,
    ) -> str:
        if not execution.outputs:
            raise ValueError("successful job must produce at least one output")
        organization_id = OrganizationId(envelope.organization_id)
        with TemporaryDirectory(prefix="roehub-job-result-") as temporary:
            bundle_root = Path(temporary)
            payload_root = bundle_root / "payload"
            entries: list[ArtifactManifestEntry] = []
            for output in execution.outputs:
                source = output_root / output.path
                payload = source.read_bytes()
                if (
                    len(payload) != output.size_bytes
                    or sha256_job_payload(payload) != output.digest
                ):
                    raise ValueError("job output changed before artifact publication")
                target = payload_root / output.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                entries.append(
                    ArtifactManifestEntry(
                        path=output.path,
                        blob=ArtifactBlobDescriptor(
                            digest=output.digest,
                            size_bytes=output.size_bytes,
                            media_type=output.media_type,
                        ),
                    )
                )
            placeholder = ArtifactBundleSignature(
                key_id=self._signing_key_id,
                value_b64=base64.b64encode(bytes(64)).decode(),
            )
            unsigned = ArtifactManifest(
                schema="ArtifactManifest/v1",
                bundle_id=f"job.{envelope.job_id.hex}",
                name=f"Job result {envelope.job_id}",
                version=f"1.0.{envelope.attempt_number - 1}",
                created_at=execution.completed_at,
                entries=tuple(entries),
                metadata={
                    "job_id": str(envelope.job_id),
                    "attempt_id": str(envelope.attempt_id),
                    "capability": envelope.capability,
                    "envelope_digest": envelope.envelope_digest,
                },
                signature=placeholder,
            )
            signature = ArtifactBundleSignature(
                key_id=self._signing_key_id,
                value_b64=base64.b64encode(
                    self._signing_key.sign(unsigned.signed_bytes())
                ).decode(),
            )
            manifest = unsigned.model_copy(update={"signature": signature})
            (bundle_root / "artifact.bundle.json").write_text(
                json.dumps(
                    manifest.model_dump(mode="json", by_alias=True),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            published = self._artifacts.install_bundle(
                organization_id=organization_id,
                bundle_root=bundle_root,
            )
            if published.manifest_digest != manifest.manifest_digest:
                raise ValueError("job result artifact digest changed during publication")
            shutil.rmtree(payload_root, ignore_errors=True)
            return published.manifest_digest

    def _cleanup_attempt_root(self, attempt_root: Path) -> None:
        if attempt_root.is_symlink() or attempt_root.parent != self._runtime_root:
            return
        resolved = attempt_root.expanduser().resolve()
        if (
            resolved.parent == self._runtime_root
            and resolved.exists()
            and resolved.is_dir()
            and not resolved.is_symlink()
        ):
            shutil.rmtree(resolved)


__all__ = ["JobAttemptExecutor"]
