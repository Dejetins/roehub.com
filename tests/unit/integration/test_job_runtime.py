from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from apps.worker.job_runtime.authority import (
    JobRuntimeAuthorityError,
    JobSubmissionService,
    TrustedRuntimeAuthority,
    TrustedRuntimeGrant,
)
from apps.worker.job_runtime.capabilities import CAPABILITY_POLICIES
from apps.worker.job_runtime.executor import JobAttemptExecutor
from apps.worker.job_runtime.oci_runner import OciJobRunner, OciRuntimeError
from trading.contexts.extensions.domain import PluginInstallation, PluginPackage
from trading.integration import (
    JobEnvelope,
    JobOutputDescriptor,
    JobResourceLimits,
    JobResultManifest,
    JobRuntimeIdentity,
    StrategyRuntimeDecision,
)
from trading.integration.job_runtime_postgres import ClaimedJobAttempt
from trading.shared_kernel.primitives import InstallationId, OrganizationId

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _envelope(*, attempt_number: int = 1, attempt_id: UUID | None = None) -> JobEnvelope:
    return JobEnvelope(
        schema="JobEnvelope/v1",
        job_id=uuid4(),
        attempt_id=attempt_id or uuid4(),
        attempt_number=attempt_number,
        organization_id=uuid4(),
        semantic_job_key="backtest:test:0001",
        capability="backtest",
        image_digest="sha256:" + "a" * 64,
        runtime=JobRuntimeIdentity(name="roehub.backtest", version="1.0.0"),
        config_snapshot={"seed": 42, "mode": "deterministic"},
        input_artifact_digests=("sha256:" + "b" * 64,),
        limits=JobResourceLimits(
            cpu_millis=500,
            memory_bytes=64 * 1024 * 1024,
            pids=64,
            wall_time_seconds=30,
            tmpfs_bytes=8 * 1024 * 1024,
            output_bytes=8 * 1024 * 1024,
        ),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        command=("/bin/sh", "-c", "printf ok > /job/output/result.txt"),
    )


def test_job_envelope_identity_is_canonical_and_retry_spec_is_stable() -> None:
    first = _envelope()
    retry = first.model_copy(
        update={
            "attempt_id": uuid4(),
            "attempt_number": 2,
            "deadline": first.deadline + timedelta(minutes=1),
        }
    )

    assert first.envelope_digest != retry.envelope_digest
    assert first.semantic_spec_digest == retry.semantic_spec_digest
    assert first.canonical_bytes() == first.canonical_bytes()


@pytest.mark.parametrize(
    "config",
    (
        {"api_token": "forbidden"},
        {"nested": {"database_dsn": "forbidden"}},
        {"ratio": 0.5},
        {"integer": 2**53},
    ),
)
def test_job_envelope_rejects_secret_shaped_or_nonportable_config(config: object) -> None:
    with pytest.raises(ValidationError):
        JobEnvelope.model_validate(
            _envelope().model_dump(mode="json", by_alias=True) | {"config_snapshot": config}
        )


def test_image_tag_and_strategy_order_payload_fail_closed() -> None:
    payload = _envelope().model_dump(mode="json", by_alias=True)
    payload["image_digest"] = "roehub/runtime:latest"
    with pytest.raises(ValidationError):
        JobEnvelope.model_validate(payload)

    with pytest.raises(ValidationError):
        StrategyRuntimeDecision.model_validate(
            {
                "kind": "intent",
                "instrument_id": "btc.usdt",
                "side": "buy",
                "strength_decimal": "0.5",
                "observed_at": datetime.now(UTC).isoformat(),
                "reason_code": "test.signal",
                "exchange_order": {"market": "mainnet"},
            }
        )


def test_all_required_capabilities_are_host_owned_and_have_no_exchange_access() -> None:
    assert set(CAPABILITY_POLICIES) == {
        "backtest",
        "optimize",
        "history_import",
        "report",
        "artifact_transform",
        "ml_training",
        "ml_inference",
        "rl_training",
        "rl_inference",
        "custom_strategy",
    }
    assert not any(policy.exchange_access for policy in CAPABILITY_POLICIES.values())
    assert CAPABILITY_POLICIES["custom_strategy"].strategy_decisions_required is True


def test_generated_job_envelope_schema_accepts_canonical_contract() -> None:
    schema = json.loads((_REPO_ROOT / "schemas/jobs/job-envelope-v1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    assert schema["x-roehub-enforcement-boundary"] == (
        "trading.integration.JobEnvelope.model_validate"
    )
    Draft202012Validator(schema).validate(_envelope().model_dump(mode="json", by_alias=True))


@pytest.mark.parametrize(
    "mutation",
    (
        {"config_snapshot": {"ratio": 0.5}},
        {"config_snapshot": {"nested": {"API_TOKEN": "forbidden"}}},
        {"input_artifact_digests": ["sha256:" + "b" * 64] * 2},
        {
            "capability": "custom_strategy",
            "runtime": {"name": "roehub.strategy", "version": "1.0.0"},
        },
    ),
)
def test_generated_envelope_schema_rejects_model_rejections(
    mutation: dict[str, object],
) -> None:
    schema = json.loads((_REPO_ROOT / "schemas/jobs/job-envelope-v1.schema.json").read_text())
    payload = _envelope().model_dump(mode="json", by_alias=True) | mutation

    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValidationError):
        JobEnvelope.model_validate(payload)


def test_generated_result_schema_enforces_outcome_and_portable_path() -> None:
    schema = json.loads(
        (_REPO_ROOT / "schemas/jobs/job-result-manifest-v1.schema.json").read_text()
    )
    result = JobResultManifest(
        schema="JobResultManifest/v1",
        job_id=uuid4(),
        attempt_id=uuid4(),
        organization_id=uuid4(),
        outcome="succeeded",
        envelope_digest="sha256:" + "a" * 64,
        output_artifact_manifest_digest="sha256:" + "b" * 64,
        outputs=(
            JobOutputDescriptor(
                path="result.json",
                digest="sha256:" + "c" * 64,
                size_bytes=2,
                media_type="application/json",
            ),
        ),
        strategy_decisions=(),
        completed_at=datetime.now(UTC),
    )
    validator = Draft202012Validator(schema)
    payload = result.model_dump(mode="json", by_alias=True)

    assert validator.is_valid(payload)
    assert not validator.is_valid(payload | {"output_artifact_manifest_digest": None})
    payload["outputs"][0]["path"] = "../result.json"
    assert not validator.is_valid(payload)


def test_model_validation_is_the_boundary_for_annotated_aggregate_limits() -> None:
    envelope_schema = json.loads(
        (_REPO_ROOT / "schemas/jobs/job-envelope-v1.schema.json").read_text()
    )
    envelope_payload = _envelope().model_dump(mode="json", by_alias=True)
    envelope_payload["command"] = ["x" * 4096] * 9

    assert Draft202012Validator(envelope_schema).is_valid(envelope_payload)
    with pytest.raises(ValidationError):
        JobEnvelope.model_validate(envelope_payload)

    result_schema = json.loads(
        (_REPO_ROOT / "schemas/jobs/job-result-manifest-v1.schema.json").read_text()
    )
    descriptor = {
        "path": "duplicate.json",
        "digest": "sha256:" + "c" * 64,
        "size_bytes": 2,
        "media_type": "application/json",
    }
    result_payload = {
        "schema": "JobResultManifest/v1",
        "job_id": str(uuid4()),
        "attempt_id": str(uuid4()),
        "organization_id": str(uuid4()),
        "outcome": "succeeded",
        "envelope_digest": "sha256:" + "a" * 64,
        "output_artifact_manifest_digest": "sha256:" + "b" * 64,
        "outputs": [descriptor, descriptor],
        "strategy_decisions": [],
        "completed_at": datetime.now(UTC).isoformat(),
        "exit_code": 0,
        "error_code": None,
    }

    assert Draft202012Validator(result_schema).is_valid(result_payload)
    with pytest.raises(ValidationError):
        JobResultManifest.model_validate(result_payload)


def test_runtime_authority_binds_capability_image_command_and_plugin_package() -> None:
    payload = _envelope().model_dump(mode="json", by_alias=True)
    payload.update(
        {
            "capability": "custom_strategy",
            "runtime": {
                "name": "roehub.strategy",
                "version": "1.0.0",
                "plugin_package_digest": "1" * 64,
            },
        }
    )
    envelope = JobEnvelope.model_validate(payload)
    package = PluginPackage(
        package_id=uuid4(),
        installation_id=InstallationId(uuid4()),
        plugin_id="strategy.fixture",
        version=envelope.runtime.version,
        package_digest=cast(str, envelope.runtime.plugin_package_digest),
        image_reference="roehub/strategy@" + envelope.image_digest,
        image_digest=envelope.image_digest,
        publisher_key_id="publisher.fixture",
        publisher_public_key_b64="cHVibGljLWtleQ==",
        publisher_key_fingerprint_sha256="1" * 64,
        manifest={},
        created_at=datetime.now(UTC),
    )
    installation = PluginInstallation(
        plugin_installation_id=uuid4(),
        installation_id=package.installation_id,
        organization_id=OrganizationId(envelope.organization_id),
        plugin_id=package.plugin_id,
        package_id=package.package_id,
        previous_package_id=None,
        granted_permissions=(),
        status="enabled",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    trust_root = {
        cast(str, package.publisher_key_id): cast(str, package.publisher_key_fingerprint_sha256)
    }
    authority = TrustedRuntimeAuthority(
        grants=(
            TrustedRuntimeGrant.for_signed_plugin(
                envelope=envelope,
                package=package,
                installation=installation,
                trusted_publisher_fingerprints=trust_root,
            ),
        ),
        plugin_trust_resolver=lambda _envelope: (package, installation, trust_root),
    )

    authority.authorize(envelope)
    for changed in (
        envelope.model_copy(update={"image_digest": "sha256:" + "f" * 64}),
        envelope.model_copy(update={"command": ("/bin/false",)}),
        envelope.model_copy(
            update={
                "runtime": envelope.runtime.model_copy(update={"plugin_package_digest": "2" * 64})
            }
        ),
    ):
        with pytest.raises(JobRuntimeAuthorityError) as raised:
            authority.authorize(changed)
        assert raised.value.code == "job.runtime_not_trusted"

    with pytest.raises(JobRuntimeAuthorityError) as raised:
        TrustedRuntimeGrant.for_signed_plugin(
            envelope=envelope,
            package=replace(
                package,
                publisher_key_id=None,
                publisher_public_key_b64=None,
                publisher_key_fingerprint_sha256=None,
            ),
            installation=installation,
            trusted_publisher_fingerprints={},
        )
    assert raised.value.code == "job.plugin_package_unsigned"

    trust_root.clear()
    with pytest.raises(JobRuntimeAuthorityError) as raised:
        authority.authorize(envelope)
    assert raised.value.code == "job.plugin_publisher_untrusted"


def test_docker_control_timeout_with_ambiguous_cleanup_stays_recoverable(
    tmp_path: Path,
) -> None:
    class _TimeoutRunner:
        def run(self, *_args: object, **_kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd="docker", timeout=0.05)

    runner = OciJobRunner(
        utility_image_digest="sha256:" + "f" * 64,
        command_runner=cast(Any, _TimeoutRunner()),
        docker_command=("/bin/sh", "-c", "sleep 1"),
        control_timeout_seconds=0.05,
    )

    with pytest.raises(OciRuntimeError) as raised:
        runner.run(
            envelope=_envelope(),
            input_root=tmp_path / "input",
            output_root=tmp_path / "output",
        )

    assert raised.value.code == "job.cleanup_boundary_failed"


def test_cleanup_boundary_failure_stays_running_for_recovery(tmp_path: Path) -> None:
    normalized = OciJobRunner._cleanup_failure(OciRuntimeError(code="job.docker_control_timeout"))
    assert normalized.code == "job.cleanup_boundary_failed"

    envelope = _envelope()
    catalog = Mock()
    runner = Mock()
    runner.run.side_effect = OciRuntimeError(code="job.cleanup_boundary_failed")
    artifacts = Mock()
    artifacts.get_manifest.return_value = Mock(entries=())
    authority = TrustedRuntimeAuthority(grants=(TrustedRuntimeGrant.for_builtin(envelope),))
    executor = JobAttemptExecutor(
        catalog=cast(Any, catalog),
        runner=cast(Any, runner),
        artifacts=cast(Any, artifacts),
        result_signing_key=Ed25519PrivateKey.generate(),
        result_signing_key_id="stage15.test",
        worker_id="stage15.test.worker",
        authority=authority,
        runtime_root=tmp_path / "runtime",
    )

    with pytest.raises(OciRuntimeError) as raised:
        executor.execute(
            claimed=ClaimedJobAttempt(
                envelope=envelope,
                worker_id="stage15.test.worker",
                claimed_at=datetime.now(UTC),
            )
        )

    assert raised.value.code == "job.cleanup_boundary_failed"
    catalog.finish_attempt.assert_not_called()


def test_submission_service_rejects_untrusted_envelope_before_catalog_write() -> None:
    envelope = _envelope()
    catalog = Mock()
    catalog.submit.return_value = envelope.job_id
    service = JobSubmissionService(
        catalog=cast(Any, catalog),
        authority=TrustedRuntimeAuthority(grants=(TrustedRuntimeGrant.for_builtin(envelope),)),
    )

    assert service.submit(envelope=envelope, created_at=datetime.now(UTC)) == envelope.job_id
    catalog.submit.assert_called_once()
    with pytest.raises(JobRuntimeAuthorityError):
        service.submit(
            envelope=envelope.model_copy(update={"command": ("/bin/false",)}),
            created_at=datetime.now(UTC),
        )
    catalog.submit.assert_called_once()
