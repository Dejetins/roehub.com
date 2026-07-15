"""Real Docker proof for installation backup, fresh restore, upgrade, and rollback."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence, TypedDict
from uuid import UUID

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apps.control_agent.auth import ServiceIdentityAuthorizer
from apps.control_agent.backup_backend import (
    InstallationBackupControlBackend,
    RecoveryControlBackend,
)
from apps.control_agent.server import (
    start_control_agent_server,
    stop_control_agent_server,
)
from apps.roehubctl.main.main import main as roehubctl_main
from infra.openbao.verify_runtime import verify as verify_openbao_runtime
from tools.backup import (
    BackupBundleError,
    BackupSource,
    create_backup,
    rollback_from_backup,
    upgrade_from_backup,
    verify_backup,
)
from trading.contexts.operations import (
    BackupCaptureEntry,
    BackupPolicySource,
    BackupStateOwner,
    ControlOperationError,
    ControlOperationService,
    InstallationBackupManifest,
    InstallationBackupPolicy,
    InstallationCaptureRecord,
    InstallationReleasePolicy,
    OperationAction,
    OperationRequest,
    ReleaseTransitionRule,
)
from trading.contexts.operations.adapters import AppendOnlyOperationJournal

ROOT = Path(__file__).resolve().parents[2]
STORAGE_COMPOSE = ROOT / "infra/docker/storage-embedded.compose.yml"
OBSERVABILITY_COMPOSE = ROOT / "configs/installation/generated/base/compose.yaml"
OBSERVABILITY_OVERRIDE = ROOT / "tests/fixtures/observability-runtime-override.yaml"
PROFILE_ROOT = ROOT / "configs/installation/generated/base"
RELEASE_MANIFEST = ROOT / "tools/release/release-metadata.json"
SERVICE_CONFIG = PROFILE_ROOT / "service-config.json"
STAGE20_EVIDENCE = (
    ROOT
    / "docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports"
    / "evidence/20-observability-runtime-proof.json"
)
DEFAULT_EVIDENCE = (
    ROOT
    / "docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports"
    / "evidence/21-backup-restore-upgrade-runtime-proof.json"
)
MONITORING_SERVICES = (
    "alertmanager",
    "blackbox",
    "grafana",
    "loki",
    "operational-health",
    "prometheus",
)
_RUNTIME_IMAGE_OVERRIDE_ENV = "ROEHUB_RUNTIME_PROOF_IMAGE_OVERRIDE"
_OPENBAO_IMAGE_OVERRIDE_ENV = "ROEHUB_RUNTIME_PROOF_OPENBAO_IMAGE_OVERRIDE"


class RecoveryRuntimeProofError(RuntimeError):
    """Sanitized real-boundary failure."""


class MonitoringState(TypedDict):
    """Sanitized monitoring state kept available during recovery."""

    running: list[str]
    operational_health_ready: bool
    web_api_running: bool


class RuntimeStateCoordinator:
    """Real disposable state-owner capture and import used by the Stage 21 proof."""

    def __init__(
        self,
        *,
        source_project: str,
        target_project: str,
        update_project: str,
        rollback_project: str,
        environ: dict[str, str],
        temporary: Path,
    ) -> None:
        self._source_project = source_project
        self._target_project = target_project
        self._update_project = update_project
        self._rollback_project = rollback_project
        self._environ = environ
        self._temporary = temporary
        self.openbao_proof: dict[str, Any] = {}
        self.restore_comparison: dict[str, object] = {}
        self.current_release = "0.0.0"
        self.fail_next_update = True
        self.release_lifecycle: dict[str, object] = {}

    def capture(self, policy: InstallationBackupPolicy) -> InstallationCaptureRecord:
        expected = {source.filename for source in policy.sources}
        actual = {path.name for path in policy.source_root.iterdir()}
        if actual:
            if actual != expected or not policy.capture_record_file.is_file():
                raise RecoveryRuntimeProofError("capture staging cannot be resumed safely")
            capture = InstallationCaptureRecord.model_validate_json(
                policy.capture_record_file.read_bytes()
            )
            captured = {entry.owner: entry for entry in capture.entries}
            if any(
                _file_sha256(policy.source_root / source.filename)
                != captured[source.owner].plaintext_sha256
                for source in policy.sources
            ):
                raise RecoveryRuntimeProofError("resumed capture digest changed")
            return capture
        quiesce_started_at = datetime.now(UTC)
        self.openbao_proof = _capture_sources(
            self._source_project,
            environ=self._environ,
            source_root=policy.source_root,
            age_identity_file=policy.age_identity_file,
            age_recipient_file=policy.age_recipient_file,
        )
        source_rows = _policy_sources()
        capture_times = {
            source.owner: datetime.fromtimestamp(
                (policy.source_root / source.filename).stat().st_mtime,
                tz=UTC,
            )
            for source in source_rows
        }
        quiesce_completed_at = datetime.now(UTC)
        capture = InstallationCaptureRecord(
            installation_fingerprint=policy.installation_fingerprint,
            source_release_version=policy.release_version,
            quiesce_started_at=quiesce_started_at,
            quiesce_completed_at=quiesce_completed_at,
            entries=tuple(
                BackupCaptureEntry(
                    owner=source.owner,
                    captured_at=capture_times[source.owner],
                    plaintext_sha256=_file_sha256(
                        policy.source_root / source.filename
                    ),
                )
                for source in source_rows
            ),
        )
        policy.capture_record_file.write_text(
            capture.model_dump_json(by_alias=True),
            encoding="utf-8",
        )
        policy.capture_record_file.chmod(0o600)
        return capture

    def restore(
        self,
        policy: InstallationBackupPolicy,
        restored_root: Path,
        manifest: InstallationBackupManifest,
    ) -> dict[str, object]:
        self.restore_comparison = self._apply_to_project(
            policy=policy,
            project=self._target_project,
            restored_root=restored_root,
            manifest=manifest,
        )
        return {
            "status": "ready",
            "state_owner_count": len(manifest.entries),
            "comparison": self.restore_comparison,
        }

    def update(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
        bundle: Path,
        transition: ReleaseTransitionRule,
    ) -> dict[str, object]:
        assert request.release_version is not None
        fail_before_commit = self.fail_next_update
        self.fail_next_update = False
        try:
            outcome = upgrade_from_backup(
                bundle=bundle,
                target_root=self._temporary / "upgrade-target",
                age_identity_file=policy.age_identity_file,
                verification_public_key_file=policy.verification_public_key_file,
                target_release_version=request.release_version,
                irreversible=not transition.reversible,
                forward_recovery_plan=transition.forward_recovery_plan_sha256,
                fail_before_commit=fail_before_commit,
                apply_restored_state=lambda root, manifest: self._ready_result(
                    policy=policy,
                    project=self._update_project,
                    restored_root=root,
                    manifest=manifest,
                ),
                operation_id=str(request.operation_id),
            )
        except BackupBundleError as error:
            if error.code == "upgrade.forward_recovery_plan_required":
                self.release_lifecycle["irreversible_migration_guard"] = (
                    "forward_recovery_plan_required"
                )
            else:
                self.release_lifecycle["injected_failure"] = (
                    "failed_before_atomic_release_commit"
                )
            raise
        self.current_release = request.release_version
        self.release_lifecycle.update(
            {
                "from_release": outcome["from_release"],
                "to_release": outcome["to_release"],
                "preupgrade_backup_verified": outcome["preupgrade_backup_verified"],
                "resume_after_failure": outcome["status"],
                "observed_upgrade_seconds": outcome["observed_upgrade_seconds"],
            }
        )
        return outcome

    def rollback(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
        bundle: Path,
        failed_release_version: str,
        transition: ReleaseTransitionRule,
    ) -> dict[str, object]:
        assert request.release_version is not None
        outcome = rollback_from_backup(
            bundle=bundle,
            target_root=self._temporary / "rollback-target",
            age_identity_file=policy.age_identity_file,
            verification_public_key_file=policy.verification_public_key_file,
            failed_release_version=failed_release_version,
            apply_restored_state=lambda root, manifest: self._ready_result(
                policy=policy,
                project=self._rollback_project,
                restored_root=root,
                manifest=manifest,
            ),
            operation_id=str(request.operation_id),
        )
        self.current_release = request.release_version
        self.release_lifecycle.update(
            {
                "rollback_release": outcome["restored_release"],
                "rollback_state_owner_count": outcome["state_owner_count"],
                "observed_rollback_seconds": outcome["observed_rollback_seconds"],
            }
        )
        return outcome

    def reconcile_release(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
    ) -> dict[str, object] | None:
        del policy
        if request.action is OperationAction.UPDATE:
            result_path = self._temporary / "upgrade-target/installed-release.json"
        elif request.action is OperationAction.ROLLBACK:
            result_path = self._temporary / "rollback-target/rollback-result.json"
        else:
            return None
        if not result_path.is_file():
            return None
        result = _read_json(result_path)
        if result.get("operation_id") != str(request.operation_id):
            return None
        return result

    def _ready_result(
        self,
        *,
        policy: InstallationBackupPolicy,
        project: str,
        restored_root: Path,
        manifest: InstallationBackupManifest,
    ) -> dict[str, object]:
        comparison = self._apply_to_project(
            policy=policy,
            project=project,
            restored_root=restored_root,
            manifest=manifest,
        )
        return {
            "status": "ready",
            "state_owner_count": len(manifest.entries),
            "comparison": comparison,
        }

    def _apply_to_project(
        self,
        *,
        policy: InstallationBackupPolicy,
        project: str,
        restored_root: Path,
        manifest: InstallationBackupManifest,
    ) -> dict[str, object]:
        _storage_up(project, environ=self._environ)
        return _restore_and_compare_storage(
            source_project=self._source_project,
            target_project=project,
            restored_root=restored_root,
            environ=self._environ,
            manifest=manifest,
            temporary=self._temporary,
            age_identity_file=policy.age_identity_file,
        )


class _UnavailableRuntimeBackend:
    def execute(self, request: Any) -> Any:
        raise ControlOperationError(code="operation.handler_unavailable")

    def reconcile(self, request: Any) -> Any:
        raise ControlOperationError(code="operation.effect_unknown")


def verify_runtime(
    *,
    project_prefix: str,
    image_override: Path | None = None,
) -> dict[str, object]:
    source_project = f"{project_prefix}-source"
    target_project = f"{project_prefix}-target"
    update_project = f"{project_prefix}-update"
    rollback_project = f"{project_prefix}-rollback"
    monitoring_project = f"{project_prefix}-monitoring"
    projects = (
        source_project,
        target_project,
        update_project,
        rollback_project,
        monitoring_project,
    )
    environ = _storage_environment()
    if image_override is not None:
        resolved_image_override = image_override.expanduser().resolve()
        if not resolved_image_override.is_file():
            raise RecoveryRuntimeProofError("runtime image override is unavailable")
        environ[_RUNTIME_IMAGE_OVERRIDE_ENV] = str(resolved_image_override)
    control_server: Any | None = None
    control_thread: Any | None = None
    safe_result: dict[str, object] | None = None
    cleanup: dict[str, object] = {"status": "pending"}
    with tempfile.TemporaryDirectory(prefix="roehub-stage21-") as temporary_name:
        temporary = Path(temporary_name).resolve()
        runtime_image_override = environ.get(_RUNTIME_IMAGE_OVERRIDE_ENV)
        if runtime_image_override:
            environ[_OPENBAO_IMAGE_OVERRIDE_ENV] = str(
                _write_scoped_image_override(
                    source=Path(runtime_image_override),
                    destination=temporary / "openbao-image-override.json",
                    services=("openbao",),
                )
            )
        try:
            _report_phase("source-storage")
            _storage_up(source_project, environ=environ)
            _seed_storage(source_project, environ=environ)
            source_root = temporary / "sources"
            source_root.mkdir(mode=0o700)
            materials = _operator_materials(temporary)
            policy_path = _write_backup_policy(
                temporary=temporary,
                source_root=source_root,
                materials=materials,
            )
            _report_phase("recovery-observability")
            _observability_up(monitoring_project, environ=environ)
            monitoring_before = _monitoring_state(monitoring_project, environ=environ)
            if set(monitoring_before["running"]) != set(MONITORING_SERVICES):
                raise RecoveryRuntimeProofError("monitoring services are not independently ready")
            if monitoring_before["web_api_running"]:
                raise RecoveryRuntimeProofError("Web or API unexpectedly runs in recovery proof")

            coordinator = RuntimeStateCoordinator(
                source_project=source_project,
                target_project=target_project,
                update_project=update_project,
                rollback_project=rollback_project,
                environ=environ,
                temporary=temporary,
            )
            _report_phase("control-agent")
            control = _start_control_agent(
                temporary=temporary,
                policy_path=policy_path,
                state_coordinator=coordinator,
                current_release=lambda: coordinator.current_release,
            )
            control_server = control["server"]
            control_thread = control["thread"]
            socket_path = control["socket"]
            owner_identity_file = control["owner_identity_file"]
            cancelled_backup_operation = UUID("00000000-0000-4000-8000-000000002101")
            backup_operation = UUID("00000000-0000-4000-8000-000000002102")
            cancelled_restore_operation = UUID("00000000-0000-4000-8000-000000002103")
            restore_operation = UUID("00000000-0000-4000-8000-000000002104")
            _report_phase("roehubctl-backup-restore-cancel")
            bundle = temporary / "backups/stage21-current"
            restored_root = temporary / "restores/stage21-current"
            with ThreadPoolExecutor(max_workers=2) as executor:
                backup_cancel_future = executor.submit(
                    _roehubctl_when_progress,
                    progress_path=bundle / "backup-progress.json",
                    socket_path=socket_path,
                    owner_identity_file=owner_identity_file,
                    command="backup-cancel",
                    subject_id=str(cancelled_backup_operation),
                    operation_id=UUID("00000000-0000-4000-8000-000000002105"),
                )
                cancelled_backup_future = executor.submit(
                    _roehubctl,
                    socket_path=socket_path,
                    owner_identity_file=owner_identity_file,
                    command="backup",
                    subject_id="stage21-current",
                    operation_id=cancelled_backup_operation,
                )
                cancelled_backup_cli = cancelled_backup_future.result(timeout=180)
                backup_cancel_cli = backup_cancel_future.result(timeout=180)
            if cancelled_backup_cli.get("detail_code") != "backup.cancelled":
                raise RecoveryRuntimeProofError(
                    "concurrent backup cancellation was not observed: "
                    f"state={cancelled_backup_cli.get('state')},"
                    f"detail={cancelled_backup_cli.get('detail_code')}"
                )
            backup_cli = _roehubctl(
                socket_path=socket_path,
                owner_identity_file=owner_identity_file,
                command="backup",
                subject_id="stage21-current",
                operation_id=backup_operation,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                restore_cancel_future = executor.submit(
                    _roehubctl_when_progress,
                    progress_path=restored_root / "restore-progress.json",
                    socket_path=socket_path,
                    owner_identity_file=owner_identity_file,
                    command="restore-cancel",
                    subject_id=str(cancelled_restore_operation),
                    operation_id=UUID("00000000-0000-4000-8000-000000002106"),
                )
                cancelled_restore_future = executor.submit(
                    _roehubctl,
                    socket_path=socket_path,
                    owner_identity_file=owner_identity_file,
                    command="restore",
                    subject_id="stage21-current",
                    operation_id=cancelled_restore_operation,
                )
                cancelled_restore_cli = cancelled_restore_future.result(timeout=180)
                restore_cancel_cli = restore_cancel_future.result(timeout=180)
            if cancelled_restore_cli.get("detail_code") != "restore.cancelled":
                raise RecoveryRuntimeProofError(
                    "concurrent restore cancellation was not observed: "
                    f"state={cancelled_restore_cli.get('state')},"
                    f"detail={cancelled_restore_cli.get('detail_code')}"
                )
            restore_cli = _roehubctl(
                socket_path=socket_path,
                owner_identity_file=owner_identity_file,
                command="restore",
                subject_id="stage21-current",
                operation_id=restore_operation,
            )
            failed_update_cli = _roehubctl(
                socket_path=socket_path,
                owner_identity_file=owner_identity_file,
                command="update",
                release_version="0.1.0",
                operation_id=UUID("00000000-0000-4000-8000-000000002107"),
            )
            resumed_update_cli = _roehubctl(
                socket_path=socket_path,
                owner_identity_file=owner_identity_file,
                command="update",
                release_version="0.1.0",
                operation_id=UUID("00000000-0000-4000-8000-000000002108"),
            )
            rollback_cli = _roehubctl(
                socket_path=socket_path,
                owner_identity_file=owner_identity_file,
                command="rollback",
                release_version="0.0.0",
                operation_id=UUID("00000000-0000-4000-8000-000000002109"),
            )
            irreversible_cli = _roehubctl(
                socket_path=socket_path,
                owner_identity_file=owner_identity_file,
                command="update",
                release_version="0.2.0",
                operation_id=UUID("00000000-0000-4000-8000-000000002110"),
            )
            monitoring_during = _monitoring_state(monitoring_project, environ=environ)
            if set(monitoring_during["running"]) != set(MONITORING_SERVICES):
                raise RecoveryRuntimeProofError("monitoring disappeared during roehubctl recovery")
            if monitoring_during["web_api_running"]:
                raise RecoveryRuntimeProofError("Web or API started during recovery proof")

            manifest = verify_backup(
                bundle=bundle,
                verification_public_key_file=materials["verification_file"],
            )
            restore_result = _read_json(restored_root / "restore-result.json")
            _report_phase("fresh-target-restore")
            comparison = coordinator.restore_comparison
            _report_phase("upgrade-rollback-lifecycle")
            lifecycle = {
                "fixture": "versioned-unpublished-n-minus-one",
                **coordinator.release_lifecycle,
                "failed_update_state": failed_update_cli.get("state"),
                "failed_update_detail": failed_update_cli.get("detail_code"),
                "resumed_update_state": resumed_update_cli.get("state"),
                "rollback_state": rollback_cli.get("state"),
                "irreversible_update_state": irreversible_cli.get("state"),
                "irreversible_migration_guard": irreversible_cli.get("detail_code"),
            }
            backup_progress = _read_json(bundle / "backup-progress.json")
            cancellation_markers = (
                temporary / "cancel" / f"{cancelled_backup_operation}.cancel",
                temporary / "cancel" / f"{cancelled_restore_operation}.cancel",
            )
            if backup_progress.get("state") != "completed" or not all(
                marker.is_file() for marker in cancellation_markers
            ):
                raise RecoveryRuntimeProofError("progress or cancellation evidence is incomplete")
            backup_cli_state = backup_cli.get("state")
            restore_cli_state = restore_cli.get("state")
            backup_cancel_state = backup_cancel_cli.get("state")
            restore_cancel_state = restore_cancel_cli.get("state")
            if (
                backup_cli_state,
                restore_cli_state,
                backup_cancel_state,
                restore_cancel_state,
            ) != (
                "succeeded",
                "succeeded",
                "succeeded",
                "succeeded",
            ):
                raise RecoveryRuntimeProofError("roehubctl recovery operation failed")
            if (
                failed_update_cli.get("state") != "failed"
                or failed_update_cli.get("detail_code")
                != "upgrade.injected_failure_before_commit"
                or resumed_update_cli.get("state") != "succeeded"
                or rollback_cli.get("state") != "succeeded"
                or irreversible_cli.get("state") != "failed"
                or irreversible_cli.get("detail_code")
                != "upgrade.forward_recovery_plan_required"
            ):
                raise RecoveryRuntimeProofError("roehubctl release lifecycle failed")
            safe_result = {
                "schema": "io.roehub.installation-recovery-runtime-proof/v1alpha1",
                "stage": "21",
                "status": "passed",
                "proof_boundary": (
                    "N/A: disposable local Docker projects, generated fixture data, "
                    "operator-generated temporary keys, no production restore"
                ),
                "source_project": source_project,
                "target_project": target_project,
                "monitoring_project": monitoring_project,
                "state_owner_coverage": [entry.owner.value for entry in manifest.entries],
                "encrypted_entries": sum(entry.encrypted for entry in manifest.entries),
                "manifest_signature": "verified",
                "source_target_separation": "passed",
                "roehubctl_without_web_api": {
                    "backup": backup_cli_state,
                    "restore": restore_cli_state,
                    "backup_cancellation": {
                        "request": backup_cancel_state,
                        "target": cancelled_backup_cli.get("state"),
                        "resume": backup_cli_state,
                    },
                    "restore_cancellation": {
                        "request": restore_cancel_state,
                        "target": cancelled_restore_cli.get("state"),
                        "resume": restore_cli_state,
                    },
                    "control_socket": "unix",
                    "web_api_running": False,
                },
                "monitoring_during_recovery": monitoring_during,
                "restore_comparison": comparison,
                "openbao": {
                    "encrypted_raft_snapshot": coordinator.openbao_proof.get("encrypted_backup"),
                    "fresh_volume_force_restore": coordinator.openbao_proof.get(
                        "fresh_volume_force_restore"
                    ),
                    "fresh_storage_guard": coordinator.openbao_proof.get("fresh_storage_guard"),
                    "forbidden_output_scan": coordinator.openbao_proof.get("forbidden_output_scan"),
                    "cleanup": coordinator.openbao_proof.get("cleanup"),
                },
                "backup_progress": backup_progress.get("state"),
                "restore_progress": _read_json(
                    restored_root / "restore-progress.json"
                ).get("state"),
                "observed_rpo_seconds": manifest.observed_rpo_seconds,
                "observed_rto_seconds": restore_result.get("observed_rto_seconds"),
                "sla_claimed": False,
                "release_lifecycle": lifecycle,
                "external_provider_writes": False,
                "real_order_effects": False,
                "production_mutation": False,
                "sensitive_data_present": False,
                "cleanup": {"status": "pending"},
            }
        finally:
            _report_phase("cleanup")
            if control_server is not None and control_thread is not None:
                stop_control_agent_server(
                    server=control_server,
                    thread=control_thread,
                    socket_path=temporary / "control-agent.sock",
                )
            cleanup = _cleanup_projects(projects, environ=environ)
    if safe_result is None:
        raise RecoveryRuntimeProofError("runtime proof produced no safe result")
    safe_result["cleanup"] = cleanup
    if cleanup.get("status") != "completed":
        raise RecoveryRuntimeProofError("runtime proof cleanup is incomplete")
    return safe_result


def _capture_sources(
    project: str,
    *,
    environ: dict[str, str],
    source_root: Path,
    age_identity_file: Path,
    age_recipient_file: Path,
) -> dict[str, Any]:
    _capture_release_config(source_root)
    postgresql_dump = _storage_exec(
        project,
        "postgresql",
        [
            "pg_dump",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            "--table=stage21_state",
        ],
        environ=environ,
    )
    (source_root / "postgresql.snapshot").write_bytes(postgresql_dump)
    clickhouse_dump = _clickhouse_query(
        project,
        "SELECT ts, value FROM stage21_timeseries ORDER BY ts FORMAT JSONEachRow",
        environ=environ,
    )
    (source_root / "clickhouse.snapshot").write_bytes(clickhouse_dump)
    checkpoint = _redis_command(project, ["GET", "stage21:checkpoint"], environ=environ)
    _write_json(
        source_root / "redis_checkpoint.snapshot",
        {"schema": "io.roehub.redis-checkpoint/v1alpha1", "checkpoint": checkpoint},
    )
    plugin_audit = _storage_exec(
        project,
        "postgresql",
        [
            "psql",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "-At",
            "-c",
            (
                "SELECT kind || ',' || item_id FROM stage21_state "
                "WHERE kind IN ('plugin','operation','audit') ORDER BY kind"
            ),
        ],
        environ=environ,
    )
    (source_root / "plugin_operation_audit.snapshot").write_bytes(plugin_audit)
    _capture_artifacts(source_root)
    stage20 = _read_json(STAGE20_EVIDENCE)
    if stage20.get("status") != "passed":
        raise RecoveryRuntimeProofError("accepted observability evidence is unavailable")
    shutil.copyfile(STAGE20_EVIDENCE, source_root / "observability.snapshot")
    openbao_staging = source_root / ".openbao-staging"
    openbao_staging.mkdir(mode=0o700)
    openbao_ciphertext = openbao_staging / "openbao.snap.age"
    openbao_proof = verify_openbao_runtime(
        export_encrypted_backup=openbao_ciphertext,
        recovery_identity_path=age_identity_file,
        recovery_recipient_path=age_recipient_file,
        compose_override=(
            Path(environ[_OPENBAO_IMAGE_OVERRIDE_ENV])
            if _OPENBAO_IMAGE_OVERRIDE_ENV in environ
            else None
        ),
    )
    openbao_metadata = openbao_ciphertext.with_suffix(
        openbao_ciphertext.suffix + ".metadata.json"
    )
    if (
        openbao_proof.get("status") != "passed"
        or not openbao_ciphertext.is_file()
        or not openbao_metadata.is_file()
    ):
        raise RecoveryRuntimeProofError("OpenBao encrypted snapshot proof failed")
    openbao_binding = {
        "schema": "io.roehub.stage21-openbao-snapshot-binding/v1alpha1",
        "ciphertext_sha256": _file_sha256(openbao_ciphertext),
        "fresh_volume_force_restore": openbao_proof.get("fresh_volume_force_restore"),
        "fresh_storage_guard": openbao_proof.get("fresh_storage_guard"),
        "forbidden_output_scan": openbao_proof.get("forbidden_output_scan"),
    }
    _write_json(openbao_staging / "restore-proof.json", openbao_binding)
    with tarfile.open(source_root / "openbao.snapshot", "w") as archive:
        archive.add(openbao_ciphertext, arcname="openbao.snap.age")
        archive.add(openbao_metadata, arcname="openbao.snap.age.metadata.json")
        archive.add(openbao_staging / "restore-proof.json", arcname="restore-proof.json")
    shutil.rmtree(openbao_staging)
    for source in _policy_sources():
        (source_root / source.filename).chmod(0o600)
    return openbao_proof


def _capture_release_config(source_root: Path) -> None:
    staging = source_root / ".release-config-staging"
    staging.mkdir(mode=0o700)
    sources = {
        "release-metadata.json": RELEASE_MANIFEST,
        "service-config.json": SERVICE_CONFIG,
        "generation-manifest.json": PROFILE_ROOT / "generation-manifest.json",
    }
    digests: dict[str, str] = {}
    for name, source in sources.items():
        target = staging / name
        shutil.copyfile(source, target)
        target.chmod(0o600)
        digests[name] = _file_sha256(target)
    _write_json(
        staging / "snapshot-manifest.json",
        {
            "schema": "io.roehub.stage21-release-config-snapshot/v1alpha1",
            "profile": "base",
            "source_release_version": "0.0.0",
            "files": digests,
        },
    )
    (staging / "snapshot-manifest.json").chmod(0o600)
    with tarfile.open(source_root / "release_config.snapshot", "w") as archive:
        for name in sorted((*sources, "snapshot-manifest.json")):
            archive.add(staging / name, arcname=name)
    shutil.rmtree(staging)


def _seed_storage(project: str, *, environ: dict[str, str]) -> None:
    sql = (
        "CREATE TABLE stage21_state ("
        "kind text NOT NULL, item_id text PRIMARY KEY, payload text NOT NULL, "
        "recorded_at timestamptz NOT NULL);"
        "INSERT INTO stage21_state(kind,item_id,payload,recorded_at) VALUES"
        "('user','user-1','fixture-user','2026-07-14T09:00:00Z'),"
        "('config','config-1','fixture-config','2026-07-14T09:01:00Z'),"
        "('plugin','plugin-1','fixture-plugin','2026-07-14T09:02:00Z'),"
        "('operation','operation-1','fixture-operation','2026-07-14T09:03:00Z'),"
        "('audit','audit-1','fixture-audit','2026-07-14T09:04:00Z');"
    )
    _storage_exec(
        project,
        "postgresql",
        ["psql", "-U", "roehub", "-d", "roehub", "-v", "ON_ERROR_STOP=1", "-c", sql],
        environ=environ,
    )
    _clickhouse_query(
        project,
        (
            "CREATE TABLE stage21_timeseries "
            "(ts DateTime64(3, 'UTC'), value Float64) "
            "ENGINE=MergeTree ORDER BY ts"
        ),
        environ=environ,
    )
    _clickhouse_query(
        project,
        (
            "INSERT INTO stage21_timeseries VALUES "
            "('2026-07-14 09:00:00.000',1.0),"
            "('2026-07-14 09:01:00.000',2.0),"
            "('2026-07-14 09:02:00.000',3.0)"
        ),
        environ=environ,
    )
    result = _redis_command(
        project,
        ["SET", "stage21:checkpoint", "checkpoint-0003"],
        environ=environ,
    )
    if result != "OK":
        raise RecoveryRuntimeProofError("Redis checkpoint seed failed")


def _restore_and_compare_storage(
    *,
    source_project: str,
    target_project: str,
    restored_root: Path,
    environ: dict[str, str],
    manifest: Any,
    temporary: Path,
    age_identity_file: Path,
) -> dict[str, object]:
    _storage_exec(
        target_project,
        "postgresql",
        ["pg_restore", "-U", "roehub", "-d", "roehub", "--no-owner", "--no-privileges"],
        environ=environ,
        input_bytes=(restored_root / "postgresql.snapshot").read_bytes(),
    )
    _clickhouse_query(
        target_project,
        (
            "CREATE TABLE stage21_timeseries "
            "(ts DateTime64(3, 'UTC'), value Float64) "
            "ENGINE=MergeTree ORDER BY ts"
        ),
        environ=environ,
    )
    _clickhouse_query(
        target_project,
        "INSERT INTO stage21_timeseries FORMAT JSONEachRow",
        environ=environ,
        input_bytes=(restored_root / "clickhouse.snapshot").read_bytes(),
    )
    redis_payload = _read_json(restored_root / "redis_checkpoint.snapshot")
    checkpoint = redis_payload.get("checkpoint")
    if not isinstance(checkpoint, str):
        raise RecoveryRuntimeProofError("restored Redis checkpoint is invalid")
    _redis_command(
        target_project,
        ["SET", "stage21:checkpoint", checkpoint],
        environ=environ,
    )
    source_rows = _postgres_rows(source_project, environ=environ)
    target_rows = _postgres_rows(target_project, environ=environ)
    source_timeseries = _clickhouse_query(
        source_project,
        "SELECT ts, value FROM stage21_timeseries ORDER BY ts FORMAT JSONEachRow",
        environ=environ,
    )
    target_timeseries = _clickhouse_query(
        target_project,
        "SELECT ts, value FROM stage21_timeseries ORDER BY ts FORMAT JSONEachRow",
        environ=environ,
    )
    target_checkpoint = _redis_command(
        target_project,
        ["GET", "stage21:checkpoint"],
        environ=environ,
    )
    if source_rows != target_rows or source_timeseries != target_timeseries:
        raise RecoveryRuntimeProofError("database restore comparison failed")
    if target_checkpoint != checkpoint:
        raise RecoveryRuntimeProofError("Redis checkpoint restore comparison failed")
    artifact_target = temporary / f"restored-artifacts-{target_project}"
    artifact_target.mkdir()
    _safe_extract_tar(restored_root / "artifacts.snapshot", artifact_target)
    artifact_catalog = _read_json(artifact_target / "catalog.json")
    artifact_digest = artifact_catalog.get("digest")
    if not isinstance(artifact_digest, str):
        raise RecoveryRuntimeProofError("artifact catalog restore failed")
    artifact_blob = artifact_target / "blobs/sha256" / artifact_digest.removeprefix("sha256:")
    if _file_sha256(artifact_blob) != artifact_digest:
        raise RecoveryRuntimeProofError("artifact digest restore failed")
    release_config = _verify_release_config_archive(
        restored_root / "release_config.snapshot",
        temporary / f"restored-release-config-{target_project}",
    )
    openbao = _verify_openbao_snapshot_archive(
        restored_root / "openbao.snapshot",
        temporary / f"restored-openbao-{target_project}",
        age_identity_file=age_identity_file,
    )
    target_plugin_audit = _storage_exec(
        target_project,
        "postgresql",
        [
            "psql",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "-At",
            "-c",
            (
                "SELECT kind || ',' || item_id FROM stage21_state "
                "WHERE kind IN ('plugin','operation','audit') ORDER BY kind"
            ),
        ],
        environ=environ,
    )
    if target_plugin_audit != (restored_root / "plugin_operation_audit.snapshot").read_bytes():
        raise RecoveryRuntimeProofError("plugin/operation/audit restore comparison failed")
    digest_matches = 0
    for entry in manifest.entries:
        restored = restored_root / f"{entry.owner.value}.snapshot"
        if _file_sha256(restored) != entry.plaintext_sha256:
            raise RecoveryRuntimeProofError("restored digest does not match manifest")
        digest_matches += 1
    return {
        "postgresql": {
            "rows": len(target_rows),
            "kinds": sorted(row.split("|", 1)[0] for row in target_rows),
            "users_config_plugin_operation_audit": "matched",
        },
        "clickhouse": {
            "rows": len(target_timeseries.decode().strip().splitlines()),
            "time_range": ["2026-07-14 09:00:00.000", "2026-07-14 09:02:00.000"],
            "digest_match": True,
        },
        "redis_checkpoint": {
            "value_match": True,
            "source_of_truth": False,
        },
        "artifacts": {"digest": artifact_digest, "match": True},
        "plugin_operation_audit": "matched",
        "release_config": release_config,
        "observability_bounded_history": "matched",
        "openbao_encrypted_snapshot": openbao,
        "all_plaintext_digests": digest_matches,
    }


def _verify_release_config_archive(path: Path, target: Path) -> dict[str, object]:
    target.mkdir(mode=0o700)
    _safe_extract_tar(path, target)
    manifest = _read_json(target / "snapshot-manifest.json")
    files = manifest.get("files")
    expected_sources = {
        "release-metadata.json": RELEASE_MANIFEST,
        "service-config.json": SERVICE_CONFIG,
        "generation-manifest.json": PROFILE_ROOT / "generation-manifest.json",
    }
    if (
        manifest.get("schema")
        != "io.roehub.stage21-release-config-snapshot/v1alpha1"
        or manifest.get("profile") != "base"
        or manifest.get("source_release_version") != "0.0.0"
        or not isinstance(files, dict)
        or set(files) != set(expected_sources)
    ):
        raise RecoveryRuntimeProofError("release/config snapshot manifest is invalid")
    for name, source in expected_sources.items():
        restored = target / name
        digest = _file_sha256(restored)
        if digest != files.get(name) or restored.read_bytes() != source.read_bytes():
            raise RecoveryRuntimeProofError("release/config restore comparison failed")
    return {"exact_files": len(expected_sources), "match": True}


def _verify_openbao_snapshot_archive(
    path: Path,
    target: Path,
    *,
    age_identity_file: Path,
) -> dict[str, object]:
    target.mkdir(mode=0o700)
    _safe_extract_tar(path, target)
    ciphertext = target / "openbao.snap.age"
    metadata = _read_json(target / "openbao.snap.age.metadata.json")
    binding = _read_json(target / "restore-proof.json")
    digest = _file_sha256(ciphertext)
    decrypted = _run(
        [
            "age",
            "--decrypt",
            "--identity",
            str(age_identity_file),
            str(ciphertext),
        ],
        timeout=300,
    )
    if (
        metadata.get("ciphertext_sha256") != digest.removeprefix("sha256:")
        or binding.get("ciphertext_sha256") != digest
        or binding.get("fresh_volume_force_restore") != "passed"
        or binding.get("fresh_storage_guard") != "passed"
        or binding.get("forbidden_output_scan") != "passed"
        or decrypted.returncode != 0
        or not decrypted.stdout
    ):
        raise RecoveryRuntimeProofError("OpenBao exact snapshot restore binding failed")
    return {
        "ciphertext_sha256": digest,
        "decryptable_by_operator_key": True,
        "same_snapshot_fresh_volume_restore": "passed",
    }


def _exercise_release_lifecycle(
    *,
    temporary: Path,
    source_root: Path,
    materials: dict[str, Path],
) -> dict[str, object]:
    def apply_materialized(root: Path, manifest: Any) -> dict[str, object]:
        if len(list(root.glob("*.snapshot"))) != len(manifest.entries):
            raise BackupBundleError("restore.state_not_ready")
        return {"status": "ready", "state_owner_count": len(manifest.entries)}

    now = datetime.now(UTC)
    sources = tuple(
        BackupSource(
            owner=source.owner,
            path=source_root / source.filename,
            media_type=source.media_type,
            consistency_mode=source.consistency_mode,
            source_schema_version="fixture-n-minus-one",
            captured_at=now,
            expected_plaintext_sha256=_file_sha256(
                source_root / source.filename
            ),
            limitations=source.limitations,
        )
        for source in _policy_sources()
    )
    create_backup(
        backup_id="stage21-n-minus-one",
        installation_fingerprint="sha256:" + hashlib.sha256(b"stage21-fixture").hexdigest(),
        source_release_version="0.0.0",
        sources=sources,
        backup_root=temporary / "backups",
        age_recipient_file=materials["age_recipient"],
        age_identity_file=materials["age_identity"],
        signing_private_key_file=materials["signing_file"],
        verification_public_key_file=materials["verification_file"],
        quiesce_started_at=now,
        quiesce_completed_at=now,
    )
    bundle = temporary / "backups/stage21-n-minus-one"
    upgrade_root = temporary / "upgrade-target"
    injected_failure = "not_exercised"
    try:
        upgrade_from_backup(
            bundle=bundle,
            target_root=upgrade_root,
            age_identity_file=materials["age_identity"],
            verification_public_key_file=materials["verification_file"],
            target_release_version="0.1.0",
            fail_before_commit=True,
            apply_restored_state=apply_materialized,
        )
    except BackupBundleError as error:
        if str(error) != "upgrade.injected_failure_before_commit":
            raise
        injected_failure = "failed_before_atomic_release_commit"
    upgraded = upgrade_from_backup(
        bundle=bundle,
        target_root=upgrade_root,
        age_identity_file=materials["age_identity"],
        verification_public_key_file=materials["verification_file"],
        target_release_version="0.1.0",
        apply_restored_state=apply_materialized,
    )
    rolled_back = rollback_from_backup(
        bundle=bundle,
        target_root=temporary / "rollback-target",
        age_identity_file=materials["age_identity"],
        verification_public_key_file=materials["verification_file"],
        failed_release_version="0.1.0",
        apply_restored_state=apply_materialized,
    )
    irreversible_guard = "not_exercised"
    try:
        upgrade_from_backup(
            bundle=bundle,
            target_root=temporary / "irreversible-target",
            age_identity_file=materials["age_identity"],
            verification_public_key_file=materials["verification_file"],
            target_release_version="0.1.0",
            irreversible=True,
            apply_restored_state=apply_materialized,
        )
    except BackupBundleError as error:
        if str(error) != "upgrade.forward_recovery_plan_required":
            raise
        irreversible_guard = "forward_recovery_plan_required"
    return {
        "fixture": "versioned-unpublished-n-minus-one",
        "from_release": upgraded["from_release"],
        "to_release": upgraded["to_release"],
        "preupgrade_backup_verified": upgraded["preupgrade_backup_verified"],
        "injected_failure": injected_failure,
        "resume_after_failure": upgraded["status"],
        "rollback_release": rolled_back["restored_release"],
        "rollback_state_owner_count": rolled_back["state_owner_count"],
        "irreversible_migration_guard": irreversible_guard,
        "observed_upgrade_seconds": upgraded["observed_upgrade_seconds"],
        "observed_rollback_seconds": rolled_back["observed_rollback_seconds"],
    }


def _operator_materials(temporary: Path) -> dict[str, Path]:
    age_identity = temporary / "operator.agekey"
    generated = _run(["age-keygen", "-o", str(age_identity)])
    if generated.returncode != 0:
        raise RecoveryRuntimeProofError("age identity generation failed")
    os.chmod(age_identity, 0o600)
    recipient = _run(["age-keygen", "-y", str(age_identity)])
    if recipient.returncode != 0 or not recipient.stdout:
        raise RecoveryRuntimeProofError("age recipient derivation failed")
    age_recipient = temporary / "operator.recipient"
    age_recipient.write_bytes(recipient.stdout)
    signer = Ed25519PrivateKey.generate()
    signing_file = temporary / "operator-signing.pem"
    verification_file = temporary / "operator-verification.pem"
    signing_file.write_bytes(
        signer.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    verification_file.write_bytes(
        signer.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    os.chmod(signing_file, 0o600)
    return {
        "age_identity": age_identity,
        "age_recipient": age_recipient,
        "signing_file": signing_file,
        "verification_file": verification_file,
    }


def _write_backup_policy(
    *,
    temporary: Path,
    source_root: Path,
    materials: dict[str, Path],
) -> Path:
    source_rows = _policy_sources()
    capture_record_file = temporary / "capture-record.json"
    installation_fingerprint = "sha256:" + hashlib.sha256(b"stage21-local").hexdigest()
    release_policy_file = temporary / "release-policy.json"
    release_policy = InstallationReleasePolicy(
        installation_fingerprint=installation_fingerprint,
        transitions=(
            ReleaseTransitionRule(
                from_release="0.0.0",
                to_release="0.1.0",
                reversible=True,
            ),
            ReleaseTransitionRule(
                from_release="0.0.0",
                to_release="0.2.0",
                reversible=False,
            ),
        ),
    )
    release_policy_file.write_text(
        release_policy.model_dump_json(by_alias=True),
        encoding="utf-8",
    )
    release_policy_file.chmod(0o600)
    policy = InstallationBackupPolicy(
        profile="base",
        installation_fingerprint=installation_fingerprint,
        release_version="0.0.0",
        source_root=source_root,
        backup_root=temporary / "backups",
        restore_root=temporary / "restores",
        age_recipient_file=materials["age_recipient"],
        age_identity_file=materials["age_identity"],
        signing_private_key_file=materials["signing_file"],
        verification_public_key_file=materials["verification_file"],
        cancellation_root=temporary / "cancel",
        capture_record_file=capture_record_file,
        release_policy_file=release_policy_file,
        sources=source_rows,
    )
    path = temporary / "backup-policy.json"
    path.write_text(
        json.dumps(policy.model_dump(mode="json", by_alias=True), sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


def _policy_sources() -> tuple[BackupPolicySource, ...]:
    return (
        BackupPolicySource(
            owner=BackupStateOwner.RELEASE_CONFIG,
            filename="release_config.snapshot",
            media_type="application/x-tar",
            consistency_mode="application_quiesced",
            source_schema_version="release-config-bundle-v1",
        ),
        BackupPolicySource(
            owner=BackupStateOwner.POSTGRESQL,
            filename="postgresql.snapshot",
            media_type="application/vnd.postgresql.custom",
            consistency_mode="database_snapshot",
            source_schema_version="postgresql-16-custom",
        ),
        BackupPolicySource(
            owner=BackupStateOwner.CLICKHOUSE,
            filename="clickhouse.snapshot",
            media_type="application/x-ndjson",
            consistency_mode="database_snapshot",
            source_schema_version="clickhouse-24-json-each-row",
        ),
        BackupPolicySource(
            owner=BackupStateOwner.REDIS_CHECKPOINT,
            filename="redis_checkpoint.snapshot",
            media_type="application/json",
            consistency_mode="durable_checkpoint",
            source_schema_version="redis-7-checkpoint-v1",
            limitations=("Redis is transport/checkpoint state, not the source of truth",),
        ),
        BackupPolicySource(
            owner=BackupStateOwner.OPENBAO,
            filename="openbao.snapshot",
            media_type="application/vnd.roehub.openbao-recovery+tar",
            consistency_mode="encrypted_raft_snapshot",
            source_schema_version="openbao-raft-v1",
        ),
        BackupPolicySource(
            owner=BackupStateOwner.ARTIFACTS,
            filename="artifacts.snapshot",
            media_type="application/x-tar",
            consistency_mode="content_addressed_snapshot",
            source_schema_version="artifact-backup-v1",
        ),
        BackupPolicySource(
            owner=BackupStateOwner.PLUGIN_OPERATION_AUDIT,
            filename="plugin_operation_audit.snapshot",
            media_type="text/csv",
            consistency_mode="application_quiesced",
            source_schema_version="plugin-operation-audit-v1",
        ),
        BackupPolicySource(
            owner=BackupStateOwner.OBSERVABILITY,
            filename="observability.snapshot",
            media_type="application/json",
            consistency_mode="bounded_history_snapshot",
            source_schema_version="operational-health-v1alpha1",
            limitations=("Bounded observability history is not product source-of-truth data",),
        ),
    )


def _start_control_agent(
    *,
    temporary: Path,
    policy_path: Path,
    state_coordinator: RuntimeStateCoordinator,
    current_release: Any,
) -> dict[str, Any]:
    socket_path = temporary / "control-agent.sock"
    journal = temporary / "operations.jsonl"
    identity_files: dict[str, Path] = {}
    for name in ("api", "owner", "job"):
        path = temporary / f"{name}.identity"
        path.write_text(secrets.token_urlsafe(48), encoding="utf-8")
        os.chmod(path, 0o600)
        identity_files[name] = path
    backup_backend = InstallationBackupControlBackend(
        policy_path=policy_path,
        receipt_root=temporary / "backup-receipts",
        state_coordinator=state_coordinator,
    )
    backend = RecoveryControlBackend(
        runtime_backend=_UnavailableRuntimeBackend(),
        backup_backend=backup_backend,
        current_release=current_release,
    )
    operation_journal = AppendOnlyOperationJournal(path=journal)
    service = ControlOperationService(
        backend=backend,
        journal=operation_journal,
        before_lock=backup_backend.request_cancellation,
    )
    authorizer = ServiceIdentityAuthorizer(
        api_token_file=identity_files["api"],
        owner_token_file=identity_files["owner"],
        job_token_file=identity_files["job"],
        replay_state_dir=temporary / "auth-replay",
    )
    server, thread = start_control_agent_server(
        socket_path=socket_path,
        service=service,
        journal=operation_journal,
        authorizer=authorizer,
    )
    return {
        "server": server,
        "thread": thread,
        "socket": socket_path,
        "owner_identity_file": identity_files["owner"],
    }


def _roehubctl(
    *,
    socket_path: Path,
    owner_identity_file: Path,
    command: str,
    subject_id: str | None = None,
    release_version: str | None = None,
    operation_id: UUID,
) -> dict[str, Any]:
    cli_args = _roehubctl_arguments(
        socket_path=socket_path,
        owner_identity_file=owner_identity_file,
        command=command,
        subject_id=subject_id,
        release_version=release_version,
        operation_id=operation_id,
    )
    completed = _run(
        [sys.executable, "-m", "apps.roehubctl.main.main", *cli_args],
        timeout=180,
    )
    if completed.returncode != 0:
        raise RecoveryRuntimeProofError(f"roehubctl {command} failed")
    return _parse_roehubctl_result(completed.stdout)


def _roehubctl_when_progress(
    *,
    progress_path: Path,
    socket_path: Path,
    owner_identity_file: Path,
    command: str,
    subject_id: str,
    operation_id: UUID,
) -> dict[str, Any]:
    _wait_for_progress(progress_path)
    cli_args = _roehubctl_arguments(
        socket_path=socket_path,
        owner_identity_file=owner_identity_file,
        command=command,
        subject_id=subject_id,
        release_version=None,
        operation_id=operation_id,
    )
    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = roehubctl_main(cli_args)
    if exit_code != 0:
        raise RecoveryRuntimeProofError(f"roehubctl {command} failed")
    return _parse_roehubctl_result(output.getvalue().encode())


def _roehubctl_arguments(
    *,
    socket_path: Path,
    owner_identity_file: Path,
    command: str,
    subject_id: str | None,
    release_version: str | None,
    operation_id: UUID,
) -> list[str]:
    command_args = [
        "--socket",
        str(socket_path),
            "--identity-file",
            str(owner_identity_file),
            command,
            "--profile",
            "base",
    ]
    if subject_id is not None:
        command_args.extend(["--subject-id", subject_id])
    if release_version is not None:
        command_args.extend(["--release-version", release_version])
    command_args.extend(["--operation-id", str(operation_id)])
    return command_args


def _parse_roehubctl_result(payload_bytes: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryRuntimeProofError("roehubctl returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RecoveryRuntimeProofError("roehubctl returned invalid result")
    return payload


def _capture_artifacts(source_root: Path) -> None:
    content_block = hashlib.sha256(b"stage21-artifact-fixture").digest() * 32_768
    artifact_root = source_root / ".artifact-staging"
    blob_root = artifact_root / "blobs/sha256"
    blob_root.mkdir(parents=True)
    blob = blob_root / "pending"
    digest_builder = hashlib.sha256()
    with blob.open("wb") as stream:
        for _ in range(32):
            stream.write(content_block)
            digest_builder.update(content_block)
    digest = "sha256:" + digest_builder.hexdigest()
    artifact_blob = blob.with_name(digest.removeprefix("sha256:"))
    blob.replace(artifact_blob)
    _write_json(
        artifact_root / "catalog.json",
        {
            "schema": "ArtifactBackup/v1",
            "digest": digest,
            "size_bytes": artifact_blob.stat().st_size,
        },
    )
    with tarfile.open(source_root / "artifacts.snapshot", "w") as archive:
        archive.add(artifact_root / "catalog.json", arcname="catalog.json")
        archive.add(artifact_root / "blobs", arcname="blobs")
    shutil.rmtree(artifact_root)


def _wait_for_progress(path: Path, *, timeout_seconds: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            progress = _read_json(path)
            if progress.get("state") == "running" and isinstance(
                progress.get("completed"), dict
            ):
                return
            if progress.get("state") in {"completed", "cancelled"}:
                raise RecoveryRuntimeProofError(
                    "operation completed before concurrent cancellation"
                )
        time.sleep(0.01)
    raise RecoveryRuntimeProofError("operation progress was not observable")


def _safe_extract_tar(path: Path, target: Path) -> None:
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
        for member in members:
            candidate = (target / member.name).resolve()
            if target.resolve() not in candidate.parents and candidate != target.resolve():
                raise RecoveryRuntimeProofError("artifact backup path traversal rejected")
            if member.issym() or member.islnk():
                raise RecoveryRuntimeProofError("artifact backup link rejected")
        archive.extractall(target, members=members, filter="data")


def _storage_environment() -> dict[str, str]:
    values = tuple(secrets.token_hex(24) for _ in range(3))
    environ = dict(os.environ)
    environ.update(
        {
            "ROEHUB_STORAGE_POSTGRES_PASSWORD": values[0],
            "ROEHUB_STORAGE_CLICKHOUSE_PASSWORD": values[1],
            "ROEHUB_STORAGE_REDIS_PASSWORD": values[2],
            "ROEHUB_STORAGE_POSTGRES_DSN": (
                "host=postgresql port=5432 dbname=roehub user=roehub password=" + values[0]
            ),
            "ROEHUB_STORAGE_CLICKHOUSE_DSN": (
                "http://roehub:" + values[1] + "@clickhouse:8123/default"
            ),
            "ROEHUB_STORAGE_REDIS_URL": (
                "redis://:" + values[2] + "@redis:6379/0"
            ),
            "ROEHUB_STORAGE_SERVICE_CONFIG": str(SERVICE_CONFIG),
        }
    )
    return environ


def _storage_up(project: str, *, environ: dict[str, str]) -> None:
    completed = _run(
        [
            *_compose_command(
                project=project,
                primary=STORAGE_COMPOSE,
                environ=environ,
            ),
            "up",
            "-d",
            "--wait",
            "postgresql",
            "clickhouse",
            "redis",
        ],
        environ=environ,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RecoveryRuntimeProofError("storage project failed to start")


def _storage_exec(
    project: str,
    service: str,
    args: list[str],
    *,
    environ: dict[str, str],
    input_bytes: bytes | None = None,
) -> bytes:
    completed = _run(
        [
            *_compose_command(
                project=project,
                primary=STORAGE_COMPOSE,
                environ=environ,
            ),
            "exec",
            "-T",
            service,
            *args,
        ],
        environ=environ,
        input_bytes=input_bytes,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RecoveryRuntimeProofError(f"storage command failed for {service}")
    return completed.stdout


def _clickhouse_query(
    project: str,
    query: str,
    *,
    environ: dict[str, str],
    input_bytes: bytes | None = None,
) -> bytes:
    return _storage_exec(
        project,
        "clickhouse",
        [
            "/bin/sh",
            "-ec",
            'clickhouse-client --user roehub --password "$CLICKHOUSE_PASSWORD" --query "$1"',
            "stage21-clickhouse",
            query,
        ],
        environ=environ,
        input_bytes=input_bytes,
    )


def _redis_command(
    project: str,
    args: list[str],
    *,
    environ: dict[str, str],
) -> str:
    command = [
        "/bin/sh",
        "-ec",
        'REDISCLI_AUTH="$ROEHUB_REDIS_PASSWORD" exec redis-cli --raw "$@"',
        "stage21-redis",
        *args,
    ]
    return _storage_exec(
        project,
        "redis",
        command,
        environ=environ,
    ).decode("utf-8").strip()


def _postgres_rows(project: str, *, environ: dict[str, str]) -> list[str]:
    output = _storage_exec(
        project,
        "postgresql",
        [
            "psql",
            "-U",
            "roehub",
            "-d",
            "roehub",
            "-At",
            "-c",
            "SELECT kind || '|' || item_id || '|' || payload FROM stage21_state ORDER BY kind",
        ],
        environ=environ,
    )
    return output.decode("utf-8").strip().splitlines()


def _observability_up(project: str, *, environ: dict[str, str]) -> None:
    compose = _compose_command(
        project=project,
        primary=OBSERVABILITY_COMPOSE,
        environ=environ,
        final_overrides=(OBSERVABILITY_OVERRIDE,),
    )
    secret_init = _run(
        [*compose, "up", "--no-build", "--pull", "never", "secret-init"],
        environ=environ,
    )
    if secret_init.returncode != 0:
        raise RecoveryRuntimeProofError("monitoring secret initialization failed")
    started = _run(
        [
            *compose,
            "up",
            "-d",
            "--no-build",
            "--pull",
            "never",
            "--no-deps",
            "--wait",
            "--wait-timeout",
            "120",
            *MONITORING_SERVICES,
        ],
        environ=environ,
        timeout=180,
    )
    if started.returncode != 0:
        raise RecoveryRuntimeProofError("monitoring project failed to start")


def _monitoring_state(
    project: str,
    *,
    environ: dict[str, str],
) -> MonitoringState:
    compose = _compose_command(
        project=project,
        primary=OBSERVABILITY_COMPOSE,
        environ=environ,
        final_overrides=(OBSERVABILITY_OVERRIDE,),
    )
    running_result = _run(
        [*compose, "ps", "--services", "--status", "running"],
        environ=environ,
    )
    if running_result.returncode != 0:
        raise RecoveryRuntimeProofError("monitoring state inspection failed")
    running = sorted(running_result.stdout.decode().splitlines())
    ready_result = _run(
        [
            *compose,
            "exec",
            "-T",
            "operational-health",
            "python",
            "-c",
            (
                "import urllib.request; "
                "response = urllib.request.urlopen("
                "'http://127.0.0.1:9300/health/ready', timeout=5); "
                "raise SystemExit(0 if response.status == 200 else 1)"
            ),
        ],
        environ=environ,
    )
    if ready_result.returncode != 0:
        raise RecoveryRuntimeProofError("operational-health readiness failed")
    return {
        "running": running,
        "operational_health_ready": True,
        "web_api_running": bool({"web", "api"} & set(running)),
    }


def _cleanup_projects(
    projects: tuple[str, ...],
    *,
    environ: dict[str, str],
) -> dict[str, object]:
    down_status: dict[str, int] = {}
    residual: dict[str, dict[str, list[str]]] = {}
    for project in projects:
        if project.endswith("-monitoring"):
            command = [
                *_compose_command(
                    project=project,
                    primary=OBSERVABILITY_COMPOSE,
                    environ=environ,
                    final_overrides=(OBSERVABILITY_OVERRIDE,),
                ),
                "down",
                "-v",
                "--remove-orphans",
            ]
            completed = _run(command, environ=environ, timeout=120)
        else:
            command = [
                *_compose_command(
                    project=project,
                    primary=STORAGE_COMPOSE,
                    environ=environ,
                ),
                "down",
                "-v",
                "--remove-orphans",
            ]
            completed = _run(command, environ=environ, timeout=120)
        down_status[project] = completed.returncode
        residual[project] = _residual_resources(project)
    completed = all(status == 0 for status in down_status.values()) and all(
        not any(resources.values()) for resources in residual.values()
    )
    return {
        "status": "completed" if completed else "failed",
        "down_exit_status": down_status,
        "residual_resources": residual,
    }


def _compose_command(
    *,
    project: str,
    primary: Path,
    environ: dict[str, str],
    final_overrides: tuple[Path, ...] = (),
) -> list[str]:
    command = ["docker", "compose", "-p", project, "-f", str(primary)]
    image_override = environ.get(_RUNTIME_IMAGE_OVERRIDE_ENV)
    if image_override:
        command.extend(("-f", image_override))
    for override in final_overrides:
        command.extend(("-f", str(override)))
    return command


def _write_scoped_image_override(
    *,
    source: Path,
    destination: Path,
    services: tuple[str, ...],
) -> Path:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryRuntimeProofError("runtime image override is invalid") from error
    source_services = payload.get("services") if isinstance(payload, dict) else None
    if not isinstance(source_services, dict):
        raise RecoveryRuntimeProofError("runtime image override is invalid")
    selected: dict[str, dict[str, str]] = {}
    for service in services:
        record = source_services.get(service)
        image = record.get("image") if isinstance(record, dict) else None
        if not isinstance(image, str) or not image.startswith("sha256:"):
            raise RecoveryRuntimeProofError("runtime image override is incomplete")
        selected[service] = {"image": image, "pull_policy": "never"}
    _write_json(destination, {"services": selected})
    return destination


def _residual_resources(project: str) -> dict[str, list[str]]:
    commands = {
        "containers": [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ],
        "networks": [
            "docker",
            "network",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ],
        "volumes": [
            "docker",
            "volume",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.Name}}",
        ],
    }
    result: dict[str, list[str]] = {}
    for kind, command in commands.items():
        completed = _run(command)
        result[kind] = completed.stdout.decode().strip().splitlines()
    return result


def _run(
    command: list[str],
    *,
    environ: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
    timeout: float = 300,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=environ,
            stdin=subprocess.DEVNULL if input_bytes is None else None,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RecoveryRuntimeProofError(
            f"runtime command timed out: {Path(command[0]).name}"
        ) from error
    except OSError as error:
        raise RecoveryRuntimeProofError(
            f"required runtime command is unavailable: {Path(command[0]).name}"
        ) from error


def _report_phase(name: str) -> None:
    print(f"stage21 runtime proof phase: {name}", file=sys.stderr, flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryRuntimeProofError("runtime artifact is invalid") from error
    if not isinstance(payload, dict):
        raise RecoveryRuntimeProofError("runtime artifact is invalid")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-prefix", default="roehub-stage21-proof")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = verify_runtime(project_prefix=args.project_prefix)
    except RecoveryRuntimeProofError as error:
        print(
            json.dumps(
                {
                    "schema": "io.roehub.installation-recovery-runtime-proof/v1alpha1",
                    "status": "failed",
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
