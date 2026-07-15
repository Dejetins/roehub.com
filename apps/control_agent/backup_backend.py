"""Typed installation backup/restore backend for the control-agent."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from tools.backup import (
    BackupBundleError,
    BackupSource,
    create_backup,
    restore_backup,
    verify_backup,
)
from trading.contexts.operations import (
    ControlOperationError,
    InstallationBackupManifest,
    InstallationBackupPolicy,
    InstallationCaptureRecord,
    InstallationReleasePolicy,
    OperationAction,
    OperationRequest,
    OperationResult,
    OperationState,
    ReleaseTransitionRule,
)
from trading.contexts.operations.ports import ControlBackendPort

_CANCEL_ACTIONS = {OperationAction.BACKUP_CANCEL, OperationAction.RESTORE_CANCEL}
_BACKUP_ACTIONS = {OperationAction.BACKUP, OperationAction.RESTORE} | _CANCEL_ACTIONS


class InstallationStateCoordinatorPort(Protocol):
    """Capture and apply every state owner inside the control-agent effect."""

    def capture(self, policy: InstallationBackupPolicy) -> InstallationCaptureRecord: ...

    def restore(
        self,
        policy: InstallationBackupPolicy,
        restored_root: Path,
        manifest: InstallationBackupManifest,
    ) -> dict[str, object]: ...

    def update(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
        bundle: Path,
        transition: ReleaseTransitionRule,
    ) -> dict[str, object]: ...

    def rollback(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
        bundle: Path,
        failed_release_version: str,
        transition: ReleaseTransitionRule,
    ) -> dict[str, object]: ...

    def reconcile_release(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
    ) -> dict[str, object] | None: ...


class InstallationBackupControlBackend:
    """Apply a closed host policy; user input selects only one bounded backup id."""

    def __init__(
        self,
        *,
        policy_path: Path,
        receipt_root: Path,
        now: Callable[[], datetime] | None = None,
        state_coordinator: InstallationStateCoordinatorPort | None = None,
    ) -> None:
        self._policy_path = policy_path
        self._receipts = _prepare_private_directory(receipt_root)
        self._clock = now or (lambda: datetime.now(UTC))
        self._state_coordinator = state_coordinator
        self._load_policy()

    def execute(self, request: OperationRequest) -> OperationResult:
        policy = self._load_policy()
        self._validate_request(request, policy)
        assert request.subject_id is not None
        if request.action in _CANCEL_ACTIONS:
            self.request_cancellation(request)
            target_operation_id = UUID(request.subject_id)
            outcome: dict[str, object] = {
                "schema": "io.roehub.backup-cancellation-result/v1alpha1",
                "status": "passed",
                "target_operation_id": str(target_operation_id),
            }
            self._record_receipt(request=request, policy=policy, outcome=outcome)
            return OperationResult(
                operation_id=request.operation_id,
                action=request.action,
                profile=request.profile,
                state=OperationState.SUCCEEDED,
                detail_code="backup.cancellation_requested",
            )
        cancellation_file = policy.cancellation_root / f"{request.operation_id}.cancel"
        try:
            if request.action is OperationAction.BACKUP:
                capture = (
                    self._state_coordinator.capture(policy)
                    if self._state_coordinator is not None
                    else self._load_capture(policy)
                )
                captured = {entry.owner: entry for entry in capture.entries}
                sources = tuple(
                    BackupSource(
                        owner=source.owner,
                        path=policy.source_root / source.filename,
                        media_type=source.media_type,
                        consistency_mode=source.consistency_mode,
                        source_schema_version=source.source_schema_version,
                        captured_at=captured[source.owner].captured_at,
                        expected_plaintext_sha256=captured[source.owner].plaintext_sha256,
                        limitations=source.limitations,
                    )
                    for source in policy.sources
                )
                outcome = create_backup(
                    backup_id=request.subject_id,
                    installation_fingerprint=policy.installation_fingerprint,
                    source_release_version=policy.release_version,
                    sources=sources,
                    backup_root=policy.backup_root,
                    age_recipient_file=policy.age_recipient_file,
                    age_identity_file=policy.age_identity_file,
                    signing_private_key_file=policy.signing_private_key_file,
                    verification_public_key_file=policy.verification_public_key_file,
                    quiesce_started_at=capture.quiesce_started_at,
                    quiesce_completed_at=capture.quiesce_completed_at,
                    cancellation_file=cancellation_file,
                    now=self._clock,
                )
                self._cleanup_plaintext_sources(policy)
                detail = "backup.completed"
            else:
                coordinator = self._state_coordinator
                if coordinator is None:
                    raise ControlOperationError(
                        code="restore.state_coordinator_required"
                    )
                outcome = restore_backup(
                    bundle=policy.backup_root / request.subject_id,
                    restore_root=policy.restore_root / request.subject_id,
                    age_identity_file=policy.age_identity_file,
                    verification_public_key_file=policy.verification_public_key_file,
                    expected_installation_fingerprint=policy.installation_fingerprint,
                    protected_source_root=policy.source_root,
                    cancellation_file=cancellation_file,
                    apply_restored_state=lambda root, manifest: (
                        coordinator.restore(policy, root, manifest)
                    ),
                )
                detail = "restore.completed"
        except BackupBundleError as error:
            raise ControlOperationError(code=error.code) from error
        try:
            self._record_receipt(request=request, policy=policy, outcome=outcome)
        except ControlOperationError as error:
            raise ControlOperationError(code="operation.effect_unknown") from error
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED,
            detail_code=detail,
        )

    def request_cancellation(self, request: OperationRequest) -> None:
        """Publish only a bounded cancellation marker before the effect lock."""

        if request.action not in _CANCEL_ACTIONS:
            return
        policy = self._load_policy()
        self._validate_request(request, policy)
        assert request.subject_id is not None
        try:
            target_operation_id = UUID(request.subject_id)
        except ValueError as error:
            raise ControlOperationError(code="backup.cancellation_target_invalid") from error
        cancellation_file = policy.cancellation_root / f"{target_operation_id}.cancel"
        _atomic_json(
            cancellation_file,
            {
                "schema": "io.roehub.backup-cancellation-result/v1alpha1",
                "status": "passed",
                "target_operation_id": str(target_operation_id),
            },
        )

    def reconcile(self, request: OperationRequest) -> OperationResult:
        policy = self._load_policy()
        self._validate_request(request, policy)
        complete = False
        if request.action in _CANCEL_ACTIONS:
            assert request.subject_id is not None
            try:
                target_operation_id = UUID(request.subject_id)
            except ValueError as error:
                raise ControlOperationError(code="backup.cancellation_target_invalid") from error
            complete = (policy.cancellation_root / f"{target_operation_id}.cancel").is_file()
            if complete:
                self._record_receipt(
                    request=request,
                    policy=policy,
                    outcome={"manifest_sha256": None},
                )
        elif request.action is OperationAction.BACKUP:
            assert request.subject_id is not None
            try:
                manifest = verify_backup(
                    bundle=policy.backup_root / request.subject_id,
                    verification_public_key_file=policy.verification_public_key_file,
                    expected_installation_fingerprint=policy.installation_fingerprint,
                    age_identity_file=policy.age_identity_file,
                )
            except BackupBundleError:
                complete = False
            else:
                self._record_receipt(
                    request=request,
                    policy=policy,
                    outcome={"manifest_sha256": manifest.manifest_sha256},
                )
                self._cleanup_plaintext_sources(policy)
                complete = True
        elif request.action is OperationAction.RESTORE:
            assert request.subject_id is not None
            result_path = policy.restore_root / request.subject_id / "restore-result.json"
            try:
                result = json.loads(_read_secure_bytes(result_path))
            except (ControlOperationError, json.JSONDecodeError):
                complete = False
            else:
                complete = (
                    isinstance(result, dict)
                    and result.get("status") == "passed"
                )
                if complete:
                    self._record_receipt(
                        request=request,
                        policy=policy,
                        outcome={"manifest_sha256": result.get("manifest_sha256")},
                    )
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED if complete else OperationState.UNKNOWN,
            detail_code="operation.reconciled" if complete else "operation.effect_unknown",
        )

    def require_release_backup(
        self,
        *,
        source_release_version: str,
    ) -> tuple[Path, InstallationBackupManifest]:
        """Fail closed before upgrade/rollback without a verified pre-operation backup."""

        policy = self._load_policy()
        latest_path = policy.backup_root / "latest-verified.json"
        try:
            latest = json.loads(_read_secure_bytes(latest_path))
        except (ControlOperationError, json.JSONDecodeError) as error:
            raise ControlOperationError(code="backup.preupgrade_required") from error
        backup_id = latest.get("backup_id") if isinstance(latest, dict) else None
        if not isinstance(backup_id, str):
            raise ControlOperationError(code="backup.preupgrade_required")
        try:
            manifest = verify_backup(
                bundle=policy.backup_root / backup_id,
                verification_public_key_file=policy.verification_public_key_file,
                expected_installation_fingerprint=policy.installation_fingerprint,
                age_identity_file=policy.age_identity_file,
            )
        except BackupBundleError as error:
            raise ControlOperationError(code="backup.preupgrade_required") from error
        if manifest.source_release_version != source_release_version:
            raise ControlOperationError(code="backup.preupgrade_version_mismatch")
        return policy.backup_root / backup_id, manifest

    def execute_release(
        self,
        request: OperationRequest,
        *,
        current_release_version: str,
    ) -> OperationResult:
        coordinator = self._state_coordinator
        if coordinator is None or request.release_version is None:
            raise ControlOperationError(code="backup.release_coordinator_required")
        policy = self._load_policy()
        release_policy = self._load_release_policy(policy)
        try:
            transition = release_policy.require_transition(
                from_release=(
                    current_release_version
                    if request.action is OperationAction.UPDATE
                    else request.release_version
                ),
                to_release=(
                    request.release_version
                    if request.action is OperationAction.UPDATE
                    else current_release_version
                ),
            )
        except ValueError as error:
            raise ControlOperationError(code="upgrade.release_transition_untrusted") from error
        if request.action is OperationAction.ROLLBACK and not transition.reversible:
            raise ControlOperationError(code="rollback.irreversible_transition")
        if (
            request.action is OperationAction.UPDATE
            and not transition.reversible
            and transition.forward_recovery_plan_sha256 is None
        ):
            raise ControlOperationError(code="upgrade.forward_recovery_plan_required")
        source_version = (
            current_release_version
            if request.action is OperationAction.UPDATE
            else request.release_version
        )
        bundle, manifest = self.require_release_backup(
            source_release_version=source_version
        )
        try:
            if request.action is OperationAction.UPDATE:
                outcome = coordinator.update(policy, request, bundle, transition)
                detail_code = "upgrade.completed"
            elif request.action is OperationAction.ROLLBACK:
                outcome = coordinator.rollback(
                    policy,
                    request,
                    bundle,
                    current_release_version,
                    transition,
                )
                detail_code = "rollback.completed"
            else:
                raise ControlOperationError(code="operation.handler_unavailable")
        except BackupBundleError as error:
            raise ControlOperationError(code=error.code) from error
        if outcome.get("status") != "passed" or (
            outcome.get("backup_manifest_sha256") != manifest.manifest_sha256
        ):
            raise ControlOperationError(code="backup.release_result_invalid")
        try:
            self._record_receipt(request=request, policy=policy, outcome=outcome)
        except ControlOperationError as error:
            raise ControlOperationError(code="operation.effect_unknown") from error
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED,
            detail_code=detail_code,
        )

    def reconcile_release(self, request: OperationRequest) -> OperationResult:
        policy = self._load_policy()
        coordinator = self._state_coordinator
        if coordinator is None or request.release_version is None:
            return self._unknown_result(request)
        if self._read_receipt(request) is not None:
            return self._reconciled_result(request)
        outcome = coordinator.reconcile_release(policy, request)
        if outcome is None or outcome.get("status") != "passed":
            return self._unknown_result(request)
        if request.action is OperationAction.UPDATE:
            source_version = outcome.get("from_release")
            from_release = source_version
            to_release = request.release_version
        else:
            source_version = outcome.get("restored_release")
            from_release = source_version
            to_release = outcome.get("failed_release")
        if not isinstance(source_version, str) or not isinstance(to_release, str):
            return self._unknown_result(request)
        try:
            release_policy = self._load_release_policy(policy)
            transition = release_policy.require_transition(
                from_release=str(from_release),
                to_release=to_release,
            )
            if request.action is OperationAction.ROLLBACK and not transition.reversible:
                return self._unknown_result(request)
            _, manifest = self.require_release_backup(
                source_release_version=source_version
            )
        except (ControlOperationError, ValueError):
            return self._unknown_result(request)
        if outcome.get("backup_manifest_sha256") != manifest.manifest_sha256:
            return self._unknown_result(request)
        self._record_receipt(request=request, policy=policy, outcome=outcome)
        return self._reconciled_result(request)

    @staticmethod
    def _unknown_result(request: OperationRequest) -> OperationResult:
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.UNKNOWN,
            detail_code="operation.effect_unknown",
        )

    @staticmethod
    def _reconciled_result(request: OperationRequest) -> OperationResult:
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED,
            detail_code="operation.reconciled",
        )

    def _validate_request(
        self,
        request: OperationRequest,
        policy: InstallationBackupPolicy,
    ) -> None:
        if request.action not in _BACKUP_ACTIONS:
            raise ControlOperationError(code="operation.handler_unavailable")
        if request.profile != policy.profile or request.subject_id is None:
            raise ControlOperationError(code="backup.request_rejected")

    @staticmethod
    def _cleanup_plaintext_sources(policy: InstallationBackupPolicy) -> None:
        root = _prepare_private_directory(policy.source_root, create=False)
        expected = {source.filename for source in policy.sources}
        actual = {path.name for path in root.iterdir()}
        if not actual:
            return
        if actual != expected:
            raise ControlOperationError(code="backup.source_staging_shape_invalid")
        for filename in sorted(expected):
            path = root / filename
            try:
                info = path.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) & 0o077
                ):
                    raise ControlOperationError(code="backup.source_staging_unsafe")
                path.unlink()
            except OSError as error:
                raise ControlOperationError(code="backup.source_cleanup_failed") from error
        _fsync_directory(root)

    def _load_policy(self) -> InstallationBackupPolicy:
        try:
            return InstallationBackupPolicy.model_validate_json(
                _read_secure_bytes(self._policy_path)
            )
        except ValueError as error:
            raise ControlOperationError(code="backup.policy_invalid") from error

    def _load_capture(self, policy: InstallationBackupPolicy) -> InstallationCaptureRecord:
        try:
            capture = InstallationCaptureRecord.model_validate_json(
                _read_secure_bytes(policy.capture_record_file)
            )
        except ValueError as error:
            raise ControlOperationError(code="backup.capture_record_invalid") from error
        if (
            capture.installation_fingerprint != policy.installation_fingerprint
            or capture.source_release_version != policy.release_version
        ):
            raise ControlOperationError(code="backup.capture_record_mismatch")
        return capture

    def _load_release_policy(
        self,
        policy: InstallationBackupPolicy,
    ) -> InstallationReleasePolicy:
        try:
            release_policy = InstallationReleasePolicy.model_validate_json(
                _read_secure_bytes(policy.release_policy_file)
            )
        except ValueError as error:
            raise ControlOperationError(code="upgrade.release_policy_invalid") from error
        if release_policy.installation_fingerprint != policy.installation_fingerprint:
            raise ControlOperationError(code="upgrade.release_policy_mismatch")
        return release_policy

    def _record_receipt(
        self,
        *,
        request: OperationRequest,
        policy: InstallationBackupPolicy,
        outcome: dict[str, object],
    ) -> None:
        payload = {
            "schema": "io.roehub.backup-control-receipt/v1alpha1",
            "operation_id": str(request.operation_id),
            "request_digest": request.request_digest,
            "action": request.action.value,
            "profile": request.profile,
            "subject_id": request.subject_id,
            "release_version": request.release_version or policy.release_version,
            "manifest_sha256": outcome.get("manifest_sha256")
            or outcome.get("backup_manifest_sha256"),
        }
        target = self._receipts / f"{request.operation_id}.json"
        _write_new_or_identical(target, payload)
        if request.action is OperationAction.BACKUP:
            _atomic_json(
                policy.backup_root / "latest-verified.json",
                {
                    "schema": "io.roehub.latest-verified-backup/v1alpha1",
                    "backup_id": request.subject_id,
                    "manifest_sha256": outcome.get("manifest_sha256"),
                    "source_release_version": policy.release_version,
                },
            )

    def _read_receipt(self, request: OperationRequest) -> dict[str, Any] | None:
        target = self._receipts / f"{request.operation_id}.json"
        if not target.exists():
            return None
        try:
            payload = json.loads(_read_secure_bytes(target))
        except json.JSONDecodeError as error:
            raise ControlOperationError(code="backup.receipt_corrupt") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "io.roehub.backup-control-receipt/v1alpha1"
            or payload.get("operation_id") != str(request.operation_id)
            or payload.get("request_digest") != request.request_digest
        ):
            raise ControlOperationError(code="backup.receipt_corrupt")
        return payload


class RecoveryControlBackend:
    """Route state recovery and Docker lifecycle through one typed service boundary."""

    def __init__(
        self,
        *,
        runtime_backend: ControlBackendPort,
        backup_backend: InstallationBackupControlBackend,
        current_release: Callable[[], str | None],
    ) -> None:
        self._runtime = runtime_backend
        self._backup = backup_backend
        self._current_release = current_release

    def execute(self, request: OperationRequest) -> OperationResult:
        if request.action in _BACKUP_ACTIONS:
            return self._backup.execute(request)
        if request.action in {OperationAction.UPDATE, OperationAction.ROLLBACK}:
            current = self._current_release()
            if current is None:
                raise ControlOperationError(code="backup.preupgrade_required")
            return self._backup.execute_release(
                request,
                current_release_version=current,
            )
        return self._runtime.execute(request)

    def reconcile(self, request: OperationRequest) -> OperationResult:
        if request.action in _BACKUP_ACTIONS:
            return self._backup.reconcile(request)
        if request.action in {OperationAction.UPDATE, OperationAction.ROLLBACK}:
            return self._backup.reconcile_release(request)
        return self._runtime.reconcile(request)

    def request_cancellation(self, request: OperationRequest) -> None:
        self._backup.request_cancellation(request)


def _read_secure_bytes(path: Path) -> bytes:
    candidate = path.expanduser()
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ControlOperationError(code="backup.file_unavailable") from error
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or info.st_size > 1_048_576
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ControlOperationError(code="backup.file_unsafe")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _prepare_private_directory(path: Path, *, create: bool = True) -> Path:
    candidate = path.expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ControlOperationError(code="backup.directory_unsafe")
    root = candidate.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
    try:
        info = root.stat()
    except OSError as error:
        raise ControlOperationError(code="backup.directory_unavailable") from error
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ControlOperationError(code="backup.directory_permissions_invalid")
    return root


def _write_new_or_identical(path: Path, payload: object) -> None:
    encoded = _json_bytes(payload)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        if _read_secure_bytes(path) != encoded:
            raise ControlOperationError(code="backup.receipt_conflict")
        return
    except OSError as error:
        raise ControlOperationError(code="backup.receipt_unavailable") from error
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            _write_all(descriptor, _json_bytes(payload))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise ControlOperationError(code="backup.metadata_unavailable") from error
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ControlOperationError(code="backup.write_failed")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "InstallationBackupControlBackend",
    "InstallationStateCoordinatorPort",
    "RecoveryControlBackend",
]
