from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator

import apps.control_agent.backup_backend as backup_backend_module
from apps.control_agent.backup_backend import (
    InstallationBackupControlBackend,
    RecoveryControlBackend,
)
from tools.backup import verify_backup
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
    OperationResult,
    OperationState,
    ReleaseTransitionRule,
)
from trading.contexts.operations.adapters import AppendOnlyOperationJournal


class _RuntimeBackend:
    def __init__(self) -> None:
        self.executed: list[OperationRequest] = []

    def execute(self, request: OperationRequest) -> OperationResult:
        self.executed.append(request)
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED,
            detail_code="runtime.completed",
        )

    def reconcile(self, request: OperationRequest) -> OperationResult:
        return self.execute(request)


class _FixtureStateCoordinator:
    def __init__(self) -> None:
        self.release_results: dict[UUID, dict[str, object]] = {}

    def capture(self, policy: InstallationBackupPolicy) -> InstallationCaptureRecord:
        return InstallationCaptureRecord.model_validate_json(
            policy.capture_record_file.read_bytes()
        )

    def restore(
        self,
        policy: InstallationBackupPolicy,
        restored_root: Path,
        manifest: InstallationBackupManifest,
    ) -> dict[str, object]:
        assert policy.profile == "base"
        assert manifest is not None
        assert len(list(restored_root.glob("*.snapshot"))) == 8
        return {"status": "ready", "state_owner_count": 8}

    def update(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
        bundle: Path,
        transition: ReleaseTransitionRule,
    ) -> dict[str, object]:
        manifest = verify_backup(
            bundle=bundle,
            verification_public_key_file=policy.verification_public_key_file,
            expected_installation_fingerprint=policy.installation_fingerprint,
            age_identity_file=policy.age_identity_file,
        )
        outcome: dict[str, object] = {
            "status": "passed",
            "from_release": transition.from_release,
            "to_release": transition.to_release,
            "backup_manifest_sha256": manifest.manifest_sha256,
        }
        self.release_results[request.operation_id] = outcome
        return outcome

    def rollback(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
        bundle: Path,
        failed_release_version: str,
        transition: ReleaseTransitionRule,
    ) -> dict[str, object]:
        manifest = verify_backup(
            bundle=bundle,
            verification_public_key_file=policy.verification_public_key_file,
            expected_installation_fingerprint=policy.installation_fingerprint,
            age_identity_file=policy.age_identity_file,
        )
        outcome: dict[str, object] = {
            "status": "passed",
            "failed_release": failed_release_version,
            "restored_release": transition.from_release,
            "backup_manifest_sha256": manifest.manifest_sha256,
        }
        self.release_results[request.operation_id] = outcome
        return outcome

    def reconcile_release(
        self,
        policy: InstallationBackupPolicy,
        request: OperationRequest,
    ) -> dict[str, object] | None:
        del policy
        return self.release_results.get(request.operation_id)


def _policy(tmp_path: Path) -> Path:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_root.chmod(0o700)
    source_rows: list[BackupPolicySource] = []
    modes = {
        BackupStateOwner.RELEASE_CONFIG: "application_quiesced",
        BackupStateOwner.POSTGRESQL: "database_snapshot",
        BackupStateOwner.CLICKHOUSE: "database_snapshot",
        BackupStateOwner.REDIS_CHECKPOINT: "durable_checkpoint",
        BackupStateOwner.OPENBAO: "encrypted_raft_snapshot",
        BackupStateOwner.ARTIFACTS: "content_addressed_snapshot",
        BackupStateOwner.PLUGIN_OPERATION_AUDIT: "application_quiesced",
        BackupStateOwner.OBSERVABILITY: "bounded_history_snapshot",
    }
    for owner in BackupStateOwner:
        filename = f"{owner.value}.snapshot"
        (source_root / filename).write_text(
            json.dumps({"owner": owner.value, "fixture": True}),
            encoding="utf-8",
        )
        (source_root / filename).chmod(0o600)
        source_rows.append(
            BackupPolicySource(
                owner=owner,
                filename=filename,
                media_type="application/json",
                consistency_mode=modes[owner],  # type: ignore[arg-type]
                source_schema_version="fixture-v1",
            )
        )

    age_identity = tmp_path / "operator.agekey"
    created = subprocess.run(
        ["age-keygen", "-o", str(age_identity)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0
    recipient = subprocess.run(
        ["age-keygen", "-y", str(age_identity)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert recipient.returncode == 0
    age_recipient = tmp_path / "operator.recipient"
    age_recipient.write_text(recipient.stdout, encoding="utf-8")
    os.chmod(age_identity, 0o600)

    signer = Ed25519PrivateKey.generate()
    signing_file = tmp_path / "operator-signing.pem"
    verification_file = tmp_path / "operator-verification.pem"
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
    release_policy_file = tmp_path / "release-policy.json"
    release_policy = InstallationReleasePolicy(
        installation_fingerprint="sha256:" + "4" * 64,
        transitions=(
            ReleaseTransitionRule(
                from_release="0.1.0",
                to_release="0.2.0",
                reversible=True,
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
        installation_fingerprint="sha256:" + "4" * 64,
        release_version="0.1.0",
        source_root=source_root,
        backup_root=tmp_path / "backups",
        restore_root=tmp_path / "restores",
        age_recipient_file=age_recipient,
        age_identity_file=age_identity,
        signing_private_key_file=signing_file,
        verification_public_key_file=verification_file,
        cancellation_root=tmp_path / "cancel",
        capture_record_file=tmp_path / "capture-record.json",
        release_policy_file=release_policy_file,
        sources=tuple(source_rows),
    )
    captured_at = datetime(2026, 7, 14, 11, 59, 59, tzinfo=UTC)
    capture = InstallationCaptureRecord(
        installation_fingerprint=policy.installation_fingerprint,
        source_release_version=policy.release_version,
        quiesce_started_at=captured_at,
        quiesce_completed_at=captured_at,
        entries=tuple(
            BackupCaptureEntry(
                owner=source.owner,
                captured_at=captured_at,
                plaintext_sha256=(
                    "sha256:"
                    + hashlib.sha256((source_root / source.filename).read_bytes()).hexdigest()
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
    policy_path = tmp_path / "backup-policy.json"
    policy_path.write_text(
        json.dumps(policy.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )
    return policy_path


def _request(
    *, action: OperationAction, operation_number: int, subject_id: str | None = None
) -> OperationRequest:
    return OperationRequest(
        operation_id=UUID(f"00000000-0000-4000-8000-{operation_number:012d}"),
        action=action,
        profile="base",
        subject_id=subject_id,
        release_version=("0.2.0" if action is OperationAction.UPDATE else None),
    )


def test_backup_backend_executes_and_reconciles_backup_and_fresh_restore(
    tmp_path: Path,
) -> None:
    policy_path = _policy(tmp_path)
    backend = InstallationBackupControlBackend(
        policy_path=policy_path,
        receipt_root=tmp_path / "receipts",
        now=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        state_coordinator=_FixtureStateCoordinator(),
    )
    backup = _request(
        action=OperationAction.BACKUP,
        operation_number=2101,
        subject_id="backup-001",
    )
    restore = _request(
        action=OperationAction.RESTORE,
        operation_number=2102,
        subject_id="backup-001",
    )

    assert backend.execute(backup).detail_code == "backup.completed"
    assert not list((tmp_path / "sources").iterdir())
    assert backend.reconcile(backup).state is OperationState.SUCCEEDED
    assert backend.execute(restore).detail_code == "restore.completed"
    assert backend.reconcile(restore).state is OperationState.SUCCEEDED
    assert len(list((tmp_path / "restores" / "backup-001").glob("*.snapshot"))) == 8


def test_recovery_backend_requires_verified_preupgrade_backup(tmp_path: Path) -> None:
    policy_path = _policy(tmp_path)
    coordinator = _FixtureStateCoordinator()
    backup_backend = InstallationBackupControlBackend(
        policy_path=policy_path,
        receipt_root=tmp_path / "receipts",
        now=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
        state_coordinator=coordinator,
    )
    runtime = _RuntimeBackend()
    recovery = RecoveryControlBackend(
        runtime_backend=runtime,
        backup_backend=backup_backend,
        current_release=lambda: "0.1.0",
    )
    update = _request(action=OperationAction.UPDATE, operation_number=2103)
    with pytest.raises(ControlOperationError, match="backup.preupgrade_required"):
        recovery.execute(update)

    recovery.execute(
        _request(
            action=OperationAction.BACKUP,
            operation_number=2104,
            subject_id="preupgrade-001",
        )
    )
    assert recovery.execute(update).detail_code == "upgrade.completed"
    assert runtime.executed == []


def test_backup_backend_rejects_profile_or_non_backup_action(tmp_path: Path) -> None:
    backend = InstallationBackupControlBackend(
        policy_path=_policy(tmp_path),
        receipt_root=tmp_path / "receipts",
    )
    wrong_profile = OperationRequest(
        operation_id=UUID("00000000-0000-4000-8000-000000002105"),
        action=OperationAction.BACKUP,
        profile="trading",
        subject_id="backup-001",
    )
    with pytest.raises(ControlOperationError, match="backup.request_rejected"):
        backend.execute(wrong_profile)


def test_backup_backend_records_typed_cancellation_marker(tmp_path: Path) -> None:
    backend = InstallationBackupControlBackend(
        policy_path=_policy(tmp_path),
        receipt_root=tmp_path / "receipts",
    )
    target_operation = UUID("00000000-0000-4000-8000-000000002106")
    request = OperationRequest(
        operation_id=UUID("00000000-0000-4000-8000-000000002107"),
        action=OperationAction.BACKUP_CANCEL,
        profile="base",
        subject_id=str(target_operation),
    )

    result = backend.execute(request)

    assert result.detail_code == "backup.cancellation_requested"
    assert backend.reconcile(request).state is OperationState.SUCCEEDED
    assert (tmp_path / "cancel" / f"{target_operation}.cancel").is_file()


def test_backup_policy_schema_rejects_duplicate_owner_semantics(tmp_path: Path) -> None:
    policy_payload = json.loads(_policy(tmp_path).read_text(encoding="utf-8"))
    policy_payload["sources"][-1]["owner"] = policy_payload["sources"][0]["owner"]
    schema = json.loads(
        (Path("schemas/backup/installation-backup-policy.schema.json")).read_text(
            encoding="utf-8"
        )
    )

    assert list(Draft202012Validator(schema).iter_errors(policy_payload))


def test_backup_reconciles_crash_after_verified_effect_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = InstallationBackupControlBackend(
        policy_path=_policy(tmp_path),
        receipt_root=tmp_path / "receipts",
        state_coordinator=_FixtureStateCoordinator(),
        now=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
    )
    service = ControlOperationService(
        backend=backend,
        journal=AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl"),
    )
    request = _request(
        action=OperationAction.BACKUP,
        operation_number=2110,
        subject_id="crash-after-effect",
    )
    original = backup_backend_module._write_new_or_identical

    def fail_receipt(path: Path, payload: object) -> None:
        del path, payload
        raise ControlOperationError(code="backup.receipt_unavailable")

    monkeypatch.setattr(backup_backend_module, "_write_new_or_identical", fail_receipt)
    assert service.submit(request).state is OperationState.UNKNOWN
    monkeypatch.setattr(backup_backend_module, "_write_new_or_identical", original)

    reconciled = service.reconcile(request.operation_id)

    assert reconciled.state is OperationState.SUCCEEDED
    assert (tmp_path / "backups/latest-verified.json").is_file()


def test_update_reconciles_crash_after_release_commit_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = _policy(tmp_path)
    coordinator = _FixtureStateCoordinator()
    backup_backend = InstallationBackupControlBackend(
        policy_path=policy_path,
        receipt_root=tmp_path / "receipts",
        state_coordinator=coordinator,
    )
    recovery = RecoveryControlBackend(
        runtime_backend=_RuntimeBackend(),
        backup_backend=backup_backend,
        current_release=lambda: "0.1.0",
    )
    service = ControlOperationService(
        backend=recovery,
        journal=AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl"),
    )
    backup = _request(
        action=OperationAction.BACKUP,
        operation_number=2111,
        subject_id="pre-update",
    )
    update = _request(action=OperationAction.UPDATE, operation_number=2112)
    assert service.submit(backup).state is OperationState.SUCCEEDED
    original = backup_backend_module._write_new_or_identical

    def fail_receipt(path: Path, payload: object) -> None:
        del path, payload
        raise ControlOperationError(code="backup.receipt_unavailable")

    monkeypatch.setattr(backup_backend_module, "_write_new_or_identical", fail_receipt)
    assert service.submit(update).state is OperationState.UNKNOWN
    monkeypatch.setattr(backup_backend_module, "_write_new_or_identical", original)

    assert service.reconcile(update.operation_id).state is OperationState.SUCCEEDED


def test_irreversible_update_without_trusted_forward_plan_is_rejected(
    tmp_path: Path,
) -> None:
    policy_path = _policy(tmp_path)
    policy = InstallationBackupPolicy.model_validate_json(policy_path.read_bytes())
    release_policy = InstallationReleasePolicy(
        installation_fingerprint=policy.installation_fingerprint,
        transitions=(
            ReleaseTransitionRule(
                from_release="0.1.0",
                to_release="0.2.0",
                reversible=False,
            ),
        ),
    )
    policy.release_policy_file.write_text(
        release_policy.model_dump_json(by_alias=True),
        encoding="utf-8",
    )
    coordinator = _FixtureStateCoordinator()
    backend = InstallationBackupControlBackend(
        policy_path=policy_path,
        receipt_root=tmp_path / "receipts",
        state_coordinator=coordinator,
    )
    backend.execute(
        _request(
            action=OperationAction.BACKUP,
            operation_number=2113,
            subject_id="pre-update",
        )
    )

    with pytest.raises(ControlOperationError, match="upgrade.forward_recovery_plan_required"):
        backend.execute_release(
            _request(action=OperationAction.UPDATE, operation_number=2114),
            current_release_version="0.1.0",
        )
