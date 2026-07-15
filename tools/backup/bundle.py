"""Signed, age-encrypted, resumable installation backup bundles."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from trading.contexts.operations import (
    REQUIRED_BACKUP_STATE_OWNERS,
    BackupManifestEntry,
    BackupManifestSignature,
    BackupStateOwner,
    ConsistencyMode,
    InstallationBackupManifest,
)

_MAX_SNAPSHOT_BYTES = 1_073_741_824
_MAX_CIPHERTEXT_BYTES = _MAX_SNAPSHOT_BYTES + 1_048_576
_MAX_METADATA_BYTES = 1_048_576
_MANIFEST_NAME = "backup-manifest.json"
_SIGNATURE_NAME = "backup-manifest.signature.json"
_PROGRESS_NAME = "backup-progress.json"
_RESTORE_PROGRESS_NAME = "restore-progress.json"
_RESTORE_RESULT_NAME = "restore-result.json"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")

class BackupBundleError(RuntimeError):
    """Stable, sanitized backup failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class BackupSource:
    owner: BackupStateOwner
    path: Path
    media_type: str
    consistency_mode: ConsistencyMode
    source_schema_version: str
    captured_at: datetime
    expected_plaintext_sha256: str
    limitations: tuple[str, ...] = ()


def create_backup(
    *,
    backup_id: str,
    installation_fingerprint: str,
    source_release_version: str,
    sources: tuple[BackupSource, ...],
    backup_root: Path,
    age_recipient_file: Path,
    age_identity_file: Path,
    signing_private_key_file: Path,
    verification_public_key_file: Path,
    quiesce_started_at: datetime,
    quiesce_completed_at: datetime,
    cancellation_file: Path | None = None,
    fail_after_entries: int | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Encrypt every state-owner snapshot and atomically publish a signed manifest."""

    clock = now or (lambda: datetime.now(UTC))
    started = time.monotonic()
    _validate_identifier(backup_id)
    if not installation_fingerprint.startswith("sha256:"):
        raise BackupBundleError("backup.installation_fingerprint_invalid")
    _validate_sources(
        sources,
        quiesce_started_at=quiesce_started_at,
        quiesce_completed_at=quiesce_completed_at,
    )
    _require_executable("age")
    recipient = _age_recipient(age_recipient_file)
    identity_file = _regular_path(age_identity_file, restricted=True)
    signer = _load_private_key(signing_private_key_file)
    bundle = _prepare_bundle_root(backup_root=backup_root, backup_id=backup_id)
    progress_path = bundle / _PROGRESS_NAME
    progress = _load_or_create_backup_progress(progress_path, backup_id=backup_id)
    completed = cast(dict[str, dict[str, object]], progress["completed"])
    resumed_entries = len(completed)

    for source in sorted(sources, key=lambda item: item.owner.value):
        if _cancelled(cancellation_file):
            progress["state"] = "cancelled"
            _atomic_json(progress_path, progress)
            raise BackupBundleError("backup.cancelled")
        owner = source.owner.value
        previous = completed.get(owner)
        if previous is not None:
            entry = BackupManifestEntry.model_validate(previous)
            _verify_ciphertext(bundle / entry.artifact_name, entry)
            continue
        plaintext = _read_regular(source.path, max_bytes=_MAX_SNAPSHOT_BYTES)
        if _sha256(plaintext) != source.expected_plaintext_sha256:
            raise BackupBundleError("backup.capture_digest_mismatch")
        artifact_name = f"{owner}.snapshot.age"
        ciphertext_path = bundle / artifact_name
        try:
            _encrypt_payload(
                payload=plaintext,
                destination=ciphertext_path,
                recipient=recipient,
                cancellation_file=cancellation_file,
            )
        except BackupBundleError as error:
            if error.code == "backup.cancelled":
                progress["state"] = "cancelled"
                _atomic_json(progress_path, progress)
            raise
        ciphertext = _read_regular(ciphertext_path, max_bytes=_MAX_CIPHERTEXT_BYTES)
        entry = BackupManifestEntry(
            owner=source.owner,
            artifact_name=artifact_name,
            media_type=source.media_type,
            consistency_mode=source.consistency_mode,
            source_schema_version=source.source_schema_version,
            captured_at=source.captured_at,
            plaintext_bytes=len(plaintext),
            plaintext_sha256=_sha256(plaintext),
            ciphertext_bytes=len(ciphertext),
            ciphertext_sha256=_sha256(ciphertext),
            limitations=source.limitations,
        )
        completed[owner] = entry.model_dump(mode="json")
        progress["completed"] = completed
        progress["state"] = "running"
        _atomic_json(progress_path, progress)
        if fail_after_entries is not None and len(completed) >= fail_after_entries:
            progress["state"] = "partial"
            _atomic_json(progress_path, progress)
            raise BackupBundleError("backup.injected_partial_failure")

    completed_at = clock()
    entries = tuple(
        BackupManifestEntry.model_validate(completed[owner.value])
        for owner in sorted(REQUIRED_BACKUP_STATE_OWNERS, key=lambda item: item.value)
    )
    verifier = signer.public_key()
    key_id = _public_key_id(verifier)
    observed_rpo = max(
        0.0,
        max((quiesce_completed_at - entry.captured_at).total_seconds() for entry in entries),
    )
    manifest = InstallationBackupManifest(
        backup_id=backup_id,
        installation_fingerprint=installation_fingerprint,
        source_release_version=source_release_version,
        created_at=completed_at,
        quiesce_started_at=quiesce_started_at,
        quiesce_completed_at=quiesce_completed_at,
        observed_rpo_seconds=observed_rpo,
        signing_key_id=key_id,
        entries=entries,
    )
    signature = BackupManifestSignature(
        key_id=key_id,
        manifest_sha256=manifest.manifest_sha256,
        signature_base64=base64.b64encode(signer.sign(manifest.canonical_bytes())).decode(),
    )
    _atomic_bytes(bundle / _MANIFEST_NAME, manifest.canonical_bytes(), mode=0o600)
    _atomic_json(
        bundle / _SIGNATURE_NAME,
        signature.model_dump(mode="json", by_alias=True),
    )
    try:
        verify_backup(
            bundle=bundle,
            verification_public_key_file=verification_public_key_file,
            expected_installation_fingerprint=installation_fingerprint,
            age_identity_file=identity_file,
            cancellation_file=cancellation_file,
        )
        if _cancelled(cancellation_file):
            raise BackupBundleError("backup.cancelled")
    except BackupBundleError as error:
        if error.code == "backup.cancelled":
            (bundle / _MANIFEST_NAME).unlink(missing_ok=True)
            (bundle / _SIGNATURE_NAME).unlink(missing_ok=True)
            progress["state"] = "cancelled"
            _atomic_json(progress_path, progress)
        raise
    progress["state"] = "completed"
    progress["manifest_sha256"] = manifest.manifest_sha256
    _atomic_json(progress_path, progress)
    return {
        "schema": "io.roehub.installation-backup-result/v1alpha1",
        "operation": "backup",
        "status": "passed",
        "backup_id": backup_id,
        "manifest_sha256": manifest.manifest_sha256,
        "state_owner_count": len(entries),
        "encrypted_entry_count": len(entries),
        "observed_rpo_seconds": observed_rpo,
        "duration_seconds": max(0.0, time.monotonic() - started),
        "resumed_entries": resumed_entries,
    }


def verify_backup(
    *,
    bundle: Path,
    verification_public_key_file: Path,
    expected_installation_fingerprint: str | None = None,
    age_identity_file: Path | None = None,
    cancellation_file: Path | None = None,
) -> InstallationBackupManifest:
    """Verify signature, exact owner coverage, ciphertext digests, and bundle shape."""

    root = _absolute_directory(bundle)
    manifest = InstallationBackupManifest.model_validate_json(
        _read_regular(root / _MANIFEST_NAME, max_bytes=_MAX_METADATA_BYTES)
    )
    if (
        expected_installation_fingerprint is not None
        and manifest.installation_fingerprint != expected_installation_fingerprint
    ):
        raise BackupBundleError("backup.installation_fingerprint_mismatch")
    signature = BackupManifestSignature.model_validate_json(
        _read_regular(root / _SIGNATURE_NAME, max_bytes=_MAX_METADATA_BYTES)
    )
    verifier = _load_public_key(verification_public_key_file)
    key_id = _public_key_id(verifier)
    if (
        signature.key_id != key_id
        or manifest.signing_key_id != key_id
        or signature.manifest_sha256 != manifest.manifest_sha256
    ):
        raise BackupBundleError("backup.signature_identity_mismatch")
    try:
        verifier.verify(
            base64.b64decode(signature.signature_base64, validate=True),
            manifest.canonical_bytes(),
        )
    except (InvalidSignature, ValueError) as error:
        raise BackupBundleError("backup.signature_invalid") from error
    if root.name != manifest.backup_id:
        raise BackupBundleError("backup.bundle_identity_mismatch")
    expected = {_MANIFEST_NAME, _SIGNATURE_NAME, _PROGRESS_NAME}
    for entry in manifest.entries:
        expected.add(entry.artifact_name)
        _verify_ciphertext(root / entry.artifact_name, entry)
    if {path.name for path in root.iterdir()} != expected:
        raise BackupBundleError("backup.bundle_shape_invalid")
    if age_identity_file is not None:
        _verify_decryptability(
            bundle=root,
            manifest=manifest,
            age_identity_file=age_identity_file,
            cancellation_file=cancellation_file,
        )
    return manifest


def restore_backup(
    *,
    bundle: Path,
    restore_root: Path,
    age_identity_file: Path,
    verification_public_key_file: Path,
    expected_installation_fingerprint: str | None = None,
    protected_source_root: Path | None = None,
    cancellation_file: Path | None = None,
    fail_after_entries: int | None = None,
    apply_restored_state: Callable[
        [Path, InstallationBackupManifest], Mapping[str, object]
    ]
    | None = None,
) -> dict[str, object]:
    """Restore only into a fresh target, supporting cancellation and safe resume."""

    started = time.monotonic()
    manifest = verify_backup(
        bundle=bundle,
        verification_public_key_file=verification_public_key_file,
        expected_installation_fingerprint=expected_installation_fingerprint,
        age_identity_file=age_identity_file,
    )
    source_root = _absolute_directory(bundle)
    target = _prepare_restore_root(
        restore_root,
        source_root=source_root,
        protected_source_root=protected_source_root,
    )
    identity_file = _regular_path(age_identity_file, restricted=True)
    progress_path = target / _RESTORE_PROGRESS_NAME
    progress = _load_or_create_restore_progress(
        progress_path,
        backup_id=manifest.backup_id,
        manifest_sha256=manifest.manifest_sha256,
    )
    completed = cast(dict[str, str], progress["completed"])
    _validate_restore_resume_shape(target, completed=completed)
    for entry in manifest.entries:
        if _cancelled(cancellation_file):
            progress["state"] = "cancelled"
            _atomic_json(progress_path, progress)
            raise BackupBundleError("restore.cancelled")
        destination = target / f"{entry.owner.value}.snapshot"
        expected = completed.get(entry.owner.value)
        if expected is not None:
            payload = _read_regular(destination, max_bytes=_MAX_SNAPSHOT_BYTES)
            if _sha256(payload) != expected or expected != entry.plaintext_sha256:
                raise BackupBundleError("restore.resume_digest_mismatch")
            continue
        ciphertext = _read_regular(
            source_root / entry.artifact_name,
            max_bytes=_MAX_CIPHERTEXT_BYTES,
        )
        if (
            len(ciphertext) != entry.ciphertext_bytes
            or _sha256(ciphertext) != entry.ciphertext_sha256
        ):
            raise BackupBundleError("backup.ciphertext_digest_mismatch")
        try:
            _decrypt_payload_to_file(
                payload=ciphertext,
                destination=destination,
                identity_file=identity_file,
                cancellation_file=cancellation_file,
            )
        except BackupBundleError as error:
            if error.code == "restore.cancelled":
                progress["state"] = "cancelled"
                _atomic_json(progress_path, progress)
            raise
        payload = _read_regular(destination, max_bytes=_MAX_SNAPSHOT_BYTES)
        if len(payload) != entry.plaintext_bytes or _sha256(payload) != entry.plaintext_sha256:
            destination.unlink(missing_ok=True)
            raise BackupBundleError("restore.plaintext_digest_mismatch")
        completed[entry.owner.value] = entry.plaintext_sha256
        progress["completed"] = completed
        progress["state"] = "running"
        _atomic_json(progress_path, progress)
        if fail_after_entries is not None and len(completed) >= fail_after_entries:
            progress["state"] = "partial"
            _atomic_json(progress_path, progress)
            raise BackupBundleError("restore.injected_partial_failure")
    if apply_restored_state is None:
        raise BackupBundleError("restore.state_coordinator_required")
    state_result = dict(apply_restored_state(target, manifest))
    if state_result.get("status") != "ready":
        raise BackupBundleError("restore.state_not_ready")
    duration = max(0.0, time.monotonic() - started)
    result = {
        "schema": "io.roehub.installation-restore-result/v1alpha1",
        "operation": "restore",
        "status": "passed",
        "backup_id": manifest.backup_id,
        "manifest_sha256": manifest.manifest_sha256,
        "fresh_target_guard": "passed",
        "source_target_separation": "passed",
        "signature_verified": True,
        "restored_state_owner_count": len(completed),
        "observed_rto_seconds": duration,
        "state_owner_import": state_result,
    }
    _atomic_json(target / _RESTORE_RESULT_NAME, result)
    progress["state"] = "completed"
    progress["result_sha256"] = _sha256(_json_bytes(result))
    _atomic_json(progress_path, progress)
    return result


def _validate_sources(
    sources: tuple[BackupSource, ...],
    *,
    quiesce_started_at: datetime,
    quiesce_completed_at: datetime,
) -> None:
    owners = [source.owner for source in sources]
    if len(owners) != len(set(owners)) or set(owners) != REQUIRED_BACKUP_STATE_OWNERS:
        raise BackupBundleError("backup.state_owner_coverage_incomplete")
    if (
        quiesce_started_at.tzinfo is None
        or quiesce_completed_at.tzinfo is None
        or quiesce_completed_at < quiesce_started_at
    ):
        raise BackupBundleError("backup.quiesce_timestamp_invalid")
    for source in sources:
        if (
            source.captured_at.tzinfo is None
            or source.captured_at < quiesce_started_at
            or source.captured_at > quiesce_completed_at
        ):
            raise BackupBundleError("backup.capture_timestamp_invalid")
        if (
            not source.media_type
            or not source.source_schema_version
            or re.fullmatch(r"sha256:[0-9a-f]{64}", source.expected_plaintext_sha256)
            is None
        ):
            raise BackupBundleError("backup.source_metadata_invalid")
        _regular_path(source.path, restricted=True)


def _load_or_create_backup_progress(path: Path, *, backup_id: str) -> dict[str, object]:
    if not path.exists():
        progress: dict[str, object] = {
            "schema": "io.roehub.installation-backup-progress/v1alpha1",
            "backup_id": backup_id,
            "state": "running",
            "completed": {},
        }
        _atomic_json(path, progress)
        return progress
    progress = _read_json(path)
    if (
        progress.get("schema") != "io.roehub.installation-backup-progress/v1alpha1"
        or progress.get("backup_id") != backup_id
        or progress.get("state") == "completed"
        or not isinstance(progress.get("completed"), dict)
    ):
        raise BackupBundleError("backup.progress_invalid")
    return progress


def _load_or_create_restore_progress(
    path: Path,
    *,
    backup_id: str,
    manifest_sha256: str,
) -> dict[str, object]:
    if not path.exists():
        progress: dict[str, object] = {
            "schema": "io.roehub.installation-restore-progress/v1alpha1",
            "backup_id": backup_id,
            "manifest_sha256": manifest_sha256,
            "state": "running",
            "completed": {},
        }
        _atomic_json(path, progress)
        return progress
    progress = _read_json(path)
    if (
        progress.get("schema") != "io.roehub.installation-restore-progress/v1alpha1"
        or progress.get("backup_id") != backup_id
        or progress.get("manifest_sha256") != manifest_sha256
        or progress.get("state") == "completed"
        or not isinstance(progress.get("completed"), dict)
    ):
        raise BackupBundleError("restore.progress_invalid")
    return progress


def _prepare_bundle_root(*, backup_root: Path, backup_id: str) -> Path:
    root = _absolute_path(backup_root)
    if os.path.lexists(root) and (root.is_symlink() or not root.is_dir()):
        raise BackupBundleError("backup.root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    bundle = root / backup_id
    if os.path.lexists(bundle) and (bundle.is_symlink() or not bundle.is_dir()):
        raise BackupBundleError("backup.destination_unsafe")
    bundle.mkdir(mode=0o700, exist_ok=True)
    os.chmod(bundle, 0o700)
    if (bundle / _MANIFEST_NAME).exists():
        raise BackupBundleError("backup.destination_exists")
    return bundle


def _prepare_restore_root(
    path: Path,
    *,
    source_root: Path,
    protected_source_root: Path | None,
) -> Path:
    target = _absolute_path(path)
    protected = [source_root]
    if protected_source_root is not None:
        protected.append(_absolute_path(protected_source_root))
    if any(
        target == root or target in root.parents or root in target.parents
        for root in protected
    ):
        raise BackupBundleError("restore.source_target_overlap")
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_dir():
            raise BackupBundleError("restore.target_unsafe")
        names = {item.name for item in target.iterdir()}
        if names and _RESTORE_PROGRESS_NAME not in names:
            raise BackupBundleError("restore.target_not_empty")
    else:
        target.mkdir(parents=True, mode=0o700)
    os.chmod(target, 0o700)
    return target


def _validate_restore_resume_shape(target: Path, *, completed: dict[str, str]) -> None:
    expected = {_RESTORE_PROGRESS_NAME}
    expected.update(f"{owner}.snapshot" for owner in completed)
    if {item.name for item in target.iterdir()} != expected:
        raise BackupBundleError("restore.target_not_fresh")


def _encrypt_payload(
    *,
    payload: bytes,
    destination: Path,
    recipient: str,
    cancellation_file: Path | None = None,
) -> None:
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as output:
            process = subprocess.Popen(
                ["age", "--encrypt", "--recipient", recipient],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.DEVNULL,
            )
            returncode = _stream_subprocess_input(
                process,
                payload=payload,
                cancellation_file=cancellation_file,
                cancellation_code="backup.cancelled",
                failure_code="backup.encryption_failed",
            )
        if returncode != 0 or temporary.stat().st_size <= 0:
            raise BackupBundleError("backup.encryption_failed")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except (OSError, subprocess.SubprocessError) as error:
        raise BackupBundleError("backup.encryption_failed") from error
    finally:
        temporary.unlink(missing_ok=True)


def _decrypt_payload_to_file(
    *,
    payload: bytes,
    destination: Path,
    identity_file: Path,
    cancellation_file: Path | None = None,
) -> None:
    temporary = _temporary_path(destination)
    descriptor = _open_regular(identity_file, restricted=True)
    try:
        with temporary.open("wb") as output:
            process = subprocess.Popen(
                ["age", "--decrypt", "--identity", f"/dev/fd/{descriptor}"],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.DEVNULL,
                pass_fds=(descriptor,),
            )
            returncode = _stream_subprocess_input(
                process,
                payload=payload,
                cancellation_file=cancellation_file,
                cancellation_code="restore.cancelled",
                failure_code="restore.decryption_failed",
            )
        if returncode != 0 or temporary.stat().st_size <= 0:
            raise BackupBundleError("restore.decryption_failed")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except (OSError, subprocess.SubprocessError) as error:
        raise BackupBundleError("restore.decryption_failed") from error
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _stream_subprocess_input(
    process: subprocess.Popen[bytes],
    *,
    payload: bytes,
    cancellation_file: Path | None,
    cancellation_code: str,
    failure_code: str,
) -> int:
    try:
        if process.stdin is None:
            raise BackupBundleError(failure_code)
        for offset in range(0, len(payload), 1024 * 1024):
            if _cancelled(cancellation_file):
                process.terminate()
                process.wait(timeout=10)
                raise BackupBundleError(cancellation_code)
            process.stdin.write(payload[offset : offset + 1024 * 1024])
        process.stdin.close()
        while process.poll() is None:
            if _cancelled(cancellation_file):
                process.terminate()
                process.wait(timeout=10)
                raise BackupBundleError(cancellation_code)
            time.sleep(0.01)
        return int(process.returncode)
    except (BrokenPipeError, OSError, subprocess.SubprocessError) as error:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        raise BackupBundleError(failure_code) from error


def _decrypt_payload(*, payload: bytes, identity_file: Path) -> bytes:
    descriptor = _open_regular(identity_file, restricted=True)
    try:
        completed = subprocess.run(
            ["age", "--decrypt", "--identity", f"/dev/fd/{descriptor}"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=300,
            pass_fds=(descriptor,),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BackupBundleError("restore.decryption_failed") from error
    finally:
        os.close(descriptor)
    if completed.returncode != 0 or not completed.stdout:
        raise BackupBundleError("restore.decryption_failed")
    return completed.stdout


def _verify_decryptability(
    *,
    bundle: Path,
    manifest: InstallationBackupManifest,
    age_identity_file: Path,
    cancellation_file: Path | None = None,
) -> None:
    for entry in manifest.entries:
        if _cancelled(cancellation_file):
            raise BackupBundleError("backup.cancelled")
        ciphertext = _read_regular(
            bundle / entry.artifact_name,
            max_bytes=_MAX_CIPHERTEXT_BYTES,
        )
        plaintext = _decrypt_payload(payload=ciphertext, identity_file=age_identity_file)
        if _cancelled(cancellation_file):
            raise BackupBundleError("backup.cancelled")
        if len(plaintext) != entry.plaintext_bytes or _sha256(plaintext) != entry.plaintext_sha256:
            raise BackupBundleError("backup.decryptability_check_failed")


def _age_recipient(path: Path) -> str:
    payload = _read_regular(_regular_path(path, restricted=False), max_bytes=4096)
    try:
        recipient = payload.decode("ascii").strip()
    except UnicodeError as error:
        raise BackupBundleError("backup.recipient_invalid") from error
    if not recipient.startswith("age1") or any(character.isspace() for character in recipient):
        raise BackupBundleError("backup.recipient_invalid")
    return recipient


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    data = _read_regular(_regular_path(path, restricted=True), max_bytes=65_536)
    try:
        loaded = serialization.load_pem_private_key(data, None)
    except (TypeError, ValueError) as error:
        raise BackupBundleError("backup.signing_key_invalid") from error
    if not isinstance(loaded, Ed25519PrivateKey):
        raise BackupBundleError("backup.signing_key_invalid")
    return loaded


def _load_public_key(path: Path) -> Ed25519PublicKey:
    data = _read_regular(_regular_path(path, restricted=False), max_bytes=65_536)
    try:
        loaded = serialization.load_pem_public_key(data)
    except ValueError as error:
        raise BackupBundleError("backup.verification_key_invalid") from error
    if not isinstance(loaded, Ed25519PublicKey):
        raise BackupBundleError("backup.verification_key_invalid")
    return loaded


def _public_key_id(verifier: Ed25519PublicKey) -> str:
    raw = verifier.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _sha256(raw)


def _verify_ciphertext(path: Path, entry: BackupManifestEntry) -> None:
    payload = _read_regular(path, max_bytes=_MAX_CIPHERTEXT_BYTES)
    if len(payload) != entry.ciphertext_bytes or _sha256(payload) != entry.ciphertext_sha256:
        raise BackupBundleError("backup.ciphertext_digest_mismatch")


def _regular_path(path: Path, *, restricted: bool) -> Path:
    candidate = _absolute_path(path)
    _reject_symlink_ancestors(candidate)
    if candidate.is_symlink() or not candidate.is_file():
        raise BackupBundleError("backup.file_unsafe")
    mode = candidate.stat().st_mode
    if restricted and (
        mode & (stat.S_IRWXG | stat.S_IRWXO)
        or candidate.stat().st_uid != os.geteuid()
    ):
        raise BackupBundleError("backup.file_permissions_unsafe")
    return candidate


def _read_regular(path: Path, *, max_bytes: int) -> bytes:
    candidate = _regular_path(path, restricted=False)
    descriptor = _open_regular(candidate, restricted=False)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > max_bytes:
            raise BackupBundleError("backup.file_size_invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise BackupBundleError("backup.file_size_invalid")
        return payload
    finally:
        os.close(descriptor)


def _open_regular(path: Path, *, restricted: bool) -> int:
    candidate = _regular_path(path, restricted=restricted)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise BackupBundleError("backup.no_follow_unavailable")
    try:
        descriptor = os.open(candidate, os.O_RDONLY | os.O_CLOEXEC | no_follow)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise BackupBundleError("backup.file_unsafe")
        if restricted and (
            info.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
            or info.st_uid != os.geteuid()
        ):
            os.close(descriptor)
            raise BackupBundleError("backup.file_permissions_unsafe")
        return descriptor
    except OSError as error:
        raise BackupBundleError("backup.file_unavailable") from error


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(_read_regular(path, max_bytes=_MAX_METADATA_BYTES))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BackupBundleError("backup.metadata_invalid") from error
    if not isinstance(payload, dict):
        raise BackupBundleError("backup.metadata_invalid")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _json_bytes(payload), mode=0o600)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _atomic_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    temporary = _temporary_path(path)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_TRUNC)
        try:
            os.fchmod(descriptor, mode)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        os.chmod(path, mode)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    path = Path(name)
    os.chmod(path, 0o600)
    return path


def _absolute_directory(path: Path) -> Path:
    candidate = _absolute_path(path)
    _reject_symlink_ancestors(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise BackupBundleError("backup.bundle_unsafe")
    return candidate


def _absolute_path(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise BackupBundleError("backup.path_not_absolute")
    return candidate


def _reject_symlink_ancestors(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            return
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise BackupBundleError("backup.path_symlink_ancestor")
        except OSError as error:
            raise BackupBundleError("backup.path_unavailable") from error


def _validate_identifier(value: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise BackupBundleError("backup.identifier_invalid")


def _cancelled(path: Path | None) -> bool:
    return path is not None and os.path.lexists(path)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise BackupBundleError("backup.encryption_tool_unavailable")
