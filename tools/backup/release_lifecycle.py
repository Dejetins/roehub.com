"""Backup-gated first-release upgrade and rollback rehearsal helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from tools.backup.bundle import BackupBundleError, restore_backup, verify_backup
from trading.contexts.operations import InstallationBackupManifest

_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def upgrade_from_backup(
    *,
    bundle: Path,
    target_root: Path,
    age_identity_file: Path,
    verification_public_key_file: Path,
    target_release_version: str,
    irreversible: bool = False,
    forward_recovery_plan: str | None = None,
    fail_before_commit: bool = False,
    apply_restored_state: Callable[[Path, InstallationBackupManifest], Mapping[str, object]],
    operation_id: str | None = None,
) -> dict[str, object]:
    """Restore N-1 to a fresh target and atomically commit a higher release marker."""

    started = time.monotonic()
    manifest = verify_backup(
        bundle=bundle,
        verification_public_key_file=verification_public_key_file,
    )
    if not _is_upgrade(manifest.source_release_version, target_release_version):
        raise BackupBundleError("upgrade.release_transition_invalid")
    if irreversible and not forward_recovery_plan:
        raise BackupBundleError("upgrade.forward_recovery_plan_required")
    restore_result_path = target_root / "restore-result.json"
    if restore_result_path.is_file():
        _verify_restored_target(target_root, manifest=manifest)
    else:
        restore_backup(
            bundle=bundle,
            restore_root=target_root,
            age_identity_file=age_identity_file,
            verification_public_key_file=verification_public_key_file,
            apply_restored_state=apply_restored_state,
        )
    progress = {
        "schema": "io.roehub.release-upgrade-progress/v1alpha1",
        "from_release": manifest.source_release_version,
        "to_release": target_release_version,
        "backup_manifest_sha256": manifest.manifest_sha256,
        "state": "failed_before_commit" if fail_before_commit else "ready_to_commit",
        "operation_id": operation_id,
    }
    _atomic_json(target_root / "upgrade-progress.json", progress)
    if fail_before_commit:
        raise BackupBundleError("upgrade.injected_failure_before_commit")
    result = {
        "schema": "io.roehub.release-upgrade-result/v1alpha1",
        "status": "passed",
        "from_release": manifest.source_release_version,
        "to_release": target_release_version,
        "preupgrade_backup_verified": True,
        "backup_manifest_sha256": manifest.manifest_sha256,
        "state_owner_count": len(manifest.entries),
        "migration_mode": "versioned-first-v1-fixture",
        "reversible": not irreversible,
        "forward_recovery_plan_recorded": bool(forward_recovery_plan),
        "operation_id": operation_id,
        "observed_upgrade_seconds": max(0.0, time.monotonic() - started),
    }
    _atomic_json(target_root / "installed-release.json", result)
    progress["state"] = "completed"
    _atomic_json(target_root / "upgrade-progress.json", progress)
    return result


def rollback_from_backup(
    *,
    bundle: Path,
    target_root: Path,
    age_identity_file: Path,
    verification_public_key_file: Path,
    failed_release_version: str,
    apply_restored_state: Callable[[Path, InstallationBackupManifest], Mapping[str, object]],
    operation_id: str | None = None,
) -> dict[str, object]:
    """Recover a failed upgrade into another fresh target from the verified N-1 backup."""

    started = time.monotonic()
    manifest = verify_backup(
        bundle=bundle,
        verification_public_key_file=verification_public_key_file,
    )
    if not _is_upgrade(manifest.source_release_version, failed_release_version):
        raise BackupBundleError("rollback.release_transition_invalid")
    restored = restore_backup(
        bundle=bundle,
        restore_root=target_root,
        age_identity_file=age_identity_file,
        verification_public_key_file=verification_public_key_file,
        apply_restored_state=apply_restored_state,
    )
    result = {
        "schema": "io.roehub.release-rollback-result/v1alpha1",
        "status": "passed",
        "failed_release": failed_release_version,
        "restored_release": manifest.source_release_version,
        "backup_manifest_sha256": manifest.manifest_sha256,
        "fresh_target_guard": restored["fresh_target_guard"],
        "state_owner_count": restored["restored_state_owner_count"],
        "observed_rollback_seconds": max(0.0, time.monotonic() - started),
        "operation_id": operation_id,
    }
    _atomic_json(target_root / "rollback-result.json", result)
    return result


def _verify_restored_target(
    target_root: Path,
    *,
    manifest: InstallationBackupManifest,
) -> None:
    for entry in manifest.entries:
        path = target_root / f"{entry.owner.value}.snapshot"
        if not path.is_file() or path.is_symlink():
            raise BackupBundleError("upgrade.restored_state_incomplete")
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.plaintext_sha256:
            raise BackupBundleError("upgrade.restored_state_mismatch")


def _is_upgrade(source: str, target: str) -> bool:
    source_match = _SEMVER.fullmatch(source)
    target_match = _SEMVER.fullmatch(target)
    if source_match is None or target_match is None:
        return False
    return tuple(map(int, target_match.groups())) > tuple(map(int, source_match.groups()))


def _atomic_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise BackupBundleError("release_lifecycle.write_failed")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["rollback_from_backup", "upgrade_from_backup"]
