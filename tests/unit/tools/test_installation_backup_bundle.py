from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.backup import (
    BackupBundleError,
    BackupSource,
    create_backup,
    restore_backup,
    rollback_from_backup,
    upgrade_from_backup,
    verify_backup,
)
from trading.contexts.operations import BackupStateOwner


def _apply_materialized(root: Path, manifest: object) -> dict[str, object]:
    assert len(list(root.glob("*.snapshot"))) == 8
    assert manifest is not None
    return {"status": "ready", "state_owner_count": 8}


def _materials(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    age_identity = root / "operator.agekey"
    age_recipient = root / "operator.recipient"
    completed = subprocess.run(
        ["age-keygen", "-o", str(age_identity)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    recipient = subprocess.run(
        ["age-keygen", "-y", str(age_identity)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert recipient.returncode == 0
    age_recipient.write_text(recipient.stdout, encoding="utf-8")
    os.chmod(age_identity, 0o600)

    signer = Ed25519PrivateKey.generate()
    signing_file = root / "operator-signing.pem"
    verification_file = root / "operator-verification.pem"
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
    return age_identity, age_recipient, signing_file, verification_file


def _sources(root: Path, captured_at: datetime) -> tuple[BackupSource, ...]:
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    result: list[BackupSource] = []
    for owner in BackupStateOwner:
        path = root / f"{owner.value}.snapshot"
        path.write_text(
            json.dumps({"owner": owner.value, "row": f"fixture-{owner.value}"}),
            encoding="utf-8",
        )
        path.chmod(0o600)
        mode = {
            BackupStateOwner.RELEASE_CONFIG: "application_quiesced",
            BackupStateOwner.POSTGRESQL: "database_snapshot",
            BackupStateOwner.CLICKHOUSE: "database_snapshot",
            BackupStateOwner.REDIS_CHECKPOINT: "durable_checkpoint",
            BackupStateOwner.OPENBAO: "encrypted_raft_snapshot",
            BackupStateOwner.ARTIFACTS: "content_addressed_snapshot",
            BackupStateOwner.PLUGIN_OPERATION_AUDIT: "application_quiesced",
            BackupStateOwner.OBSERVABILITY: "bounded_history_snapshot",
        }[owner]
        result.append(
            BackupSource(
                owner=owner,
                path=path,
                media_type="application/json",
                consistency_mode=mode,  # type: ignore[arg-type]
                source_schema_version="fixture-v1",
                captured_at=captured_at,
                expected_plaintext_sha256=(
                    "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                ),
                limitations=(
                    ("Redis is a recoverable checkpoint, not the source of truth",)
                    if owner is BackupStateOwner.REDIS_CHECKPOINT
                    else ()
                ),
            )
        )
    return tuple(result)


def _create(
    tmp_path: Path,
    *,
    release_version: str = "0.1.0",
) -> tuple[Path, Path, Path, tuple[BackupSource, ...]]:
    age_identity, age_recipient, signing_file, verification_file = _materials(tmp_path)
    quiesce = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    sources = _sources(tmp_path / "sources", quiesce + timedelta(seconds=1))
    create_backup(
        backup_id="backup-001",
        installation_fingerprint="sha256:" + "1" * 64,
        source_release_version=release_version,
        sources=sources,
        backup_root=tmp_path / "backups",
        age_recipient_file=age_recipient,
        age_identity_file=age_identity,
        signing_private_key_file=signing_file,
        verification_public_key_file=verification_file,
        quiesce_started_at=quiesce,
        quiesce_completed_at=quiesce + timedelta(seconds=2),
        now=lambda: quiesce + timedelta(seconds=3),
    )
    return (
        tmp_path / "backups" / "backup-001",
        age_identity,
        verification_file,
        sources,
    )


def test_backup_encrypts_signs_and_restores_every_state_owner(tmp_path: Path) -> None:
    bundle, age_identity, verification_file, sources = _create(tmp_path)

    manifest = verify_backup(
        bundle=bundle,
        verification_public_key_file=verification_file,
    )
    result = restore_backup(
        bundle=bundle,
        restore_root=tmp_path / "restored",
        age_identity_file=age_identity,
        verification_public_key_file=verification_file,
        apply_restored_state=_apply_materialized,
    )

    assert {entry.owner for entry in manifest.entries} == set(BackupStateOwner)
    assert all(entry.encrypted for entry in manifest.entries)
    assert not list(bundle.glob("*.snapshot"))
    assert result["restored_state_owner_count"] == 8
    for source in sources:
        restored = tmp_path / "restored" / f"{source.owner.value}.snapshot"
        assert restored.read_bytes() == source.path.read_bytes()


def test_backup_and_restore_resume_after_injected_partial_failure(tmp_path: Path) -> None:
    age_identity, age_recipient, signing_file, verification_file = _materials(tmp_path)
    quiesce = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    sources = _sources(tmp_path / "sources", quiesce + timedelta(seconds=1))

    with pytest.raises(BackupBundleError, match="backup.injected_partial_failure"):
        create_backup(
            backup_id="backup-resume",
            installation_fingerprint="sha256:" + "2" * 64,
            source_release_version="0.1.0",
            sources=sources,
            backup_root=tmp_path / "backups",
            age_recipient_file=age_recipient,
            age_identity_file=age_identity,
            signing_private_key_file=signing_file,
            verification_public_key_file=verification_file,
            quiesce_started_at=quiesce,
            quiesce_completed_at=quiesce + timedelta(seconds=2),
            fail_after_entries=3,
            now=lambda: quiesce + timedelta(seconds=3),
        )
    create_backup(
        backup_id="backup-resume",
        installation_fingerprint="sha256:" + "2" * 64,
        source_release_version="0.1.0",
        sources=sources,
        backup_root=tmp_path / "backups",
        age_recipient_file=age_recipient,
        age_identity_file=age_identity,
        signing_private_key_file=signing_file,
        verification_public_key_file=verification_file,
        quiesce_started_at=quiesce,
        quiesce_completed_at=quiesce + timedelta(seconds=2),
        now=lambda: quiesce + timedelta(seconds=4),
    )
    bundle = tmp_path / "backups" / "backup-resume"

    with pytest.raises(BackupBundleError, match="restore.injected_partial_failure"):
        restore_backup(
            bundle=bundle,
            restore_root=tmp_path / "restored",
            age_identity_file=age_identity,
            verification_public_key_file=verification_file,
            fail_after_entries=4,
        )
    result = restore_backup(
        bundle=bundle,
        restore_root=tmp_path / "restored",
        age_identity_file=age_identity,
        verification_public_key_file=verification_file,
        apply_restored_state=_apply_materialized,
    )
    assert result["status"] == "passed"


def test_backup_rejects_missing_owner_and_tampered_ciphertext(tmp_path: Path) -> None:
    age_identity, age_recipient, signing_file, verification_file = _materials(tmp_path)
    quiesce = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    sources = _sources(tmp_path / "sources", quiesce + timedelta(seconds=1))
    with pytest.raises(BackupBundleError, match="backup.state_owner_coverage_incomplete"):
        create_backup(
            backup_id="incomplete",
            installation_fingerprint="sha256:" + "3" * 64,
            source_release_version="0.1.0",
            sources=sources[:-1],
            backup_root=tmp_path / "backups",
            age_recipient_file=age_recipient,
            age_identity_file=age_identity,
            signing_private_key_file=signing_file,
            verification_public_key_file=verification_file,
            quiesce_started_at=quiesce,
            quiesce_completed_at=quiesce + timedelta(seconds=2),
        )

    bundle, _identity, _verification, _sources_value = _create(tmp_path / "valid")
    ciphertext = next(bundle.glob("*.age"))
    ciphertext.write_bytes(ciphertext.read_bytes() + b"tamper")
    with pytest.raises(BackupBundleError, match="backup.ciphertext_digest_mismatch"):
        verify_backup(bundle=bundle, verification_public_key_file=_verification)
    assert age_identity.is_file()


def test_restore_rejects_nonempty_or_overlapping_target(tmp_path: Path) -> None:
    bundle, age_identity, verification_file, _sources_value = _create(tmp_path)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "foreign").write_text("preserve", encoding="utf-8")
    with pytest.raises(BackupBundleError, match="restore.target_not_empty"):
        restore_backup(
            bundle=bundle,
            restore_root=nonempty,
            age_identity_file=age_identity,
            verification_public_key_file=verification_file,
            apply_restored_state=_apply_materialized,
        )
    with pytest.raises(BackupBundleError, match="restore.source_target_overlap"):
        restore_backup(
            bundle=bundle,
            restore_root=bundle / "nested",
            age_identity_file=age_identity,
            verification_public_key_file=verification_file,
        )


def test_n_minus_one_upgrade_failure_resume_and_rollback(tmp_path: Path) -> None:
    bundle, age_identity, verification_file, _sources_value = _create(
        tmp_path,
        release_version="0.0.0",
    )
    upgrade_root = tmp_path / "upgrade"
    with pytest.raises(BackupBundleError, match="upgrade.injected_failure_before_commit"):
        upgrade_from_backup(
            bundle=bundle,
            target_root=upgrade_root,
            age_identity_file=age_identity,
            verification_public_key_file=verification_file,
            target_release_version="0.1.0",
            fail_before_commit=True,
            apply_restored_state=_apply_materialized,
        )
    upgraded = upgrade_from_backup(
        bundle=bundle,
        target_root=upgrade_root,
        age_identity_file=age_identity,
        verification_public_key_file=verification_file,
        target_release_version="0.1.0",
        apply_restored_state=_apply_materialized,
    )
    rolled_back = rollback_from_backup(
        bundle=bundle,
        target_root=tmp_path / "rollback",
        age_identity_file=age_identity,
        verification_public_key_file=verification_file,
        failed_release_version="0.1.0",
        apply_restored_state=_apply_materialized,
    )

    assert upgraded["preupgrade_backup_verified"] is True
    assert upgraded["from_release"] == "0.0.0"
    assert upgraded["to_release"] == "0.1.0"
    assert rolled_back["restored_release"] == "0.0.0"
    assert rolled_back["state_owner_count"] == 8


def test_irreversible_upgrade_requires_forward_recovery_plan(tmp_path: Path) -> None:
    bundle, age_identity, verification_file, _sources_value = _create(
        tmp_path,
        release_version="0.0.0",
    )
    with pytest.raises(BackupBundleError, match="upgrade.forward_recovery_plan_required"):
        upgrade_from_backup(
            bundle=bundle,
            target_root=tmp_path / "upgrade",
            age_identity_file=age_identity,
            verification_public_key_file=verification_file,
            target_release_version="0.1.0",
            irreversible=True,
            apply_restored_state=_apply_materialized,
        )


def test_backup_honors_cancellation_before_writing_ciphertext(tmp_path: Path) -> None:
    age_identity, age_recipient, signing_file, verification_file = _materials(
        tmp_path
    )
    quiesce = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    sources = _sources(tmp_path / "sources", quiesce + timedelta(seconds=1))
    cancellation = tmp_path / "cancel"
    cancellation.write_text("requested", encoding="utf-8")

    with pytest.raises(BackupBundleError, match="backup.cancelled"):
        create_backup(
            backup_id="cancelled",
            installation_fingerprint="sha256:" + "5" * 64,
            source_release_version="0.1.0",
            sources=sources,
            backup_root=tmp_path / "backups",
            age_recipient_file=age_recipient,
            age_identity_file=age_identity,
            signing_private_key_file=signing_file,
            verification_public_key_file=verification_file,
            quiesce_started_at=quiesce,
            quiesce_completed_at=quiesce + timedelta(seconds=2),
            cancellation_file=cancellation,
        )
    assert not list((tmp_path / "backups" / "cancelled").glob("*.age"))


def test_backup_rejects_mismatched_trust_key_and_symlink_ancestor(
    tmp_path: Path,
) -> None:
    age_identity, age_recipient, signing_file, _verification_file = _materials(
        tmp_path / "operator-a"
    )
    _other_identity, _other_recipient, _other_signing, other_verification = _materials(
        tmp_path / "operator-b"
    )
    quiesce = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    sources = _sources(tmp_path / "sources", quiesce + timedelta(seconds=1))
    with pytest.raises(BackupBundleError, match="backup.signature_identity_mismatch"):
        create_backup(
            backup_id="wrong-trust",
            installation_fingerprint="sha256:" + "6" * 64,
            source_release_version="0.1.0",
            sources=sources,
            backup_root=tmp_path / "backups",
            age_recipient_file=age_recipient,
            age_identity_file=age_identity,
            signing_private_key_file=signing_file,
            verification_public_key_file=other_verification,
            quiesce_started_at=quiesce,
            quiesce_completed_at=quiesce + timedelta(seconds=2),
        )

    alias = tmp_path / "source-alias"
    alias.symlink_to(tmp_path / "sources", target_is_directory=True)
    aliased = (replace(sources[0], path=alias / sources[0].path.name), *sources[1:])
    with pytest.raises(BackupBundleError, match="backup.path_symlink_ancestor"):
        create_backup(
            backup_id="symlink-source",
            installation_fingerprint="sha256:" + "7" * 64,
            source_release_version="0.1.0",
            sources=aliased,
            backup_root=tmp_path / "other-backups",
            age_recipient_file=age_recipient,
            age_identity_file=age_identity,
            signing_private_key_file=signing_file,
            verification_public_key_file=_verification_file,
            quiesce_started_at=quiesce,
            quiesce_completed_at=quiesce + timedelta(seconds=2),
        )


def test_restore_resume_rejects_unknown_file(tmp_path: Path) -> None:
    bundle, age_identity, verification_file, _sources_value = _create(tmp_path)
    restored = tmp_path / "restored"
    with pytest.raises(BackupBundleError, match="restore.injected_partial_failure"):
        restore_backup(
            bundle=bundle,
            restore_root=restored,
            age_identity_file=age_identity,
            verification_public_key_file=verification_file,
            fail_after_entries=2,
            apply_restored_state=_apply_materialized,
        )
    foreign = restored / "foreign-state"
    foreign.write_text("must-not-survive", encoding="utf-8")
    foreign.chmod(0o600)
    with pytest.raises(BackupBundleError, match="restore.target_not_fresh"):
        restore_backup(
            bundle=bundle,
            restore_root=restored,
            age_identity_file=age_identity,
            verification_public_key_file=verification_file,
            apply_restored_state=_apply_materialized,
        )
