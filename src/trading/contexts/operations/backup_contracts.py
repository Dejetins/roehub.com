"""Versioned installation backup contracts owned by the operations context."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BackupStateOwner(StrEnum):
    RELEASE_CONFIG = "release_config"
    POSTGRESQL = "postgresql"
    CLICKHOUSE = "clickhouse"
    REDIS_CHECKPOINT = "redis_checkpoint"
    OPENBAO = "openbao"
    ARTIFACTS = "artifacts"
    PLUGIN_OPERATION_AUDIT = "plugin_operation_audit"
    OBSERVABILITY = "observability"


REQUIRED_BACKUP_STATE_OWNERS = frozenset(BackupStateOwner)

ConsistencyMode = Literal[
    "application_quiesced",
    "database_snapshot",
    "durable_checkpoint",
    "encrypted_raft_snapshot",
    "content_addressed_snapshot",
    "bounded_history_snapshot",
]

REQUIRED_CONSISTENCY_MODES: dict[BackupStateOwner, ConsistencyMode] = {
    BackupStateOwner.RELEASE_CONFIG: "application_quiesced",
    BackupStateOwner.POSTGRESQL: "database_snapshot",
    BackupStateOwner.CLICKHOUSE: "database_snapshot",
    BackupStateOwner.REDIS_CHECKPOINT: "durable_checkpoint",
    BackupStateOwner.OPENBAO: "encrypted_raft_snapshot",
    BackupStateOwner.ARTIFACTS: "content_addressed_snapshot",
    BackupStateOwner.PLUGIN_OPERATION_AUDIT: "application_quiesced",
    BackupStateOwner.OBSERVABILITY: "bounded_history_snapshot",
}


class BackupPolicySource(BaseModel):
    """One fixed source path in an owner-protected host backup policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: BackupStateOwner
    filename: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    media_type: str = Field(min_length=1, max_length=128)
    consistency_mode: ConsistencyMode
    source_schema_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")
    limitations: tuple[str, ...] = Field(default=(), max_length=16)


class BackupCaptureEntry(BaseModel):
    """Digest-bound capture timestamp emitted by a state-owner coordinator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: BackupStateOwner
    captured_at: datetime
    plaintext_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class InstallationCaptureRecord(BaseModel):
    """One closed quiesce window covering all installation state owners."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.installation-capture/v1alpha1"] = Field(
        default="io.roehub.installation-capture/v1alpha1",
        alias="schema",
    )
    installation_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_release_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")
    quiesce_started_at: datetime
    quiesce_completed_at: datetime
    entries: tuple[BackupCaptureEntry, ...] = Field(
        min_length=8,
        max_length=8,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def _validate_capture(self) -> InstallationCaptureRecord:
        owners = [entry.owner for entry in self.entries]
        if len(owners) != len(set(owners)) or set(owners) != REQUIRED_BACKUP_STATE_OWNERS:
            raise ValueError("capture record must cover every required state owner")
        if self.quiesce_completed_at < self.quiesce_started_at:
            raise ValueError("capture quiesce interval is invalid")
        if any(
            entry.captured_at < self.quiesce_started_at
            or entry.captured_at > self.quiesce_completed_at
            for entry in self.entries
        ):
            raise ValueError("capture timestamp is outside the quiesce interval")
        return self


class ReleaseTransitionRule(BaseModel):
    """One trusted release transition and its recovery classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_release: str = Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")
    to_release: str = Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")
    reversible: bool
    forward_recovery_plan_sha256: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _validate_transition(self) -> ReleaseTransitionRule:
        if self.from_release == self.to_release:
            raise ValueError("release transition must change the release version")
        if self.reversible and self.forward_recovery_plan_sha256 is not None:
            raise ValueError("reversible transition must not declare a forward-only plan")
        return self


class InstallationReleasePolicy(BaseModel):
    """Installation-bound authority for update and rollback transitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.installation-release-policy/v1alpha1"] = Field(
        default="io.roehub.installation-release-policy/v1alpha1",
        alias="schema",
    )
    installation_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    transitions: tuple[ReleaseTransitionRule, ...] = Field(
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def _validate_transitions(self) -> InstallationReleasePolicy:
        identities = {
            (transition.from_release, transition.to_release)
            for transition in self.transitions
        }
        if len(identities) != len(self.transitions):
            raise ValueError("release transitions must be unique")
        return self

    def require_transition(
        self,
        *,
        from_release: str,
        to_release: str,
    ) -> ReleaseTransitionRule:
        for transition in self.transitions:
            if (
                transition.from_release == from_release
                and transition.to_release == to_release
            ):
                return transition
        raise ValueError("release transition is not trusted")


class InstallationBackupPolicy(BaseModel):
    """Closed host-side policy for installation backup and fresh restore."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.installation-backup-policy/v1alpha1"] = Field(
        default="io.roehub.installation-backup-policy/v1alpha1",
        alias="schema",
    )
    profile: Literal["base", "trading", "ml"]
    installation_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    release_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")
    source_root: Path
    backup_root: Path
    restore_root: Path
    age_recipient_file: Path
    age_identity_file: Path
    signing_private_key_file: Path
    verification_public_key_file: Path
    cancellation_root: Path
    capture_record_file: Path
    release_policy_file: Path
    cleanup_plaintext_sources: Literal[True] = True
    sources: tuple[BackupPolicySource, ...] = Field(
        min_length=8,
        max_length=8,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def _validate_policy(self) -> InstallationBackupPolicy:
        paths = (
            self.source_root,
            self.backup_root,
            self.restore_root,
            self.age_recipient_file,
            self.age_identity_file,
            self.signing_private_key_file,
            self.verification_public_key_file,
            self.cancellation_root,
            self.capture_record_file,
            self.release_policy_file,
        )
        if any(not path.is_absolute() for path in paths):
            raise ValueError("backup policy paths must be absolute")
        owners = [source.owner for source in self.sources]
        if len(owners) != len(set(owners)) or set(owners) != REQUIRED_BACKUP_STATE_OWNERS:
            raise ValueError("backup policy must cover every required state owner")
        if any(
            source.consistency_mode != REQUIRED_CONSISTENCY_MODES[source.owner]
            for source in self.sources
        ):
            raise ValueError("backup owner consistency mode is invalid")
        if any(_has_symlink_ancestor(path) for path in paths):
            raise ValueError("backup policy paths must not traverse symlinks")
        state_roots = tuple(
            path.resolve(strict=False)
            for path in (self.source_root, self.backup_root, self.restore_root)
        )
        if any(
            left == right or left in right.parents or right in left.parents
            for index, left in enumerate(state_roots)
            for right in state_roots[index + 1 :]
        ):
            raise ValueError("backup source, bundle, and restore roots must be separate")
        return self


class BackupManifestEntry(BaseModel):
    """One encrypted, digest-bound state-owner snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: BackupStateOwner
    artifact_name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}\.age$")
    media_type: str = Field(min_length=1, max_length=128)
    consistency_mode: ConsistencyMode
    source_schema_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")
    captured_at: datetime
    plaintext_bytes: int = Field(ge=1, le=1_073_741_824)
    plaintext_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ciphertext_bytes: int = Field(ge=1, le=1_074_790_400)
    ciphertext_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    encrypted: Literal[True] = True
    limitations: tuple[str, ...] = Field(default=(), max_length=16)


class InstallationBackupManifest(BaseModel):
    """Signed installation-wide backup inventory without secret material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.installation-backup/v1alpha1"] = Field(
        default="io.roehub.installation-backup/v1alpha1",
        alias="schema",
    )
    backup_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    installation_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_release_version: str = Field(pattern=r"^[a-z0-9][a-z0-9._+-]{0,63}$")
    compatibility: Literal["greenfield-self-hosted-v1"] = "greenfield-self-hosted-v1"
    created_at: datetime
    quiesce_started_at: datetime
    quiesce_completed_at: datetime
    observed_rpo_seconds: float = Field(ge=0)
    signing_key_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    entries: tuple[BackupManifestEntry, ...] = Field(
        min_length=8,
        max_length=8,
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def _validate_coverage(self) -> InstallationBackupManifest:
        owners = [entry.owner for entry in self.entries]
        if len(owners) != len(set(owners)):
            raise ValueError("backup state owners must be unique")
        if set(owners) != REQUIRED_BACKUP_STATE_OWNERS:
            raise ValueError("backup must cover every required state owner")
        if any(
            entry.consistency_mode != REQUIRED_CONSISTENCY_MODES[entry.owner]
            for entry in self.entries
        ):
            raise ValueError("backup owner consistency mode is invalid")
        if self.quiesce_completed_at < self.quiesce_started_at:
            raise ValueError("backup quiesce interval is invalid")
        if any(entry.captured_at < self.quiesce_started_at for entry in self.entries):
            raise ValueError("backup entry predates the quiesce boundary")
        if any(entry.captured_at > self.quiesce_completed_at for entry in self.entries):
            raise ValueError("backup entry exceeds the quiesce boundary")
        return self

    def canonical_bytes(self) -> bytes:
        payload = self.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    @property
    def manifest_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()


class BackupManifestSignature(BaseModel):
    """Detached Ed25519 signature for one canonical manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["io.roehub.installation-backup-signature/v1alpha1"] = Field(
        default="io.roehub.installation-backup-signature/v1alpha1",
        alias="schema",
    )
    algorithm: Literal["Ed25519"] = "Ed25519"
    key_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    signature_base64: str = Field(min_length=88, max_length=88)


__all__ = [
    "BackupManifestEntry",
    "BackupManifestSignature",
    "BackupPolicySource",
    "BackupCaptureEntry",
    "BackupStateOwner",
    "ConsistencyMode",
    "InstallationBackupPolicy",
    "InstallationBackupManifest",
    "InstallationCaptureRecord",
    "InstallationReleasePolicy",
    "ReleaseTransitionRule",
    "REQUIRED_BACKUP_STATE_OWNERS",
    "REQUIRED_CONSISTENCY_MODES",
]


def _has_symlink_ancestor(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not os.path.lexists(current):
            return False
        if stat.S_ISLNK(os.lstat(current).st_mode):
            return True
    return False
