from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .hf_reproducibility import compute_file_sha256
from .model_registry import (
    RL_TRADING_ARTIFACT_ROOT_V1,
    STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    registry_contract_hash_v1,
)
from .raw_feature_dataset import hash_json_payload_v1

STAGE09B_SCHEMA_VERSION_V1 = 1
STAGE09B_BACKUP_KIND_V1 = "rl_trading_stage09b_local_artifact_backup_v1"
STAGE09B_REGISTRY_DUMP_KIND_V1 = "rl_trading_stage09b_registry_metadata_dump_v1"
STAGE09B_RESTORE_DRILL_KIND_V1 = "rl_trading_stage09b_restore_drill_v1"
STAGE09B_ROLLBACK_KIND_V1 = "rl_trading_stage09b_rollback_command_v1"
STAGE09B_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage09b_local_artifact_backup_restore_v1"
DEFAULT_STAGE09B_BACKUP_ROOT_V1 = (
    f"{RL_TRADING_ARTIFACT_ROOT_V1}/backups/{STAGE09B_RUNTIME_ARTIFACT_SUBDIR_V1}"
)
DEFAULT_STAGE09B_RESTORE_ROOT_V1 = (
    f"{RL_TRADING_ARTIFACT_ROOT_V1}/restore_drills/"
    f"{STAGE09B_RUNTIME_ARTIFACT_SUBDIR_V1}"
)
STAGE09B_PREVIOUS_CHAMPION_FIXTURE_ID_V1 = (
    "stage09b_previous_accepted_champion_restore_drill"
)
STAGE09B_DEFAULT_RESTORE_RETENTION_DAYS_V1 = 30

ArtifactRole = Literal[
    "accepted_champion_manifest",
    "accepted_champion_scorecard",
    "calibration_status_manifest",
    "previous_champion_manifest",
    "registry_metadata_dump",
    "rollback_manifest",
    "source_manifest",
]
RetentionClass = Literal["retain_forever", "retain_days", "removable_after_restore_drill"]


class ArtifactBackupError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Stage09BArtifactSpec:
    artifact_id: str
    role: ArtifactRole
    source_path: str
    expected_sha256: str
    retention_class: RetentionClass
    retention_days: int | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.artifact_id, "artifact_id")
        _validate_store_path_text(self.source_path, "source_path")
        _validate_sha256(self.expected_sha256, "expected_sha256")
        if self.retention_class == "retain_days":
            if self.retention_days is None or self.retention_days <= 0:
                raise ArtifactBackupError(
                    reason="retention_days_required",
                    field=self.artifact_id,
                )
        elif self.retention_days is not None:
            raise ArtifactBackupError(reason="retention_days_unexpected", field=self.artifact_id)

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "expected_sha256": self.expected_sha256,
            "retention_class": self.retention_class,
            "retention_days": self.retention_days,
            "role": self.role,
            "source_path": self.source_path,
        }


@dataclass(frozen=True, slots=True)
class Stage09BDrillConfig:
    artifact_root: Path
    backup_root: Path
    restore_root: Path
    run_id: str
    generated_at_utc: datetime
    current_champion_manifest_path: Path
    expected_current_champion_manifest_sha256: str
    current_champion_scorecard_path: Path
    expected_current_champion_scorecard_sha256: str
    source_manifest_path: Path
    expected_source_manifest_sha256: str
    research_source_summary_path: Path
    expected_research_source_summary_sha256: str
    current_champion_id: str = STAGE09_ACCEPTED_CANDIDATE_ID_V1
    previous_champion_id: str = STAGE09B_PREVIOUS_CHAMPION_FIXTURE_ID_V1
    restore_retention_days: int = STAGE09B_DEFAULT_RESTORE_RETENTION_DAYS_V1
    same_physical_disk: bool = True

    def __post_init__(self) -> None:
        _validate_artifact_root(self.artifact_root)
        _validate_run_id(self.run_id)
        if self.generated_at_utc.tzinfo is None:
            raise ArtifactBackupError(reason="generated_at_utc_must_be_timezone_aware")
        _validate_sha256(
            self.expected_current_champion_manifest_sha256,
            "expected_current_champion_manifest_sha256",
        )
        _validate_sha256(
            self.expected_current_champion_scorecard_sha256,
            "expected_current_champion_scorecard_sha256",
        )
        _validate_sha256(self.expected_source_manifest_sha256, "expected_source_manifest_sha256")
        _validate_sha256(
            self.expected_research_source_summary_sha256,
            "expected_research_source_summary_sha256",
        )
        if self.restore_retention_days <= 0:
            raise ArtifactBackupError(reason="restore_retention_days_required")


def run_stage09b_backup_restore_drill_v1(config: Stage09BDrillConfig) -> dict[str, object]:
    root = _validate_artifact_root(config.artifact_root)
    backup_run_root = _resolve_run_root(config.backup_root, config.run_id)
    restore_run_root = _resolve_run_root(config.restore_root, config.run_id)
    metadata_root = backup_run_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    generated_at = _format_utc(config.generated_at_utc)
    calibration_manifest = _build_calibration_status_manifest(
        current_champion_id=config.current_champion_id,
        generated_at_utc=config.generated_at_utc,
    )
    calibration_manifest_path = metadata_root / "stage09b_calibration_status_manifest.json"
    _atomic_write_json(calibration_manifest_path, calibration_manifest)
    calibration_manifest_sha256 = compute_file_sha256(calibration_manifest_path)

    previous_manifest = _build_previous_champion_manifest(
        previous_champion_id=config.previous_champion_id,
        current_champion_id=config.current_champion_id,
        generated_at_utc=config.generated_at_utc,
    )
    previous_manifest_path = metadata_root / "stage09b_previous_champion_manifest.json"
    _atomic_write_json(previous_manifest_path, previous_manifest)
    previous_manifest_sha256 = compute_file_sha256(previous_manifest_path)

    registry_dump = build_stage09b_registry_metadata_dump_v1(
        current_champion_id=config.current_champion_id,
        current_champion_manifest_path=config.current_champion_manifest_path,
        current_champion_manifest_sha256=config.expected_current_champion_manifest_sha256,
        current_champion_scorecard_path=config.current_champion_scorecard_path,
        current_champion_scorecard_sha256=config.expected_current_champion_scorecard_sha256,
        previous_champion_id=config.previous_champion_id,
        previous_champion_manifest_path=previous_manifest_path,
        previous_champion_manifest_sha256=previous_manifest_sha256,
        source_manifest_path=config.source_manifest_path,
        source_manifest_sha256=config.expected_source_manifest_sha256,
        research_source_summary_path=config.research_source_summary_path,
        research_source_summary_sha256=config.expected_research_source_summary_sha256,
        calibration_status_manifest_path=calibration_manifest_path,
        calibration_status_manifest_sha256=calibration_manifest_sha256,
        restore_retention_days=config.restore_retention_days,
        same_physical_disk=config.same_physical_disk,
        generated_at_utc=config.generated_at_utc,
    )
    registry_dump_path = metadata_root / "stage09b_registry_metadata_dump.json"
    _atomic_write_json(registry_dump_path, registry_dump)
    registry_dump_sha256 = compute_file_sha256(registry_dump_path)

    rollback_manifest = build_stage09b_rollback_manifest_v1(
        current_champion_id=config.current_champion_id,
        previous_champion_id=config.previous_champion_id,
        previous_champion_manifest_sha256=previous_manifest_sha256,
        registry_metadata_dump_sha256=registry_dump_sha256,
        generated_at_utc=config.generated_at_utc,
    )
    rollback_manifest_path = metadata_root / "stage09b_rollback_manifest.json"
    _atomic_write_json(rollback_manifest_path, rollback_manifest)
    rollback_manifest_sha256 = compute_file_sha256(rollback_manifest_path)

    specs = (
        Stage09BArtifactSpec(
            artifact_id=config.current_champion_id,
            role="accepted_champion_manifest",
            source_path=str(config.current_champion_manifest_path),
            expected_sha256=config.expected_current_champion_manifest_sha256,
            retention_class="retain_forever",
        ),
        Stage09BArtifactSpec(
            artifact_id=f"{config.current_champion_id}:scorecard",
            role="accepted_champion_scorecard",
            source_path=str(config.current_champion_scorecard_path),
            expected_sha256=config.expected_current_champion_scorecard_sha256,
            retention_class="retain_forever",
        ),
        Stage09BArtifactSpec(
            artifact_id="stage08j_article_sessionized_manifest",
            role="source_manifest",
            source_path=str(config.source_manifest_path),
            expected_sha256=config.expected_source_manifest_sha256,
            retention_class="retain_forever",
        ),
        Stage09BArtifactSpec(
            artifact_id="stage08l_reward_warm_start_research_summary",
            role="source_manifest",
            source_path=str(config.research_source_summary_path),
            expected_sha256=config.expected_research_source_summary_sha256,
            retention_class="retain_forever",
        ),
        Stage09BArtifactSpec(
            artifact_id="stage09b_calibration_status_manifest",
            role="calibration_status_manifest",
            source_path=str(calibration_manifest_path),
            expected_sha256=calibration_manifest_sha256,
            retention_class="retain_days",
            retention_days=config.restore_retention_days,
        ),
        Stage09BArtifactSpec(
            artifact_id=config.previous_champion_id,
            role="previous_champion_manifest",
            source_path=str(previous_manifest_path),
            expected_sha256=previous_manifest_sha256,
            retention_class="retain_forever",
        ),
        Stage09BArtifactSpec(
            artifact_id="stage09b_registry_metadata_dump",
            role="registry_metadata_dump",
            source_path=str(registry_dump_path),
            expected_sha256=registry_dump_sha256,
            retention_class="retain_forever",
        ),
        Stage09BArtifactSpec(
            artifact_id="stage09b_rollback_manifest",
            role="rollback_manifest",
            source_path=str(rollback_manifest_path),
            expected_sha256=rollback_manifest_sha256,
            retention_class="retain_forever",
        ),
    )
    backup_manifest = backup_stage09b_artifacts_v1(
        artifact_root=root,
        backup_run_root=backup_run_root,
        restore_run_root=restore_run_root,
        specs=specs,
        generated_at_utc=config.generated_at_utc,
        registry_metadata_dump_path=registry_dump_path,
        registry_metadata_dump_sha256=registry_dump_sha256,
        rollback_manifest_path=rollback_manifest_path,
        rollback_manifest_sha256=rollback_manifest_sha256,
        same_physical_disk=config.same_physical_disk,
    )
    backup_manifest_path = backup_run_root / "stage09b_backup_manifest.json"
    _atomic_write_json(backup_manifest_path, backup_manifest)
    backup_manifest_sha256 = compute_file_sha256(backup_manifest_path)

    restore_report = restore_stage09b_backup_v1(
        backup_manifest_path=backup_manifest_path,
        restore_run_root=restore_run_root,
        generated_at_utc=config.generated_at_utc,
    )
    restore_report_path = restore_run_root / "stage09b_restore_report.json"
    _atomic_write_json(restore_report_path, restore_report)
    restore_report_sha256 = compute_file_sha256(restore_report_path)

    payload: dict[str, object] = {
        "artifact_count": len(specs),
        "artifact_root": str(root),
        "backup_manifest_path": str(backup_manifest_path),
        "backup_manifest_sha256": backup_manifest_sha256,
        "backup_root": str(backup_run_root),
        "generated_at_utc": generated_at,
        "kind": "rl_trading_stage09b_backup_restore_drill_result_v1",
        "registry_metadata_dump_path": str(registry_dump_path),
        "registry_metadata_dump_sha256": registry_dump_sha256,
        "restore_report_path": str(restore_report_path),
        "restore_report_sha256": restore_report_sha256,
        "restore_root": str(restore_run_root),
        "rollback_manifest_path": str(rollback_manifest_path),
        "rollback_manifest_sha256": rollback_manifest_sha256,
        "run_id": config.run_id,
        "same_physical_disk": config.same_physical_disk,
        "schema_version": STAGE09B_SCHEMA_VERSION_V1,
        "status": "accepted",
    }
    return {**payload, "drill_result_hash": hash_json_payload_v1(payload)}


def build_stage09b_registry_metadata_dump_v1(
    *,
    current_champion_id: str,
    current_champion_manifest_path: Path,
    current_champion_manifest_sha256: str,
    current_champion_scorecard_path: Path,
    current_champion_scorecard_sha256: str,
    previous_champion_id: str,
    previous_champion_manifest_path: Path,
    previous_champion_manifest_sha256: str,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    research_source_summary_path: Path,
    research_source_summary_sha256: str,
    calibration_status_manifest_path: Path,
    calibration_status_manifest_sha256: str,
    restore_retention_days: int,
    same_physical_disk: bool,
    generated_at_utc: datetime,
) -> dict[str, object]:
    _non_empty_text(current_champion_id, "current_champion_id")
    _non_empty_text(previous_champion_id, "previous_champion_id")
    _validate_sha256(current_champion_manifest_sha256, "current_champion_manifest_sha256")
    _validate_sha256(current_champion_scorecard_sha256, "current_champion_scorecard_sha256")
    _validate_sha256(previous_champion_manifest_sha256, "previous_champion_manifest_sha256")
    _validate_sha256(source_manifest_sha256, "source_manifest_sha256")
    _validate_sha256(research_source_summary_sha256, "research_source_summary_sha256")
    _validate_sha256(calibration_status_manifest_sha256, "calibration_status_manifest_sha256")
    if generated_at_utc.tzinfo is None:
        raise ArtifactBackupError(reason="generated_at_utc_must_be_timezone_aware")
    if restore_retention_days <= 0:
        raise ArtifactBackupError(reason="restore_retention_days_required")
    payload: dict[str, object] = {
        "active_champion": {
            "manifest_path": str(current_champion_manifest_path),
            "manifest_sha256": current_champion_manifest_sha256,
            "model_version_id": current_champion_id,
            "production_activation": False,
            "scorecard_path": str(current_champion_scorecard_path),
            "scorecard_sha256": current_champion_scorecard_sha256,
            "status": "accepted_champion_for_local_restore_drill",
        },
        "artifact_root": RL_TRADING_ARTIFACT_ROOT_V1,
        "calibration_pack": {
            "blocking_runtime_activation": True,
            "manifest_path": str(calibration_status_manifest_path),
            "manifest_sha256": calibration_status_manifest_sha256,
            "reason": "per_ticker_calibration_pack_is_stage10_scope",
            "status": "not_created_pre_stage10",
        },
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": STAGE09B_REGISTRY_DUMP_KIND_V1,
        "previous_champion": {
            "manifest_path": str(previous_champion_manifest_path),
            "manifest_sha256": previous_champion_manifest_sha256,
            "model_version_id": previous_champion_id,
            "production_activation": False,
            "status": "rollback_candidate_for_restore_drill",
        },
        "registry_contract_hash": registry_contract_hash_v1(),
        "retention_policy": {
            "removable_after_evidence": [
                "restore_drill_restored_file_copies",
                "operator_scratch_outputs_not_referenced_by_manifest",
            ],
            "retain_days": [
                {
                    "days": restore_retention_days,
                    "name": "stage09b_restore_drill_copies",
                    "reason": "operator replay window after accepted drill",
                }
            ],
            "retain_forever": [
                "accepted_champion_manifest",
                "accepted_champion_scorecard",
                "source_manifests",
                "registry_metadata_dump",
                "rollback_manifest",
                "backup_manifest",
            ],
        },
        "safety": {
            "cloud_storage": False,
            "exchange_side_effects": False,
            "model_activation": False,
            "raw_checkpoint_tensors_in_dump": False,
            "raw_provider_payloads_in_dump": False,
            "secrets_in_dump": False,
        },
        "same_physical_disk": same_physical_disk,
        "schema_version": STAGE09B_SCHEMA_VERSION_V1,
        "source_manifests": [
            {
                "path": str(source_manifest_path),
                "role": "stage08j_article_sessionized_manifest",
                "sha256": source_manifest_sha256,
            },
            {
                "path": str(research_source_summary_path),
                "role": "stage08l_reward_warm_start_research_summary",
                "sha256": research_source_summary_sha256,
            },
        ],
    }
    return {**payload, "registry_metadata_dump_hash": hash_json_payload_v1(payload)}


def build_stage09b_rollback_manifest_v1(
    *,
    current_champion_id: str,
    previous_champion_id: str,
    previous_champion_manifest_sha256: str,
    registry_metadata_dump_sha256: str,
    generated_at_utc: datetime,
) -> dict[str, object]:
    _non_empty_text(current_champion_id, "current_champion_id")
    _non_empty_text(previous_champion_id, "previous_champion_id")
    _validate_sha256(previous_champion_manifest_sha256, "previous_champion_manifest_sha256")
    _validate_sha256(registry_metadata_dump_sha256, "registry_metadata_dump_sha256")
    if generated_at_utc.tzinfo is None:
        raise ArtifactBackupError(reason="generated_at_utc_must_be_timezone_aware")
    payload: dict[str, object] = {
        "command": (
            "uv run python scripts/rl_trading/stage09b_local_artifact_backup_restore.py "
            "rollback-dry-run "
            f"--to-model-version-id {previous_champion_id} "
            f"--expected-current-model-version-id {current_champion_id} "
            "--reason stage09b_restore_drill"
        ),
        "current_champion_id": current_champion_id,
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": STAGE09B_ROLLBACK_KIND_V1,
        "no_artifact_deletion": True,
        "previous_champion_id": previous_champion_id,
        "previous_champion_manifest_sha256": previous_champion_manifest_sha256,
        "registry_metadata_dump_sha256": registry_metadata_dump_sha256,
        "schema_version": STAGE09B_SCHEMA_VERSION_V1,
        "status": "rollback_drill_ready",
    }
    return {**payload, "rollback_manifest_hash": hash_json_payload_v1(payload)}


def backup_stage09b_artifacts_v1(
    *,
    artifact_root: Path,
    backup_run_root: Path,
    restore_run_root: Path,
    specs: Sequence[Stage09BArtifactSpec],
    generated_at_utc: datetime,
    registry_metadata_dump_path: Path,
    registry_metadata_dump_sha256: str,
    rollback_manifest_path: Path,
    rollback_manifest_sha256: str,
    same_physical_disk: bool,
) -> dict[str, object]:
    root = _validate_artifact_root(artifact_root)
    if generated_at_utc.tzinfo is None:
        raise ArtifactBackupError(reason="generated_at_utc_must_be_timezone_aware")
    _validate_sha256(registry_metadata_dump_sha256, "registry_metadata_dump_sha256")
    _validate_sha256(rollback_manifest_sha256, "rollback_manifest_sha256")
    files_root = backup_run_root / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for spec in specs:
        source_path = _resolve_under_root(Path(spec.source_path), root, field=spec.artifact_id)
        if not source_path.is_file():
            raise ArtifactBackupError(reason="source_artifact_missing", field=str(source_path))
        actual_sha256 = compute_file_sha256(source_path)
        if actual_sha256 != spec.expected_sha256:
            raise ArtifactBackupError(reason="source_sha256_mismatch", field=spec.artifact_id)
        relative_path = source_path.relative_to(root)
        backup_path = files_root / relative_path
        _copy_file_atomic(source_path, backup_path)
        backup_sha256 = compute_file_sha256(backup_path)
        if backup_sha256 != spec.expected_sha256:
            raise ArtifactBackupError(reason="backup_sha256_mismatch", field=spec.artifact_id)
        entries.append(
            {
                **spec.as_payload(),
                "backup_path": str(backup_path),
                "backup_relative_path": str(backup_path.relative_to(backup_run_root)),
                "bytes": backup_path.stat().st_size,
                "hash_validated_after_backup": True,
                "hash_validated_before_backup": True,
                "source_relative_path": str(relative_path),
            }
        )

    payload: dict[str, object] = {
        "artifact_count": len(entries),
        "artifact_root": str(root),
        "backup_entries": sorted(entries, key=lambda item: str(item["artifact_id"])),
        "backup_root": str(backup_run_root),
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": STAGE09B_BACKUP_KIND_V1,
        "registry_metadata_dump_path": str(registry_metadata_dump_path),
        "registry_metadata_dump_sha256": registry_metadata_dump_sha256,
        "restore_root": str(restore_run_root),
        "rollback_manifest_path": str(rollback_manifest_path),
        "rollback_manifest_sha256": rollback_manifest_sha256,
        "same_physical_disk": same_physical_disk,
        "schema_version": STAGE09B_SCHEMA_VERSION_V1,
        "status": "accepted",
    }
    return {**payload, "backup_manifest_hash": hash_json_payload_v1(payload)}


def restore_stage09b_backup_v1(
    *,
    backup_manifest_path: Path,
    restore_run_root: Path,
    generated_at_utc: datetime,
) -> dict[str, object]:
    if generated_at_utc.tzinfo is None:
        raise ArtifactBackupError(reason="generated_at_utc_must_be_timezone_aware")
    manifest = _read_json(backup_manifest_path)
    if manifest.get("kind") != STAGE09B_BACKUP_KIND_V1:
        raise ArtifactBackupError(reason="unexpected_backup_manifest_kind")
    backup_root = Path(_required_str(manifest, "backup_root"))
    entries = _required_sequence(manifest, "backup_entries")
    restore_files_root = restore_run_root / "files"
    restored: list[dict[str, object]] = []
    for raw_entry in entries:
        entry = _mapping(raw_entry, "backup_entry")
        backup_relative_path = _required_str(entry, "backup_relative_path")
        expected_sha256 = _required_str(entry, "expected_sha256")
        artifact_id = _required_str(entry, "artifact_id")
        source = (backup_root / backup_relative_path).resolve(strict=False)
        if not source.is_file():
            raise ArtifactBackupError(reason="backup_artifact_missing", field=artifact_id)
        if compute_file_sha256(source) != expected_sha256:
            raise ArtifactBackupError(reason="backup_artifact_sha256_mismatch", field=artifact_id)
        destination = restore_files_root / backup_relative_path
        _copy_file_atomic(source, destination)
        restored_sha256 = compute_file_sha256(destination)
        if restored_sha256 != expected_sha256:
            raise ArtifactBackupError(reason="restored_artifact_sha256_mismatch", field=artifact_id)
        restored.append(
            {
                "artifact_id": artifact_id,
                "bytes": destination.stat().st_size,
                "expected_sha256": expected_sha256,
                "restored_path": str(destination),
                "restored_sha256": restored_sha256,
                "role": _required_str(entry, "role"),
            }
        )

    reference_validation = validate_stage09b_registry_references_v1(manifest)
    payload: dict[str, object] = {
        "backup_manifest_path": str(backup_manifest_path),
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": STAGE09B_RESTORE_DRILL_KIND_V1,
        "reference_validation": reference_validation,
        "restored_artifact_count": len(restored),
        "restored_artifacts": sorted(restored, key=lambda item: str(item["artifact_id"])),
        "restore_root": str(restore_run_root),
        "schema_version": STAGE09B_SCHEMA_VERSION_V1,
        "status": "accepted",
    }
    return {**payload, "restore_report_hash": hash_json_payload_v1(payload)}


def validate_stage09b_registry_references_v1(
    backup_manifest: Mapping[str, object],
) -> dict[str, object]:
    entries = [
        _mapping(entry, "backup_entry")
        for entry in _required_sequence(backup_manifest, "backup_entries")
    ]
    entries_by_role: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        entries_by_role.setdefault(_required_str(entry, "role"), []).append(entry)
    required_roles = {
        "accepted_champion_manifest",
        "accepted_champion_scorecard",
        "calibration_status_manifest",
        "previous_champion_manifest",
        "registry_metadata_dump",
        "rollback_manifest",
        "source_manifest",
    }
    missing_roles = sorted(role for role in required_roles if role not in entries_by_role)
    if missing_roles:
        raise ArtifactBackupError(
            reason="backup_manifest_missing_roles",
            field=",".join(missing_roles),
        )
    backup_root = Path(_required_str(backup_manifest, "backup_root"))
    registry_entry = _single_role_entry(entries_by_role, "registry_metadata_dump")
    registry_path = backup_root / _required_str(registry_entry, "backup_relative_path")
    registry_dump = _read_json(registry_path)
    active = _mapping(registry_dump.get("active_champion"), "active_champion")
    previous = _mapping(registry_dump.get("previous_champion"), "previous_champion")
    calibration = _mapping(registry_dump.get("calibration_pack"), "calibration_pack")
    _assert_expected_value(
        active.get("manifest_sha256"),
        _role_sha256(entries_by_role, "accepted_champion_manifest"),
        field="active_champion.manifest_sha256",
    )
    _assert_expected_value(
        active.get("scorecard_sha256"),
        _role_sha256(entries_by_role, "accepted_champion_scorecard"),
        field="active_champion.scorecard_sha256",
    )
    _assert_expected_value(
        previous.get("manifest_sha256"),
        _role_sha256(entries_by_role, "previous_champion_manifest"),
        field="previous_champion.manifest_sha256",
    )
    _assert_expected_value(
        calibration.get("manifest_sha256"),
        _role_sha256(entries_by_role, "calibration_status_manifest"),
        field="calibration_pack.manifest_sha256",
    )
    source_sha256s = {
        _required_str(entry, "expected_sha256") for entry in entries_by_role["source_manifest"]
    }
    source_manifest_rows = _required_sequence(registry_dump, "source_manifests")
    registry_source_sha256s = {
        _required_str(_mapping(row, "source_manifest"), "sha256")
        for row in source_manifest_rows
    }
    if not registry_source_sha256s.issubset(source_sha256s):
        raise ArtifactBackupError(reason="registry_source_manifest_not_backed_up")
    return {
        "accepted_champion_manifest_sha256": _role_sha256(
            entries_by_role,
            "accepted_champion_manifest",
        ),
        "calibration_status_manifest_sha256": _role_sha256(
            entries_by_role,
            "calibration_status_manifest",
        ),
        "previous_champion_manifest_sha256": _role_sha256(
            entries_by_role,
            "previous_champion_manifest",
        ),
        "registry_metadata_dump_sha256": _required_str(
            backup_manifest,
            "registry_metadata_dump_sha256",
        ),
        "rollback_manifest_sha256": _required_str(backup_manifest, "rollback_manifest_sha256"),
        "source_manifest_entries": sum(
            1 for entry in entries if entry.get("role") == "source_manifest"
        ),
        "status": "accepted",
    }


def _single_role_entry(
    entries_by_role: Mapping[str, Sequence[Mapping[str, object]]],
    role: str,
) -> Mapping[str, object]:
    entries = entries_by_role.get(role, ())
    if len(entries) != 1:
        raise ArtifactBackupError(reason="unexpected_role_entry_count", field=role)
    return entries[0]


def _role_sha256(
    entries_by_role: Mapping[str, Sequence[Mapping[str, object]]],
    role: str,
) -> str:
    return _required_str(_single_role_entry(entries_by_role, role), "expected_sha256")


def _assert_expected_value(actual: object, expected: str, *, field: str) -> None:
    if actual != expected:
        raise ArtifactBackupError(reason="registry_reference_sha256_mismatch", field=field)


def _build_calibration_status_manifest(
    *, current_champion_id: str, generated_at_utc: datetime
) -> dict[str, object]:
    payload: dict[str, object] = {
        "blocking_runtime_activation": True,
        "current_champion_id": current_champion_id,
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": "rl_trading_stage09b_calibration_status_manifest_v1",
        "reason": "per_ticker_calibration_pack_is_stage10_scope",
        "schema_version": STAGE09B_SCHEMA_VERSION_V1,
        "status": "not_created_pre_stage10",
    }
    return {**payload, "calibration_status_manifest_hash": hash_json_payload_v1(payload)}


def _build_previous_champion_manifest(
    *,
    previous_champion_id: str,
    current_champion_id: str,
    generated_at_utc: datetime,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "current_champion_id": current_champion_id,
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": "rl_trading_stage09b_previous_champion_manifest_v1",
        "model_version_id": previous_champion_id,
        "production_activation": False,
        "reason": (
            "metadata_only_previous_champion_for_restore_drill_until_real_champion_history_exists"
        ),
        "schema_version": STAGE09B_SCHEMA_VERSION_V1,
        "status": "rollback_candidate_for_restore_drill",
    }
    return {**payload, "previous_champion_manifest_hash": hash_json_payload_v1(payload)}


def _resolve_run_root(root: Path, run_id: str) -> Path:
    _validate_run_id(run_id)
    artifact_root = Path(RL_TRADING_ARTIFACT_ROOT_V1).expanduser().resolve(strict=False)
    resolved = root.expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise ArtifactBackupError(reason="path_outside_artifact_root", field=str(root)) from exc
    return artifact_root / relative / run_id


def _validate_artifact_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=False)
    expected = Path(RL_TRADING_ARTIFACT_ROOT_V1).expanduser().resolve(strict=False)
    if root != expected:
        raise ArtifactBackupError(reason="unexpected_artifact_root", field=str(root))
    return root


def _resolve_under_root(path: Path, root: Path, *, field: str) -> Path:
    if not path.is_absolute():
        raise ArtifactBackupError(reason="path_must_be_absolute", field=field)
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactBackupError(reason="path_outside_artifact_root", field=field) from exc
    return resolved


def _validate_store_path_text(value: str, field: str) -> None:
    _resolve_under_root(
        Path(value),
        Path(RL_TRADING_ARTIFACT_ROOT_V1).resolve(strict=False),
        field=field,
    )


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(f"{destination.suffix}.tmp")
    shutil.copy2(source, tmp)
    tmp.replace(destination)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), "json")


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactBackupError(reason="expected_mapping", field=field)
    return value


def _required_sequence(payload: Mapping[str, object], field: str) -> Sequence[object]:
    value = payload.get(field)
    if not isinstance(value, list | tuple):
        raise ArtifactBackupError(reason="missing_sequence", field=field)
    return value


def _required_str(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ArtifactBackupError(reason="missing_text", field=field)
    return value


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ArtifactBackupError(reason="invalid_sha256", field=field)


def _non_empty_text(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ArtifactBackupError(reason="missing_text", field=field)


def _validate_run_id(value: str) -> None:
    _non_empty_text(value, "run_id")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if any(char not in allowed for char in value) or value.startswith(("-", "_")):
        raise ArtifactBackupError(reason="invalid_run_id", field=value)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
