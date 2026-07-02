from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .hf_reproducibility import compute_file_sha256
from .raw_feature_dataset import hash_json_payload_v1

MODEL_REGISTRY_SCHEMA_VERSION_V1 = 1
MODEL_REGISTRY_KIND_V1 = "rl_trading_model_registry_v1"
RL_TRADING_ARTIFACT_ROOT_V1 = "/opt/roehub/state/rl_trading"
STAGE09_ACCEPTED_CANDIDATE_ID_V1 = "stage08m_a3823cbd01143878_fd7c614b"
STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1 = (
    "9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c"
)
STAGE09_ACCEPTED_CANDIDATE_POLICY_V1 = (
    "supervised_oracle_label_warm_start_contextual_bandit"
)
TRUSTED_TRAINER_PRODUCER_V1 = "roehub_trainer_service"

RegistryEntity = Literal[
    "dataset_version",
    "training_run",
    "model_version",
    "calibration_pack",
    "activation",
]
DatasetVersionStatus = Literal[
    "building",
    "qa_failed",
    "accepted",
    "missing_artifact",
    "superseded",
]
TrainingRunStatus = Literal[
    "planned",
    "running",
    "failed",
    "completed",
    "rejected",
    "candidate",
]
ModelVersionStatus = Literal[
    "candidate",
    "rejected",
    "accepted_champion",
    "rollback_candidate",
    "missing_artifact",
]
CalibrationPackStatus = Literal[
    "candidate",
    "accepted",
    "rejected",
    "superseded",
    "missing_artifact",
]
ActivationStatus = Literal[
    "inactive",
    "shadow",
    "monitor_only",
    "paper",
    "testnet",
    "live",
    "paused",
    "rolled_back",
]
ArtifactEntity = Literal[
    "dataset_version",
    "training_run",
    "model_version",
    "calibration_pack",
    "source_manifest",
]

_ALLOWED_TRANSITIONS: dict[RegistryEntity, dict[str, frozenset[str]]] = {
    "dataset_version": {
        "building": frozenset({"qa_failed", "accepted", "missing_artifact"}),
        "qa_failed": frozenset({"building", "missing_artifact"}),
        "accepted": frozenset({"superseded", "missing_artifact"}),
        "missing_artifact": frozenset({"accepted", "superseded"}),
        "superseded": frozenset({"missing_artifact"}),
    },
    "training_run": {
        "planned": frozenset({"running", "failed"}),
        "running": frozenset({"completed", "failed"}),
        "completed": frozenset({"candidate", "rejected"}),
        "failed": frozenset(),
        "rejected": frozenset(),
        "candidate": frozenset({"rejected"}),
    },
    "model_version": {
        "candidate": frozenset({"accepted_champion", "rejected", "missing_artifact"}),
        "accepted_champion": frozenset({"rollback_candidate", "missing_artifact"}),
        "rollback_candidate": frozenset({"accepted_champion", "rejected", "missing_artifact"}),
        "rejected": frozenset({"missing_artifact"}),
        "missing_artifact": frozenset({"accepted_champion", "rollback_candidate"}),
    },
    "calibration_pack": {
        "candidate": frozenset({"accepted", "rejected", "missing_artifact"}),
        "accepted": frozenset({"superseded", "missing_artifact"}),
        "rejected": frozenset({"missing_artifact"}),
        "superseded": frozenset({"missing_artifact"}),
        "missing_artifact": frozenset({"accepted", "superseded"}),
    },
    "activation": {
        "inactive": frozenset({"shadow"}),
        "shadow": frozenset({"monitor_only", "paused", "inactive", "rolled_back"}),
        "monitor_only": frozenset({"paper", "paused", "inactive", "rolled_back"}),
        "paper": frozenset({"testnet", "paused", "inactive", "rolled_back"}),
        "testnet": frozenset({"live", "paused", "inactive", "rolled_back"}),
        "live": frozenset({"paused", "rolled_back"}),
        "paused": frozenset(
            {"monitor_only", "paper", "testnet", "live", "inactive", "rolled_back"}
        ),
        "rolled_back": frozenset({"inactive", "shadow"}),
    },
}

_RUNTIME_LOADABLE_MODEL_STATUSES = frozenset({"accepted_champion", "rollback_candidate"})
_CLEANUP_ELIGIBLE_STATUSES = frozenset(
    {"failed", "rejected", "qa_failed", "superseded"}
)


class ModelRegistryError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class DatasetVersionRecord:
    dataset_version_id: str
    dataset_hash: str
    manifest_path: str
    manifest_sha256: str
    feature_contract_hash: str
    status: DatasetVersionStatus = "accepted"

    def __post_init__(self) -> None:
        _non_empty_text(self.dataset_version_id, "dataset_version_id")
        _validate_sha256(self.dataset_hash, "dataset_hash")
        _validate_store_path_text(self.manifest_path, "manifest_path")
        _validate_sha256(self.manifest_sha256, "manifest_sha256")
        _validate_sha256(self.feature_contract_hash, "feature_contract_hash")

    def as_payload(self) -> dict[str, object]:
        return {
            "dataset_hash": self.dataset_hash,
            "dataset_version_id": self.dataset_version_id,
            "feature_contract_hash": self.feature_contract_hash,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunRecord:
    training_run_id: str
    dataset_version_id: str
    model_family: str
    run_config_hash: str
    run_manifest_path: str
    run_manifest_sha256: str
    status: TrainingRunStatus

    def __post_init__(self) -> None:
        _non_empty_text(self.training_run_id, "training_run_id")
        _non_empty_text(self.dataset_version_id, "dataset_version_id")
        _non_empty_text(self.model_family, "model_family")
        _validate_sha256(self.run_config_hash, "run_config_hash")
        _validate_store_path_text(self.run_manifest_path, "run_manifest_path")
        _validate_sha256(self.run_manifest_sha256, "run_manifest_sha256")

    def as_payload(self) -> dict[str, object]:
        return {
            "dataset_version_id": self.dataset_version_id,
            "model_family": self.model_family,
            "run_config_hash": self.run_config_hash,
            "run_manifest_path": self.run_manifest_path,
            "run_manifest_sha256": self.run_manifest_sha256,
            "status": self.status,
            "training_run_id": self.training_run_id,
        }


@dataclass(frozen=True, slots=True)
class ModelVersionRecord:
    model_version_id: str
    model_family: str
    feature_contract_hash: str
    dataset_version_id: str
    dataset_hash: str
    checkpoint_path: str
    checkpoint_sha256: str
    model_state_hash: str
    status: ModelVersionStatus
    producer: str = TRUSTED_TRAINER_PRODUCER_V1

    def __post_init__(self) -> None:
        _non_empty_text(self.model_version_id, "model_version_id")
        _non_empty_text(self.model_family, "model_family")
        _validate_sha256(self.feature_contract_hash, "feature_contract_hash")
        _non_empty_text(self.dataset_version_id, "dataset_version_id")
        _validate_sha256(self.dataset_hash, "dataset_hash")
        _validate_store_path_text(self.checkpoint_path, "checkpoint_path")
        _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _validate_sha256(self.model_state_hash, "model_state_hash")
        if self.producer != TRUSTED_TRAINER_PRODUCER_V1:
            raise ModelRegistryError(reason="untrusted_checkpoint_producer", field="producer")

    def as_payload(self) -> dict[str, object]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_hash": self.dataset_hash,
            "dataset_version_id": self.dataset_version_id,
            "feature_contract_hash": self.feature_contract_hash,
            "model_family": self.model_family,
            "model_state_hash": self.model_state_hash,
            "model_version_id": self.model_version_id,
            "producer": self.producer,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class CalibrationPackRecord:
    calibration_pack_id: str
    model_version_id: str
    feature_contract_hash: str
    dataset_hash: str
    calibration_pack_hash: str
    calibration_path: str
    calibration_sha256: str
    status: CalibrationPackStatus

    def __post_init__(self) -> None:
        _non_empty_text(self.calibration_pack_id, "calibration_pack_id")
        _non_empty_text(self.model_version_id, "model_version_id")
        _validate_sha256(self.feature_contract_hash, "feature_contract_hash")
        _validate_sha256(self.dataset_hash, "dataset_hash")
        _validate_sha256(self.calibration_pack_hash, "calibration_pack_hash")
        _validate_store_path_text(self.calibration_path, "calibration_path")
        _validate_sha256(self.calibration_sha256, "calibration_sha256")

    def as_payload(self) -> dict[str, object]:
        return {
            "calibration_pack_hash": self.calibration_pack_hash,
            "calibration_pack_id": self.calibration_pack_id,
            "calibration_path": self.calibration_path,
            "calibration_sha256": self.calibration_sha256,
            "dataset_hash": self.dataset_hash,
            "feature_contract_hash": self.feature_contract_hash,
            "model_version_id": self.model_version_id,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ActivationScope:
    model_family: str
    feature_contract_hash: str
    exchange: str
    market_type: str
    symbol: str

    def __post_init__(self) -> None:
        _non_empty_text(self.model_family, "model_family")
        _validate_sha256(self.feature_contract_hash, "feature_contract_hash")
        if self.exchange.strip().lower() != self.exchange:
            raise ModelRegistryError(reason="exchange_must_be_lowercase", field="exchange")
        if self.market_type not in {"spot", "futures"}:
            raise ModelRegistryError(reason="unsupported_market_type", field="market_type")
        if self.symbol.strip().upper() != self.symbol or not self.symbol:
            raise ModelRegistryError(reason="symbol_must_be_uppercase", field="symbol")

    def as_payload(self) -> dict[str, object]:
        return {
            "exchange": self.exchange,
            "feature_contract_hash": self.feature_contract_hash,
            "market_type": self.market_type,
            "model_family": self.model_family,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class ActivationRecord:
    activation_id: str
    scope: ActivationScope
    activation_state: ActivationStatus
    model_version_id: str
    calibration_pack_id: str
    dataset_version_id: str
    model_checkpoint_sha256: str
    calibration_pack_hash: str
    dataset_hash: str
    activation_matrix_hash: str
    is_current: bool = True

    def __post_init__(self) -> None:
        _non_empty_text(self.activation_id, "activation_id")
        _non_empty_text(self.model_version_id, "model_version_id")
        _non_empty_text(self.calibration_pack_id, "calibration_pack_id")
        _non_empty_text(self.dataset_version_id, "dataset_version_id")
        _validate_sha256(self.model_checkpoint_sha256, "model_checkpoint_sha256")
        _validate_sha256(self.calibration_pack_hash, "calibration_pack_hash")
        _validate_sha256(self.dataset_hash, "dataset_hash")
        _validate_sha256(self.activation_matrix_hash, "activation_matrix_hash")

    def as_payload(self) -> dict[str, object]:
        return {
            "activation_id": self.activation_id,
            "activation_matrix_hash": self.activation_matrix_hash,
            "activation_state": self.activation_state,
            "calibration_pack_hash": self.calibration_pack_hash,
            "calibration_pack_id": self.calibration_pack_id,
            "dataset_hash": self.dataset_hash,
            "dataset_version_id": self.dataset_version_id,
            "is_current": self.is_current,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "model_version_id": self.model_version_id,
            "scope": self.scope.as_payload(),
        }


@dataclass(frozen=True, slots=True)
class ArtifactLifecyclePolicy:
    rejected_run_retention_days: int
    disk_quota_bytes: int
    disk_watermark_pct: float
    artifact_root: str = RL_TRADING_ARTIFACT_ROOT_V1

    def __post_init__(self) -> None:
        if self.rejected_run_retention_days <= 0:
            raise ModelRegistryError(
                reason="rejected_run_retention_days_required",
                field="rejected_run_retention_days",
            )
        if self.disk_quota_bytes <= 0:
            raise ModelRegistryError(reason="disk_quota_bytes_required", field="disk_quota_bytes")
        if not 0.0 < self.disk_watermark_pct < 100.0:
            raise ModelRegistryError(
                reason="disk_watermark_pct_out_of_range",
                field="disk_watermark_pct",
            )
        _validate_artifact_root(Path(self.artifact_root))

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact_root": self.artifact_root,
            "disk_quota_bytes": self.disk_quota_bytes,
            "disk_watermark_pct": self.disk_watermark_pct,
            "rejected_run_retention_days": self.rejected_run_retention_days,
        }

    def policy_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


@dataclass(frozen=True, slots=True)
class RegistryArtifactRecord:
    artifact_id: str
    entity: ArtifactEntity
    status: str
    artifact_path: str
    artifact_sha256: str
    updated_at_utc: datetime
    referenced_by_active_metadata: bool = False
    source_manifest: bool = False

    def __post_init__(self) -> None:
        _non_empty_text(self.artifact_id, "artifact_id")
        _validate_store_path_text(self.artifact_path, "artifact_path")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if self.updated_at_utc.tzinfo is None:
            raise ModelRegistryError(reason="updated_at_utc_must_be_timezone_aware")

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "entity": self.entity,
            "referenced_by_active_metadata": self.referenced_by_active_metadata,
            "source_manifest": self.source_manifest,
            "status": self.status,
            "updated_at_utc": _format_utc(self.updated_at_utc),
        }


def validate_stage09_accepted_candidate_input_v1(
    *, candidate_id: str, candidate_manifest_sha256: str
) -> dict[str, object]:
    if candidate_id != STAGE09_ACCEPTED_CANDIDATE_ID_V1:
        raise ModelRegistryError(reason="unexpected_stage09_candidate_id", field=candidate_id)
    if candidate_manifest_sha256 != STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1:
        raise ModelRegistryError(
            reason="unexpected_stage09_candidate_manifest_sha256",
            field=candidate_manifest_sha256,
        )
    return {
        "candidate_id": candidate_id,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "policy_id": STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
        "stage": "09",
        "stage09_allowed": True,
    }


def validate_registry_transition_v1(
    *, entity: RegistryEntity, current_status: str, next_status: str
) -> None:
    allowed_by_status = _ALLOWED_TRANSITIONS[entity]
    if current_status not in allowed_by_status:
        raise ModelRegistryError(reason="unknown_registry_status", field=current_status)
    if next_status not in allowed_by_status[current_status]:
        raise ModelRegistryError(
            reason="invalid_registry_transition",
            field=f"{entity}:{current_status}->{next_status}",
        )


def assert_model_runtime_load_allowed_v1(
    model: ModelVersionRecord, *, explicit_rollback_selected: bool = False
) -> None:
    if model.status not in _RUNTIME_LOADABLE_MODEL_STATUSES:
        raise ModelRegistryError(reason="model_status_not_runtime_loadable", field=model.status)
    if model.status == "rollback_candidate" and not explicit_rollback_selected:
        raise ModelRegistryError(reason="rollback_candidate_requires_explicit_selection")


def assert_activation_ready_v1(
    *,
    dataset: DatasetVersionRecord,
    model: ModelVersionRecord,
    calibration: CalibrationPackRecord,
    scope: ActivationScope,
    activation_matrix_hash: str,
    explicit_rollback_selected: bool = False,
) -> None:
    _validate_sha256(activation_matrix_hash, "activation_matrix_hash")
    if dataset.status != "accepted":
        raise ModelRegistryError(reason="dataset_not_accepted", field=dataset.status)
    if calibration.status != "accepted":
        raise ModelRegistryError(reason="calibration_not_accepted", field=calibration.status)
    assert_model_runtime_load_allowed_v1(
        model,
        explicit_rollback_selected=explicit_rollback_selected,
    )
    if model.model_family != scope.model_family:
        raise ModelRegistryError(reason="model_family_mismatch")
    if model.feature_contract_hash != scope.feature_contract_hash:
        raise ModelRegistryError(reason="feature_contract_hash_mismatch")
    if calibration.feature_contract_hash != scope.feature_contract_hash:
        raise ModelRegistryError(reason="calibration_feature_contract_hash_mismatch")
    if model.dataset_version_id != dataset.dataset_version_id:
        raise ModelRegistryError(reason="model_dataset_version_mismatch")
    if calibration.model_version_id != model.model_version_id:
        raise ModelRegistryError(reason="calibration_model_version_mismatch")
    if model.dataset_hash != dataset.dataset_hash:
        raise ModelRegistryError(reason="model_dataset_hash_mismatch")
    if calibration.dataset_hash != dataset.dataset_hash:
        raise ModelRegistryError(reason="calibration_dataset_hash_mismatch")


def build_activation_record_v1(
    *,
    activation_id: str,
    scope: ActivationScope,
    next_state: ActivationStatus,
    dataset: DatasetVersionRecord,
    model: ModelVersionRecord,
    calibration: CalibrationPackRecord,
    activation_matrix_hash: str,
    previous_activation: ActivationRecord | None = None,
    explicit_rollback_selected: bool = False,
) -> ActivationRecord:
    if previous_activation is not None:
        validate_registry_transition_v1(
            entity="activation",
            current_status=previous_activation.activation_state,
            next_status=next_state,
        )
        if previous_activation.scope != scope:
            raise ModelRegistryError(reason="activation_scope_mismatch")
    elif next_state != "shadow":
        raise ModelRegistryError(reason="initial_activation_must_start_shadow", field=next_state)
    assert_activation_ready_v1(
        dataset=dataset,
        model=model,
        calibration=calibration,
        scope=scope,
        activation_matrix_hash=activation_matrix_hash,
        explicit_rollback_selected=explicit_rollback_selected,
    )
    return ActivationRecord(
        activation_id=activation_id,
        scope=scope,
        activation_state=next_state,
        model_version_id=model.model_version_id,
        calibration_pack_id=calibration.calibration_pack_id,
        dataset_version_id=dataset.dataset_version_id,
        model_checkpoint_sha256=model.checkpoint_sha256,
        calibration_pack_hash=calibration.calibration_pack_hash,
        dataset_hash=dataset.dataset_hash,
        activation_matrix_hash=activation_matrix_hash,
    )


def build_activation_audit_event_v1(
    *,
    event_type: Literal["activate", "deactivate", "promote", "rollback"],
    scope: ActivationScope,
    next_activation: ActivationRecord,
    reason_code: str,
    operator_ref_hash: str,
    generated_at_utc: datetime,
    previous_activation: ActivationRecord | None = None,
) -> dict[str, object]:
    _validate_reason_code(reason_code, "reason_code")
    _validate_sha256(operator_ref_hash, "operator_ref_hash")
    if generated_at_utc.tzinfo is None:
        raise ModelRegistryError(reason="generated_at_utc_must_be_timezone_aware")
    if next_activation.scope != scope:
        raise ModelRegistryError(reason="activation_scope_mismatch")
    previous_payload: Mapping[str, object] | None = (
        None if previous_activation is None else previous_activation.as_payload()
    )
    payload: dict[str, object] = {
        "event_type": event_type,
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": "rl_trading_registry_activation_audit",
        "model_registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION_V1,
        "next_activation": next_activation.as_payload(),
        "operator_ref_hash": operator_ref_hash,
        "previous_activation": previous_payload,
        "reason_code": reason_code,
        "scope": scope.as_payload(),
    }
    return {**payload, "event_payload_hash": hash_json_payload_v1(payload)}


def load_trusted_checkpoint_weights_v1(
    *,
    model: ModelVersionRecord,
    torch_module: Any,
    artifact_root: Path = Path(RL_TRADING_ARTIFACT_ROOT_V1),
    map_location: str = "cpu",
    explicit_rollback_selected: bool = False,
) -> Any:
    checkpoint_path = validate_checkpoint_before_load_v1(
        model=model,
        artifact_root=artifact_root,
        explicit_rollback_selected=explicit_rollback_selected,
    )
    try:
        return torch_module.load(
            str(checkpoint_path),
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch_module.load(str(checkpoint_path), map_location=map_location)


def validate_checkpoint_before_load_v1(
    *,
    model: ModelVersionRecord,
    artifact_root: Path = Path(RL_TRADING_ARTIFACT_ROOT_V1),
    explicit_rollback_selected: bool = False,
) -> Path:
    assert_model_runtime_load_allowed_v1(
        model,
        explicit_rollback_selected=explicit_rollback_selected,
    )
    root = _validate_artifact_root(artifact_root)
    checkpoint_path = Path(model.checkpoint_path)
    if not checkpoint_path.is_absolute():
        raise ModelRegistryError(reason="checkpoint_path_must_be_absolute")
    resolved = checkpoint_path.resolve(strict=False)
    _assert_under_root(resolved, root, field="checkpoint_path")
    if not resolved.is_file():
        raise ModelRegistryError(reason="checkpoint_file_missing", field=str(resolved))
    actual_sha256 = compute_file_sha256(resolved)
    if actual_sha256 != model.checkpoint_sha256:
        raise ModelRegistryError(reason="checkpoint_sha256_mismatch", field=str(resolved))
    return resolved


def mark_model_artifact_missing_v1(
    model: ModelVersionRecord, *, reason: str = "checkpoint_file_missing"
) -> ModelVersionRecord:
    _validate_reason_code(reason, "reason")
    validate_registry_transition_v1(
        entity="model_version",
        current_status=model.status,
        next_status="missing_artifact",
    )
    return replace(model, status="missing_artifact")


def plan_registry_cleanup_v1(
    *,
    artifacts: Sequence[RegistryArtifactRecord],
    policy: ArtifactLifecyclePolicy,
    now_utc: datetime,
) -> dict[str, object]:
    if now_utc.tzinfo is None:
        raise ModelRegistryError(reason="now_utc_must_be_timezone_aware")
    root = _validate_artifact_root(Path(policy.artifact_root))
    delete: list[dict[str, object]] = []
    retain: list[dict[str, object]] = []
    retention_seconds = policy.rejected_run_retention_days * 24 * 60 * 60

    for artifact in artifacts:
        resolved = Path(artifact.artifact_path).resolve(strict=False)
        _assert_under_root(resolved, root, field=f"artifact:{artifact.artifact_id}")
        age_seconds = (now_utc - artifact.updated_at_utc.astimezone(UTC)).total_seconds()
        reason = _cleanup_retention_reason(
            artifact=artifact,
            age_seconds=age_seconds,
            retention_seconds=retention_seconds,
        )
        row = {
            **artifact.as_payload(),
            "age_seconds": int(age_seconds),
            "retention_reason": reason,
        }
        if reason == "eligible_rejected_or_superseded_after_retention":
            delete.append(row)
        else:
            retain.append(row)

    payload: dict[str, object] = {
        "delete_candidates": sorted(delete, key=lambda item: str(item["artifact_id"])),
        "kind": "rl_trading_registry_cleanup_plan",
        "model_registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION_V1,
        "policy": policy.as_payload(),
        "policy_hash": policy.policy_hash(),
        "retain": sorted(retain, key=lambda item: str(item["artifact_id"])),
    }
    return {**payload, "cleanup_plan_hash": hash_json_payload_v1(payload)}


def registry_contract_payload_v1() -> dict[str, object]:
    return {
        "accepted_stage09_candidate": {
            "candidate_id": STAGE09_ACCEPTED_CANDIDATE_ID_V1,
            "candidate_manifest_sha256": STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1,
            "policy_id": STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
        },
        "artifact_root": RL_TRADING_ARTIFACT_ROOT_V1,
        "entities": {
            entity: {status: sorted(next_values) for status, next_values in statuses.items()}
            for entity, statuses in sorted(_ALLOWED_TRANSITIONS.items())
        },
        "kind": MODEL_REGISTRY_KIND_V1,
        "model_registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION_V1,
        "runtime_loadable_model_statuses": sorted(_RUNTIME_LOADABLE_MODEL_STATUSES),
        "trusted_checkpoint_producer": TRUSTED_TRAINER_PRODUCER_V1,
    }


def registry_contract_hash_v1() -> str:
    return hash_json_payload_v1(registry_contract_payload_v1())


def _cleanup_retention_reason(
    *, artifact: RegistryArtifactRecord, age_seconds: float, retention_seconds: int
) -> str:
    if artifact.source_manifest:
        return "source_manifest_retained"
    if artifact.referenced_by_active_metadata:
        return "active_metadata_reference_retained"
    if artifact.status in {"accepted", "accepted_champion", "rollback_candidate"}:
        return "accepted_or_rollback_artifact_retained"
    if artifact.status not in _CLEANUP_ELIGIBLE_STATUSES:
        return "status_not_cleanup_eligible"
    if age_seconds < retention_seconds:
        return "retention_window_not_elapsed"
    return "eligible_rejected_or_superseded_after_retention"


def _validate_artifact_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=False)
    if root != Path(RL_TRADING_ARTIFACT_ROOT_V1):
        raise ModelRegistryError(reason="unexpected_artifact_root", field=str(root))
    return root


def _assert_under_root(path: Path, root: Path, *, field: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelRegistryError(reason="path_outside_artifact_root", field=field) from exc


def _validate_store_path_text(value: str, field: str) -> None:
    _non_empty_text(value, field)
    path = Path(value)
    if not path.is_absolute():
        raise ModelRegistryError(reason="path_must_be_absolute", field=field)
    root = Path(RL_TRADING_ARTIFACT_ROOT_V1)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModelRegistryError(reason="path_outside_artifact_root", field=field) from exc


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ModelRegistryError(reason="invalid_sha256", field=field)


def _non_empty_text(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ModelRegistryError(reason="missing_text", field=field)


def _validate_reason_code(value: str, field: str) -> None:
    _non_empty_text(value, field)
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_:-")
    if len(value) > 96 or value[0] not in "abcdefghijklmnopqrstuvwxyz0123456789":
        raise ModelRegistryError(reason="invalid_reason_code", field=field)
    if any(char not in allowed for char in value):
        raise ModelRegistryError(reason="invalid_reason_code", field=field)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
