from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .feature_contract import FEATURE_CONTRACT_HASH_V1
from .hf_reproducibility import compute_file_sha256
from .model_registry import (
    RL_TRADING_ARTIFACT_ROOT_V1,
    STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    registry_contract_hash_v1,
)
from .raw_feature_dataset import hash_json_payload_v1

STAGE10A_SCHEMA_VERSION_V1 = 1
STAGE10A_RETRAIN_TASK_KIND_V1 = "rl_trading_stage10a_retrain_task_v1"
STAGE10A_PROMOTION_PROFILE_KIND_V1 = "rl_trading_stage10a_promotion_threshold_profile_v1"
STAGE10A_PROMOTION_CHECK_KIND_V1 = "rl_trading_stage10a_promotion_check_v1"
STAGE10A_ROLLBACK_KIND_V1 = "rl_trading_stage10a_rollback_manifest_v1"
STAGE10A_SUMMARY_KIND_V1 = "rl_trading_stage10a_lifecycle_summary_v1"
STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage10a_retraining_promotion_lifecycle_v1"
DEFAULT_STAGE10A_OUTPUT_ROOT_V1 = (
    f"{RL_TRADING_ARTIFACT_ROOT_V1}/lifecycle_runs/"
    f"{STAGE10A_RUNTIME_ARTIFACT_SUBDIR_V1}"
)
DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1 = (
    "stage10_stage08m_a3823cbd01143878_fd7c614b_fd7c614b_per_ticker"
)
DEFAULT_STAGE10A_CALIBRATION_PACK_HASH_V1 = (
    "7650c16337cb7ea8d95882ca0942c97c5846f65827d573733294e43ce3d19f42"
)
DEFAULT_STAGE10A_SOURCE_MANIFEST_SHA256_V1 = (
    "fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a"
)

RetrainMode = Literal["full_retrain", "fine_tune"]
RetrainTrigger = Literal["manual", "scheduled", "drift"]
PromotionGateSeverity = Literal["hard", "warn"]
PromotionGateStatus = Literal["passed", "blocked", "warn"]


class Stage10ALifecycleError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Stage10ARetrainTaskConfig:
    artifact_root: Path
    output_root: Path
    run_id: str
    generated_at_utc: datetime
    retrain_mode: RetrainMode
    trigger: RetrainTrigger
    base_model_version_id: str = STAGE09_ACCEPTED_CANDIDATE_ID_V1
    calibration_pack_id: str = DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1
    calibration_pack_hash: str = DEFAULT_STAGE10A_CALIBRATION_PACK_HASH_V1
    source_manifest_sha256: str = DEFAULT_STAGE10A_SOURCE_MANIFEST_SHA256_V1
    feature_contract_hash: str = FEATURE_CONTRACT_HASH_V1
    schedule_enabled: bool = False
    schedule_id: str | None = None
    drift_signal_id: str | None = None
    requested_by_ref_hash: str | None = None
    auto_promote_requested: bool = False

    def __post_init__(self) -> None:
        _validate_artifact_root(self.artifact_root)
        _validate_output_root(self.output_root)
        _validate_run_id(self.run_id)
        if self.generated_at_utc.tzinfo is None:
            raise Stage10ALifecycleError(reason="generated_at_utc_must_be_timezone_aware")
        if self.retrain_mode not in {"full_retrain", "fine_tune"}:
            raise Stage10ALifecycleError(reason="unsupported_retrain_mode")
        if self.trigger not in {"manual", "scheduled", "drift"}:
            raise Stage10ALifecycleError(reason="unsupported_retrain_trigger")
        _non_empty_text(self.base_model_version_id, "base_model_version_id")
        _non_empty_text(self.calibration_pack_id, "calibration_pack_id")
        _validate_sha256(self.calibration_pack_hash, "calibration_pack_hash")
        _validate_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _validate_sha256(self.feature_contract_hash, "feature_contract_hash")
        if self.schedule_id is not None:
            _validate_reason_code(self.schedule_id, "schedule_id")
        if self.drift_signal_id is not None:
            _validate_reason_code(self.drift_signal_id, "drift_signal_id")
        if self.requested_by_ref_hash is not None:
            _validate_sha256(self.requested_by_ref_hash, "requested_by_ref_hash")


@dataclass(frozen=True, slots=True)
class Stage10APromotionScorecard:
    pnl_after_fees_funding_slippage_quote: float
    max_drawdown_quote: float
    trades_count: int
    ticker_positive_group_ratio: float
    out_of_sample_days: int
    overfit_ratio: float
    latency_p95_ms: float
    resource_rss_mb: float
    artifact_integrity_ok: bool
    registry_integrity_ok: bool

    def __post_init__(self) -> None:
        if self.trades_count < 0:
            raise Stage10ALifecycleError(reason="negative_trades_count")
        if self.out_of_sample_days < 0:
            raise Stage10ALifecycleError(reason="negative_out_of_sample_days")
        if not 0.0 <= self.ticker_positive_group_ratio <= 1.0:
            raise Stage10ALifecycleError(reason="ticker_positive_group_ratio_out_of_range")
        if min(
            self.max_drawdown_quote,
            self.overfit_ratio,
            self.latency_p95_ms,
            self.resource_rss_mb,
        ) < 0.0:
            raise Stage10ALifecycleError(reason="negative_metric_value")

    def as_payload(self) -> dict[str, object]:
        return {
            "artifact_integrity_ok": self.artifact_integrity_ok,
            "latency_p95_ms": _round_float(self.latency_p95_ms),
            "max_drawdown_quote": _round_float(self.max_drawdown_quote),
            "out_of_sample_days": self.out_of_sample_days,
            "overfit_ratio": _round_float(self.overfit_ratio),
            "pnl_after_fees_funding_slippage_quote": _round_float(
                self.pnl_after_fees_funding_slippage_quote
            ),
            "registry_integrity_ok": self.registry_integrity_ok,
            "resource_rss_mb": _round_float(self.resource_rss_mb),
            "ticker_positive_group_ratio": _round_float(self.ticker_positive_group_ratio),
            "trades_count": self.trades_count,
        }


@dataclass(frozen=True, slots=True)
class Stage10APromotionCheckConfig:
    artifact_root: Path
    output_root: Path
    run_id: str
    generated_at_utc: datetime
    candidate_model_version_id: str
    current_champion_model_version_id: str
    candidate_manifest_path: Path
    expected_candidate_manifest_sha256: str
    calibration_pack_path: Path
    expected_calibration_pack_sha256: str
    calibration_pack_id: str
    calibration_pack_hash: str
    scorecard: Stage10APromotionScorecard
    operator_ref_hash: str | None
    admin_ref_hash: str | None
    approval_reason: str
    auto_promote_requested: bool = False

    def __post_init__(self) -> None:
        _validate_artifact_root(self.artifact_root)
        _validate_output_root(self.output_root)
        _validate_run_id(self.run_id)
        if self.generated_at_utc.tzinfo is None:
            raise Stage10ALifecycleError(reason="generated_at_utc_must_be_timezone_aware")
        _non_empty_text(self.candidate_model_version_id, "candidate_model_version_id")
        _non_empty_text(self.current_champion_model_version_id, "current_champion_model_version_id")
        _validate_sha256(
            self.expected_candidate_manifest_sha256,
            "expected_candidate_manifest_sha256",
        )
        _validate_sha256(
            self.expected_calibration_pack_sha256,
            "expected_calibration_pack_sha256",
        )
        _non_empty_text(self.calibration_pack_id, "calibration_pack_id")
        _validate_sha256(self.calibration_pack_hash, "calibration_pack_hash")
        if self.operator_ref_hash is not None:
            _validate_sha256(self.operator_ref_hash, "operator_ref_hash")
        if self.admin_ref_hash is not None:
            _validate_sha256(self.admin_ref_hash, "admin_ref_hash")
        _validate_reason_code(self.approval_reason, "approval_reason")


@dataclass(frozen=True, slots=True)
class Stage10ARollbackConfig:
    artifact_root: Path
    output_root: Path
    run_id: str
    generated_at_utc: datetime
    current_champion_model_version_id: str
    previous_champion_model_version_id: str
    current_calibration_pack_id: str
    previous_calibration_pack_id: str
    current_registry_metadata_sha256: str
    previous_champion_manifest_sha256: str
    previous_calibration_pack_sha256: str
    operator_ref_hash: str
    reason: str

    def __post_init__(self) -> None:
        _validate_artifact_root(self.artifact_root)
        _validate_output_root(self.output_root)
        _validate_run_id(self.run_id)
        if self.generated_at_utc.tzinfo is None:
            raise Stage10ALifecycleError(reason="generated_at_utc_must_be_timezone_aware")
        _non_empty_text(self.current_champion_model_version_id, "current_champion_model_version_id")
        _non_empty_text(
            self.previous_champion_model_version_id,
            "previous_champion_model_version_id",
        )
        _non_empty_text(self.current_calibration_pack_id, "current_calibration_pack_id")
        _non_empty_text(self.previous_calibration_pack_id, "previous_calibration_pack_id")
        _validate_sha256(self.current_registry_metadata_sha256, "current_registry_metadata_sha256")
        _validate_sha256(
            self.previous_champion_manifest_sha256,
            "previous_champion_manifest_sha256",
        )
        _validate_sha256(self.previous_calibration_pack_sha256, "previous_calibration_pack_sha256")
        _validate_sha256(self.operator_ref_hash, "operator_ref_hash")
        _validate_reason_code(self.reason, "reason")


def run_stage10a_retrain_task_plan_v1(
    config: Stage10ARetrainTaskConfig,
) -> dict[str, object]:
    run_dir = _run_dir(config.output_root, config.run_id)
    profile = build_stage10a_promotion_threshold_profile_v1(
        generated_at_utc=config.generated_at_utc,
    )
    profile_path = run_dir / "stage10a_promotion_threshold_profile.json"
    _atomic_write_json(profile_path, profile)
    profile_sha256 = compute_file_sha256(profile_path)

    task = build_stage10a_retrain_task_payload_v1(config)
    task_path = run_dir / "stage10a_retrain_task_manifest.json"
    _atomic_write_json(task_path, task)
    task_sha256 = compute_file_sha256(task_path)

    status = "accepted" if task["status"] == "planned_candidate" else "blocked"
    summary_payload = {
        "auto_promote": False,
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "kind": STAGE10A_SUMMARY_KIND_V1,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha256,
        "proof_boundary": "target_host_readiness_pre_main",
        "registry_write_performed": False,
        "retrain_task_path": str(task_path),
        "retrain_task_sha256": task_sha256,
        "run_dir": str(run_dir),
        "run_id": config.run_id,
        "schema_version": STAGE10A_SCHEMA_VERSION_V1,
        "stage": "10A",
        "status": status,
        "trigger": config.trigger,
    }
    summary = {**summary_payload, "summary_hash": hash_json_payload_v1(summary_payload)}
    summary_path = run_dir / "stage10a_retrain_lifecycle_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": compute_file_sha256(summary_path),
    }


def build_stage10a_retrain_task_payload_v1(
    config: Stage10ARetrainTaskConfig,
) -> dict[str, object]:
    blockers: list[str] = []
    if config.trigger == "scheduled" and not config.schedule_enabled:
        blockers.append("schedule_disabled_by_default")
    if config.trigger == "drift" and config.drift_signal_id is None:
        blockers.append("drift_signal_id_required")
    if config.auto_promote_requested:
        blockers.append("auto_promotion_requested_forbidden")

    run_config = {
        "base_model_version_id": config.base_model_version_id,
        "calibration_pack_hash": config.calibration_pack_hash,
        "calibration_pack_id": config.calibration_pack_id,
        "feature_contract_hash": config.feature_contract_hash,
        "retrain_mode": config.retrain_mode,
        "source_manifest_sha256": config.source_manifest_sha256,
        "trigger": config.trigger,
    }
    status = "planned_candidate" if not blockers else "blocked"
    body = {
        "artifact_root": RL_TRADING_ARTIFACT_ROOT_V1,
        "auto_promote": False,
        "base_model_version_id": config.base_model_version_id,
        "blockers": blockers,
        "candidate_output_state": "candidate_pending_promotion_scorecard",
        "calibration_pack_hash": config.calibration_pack_hash,
        "calibration_pack_id": config.calibration_pack_id,
        "drift": {
            "creates_candidate_task": config.trigger == "drift",
            "drift_signal_id": config.drift_signal_id,
            "mutates_champion": False,
        },
        "exchange_side_effects": False,
        "feature_contract_hash": config.feature_contract_hash,
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "kind": STAGE10A_RETRAIN_TASK_KIND_V1,
        "manual_trigger_supported": True,
        "model_registry_contract_hash": registry_contract_hash_v1(),
        "paper_testnet_live_enabled": False,
        "registry_write_performed": False,
        "requested_by_ref_hash": config.requested_by_ref_hash,
        "retrain_mode": config.retrain_mode,
        "run_config": run_config,
        "run_config_hash": hash_json_payload_v1(run_config),
        "run_id": config.run_id,
        "schedule": {
            "enabled": config.schedule_enabled,
            "schedule_id": config.schedule_id,
            "status": "enabled_explicitly" if config.schedule_enabled else "disabled_by_default",
        },
        "schema_version": STAGE10A_SCHEMA_VERSION_V1,
        "source_manifest_sha256": config.source_manifest_sha256,
        "stage": "10A",
        "status": status,
        "supported_retrain_modes": ["fine_tune", "full_retrain"],
        "target_artifact_root": RL_TRADING_ARTIFACT_ROOT_V1,
        "trigger": config.trigger,
    }
    return {**body, "retrain_task_hash": hash_json_payload_v1(body)}


def build_stage10a_promotion_threshold_profile_v1(
    *, generated_at_utc: datetime
) -> dict[str, object]:
    if generated_at_utc.tzinfo is None:
        raise Stage10ALifecycleError(reason="generated_at_utc_must_be_timezone_aware")
    body: dict[str, object] = {
        "approval_contract": {
            "admin_ref_hash_required": True,
            "auto_promote_allowed": False,
            "operator_ref_hash_required": True,
            "registry_write_requires_explicit_command": True,
        },
        "generated_at_utc": _format_utc(generated_at_utc),
        "kind": STAGE10A_PROMOTION_PROFILE_KIND_V1,
        "metric_units": {
            "latency_p95_ms": "milliseconds",
            "max_drawdown_quote": "quote_currency",
            "pnl_after_fees_funding_slippage_quote": "quote_currency",
            "resource_rss_mb": "MiB",
        },
        "numeric_thresholds": {
            "latency_p95_ms": {"max": 750.0, "severity": "warn"},
            "max_drawdown_quote": {"max": 25000.0, "severity": "warn"},
            "out_of_sample_days": {"min": 30, "severity": "hard"},
            "overfit_ratio": {"max": 1.5, "severity": "warn"},
            "pnl_after_fees_funding_slippage_quote": {"min": 0.0, "severity": "hard"},
            "resource_rss_mb": {"max": 32768.0, "severity": "warn"},
            "ticker_positive_group_ratio": {"min": 0.25, "severity": "hard"},
            "trades_count": {"min": 100, "severity": "hard"},
        },
        "registry_contract_hash": registry_contract_hash_v1(),
        "schema_version": STAGE10A_SCHEMA_VERSION_V1,
        "stage": "10A",
    }
    return {**body, "threshold_profile_hash": hash_json_payload_v1(body)}


def run_stage10a_promotion_check_v1(
    config: Stage10APromotionCheckConfig,
) -> dict[str, object]:
    run_dir = _run_dir(config.output_root, config.run_id)
    payload = build_stage10a_promotion_check_payload_v1(config)
    check_path = run_dir / "stage10a_promotion_check.json"
    _atomic_write_json(check_path, payload)
    check_sha256 = compute_file_sha256(check_path)
    summary_payload = {
        "activation_mutation": False,
        "auto_promote": False,
        "check_path": str(check_path),
        "check_sha256": check_sha256,
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "kind": STAGE10A_SUMMARY_KIND_V1,
        "proof_boundary": "target_host_readiness_pre_main",
        "registry_write_performed": False,
        "run_dir": str(run_dir),
        "run_id": config.run_id,
        "schema_version": STAGE10A_SCHEMA_VERSION_V1,
        "stage": "10A",
        "status": "accepted" if payload["status"] == "promotion_ready" else "blocked",
    }
    summary = {**summary_payload, "summary_hash": hash_json_payload_v1(summary_payload)}
    summary_path = run_dir / "stage10a_promotion_check_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": compute_file_sha256(summary_path),
    }


def build_stage10a_promotion_check_payload_v1(
    config: Stage10APromotionCheckConfig,
) -> dict[str, object]:
    candidate_manifest_sha256 = _check_file_hash(
        path=config.candidate_manifest_path,
        expected_sha256=config.expected_candidate_manifest_sha256,
        field="candidate_manifest",
    )
    calibration_pack_sha256 = _check_file_hash(
        path=config.calibration_pack_path,
        expected_sha256=config.expected_calibration_pack_sha256,
        field="calibration_pack",
    )
    profile = build_stage10a_promotion_threshold_profile_v1(
        generated_at_utc=config.generated_at_utc,
    )
    thresholds = _mapping(profile["numeric_thresholds"], "numeric_thresholds")
    gates = _promotion_metric_gates(config.scorecard, thresholds)
    gates.extend(
        [
            _boolean_gate(
                name="artifact_registry_integrity",
                passed=config.scorecard.artifact_integrity_ok
                and config.scorecard.registry_integrity_ok,
                severity="hard",
            ),
            _boolean_gate(
                name="candidate_differs_from_current_champion",
                passed=config.candidate_model_version_id
                != config.current_champion_model_version_id,
                severity="hard",
            ),
            _boolean_gate(
                name="operator_approval_present",
                passed=config.operator_ref_hash is not None,
                severity="hard",
            ),
            _boolean_gate(
                name="admin_approval_present",
                passed=config.admin_ref_hash is not None,
                severity="hard",
            ),
            _boolean_gate(
                name="auto_promotion_not_requested",
                passed=not config.auto_promote_requested,
                severity="hard",
            ),
        ]
    )
    hard_blockers = sorted(
        str(gate["name"])
        for gate in gates
        if gate["severity"] == "hard" and gate["status"] == "blocked"
    )
    warn_gates = sorted(str(gate["name"]) for gate in gates if gate["status"] == "warn")
    body = {
        "activation_mutation": False,
        "admin_ref_hash": config.admin_ref_hash,
        "approval_reason": config.approval_reason,
        "artifact_inputs": {
            "calibration_pack_hash": config.calibration_pack_hash,
            "calibration_pack_id": config.calibration_pack_id,
            "calibration_pack_path": str(config.calibration_pack_path),
            "calibration_pack_sha256": calibration_pack_sha256,
            "candidate_manifest_path": str(config.candidate_manifest_path),
            "candidate_manifest_sha256": candidate_manifest_sha256,
        },
        "auto_promote": False,
        "blockers": hard_blockers,
        "candidate_model_version_id": config.candidate_model_version_id,
        "current_champion_model_version_id": config.current_champion_model_version_id,
        "gates": sorted(gates, key=lambda gate: str(gate["name"])),
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "kind": STAGE10A_PROMOTION_CHECK_KIND_V1,
        "operator_ref_hash": config.operator_ref_hash,
        "registry_contract_hash": registry_contract_hash_v1(),
        "registry_write_performed": False,
        "schema_version": STAGE10A_SCHEMA_VERSION_V1,
        "scorecard": config.scorecard.as_payload(),
        "stage": "10A",
        "status": "promotion_ready" if not hard_blockers else "blocked",
        "threshold_profile_hash": profile["threshold_profile_hash"],
        "warnings": warn_gates,
    }
    return {**body, "promotion_check_hash": hash_json_payload_v1(body)}


def run_stage10a_rollback_dry_run_v1(config: Stage10ARollbackConfig) -> dict[str, object]:
    run_dir = _run_dir(config.output_root, config.run_id)
    payload = build_stage10a_rollback_manifest_v1(config)
    rollback_path = run_dir / "stage10a_rollback_manifest.json"
    _atomic_write_json(rollback_path, payload)
    rollback_sha256 = compute_file_sha256(rollback_path)
    summary_payload = {
        "activation_mutation": False,
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "kind": STAGE10A_SUMMARY_KIND_V1,
        "no_artifact_deletion": True,
        "proof_boundary": "target_host_readiness_pre_main",
        "registry_write_performed": False,
        "rollback_manifest_path": str(rollback_path),
        "rollback_manifest_sha256": rollback_sha256,
        "run_dir": str(run_dir),
        "run_id": config.run_id,
        "schema_version": STAGE10A_SCHEMA_VERSION_V1,
        "stage": "10A",
        "status": "accepted",
    }
    summary = {**summary_payload, "summary_hash": hash_json_payload_v1(summary_payload)}
    summary_path = run_dir / "stage10a_rollback_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": compute_file_sha256(summary_path),
    }


def build_stage10a_rollback_manifest_v1(config: Stage10ARollbackConfig) -> dict[str, object]:
    body = {
        "activation_mutation": False,
        "command": (
            "uv run python scripts/rl_trading/stage10a_retraining_promotion_lifecycle.py "
            "rollback-dry-run "
            f"--to-model-version-id {config.previous_champion_model_version_id} "
            f"--expected-current-model-version-id {config.current_champion_model_version_id} "
            f"--to-calibration-pack-id {config.previous_calibration_pack_id} "
            f"--expected-current-calibration-pack-id {config.current_calibration_pack_id} "
            f"--operator-ref-hash {config.operator_ref_hash} "
            f"--reason {config.reason}"
        ),
        "current_calibration_pack_id": config.current_calibration_pack_id,
        "current_champion_model_version_id": config.current_champion_model_version_id,
        "current_registry_metadata_sha256": config.current_registry_metadata_sha256,
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "kind": STAGE10A_ROLLBACK_KIND_V1,
        "no_artifact_deletion": True,
        "operator_ref_hash": config.operator_ref_hash,
        "previous_calibration_pack_id": config.previous_calibration_pack_id,
        "previous_calibration_pack_sha256": config.previous_calibration_pack_sha256,
        "previous_champion_manifest_sha256": config.previous_champion_manifest_sha256,
        "previous_champion_model_version_id": config.previous_champion_model_version_id,
        "reason": config.reason,
        "registry_write_performed": False,
        "schema_version": STAGE10A_SCHEMA_VERSION_V1,
        "stage": "10A",
        "status": "rollback_dry_run_ready",
    }
    return {**body, "rollback_manifest_hash": hash_json_payload_v1(body)}


def _promotion_metric_gates(
    scorecard: Stage10APromotionScorecard,
    thresholds: Mapping[str, Any],
) -> list[dict[str, object]]:
    return [
        _min_gate(
            name="pnl_after_fees_funding_slippage_quote",
            value=scorecard.pnl_after_fees_funding_slippage_quote,
            threshold=_threshold(thresholds, "pnl_after_fees_funding_slippage_quote", "min"),
        ),
        _max_gate(
            name="max_drawdown_quote",
            value=scorecard.max_drawdown_quote,
            threshold=_threshold(thresholds, "max_drawdown_quote", "max"),
        ),
        _min_gate(
            name="trades_count",
            value=float(scorecard.trades_count),
            threshold=_threshold(thresholds, "trades_count", "min"),
        ),
        _min_gate(
            name="ticker_positive_group_ratio",
            value=scorecard.ticker_positive_group_ratio,
            threshold=_threshold(thresholds, "ticker_positive_group_ratio", "min"),
        ),
        _min_gate(
            name="out_of_sample_days",
            value=float(scorecard.out_of_sample_days),
            threshold=_threshold(thresholds, "out_of_sample_days", "min"),
        ),
        _max_gate(
            name="overfit_ratio",
            value=scorecard.overfit_ratio,
            threshold=_threshold(thresholds, "overfit_ratio", "max"),
        ),
        _max_gate(
            name="latency_p95_ms",
            value=scorecard.latency_p95_ms,
            threshold=_threshold(thresholds, "latency_p95_ms", "max"),
        ),
        _max_gate(
            name="resource_rss_mb",
            value=scorecard.resource_rss_mb,
            threshold=_threshold(thresholds, "resource_rss_mb", "max"),
        ),
    ]


def _threshold(
    thresholds: Mapping[str, Any],
    name: str,
    boundary: Literal["min", "max"],
) -> tuple[float, PromotionGateSeverity]:
    row = _mapping(thresholds.get(name), name)
    value = _float_value(row.get(boundary), f"{name}.{boundary}")
    severity_raw = row.get("severity")
    if severity_raw not in {"hard", "warn"}:
        raise Stage10ALifecycleError(reason="unsupported_threshold_severity", field=name)
    return value, cast(PromotionGateSeverity, severity_raw)


def _min_gate(
    *,
    name: str,
    value: float,
    threshold: tuple[float, PromotionGateSeverity],
) -> dict[str, object]:
    limit, severity = threshold
    passed = value >= limit
    return _metric_gate(
        name=name,
        passed=passed,
        severity=severity,
        threshold={"min": _round_float(limit)},
        value=value,
    )


def _max_gate(
    *,
    name: str,
    value: float,
    threshold: tuple[float, PromotionGateSeverity],
) -> dict[str, object]:
    limit, severity = threshold
    passed = value <= limit
    return _metric_gate(
        name=name,
        passed=passed,
        severity=severity,
        threshold={"max": _round_float(limit)},
        value=value,
    )


def _boolean_gate(
    *,
    name: str,
    passed: bool,
    severity: PromotionGateSeverity,
) -> dict[str, object]:
    return _metric_gate(
        name=name,
        passed=passed,
        severity=severity,
        threshold={"required": True},
        value=passed,
    )


def _metric_gate(
    *,
    name: str,
    passed: bool,
    severity: PromotionGateSeverity,
    threshold: Mapping[str, object],
    value: object,
) -> dict[str, object]:
    status: PromotionGateStatus
    if passed:
        status = "passed"
    elif severity == "warn":
        status = "warn"
    else:
        status = "blocked"
    return {
        "name": name,
        "severity": severity,
        "status": status,
        "threshold": dict(threshold),
        "value": _round_float(value) if isinstance(value, float) else value,
    }


def _run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = _validate_output_root(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _check_file_hash(*, path: Path, expected_sha256: str, field: str) -> str:
    _validate_store_path(path, field)
    actual = compute_file_sha256(path)
    if actual != expected_sha256:
        raise Stage10ALifecycleError(reason=f"{field}_sha256_mismatch", field=str(path))
    return actual


def _validate_artifact_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=False)
    expected = Path(RL_TRADING_ARTIFACT_ROOT_V1).resolve(strict=False)
    if root != expected:
        raise Stage10ALifecycleError(reason="unexpected_artifact_root", field=str(root))
    return root


def _validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    root = _validate_artifact_root(Path(RL_TRADING_ARTIFACT_ROOT_V1))
    _assert_under_root(resolved, root, field="output_root")
    return resolved


def _validate_store_path(path: Path, field: str) -> Path:
    if not path.is_absolute():
        raise Stage10ALifecycleError(reason="path_must_be_absolute", field=field)
    resolved = path.expanduser().resolve(strict=False)
    root = _validate_artifact_root(Path(RL_TRADING_ARTIFACT_ROOT_V1))
    _assert_under_root(resolved, root, field=field)
    if not resolved.is_file():
        raise Stage10ALifecycleError(reason="file_missing", field=str(resolved))
    return resolved


def _assert_under_root(path: Path, root: Path, *, field: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Stage10ALifecycleError(reason="path_outside_artifact_root", field=field) from exc


def _validate_run_id(value: str) -> None:
    _non_empty_text(value, "run_id")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if any(char not in allowed for char in value):
        raise Stage10ALifecycleError(reason="invalid_run_id", field=value)


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Stage10ALifecycleError(reason="invalid_sha256", field=field)


def _validate_reason_code(value: str, field: str) -> None:
    _non_empty_text(value, field)
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_:-")
    if len(value) > 96 or value[0] not in "abcdefghijklmnopqrstuvwxyz0123456789":
        raise Stage10ALifecycleError(reason="invalid_reason_code", field=field)
    if any(char not in allowed for char in value):
        raise Stage10ALifecycleError(reason="invalid_reason_code", field=field)


def _float_value(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise Stage10ALifecycleError(reason="expected_float", field=field)
    try:
        return float(cast(float, value))
    except (TypeError, ValueError) as exc:
        raise Stage10ALifecycleError(reason="expected_float", field=field) from exc


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Stage10ALifecycleError(reason="expected_mapping", field=field)
    return cast(dict[str, Any], value)


def _non_empty_text(value: str, field: str) -> None:
    if not value or not value.strip():
        raise Stage10ALifecycleError(reason="missing_text", field=field)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _round_float(value: object) -> float:
    return float(round(float(cast(float, value)), 10))


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
