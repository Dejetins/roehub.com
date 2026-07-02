from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .feature_contract import FEATURE_CONTRACT_HASH_V1
from .hf_reproducibility import compute_file_sha256
from .model_registry import (
    RL_TRADING_ARTIFACT_ROOT_V1,
    STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1,
    STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
    CalibrationPackRecord,
    registry_contract_hash_v1,
)
from .raw_feature_dataset import hash_json_payload_v1

STAGE10_SCHEMA_VERSION_V1 = 1
STAGE10_CALIBRATION_KIND_V1 = "rl_trading_stage10_per_ticker_calibration_pack_v1"
STAGE10_REGISTRY_RECORD_KIND_V1 = "rl_trading_stage10_calibration_registry_record_v1"
STAGE10_SUMMARY_KIND_V1 = "rl_trading_stage10_per_ticker_calibration_summary_v1"
STAGE10_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage10_per_ticker_calibration_v1"
DEFAULT_STAGE10_OUTPUT_ROOT_V1 = (
    f"{RL_TRADING_ARTIFACT_ROOT_V1}/calibration_packs/"
    f"{STAGE10_RUNTIME_ARTIFACT_SUBDIR_V1}"
)
DEFAULT_STAGE10_MIN_TICKER_SESSIONS_V1 = 10
DEFAULT_STAGE10_MIN_TICKER_POSITIVE_RATIO_V1 = 0.50
DEFAULT_STAGE10_MIN_TICKER_NET_PNL_AFTER_COSTS_QUOTE_V1 = 0.0
DEFAULT_STAGE10_FULL_CONFIDENCE_SESSIONS_V1 = 30
DEFAULT_STAGE10_DOMINANCE_SHARE_LIMIT_V1 = 0.80

CalibrationTickerStatus = Literal["accepted", "blocked"]


class Stage10CalibrationError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Stage10CalibrationConfig:
    artifact_root: Path
    output_root: Path
    run_id: str
    generated_at_utc: datetime
    candidate_summary_path: Path
    expected_candidate_summary_sha256: str
    candidate_manifest_path: Path
    expected_candidate_manifest_sha256: str
    source_manifest_path: Path
    expected_source_manifest_sha256: str
    model_version_id: str = STAGE09_ACCEPTED_CANDIDATE_ID_V1
    exchange: str = "binance"
    market_type: str = "futures"
    min_ticker_sessions: int = DEFAULT_STAGE10_MIN_TICKER_SESSIONS_V1
    min_ticker_positive_ratio: float = DEFAULT_STAGE10_MIN_TICKER_POSITIVE_RATIO_V1
    min_ticker_net_pnl_after_costs_quote: float = (
        DEFAULT_STAGE10_MIN_TICKER_NET_PNL_AFTER_COSTS_QUOTE_V1
    )
    full_confidence_sessions: int = DEFAULT_STAGE10_FULL_CONFIDENCE_SESSIONS_V1
    dominance_share_limit: float = DEFAULT_STAGE10_DOMINANCE_SHARE_LIMIT_V1
    pnl_weight: float = 0.45
    positive_ratio_weight: float = 0.25
    turnover_weight: float = 0.15
    risk_concentration_weight: float = 0.15

    def __post_init__(self) -> None:
        _validate_artifact_root(self.artifact_root)
        _validate_output_root(self.output_root)
        _validate_run_id(self.run_id)
        if self.generated_at_utc.tzinfo is None:
            raise Stage10CalibrationError(reason="generated_at_utc_must_be_timezone_aware")
        _validate_sha256(
            self.expected_candidate_summary_sha256,
            "expected_candidate_summary_sha256",
        )
        _validate_sha256(
            self.expected_candidate_manifest_sha256,
            "expected_candidate_manifest_sha256",
        )
        _validate_sha256(
            self.expected_source_manifest_sha256,
            "expected_source_manifest_sha256",
        )
        _non_empty_text(self.model_version_id, "model_version_id")
        if self.exchange != self.exchange.strip().lower() or not self.exchange:
            raise Stage10CalibrationError(reason="exchange_must_be_lowercase")
        if self.market_type not in {"spot", "futures"}:
            raise Stage10CalibrationError(reason="unsupported_market_type")
        if self.min_ticker_sessions <= 0:
            raise Stage10CalibrationError(reason="min_ticker_sessions_required")
        if not 0.0 < self.min_ticker_positive_ratio <= 1.0:
            raise Stage10CalibrationError(reason="min_ticker_positive_ratio_out_of_range")
        if self.full_confidence_sessions < self.min_ticker_sessions:
            raise Stage10CalibrationError(reason="full_confidence_sessions_too_low")
        if not 0.0 < self.dominance_share_limit <= 1.0:
            raise Stage10CalibrationError(reason="dominance_share_limit_out_of_range")
        if min(
            self.pnl_weight,
            self.positive_ratio_weight,
            self.turnover_weight,
            self.risk_concentration_weight,
        ) < 0.0:
            raise Stage10CalibrationError(reason="negative_objective_weight")
        if self._active_weight_sum() <= 0.0:
            raise Stage10CalibrationError(reason="objective_weights_required")

    def objective_weights(self) -> dict[str, float]:
        total = self._active_weight_sum()
        return {
            "drawdown": 0.0,
            "pnl_after_costs": _round_float(self.pnl_weight / total),
            "positive_ratio_stability": _round_float(self.positive_ratio_weight / total),
            "risk_concentration": _round_float(self.risk_concentration_weight / total),
            "turnover_evidence": _round_float(self.turnover_weight / total),
        }

    def _active_weight_sum(self) -> float:
        return (
            self.pnl_weight
            + self.positive_ratio_weight
            + self.turnover_weight
            + self.risk_concentration_weight
        )


def run_stage10_per_ticker_calibration_v1(
    config: Stage10CalibrationConfig,
) -> dict[str, object]:
    run_dir = _validate_output_root(config.output_root) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pack = build_stage10_calibration_pack_payload_v1(config)
    pack_path = run_dir / "stage10_per_ticker_calibration_pack.json"
    _atomic_write_json(pack_path, pack)
    pack_sha256 = compute_file_sha256(pack_path)
    registry_record = CalibrationPackRecord(
        calibration_pack_id=str(pack["calibration_pack_id"]),
        model_version_id=config.model_version_id,
        feature_contract_hash=FEATURE_CONTRACT_HASH_V1,
        dataset_hash=str(pack["dataset_hash"]),
        calibration_pack_hash=str(pack["calibration_pack_hash"]),
        calibration_path=str(pack_path),
        calibration_sha256=pack_sha256,
        status="accepted" if pack["status"] == "accepted" else "rejected",
    )
    registry_payload = {
        "calibration_pack": registry_record.as_payload(),
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "kind": STAGE10_REGISTRY_RECORD_KIND_V1,
        "model_registry_contract_hash": registry_contract_hash_v1(),
        "schema_version": STAGE10_SCHEMA_VERSION_V1,
        "source_calibration_pack_hash": pack["calibration_pack_hash"],
        "stage": "10",
    }
    registry_path = run_dir / "stage10_calibration_registry_record.json"
    _atomic_write_json(registry_path, registry_payload)
    registry_sha256 = compute_file_sha256(registry_path)
    summary_payload = {
        "accepted_ticker_count": pack["accepted_ticker_count"],
        "blocked_ticker_count": pack["blocked_ticker_count"],
        "calibration_pack_hash": pack["calibration_pack_hash"],
        "calibration_pack_id": pack["calibration_pack_id"],
        "calibration_pack_path": str(pack_path),
        "calibration_pack_sha256": pack_sha256,
        "delivery_state": "local_or_target_host_artifact_only_no_activation",
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "global_only_threshold_activated": False,
        "kind": STAGE10_SUMMARY_KIND_V1,
        "model_version_id": config.model_version_id,
        "proof_boundary": "target_host_readiness_pre_main",
        "registry_record_path": str(registry_path),
        "registry_record_sha256": registry_sha256,
        "run_dir": str(run_dir),
        "run_id": config.run_id,
        "schema_version": STAGE10_SCHEMA_VERSION_V1,
        "stage": "10",
        "status": pack["status"],
        "ticker_count": pack["ticker_count"],
    }
    summary = {**summary_payload, "summary_hash": hash_json_payload_v1(summary_payload)}
    summary_path = run_dir / "stage10_per_ticker_calibration_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        **summary,
        "summary_path": str(summary_path),
        "summary_sha256": compute_file_sha256(summary_path),
    }


def build_stage10_calibration_pack_payload_v1(
    config: Stage10CalibrationConfig,
) -> dict[str, object]:
    source = _load_and_validate_inputs(config)
    scorecard = source["candidate_scorecard"]
    ticker_rows = _ticker_rows(scorecard)
    if not ticker_rows:
        raise Stage10CalibrationError(reason="ticker_rows_missing")
    cost_model = _mapping(source["summary"].get("cost_model"), "cost_model")
    round_trip_cost_ratio = _float_value(
        cost_model.get("round_trip_cost_ratio", 0.0),
        "round_trip_cost_ratio",
    )
    total_abs_pnl = sum(abs(_row_pnl(row)) for row in ticker_rows) or 1.0
    max_positive_pnl = max((_row_pnl(row) for row in ticker_rows), default=1.0)
    if max_positive_pnl <= 0.0:
        max_positive_pnl = 1.0
    calibrated_rows = [
        _calibration_row(
            config=config,
            row=row,
            max_positive_pnl=max_positive_pnl,
            round_trip_cost_ratio=round_trip_cost_ratio,
            total_abs_pnl=total_abs_pnl,
        )
        for row in ticker_rows
    ]
    accepted_count = sum(1 for row in calibrated_rows if row["status"] == "accepted")
    blocked_count = len(calibrated_rows) - accepted_count
    calibration_pack_id = _calibration_pack_id(config=config)
    candidate_manifest = source["candidate_manifest"]
    body = {
        "accepted_ticker_count": accepted_count,
        "artifact_kind": STAGE10_CALIBRATION_KIND_V1,
        "blocked_ticker_count": blocked_count,
        "calibration_objective": {
            "metric_status": {
                "drawdown": "not_available_in_stage08m_scorecard_fail_closed_for_stage10",
                "pnl_after_costs": "used",
                "risk_concentration": "used_as_ticker_abs_pnl_concentration_proxy",
                "turnover_evidence": "used_as_final_holdout_session_count_proxy",
            },
            "weights": config.objective_weights(),
        },
        "calibration_pack_id": calibration_pack_id,
        "dataset_hash": config.expected_source_manifest_sha256,
        "evidence_thresholds": {
            "dominance_share_limit": config.dominance_share_limit,
            "full_confidence_sessions": config.full_confidence_sessions,
            "min_ticker_net_pnl_after_costs_quote": (
                config.min_ticker_net_pnl_after_costs_quote
            ),
            "min_ticker_positive_ratio": config.min_ticker_positive_ratio,
            "min_ticker_sessions": config.min_ticker_sessions,
        },
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(config.generated_at_utc),
        "global_policy": {
            "blocked_ticker_action": "skip_signal",
            "global_only_threshold_activated": False,
            "reason": "Stage 10 requires ticker/market evidence before actionable calibration",
            "skipped_action_reason": "ticker_calibration_not_accepted",
        },
        "lineage": _lineage_payload(config=config, source=source),
        "market_scope": {
            "exchange": config.exchange,
            "market_type": config.market_type,
            "scope_key": f"{config.exchange}:{config.market_type}",
        },
        "model_family": "rl_platform_warm_start_v1",
        "model_version_id": config.model_version_id,
        "normalization_reference": _normalization_reference(candidate_manifest),
        "proof_boundary": "target_host_readiness_pre_main",
        "registry": {
            "calibration_pack_entity": "calibration_pack",
            "model_registry_contract_hash": registry_contract_hash_v1(),
            "stage09_model_version_id": config.model_version_id,
            "status_if_written": "accepted" if accepted_count > 0 else "rejected",
        },
        "safety": {
            "browser_auth_used": False,
            "contains_raw_provider_payloads": False,
            "contains_secrets": False,
            "exchange_side_effects": False,
            "global_only_threshold_activated": False,
            "mainnet_submit": False,
            "model_registry_promotion": False,
            "paper_testnet_live_enabled": False,
            "raw_checkpoint_tensors_embedded": False,
        },
        "schema_version": STAGE10_SCHEMA_VERSION_V1,
        "stage": "10",
        "status": "accepted" if accepted_count > 0 else "blocked",
        "ticker_calibrations": calibrated_rows,
        "ticker_count": len(calibrated_rows),
    }
    return {**body, "calibration_pack_hash": hash_json_payload_v1(body)}


def _load_and_validate_inputs(config: Stage10CalibrationConfig) -> dict[str, Any]:
    summary_sha256 = _check_file_hash(
        path=config.candidate_summary_path,
        expected_sha256=config.expected_candidate_summary_sha256,
        field="candidate_summary",
    )
    manifest_sha256 = _check_file_hash(
        path=config.candidate_manifest_path,
        expected_sha256=config.expected_candidate_manifest_sha256,
        field="candidate_manifest",
    )
    source_manifest_sha256 = _check_file_hash(
        path=config.source_manifest_path,
        expected_sha256=config.expected_source_manifest_sha256,
        field="source_manifest",
    )
    summary = _read_json(config.candidate_summary_path)
    candidate_manifest = _read_json(config.candidate_manifest_path)
    if summary.get("stage") != "08M" or summary.get("status") != "accepted":
        raise Stage10CalibrationError(reason="candidate_summary_not_accepted")
    if summary.get("stage09_allowed") is not True:
        raise Stage10CalibrationError(reason="candidate_summary_stage09_not_allowed")
    final_gate = _mapping(summary.get("final_holdout_gate"), "final_holdout_gate")
    if final_gate.get("stage09_allowed") is not True:
        raise Stage10CalibrationError(reason="final_holdout_gate_not_accepted")
    if final_gate.get("blockers") not in ([], ()):
        raise Stage10CalibrationError(reason="final_holdout_gate_has_blockers")
    candidate_artifact = _mapping(summary.get("candidate_artifact"), "candidate_artifact")
    if candidate_artifact.get("candidate_id") != config.model_version_id:
        raise Stage10CalibrationError(reason="candidate_id_mismatch")
    if candidate_artifact.get("manifest_sha256") != STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1:
        raise Stage10CalibrationError(reason="unexpected_stage09_candidate_manifest_sha256")
    if candidate_artifact.get("manifest_sha256") != manifest_sha256:
        raise Stage10CalibrationError(reason="summary_manifest_sha256_mismatch")
    if (
        candidate_manifest.get("stage") != "08M"
        or candidate_manifest.get("stage09_allowed") is not True
    ):
        raise Stage10CalibrationError(reason="candidate_manifest_not_accepted")
    if candidate_manifest.get("candidate_id") != config.model_version_id:
        raise Stage10CalibrationError(reason="candidate_manifest_id_mismatch")
    if config.model_version_id != STAGE09_ACCEPTED_CANDIDATE_ID_V1:
        raise Stage10CalibrationError(reason="unexpected_stage10_model_version_id")
    data_quality = _mapping(summary.get("data_quality"), "data_quality")
    if data_quality.get("article_manifest_sha256") != source_manifest_sha256:
        raise Stage10CalibrationError(reason="summary_source_manifest_sha256_mismatch")
    data_lineage = _mapping(candidate_manifest.get("data_lineage"), "data_lineage")
    if data_lineage.get("article_manifest_sha256") != source_manifest_sha256:
        raise Stage10CalibrationError(reason="manifest_source_manifest_sha256_mismatch")
    if candidate_manifest.get("policy_name") != STAGE09_ACCEPTED_CANDIDATE_POLICY_V1:
        raise Stage10CalibrationError(reason="unexpected_candidate_policy")
    candidate_scorecard = _candidate_scorecard(summary)
    return {
        "candidate_manifest": candidate_manifest,
        "candidate_scorecard": candidate_scorecard,
        "candidate_summary_sha256": summary_sha256,
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "summary": summary,
    }


def _candidate_scorecard(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    comparison = _mapping(summary.get("comparison"), "comparison")
    scorecards = _sequence(
        comparison.get("final_holdout_scorecards"),
        "final_holdout_scorecards",
    )
    for item in scorecards:
        row = _mapping(item, "scorecard")
        if row.get("policy_name") == STAGE09_ACCEPTED_CANDIDATE_POLICY_V1:
            return row
    raise Stage10CalibrationError(reason="candidate_scorecard_missing")


def _ticker_rows(scorecard: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = _sequence(scorecard.get("stability_by_ticker"), "stability_by_ticker")
    return [_mapping(item, "ticker_row") for item in rows]


def _calibration_row(
    *,
    config: Stage10CalibrationConfig,
    row: Mapping[str, Any],
    max_positive_pnl: float,
    round_trip_cost_ratio: float,
    total_abs_pnl: float,
) -> dict[str, object]:
    symbol = _symbol(row.get("symbol"))
    session_count = _int_value(row.get("session_count"), "session_count")
    positive_ratio = _float_value(row.get("positive_ratio"), "positive_ratio")
    net_pnl = _float_value(row.get("net_pnl_after_costs_quote"), "net_pnl_after_costs_quote")
    blockers = _ticker_blockers(
        config=config,
        net_pnl=net_pnl,
        positive_ratio=positive_ratio,
        session_count=session_count,
    )
    status: CalibrationTickerStatus = "accepted" if not blockers else "blocked"
    components = _score_components(
        config=config,
        max_positive_pnl=max_positive_pnl,
        net_pnl=net_pnl,
        positive_ratio=positive_ratio,
        session_count=session_count,
        total_abs_pnl=total_abs_pnl,
    )
    score = _weighted_score(config.objective_weights(), components)
    if status == "blocked":
        score = 0.0
    return {
        "action_thresholds": _action_thresholds(
            calibration_score=score,
            blockers=blockers,
            round_trip_cost_ratio=round_trip_cost_ratio,
        ),
        "calibration_score": _round_float(score),
        "confidence": _confidence_payload(score=score, blockers=blockers),
        "evidence": {
            "net_pnl_after_costs_quote": _round_float(net_pnl),
            "positive_ratio": _round_float(positive_ratio),
            "session_count": session_count,
        },
        "exchange": config.exchange,
        "market_type": config.market_type,
        "risk_sizing_inputs": _risk_sizing_inputs(score=score, blockers=blockers),
        "score_components": components,
        "skipped_action_reasons": _skip_reasons(blockers),
        "status": status,
        "symbol": symbol,
    }


def _ticker_blockers(
    *,
    config: Stage10CalibrationConfig,
    net_pnl: float,
    positive_ratio: float,
    session_count: int,
) -> list[str]:
    blockers: list[str] = []
    if session_count < config.min_ticker_sessions:
        blockers.append("insufficient_ticker_sessions")
    if positive_ratio < config.min_ticker_positive_ratio:
        blockers.append("ticker_positive_ratio_below_minimum")
    if net_pnl <= config.min_ticker_net_pnl_after_costs_quote:
        blockers.append("non_positive_ticker_pnl_after_costs")
    return blockers


def _score_components(
    *,
    config: Stage10CalibrationConfig,
    max_positive_pnl: float,
    net_pnl: float,
    positive_ratio: float,
    session_count: int,
    total_abs_pnl: float,
) -> dict[str, float]:
    pnl_score = min(1.0, max(0.0, net_pnl) / max_positive_pnl)
    stability_denominator = max(1e-12, 1.0 - config.min_ticker_positive_ratio)
    stability_score = min(
        1.0,
        max(0.0, positive_ratio - config.min_ticker_positive_ratio)
        / stability_denominator,
    )
    turnover_score = min(1.0, session_count / config.full_confidence_sessions)
    concentration_share = abs(net_pnl) / total_abs_pnl
    risk_score = max(0.0, 1.0 - min(1.0, concentration_share / config.dominance_share_limit))
    return {
        "drawdown": 0.0,
        "pnl_after_costs": _round_float(pnl_score),
        "positive_ratio_stability": _round_float(stability_score),
        "risk_concentration": _round_float(risk_score),
        "turnover_evidence": _round_float(turnover_score),
    }


def _weighted_score(weights: Mapping[str, float], components: Mapping[str, float]) -> float:
    return sum(float(weights[key]) * float(components[key]) for key in weights)


def _action_thresholds(
    *,
    calibration_score: float,
    blockers: Sequence[str],
    round_trip_cost_ratio: float,
) -> dict[str, object]:
    if blockers:
        return {
            "global_only_threshold_active": False,
            "minimum_confidence_to_open": 1.0,
            "minimum_edge_after_costs": None,
            "threshold_mode": "blocked_fail_closed",
        }
    minimum_confidence = 0.85 - (0.25 * calibration_score)
    minimum_edge = round_trip_cost_ratio * (1.0 + (1.0 - calibration_score))
    return {
        "global_only_threshold_active": False,
        "minimum_confidence_to_open": _round_float(minimum_confidence),
        "minimum_edge_after_costs": _round_float(minimum_edge),
        "threshold_mode": "ticker_market_calibrated",
    }


def _confidence_payload(*, score: float, blockers: Sequence[str]) -> dict[str, object]:
    if blockers:
        return {
            "confidence_multiplier": 0.0,
            "skip_action_reasons_if_not_met": ["ticker_calibration_not_accepted"],
        }
    return {
        "confidence_multiplier": _round_float(0.5 + (0.5 * score)),
        "skip_action_reasons_if_not_met": [
            "confidence_below_ticker_threshold",
            "edge_after_costs_below_ticker_minimum",
        ],
    }


def _risk_sizing_inputs(*, score: float, blockers: Sequence[str]) -> dict[str, object]:
    if blockers:
        return {
            "max_position_fraction_multiplier": 0.0,
            "risk_policy": "do_not_size_blocked_ticker",
        }
    return {
        "max_position_fraction_multiplier": _round_float(0.25 + (0.75 * score)),
        "risk_policy": "stage10_score_scaled_position_fraction_cap",
    }


def _skip_reasons(blockers: Sequence[str]) -> list[str]:
    if not blockers:
        return []
    return ["ticker_calibration_not_accepted", *sorted(blockers)]


def _lineage_payload(
    *,
    config: Stage10CalibrationConfig,
    source: Mapping[str, Any],
) -> dict[str, object]:
    summary = _mapping(source["summary"], "summary")
    candidate_manifest = _mapping(source["candidate_manifest"], "candidate_manifest")
    return {
        "candidate_manifest_path": str(config.candidate_manifest_path),
        "candidate_manifest_sha256": source["manifest_sha256"],
        "candidate_summary_path": str(config.candidate_summary_path),
        "candidate_summary_sha256": source["candidate_summary_sha256"],
        "model_state_hash": candidate_manifest.get("model_state_hash"),
        "policy_name": candidate_manifest.get("policy_name"),
        "source_manifest_path": str(config.source_manifest_path),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "stage08m_summary_hash": summary.get("summary_hash"),
    }


def _normalization_reference(candidate_manifest: Mapping[str, Any]) -> dict[str, object]:
    model_state = _mapping(candidate_manifest.get("model_state"), "model_state")
    scaler_mean = _sequence(model_state.get("scaler_mean"), "scaler_mean")
    scaler_std = _sequence(model_state.get("scaler_std"), "scaler_std")
    if len(scaler_mean) != len(scaler_std):
        raise Stage10CalibrationError(reason="normalization_stats_length_mismatch")
    return {
        "feature_count": int(model_state.get("feature_count", len(scaler_mean))),
        "model_state_hash": candidate_manifest.get("model_state_hash"),
        "raw_values_embedded": False,
        "scaler_mean_hash": hash_json_payload_v1(list(scaler_mean)),
        "scaler_std_hash": hash_json_payload_v1(list(scaler_std)),
        "source": "stage08m_candidate_manifest.model_state",
    }


def _calibration_pack_id(*, config: Stage10CalibrationConfig) -> str:
    return (
        f"stage10_{config.model_version_id}_"
        f"{config.expected_source_manifest_sha256[:8]}_per_ticker"
    )


def _check_file_hash(*, path: Path, expected_sha256: str, field: str) -> str:
    _validate_store_path(path, field)
    actual = compute_file_sha256(path)
    if actual != expected_sha256:
        raise Stage10CalibrationError(reason=f"{field}_sha256_mismatch", field=str(path))
    return actual


def _validate_artifact_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=False)
    expected = Path(RL_TRADING_ARTIFACT_ROOT_V1).resolve(strict=False)
    if root != expected:
        raise Stage10CalibrationError(reason="unexpected_artifact_root", field=str(root))
    return root


def _validate_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    root = _validate_artifact_root(Path(RL_TRADING_ARTIFACT_ROOT_V1))
    _assert_under_root(resolved, root, field="output_root")
    return resolved


def _validate_store_path(path: Path, field: str) -> Path:
    if not path.is_absolute():
        raise Stage10CalibrationError(reason="path_must_be_absolute", field=field)
    resolved = path.expanduser().resolve(strict=False)
    root = _validate_artifact_root(Path(RL_TRADING_ARTIFACT_ROOT_V1))
    _assert_under_root(resolved, root, field=field)
    if not resolved.is_file():
        raise Stage10CalibrationError(reason="file_missing", field=str(resolved))
    return resolved


def _assert_under_root(path: Path, root: Path, *, field: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Stage10CalibrationError(reason="path_outside_artifact_root", field=field) from exc


def _validate_run_id(value: str) -> None:
    _non_empty_text(value, "run_id")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if any(char not in allowed for char in value):
        raise Stage10CalibrationError(reason="invalid_run_id", field=value)


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Stage10CalibrationError(reason="invalid_sha256", field=field)


def _symbol(value: object) -> str:
    text = str(value or "")
    if not text or text.strip().upper() != text:
        raise Stage10CalibrationError(reason="symbol_must_be_uppercase", field=text)
    return text


def _row_pnl(row: Mapping[str, Any]) -> float:
    return _float_value(row.get("net_pnl_after_costs_quote"), "net_pnl_after_costs_quote")


def _int_value(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise Stage10CalibrationError(reason="expected_int", field=field)
    try:
        parsed = int(cast(int, value))
    except (TypeError, ValueError) as exc:
        raise Stage10CalibrationError(reason="expected_int", field=field) from exc
    return parsed


def _float_value(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise Stage10CalibrationError(reason="expected_float", field=field)
    try:
        parsed = float(cast(float, value))
    except (TypeError, ValueError) as exc:
        raise Stage10CalibrationError(reason="expected_float", field=field) from exc
    return parsed


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Stage10CalibrationError(reason="expected_mapping", field=field)
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise Stage10CalibrationError(reason="expected_list", field=field)
    return list(value)


def _non_empty_text(value: str, field: str) -> None:
    if not value or not value.strip():
        raise Stage10CalibrationError(reason="missing_text", field=field)


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _round_float(value: float) -> float:
    return float(round(float(value), 10))


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
