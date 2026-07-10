from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .action_state_reward_contract import ACTION_NAMES_BY_ID_V1
from .feature_contract import FEATURE_CONTRACT_HASH_V1, FEATURE_NAMES_V1, RlFeatureCandle
from .raw_feature_dataset import hash_json_payload_v1
from .sessionized_dataset import (
    SESSIONIZED_ARTICLE_POLICY_ID_V1,
    article_session_extraction_policy_v1,
)
from .upstream_methodology import (
    UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
    FilteredBacktestPolicy,
    NormalizationStats,
    TorchD3qnPerAgent,
    UpstreamAlphaConfig,
    build_upstream_entry_state_from_history_v1,
    mask_upstream_training_action_v1,
)

STAGE08K_MONITOR_POLICY_ID_V1 = "stage08k_long_only_hold_1m_monitor_v1"
STAGE08K_MONITOR_MODEL_VERSION_V1 = "stage08k_roehub_native_best_3e033951"
STAGE08K_MONITOR_REQUIRED_CANDIDATE_SHA256_V1 = (
    "03fd26aa9cbf3ee4d4d3f50e62301408dccfa443e10a2cf9875014b064b444cc"
)
STAGE08K_MONITOR_REQUIRED_EVALUATION_SHA256_V1 = (
    "c16a08dffa6f49fd8f24d9a1af6737114d54363b4cca3630a4e83c2661880b07"
)
STAGE08K_MONITOR_REQUIRED_CHECKPOINT_SHA256_V1 = (
    "3e0339514d808a34a20d36a3e7e4035c5e722097046c2fc817bb5a4b93a03199"
)
STAGE08K_MONITOR_REQUIRED_NORMALIZATION_FILE_SHA256_V1 = (
    "e3f787c3cbecb9d39d5a87399a235d4f3a7efdb4625ec8ff8cd437af344c20fd"
)

Stage08kMonitorAction = Literal["hold", "open_long"]


class Stage08kMonitorPolicyError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        super().__init__(reason if field is None else f"{reason}: {field}")


@dataclass(frozen=True, slots=True)
class Stage08kArtifactContract:
    artifact_root: Path
    candidate_manifest_path: Path
    candidate_manifest_sha256: str
    evaluation_manifest_path: Path
    evaluation_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    normalization_stats_path: Path
    normalization_stats_file_sha256: str


@dataclass(frozen=True, slots=True)
class Stage08kMonitorPolicyConfig:
    policy_id: str = STAGE08K_MONITOR_POLICY_ID_V1
    direction_policy: str = "long_only"
    virtual_hold_minutes: int = 1
    taker_fee_rate: float = 0.0005
    slippage_rate: float = 0.00025
    virtual_notional_quote: float = 5_000.0
    funding_model: str = "not_modeled_for_1m_monitor"

    def __post_init__(self) -> None:
        if self.policy_id != STAGE08K_MONITOR_POLICY_ID_V1:
            raise Stage08kMonitorPolicyError(reason="unexpected_monitor_policy_id")
        if self.direction_policy != "long_only":
            raise Stage08kMonitorPolicyError(reason="monitor_direction_must_be_long_only")
        if self.virtual_hold_minutes != 1:
            raise Stage08kMonitorPolicyError(reason="monitor_hold_must_be_one_minute")
        for name in ("taker_fee_rate", "slippage_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise Stage08kMonitorPolicyError(reason="invalid_monitor_cost", field=name)
        if not math.isfinite(self.virtual_notional_quote) or self.virtual_notional_quote <= 0:
            raise Stage08kMonitorPolicyError(reason="invalid_virtual_notional_quote")

    def policy_hash(self) -> str:
        return hash_json_payload_v1(
            {
                "direction_policy": self.direction_policy,
                "funding_model": self.funding_model,
                "policy_id": self.policy_id,
                "slippage_rate": self.slippage_rate,
                "taker_fee_rate": self.taker_fee_rate,
                "virtual_hold_minutes": self.virtual_hold_minutes,
                "virtual_notional_quote": self.virtual_notional_quote,
            }
        )


@dataclass(frozen=True, slots=True)
class Stage08kArticleSignal:
    eligible: bool
    event_return: float
    volatility_score: float
    contrast_max_abs_return: float
    reason: str


@dataclass(frozen=True, slots=True)
class Stage08kMonitorDecision:
    requested_action_id: int
    requested_action_name: str
    action_id: int
    action_name: Stage08kMonitorAction
    confidence: float
    q_values: tuple[float, ...]
    feature_hash: str
    policy_reason: str
    signal: Stage08kArticleSignal


class Stage08kPreloadedMonitorPolicy:
    def __init__(
        self,
        *,
        agent: TorchD3qnPerAgent,
        alpha: UpstreamAlphaConfig,
        normalization_stats: NormalizationStats,
        policy_config: Stage08kMonitorPolicyConfig,
    ) -> None:
        self.agent = agent
        self.alpha = alpha
        self.normalization_stats = normalization_stats
        self.policy_config = policy_config
        self.model_version_id = STAGE08K_MONITOR_MODEL_VERSION_V1
        self._filter = FilteredBacktestPolicy.from_config(
            alpha,
            selection_strategy="advantage_based_filter",
        )

    def decide(self, candles: Sequence[RlFeatureCandle]) -> Stage08kMonitorDecision:
        signal = score_stage08k_live_signal_v1(candles)
        if not signal.eligible:
            return Stage08kMonitorDecision(
                requested_action_id=0,
                requested_action_name="hold",
                action_id=0,
                action_name="hold",
                confidence=0.0,
                q_values=(0.0, 0.0, 0.0, 0.0),
                feature_hash=_monitor_feature_hash(candles, signal=signal),
                policy_reason=signal.reason,
                signal=signal,
            )
        history = _feature_matrix(candles[-self.alpha.pre_signal_len :])
        state = build_upstream_entry_state_from_history_v1(
            history=history,
            normalization_stats=self.normalization_stats,
            config=self.alpha,
        )
        q_values = self.agent.predict_q_values(state)
        filtered = self._filter.select_from_q_values(q_values)
        masked_action = mask_upstream_training_action_v1(
            action_id=filtered.effective_action_id,
            position_side=None,
            is_last_step=False,
        )
        requested_name = ACTION_NAMES_BY_ID_V1[filtered.requested_action_id]
        policy_reason = filtered.rejection_reason or "model_action_allowed"
        effective_action = masked_action
        if masked_action == 2:
            effective_action = 0
            policy_reason = "short_blocked_by_monitor_policy"
        elif masked_action not in {0, 1}:
            effective_action = 0
            policy_reason = "invalid_entry_action_masked_to_hold"
        action_name = cast(Stage08kMonitorAction, ACTION_NAMES_BY_ID_V1[effective_action])
        return Stage08kMonitorDecision(
            requested_action_id=filtered.requested_action_id,
            requested_action_name=requested_name,
            action_id=effective_action,
            action_name=action_name,
            confidence=filtered.confidence,
            q_values=tuple(float(value) for value in q_values),
            feature_hash=hash_json_payload_v1(
                {
                    "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
                    "policy_hash": self.policy_config.policy_hash(),
                    "signal": {
                        "contrast_max_abs_return": signal.contrast_max_abs_return,
                        "event_return": signal.event_return,
                        "volatility_score": signal.volatility_score,
                    },
                    "state": [round(float(value), 10) for value in state.tolist()],
                }
            ),
            policy_reason=policy_reason,
            signal=signal,
        )


def preload_stage08k_monitor_policy_v1(
    *,
    artifacts: Stage08kArtifactContract,
    policy_config: Stage08kMonitorPolicyConfig,
    torch_num_threads: int,
    torch_num_interop_threads: int,
) -> Stage08kPreloadedMonitorPolicy:
    required_hashes = {
        "candidate_manifest": (
            artifacts.candidate_manifest_sha256,
            STAGE08K_MONITOR_REQUIRED_CANDIDATE_SHA256_V1,
        ),
        "evaluation_manifest": (
            artifacts.evaluation_manifest_sha256,
            STAGE08K_MONITOR_REQUIRED_EVALUATION_SHA256_V1,
        ),
        "checkpoint": (
            artifacts.checkpoint_sha256,
            STAGE08K_MONITOR_REQUIRED_CHECKPOINT_SHA256_V1,
        ),
        "normalization_stats": (
            artifacts.normalization_stats_file_sha256,
            STAGE08K_MONITOR_REQUIRED_NORMALIZATION_FILE_SHA256_V1,
        ),
    }
    for field_name, (configured, required) in required_hashes.items():
        if configured != required:
            raise Stage08kMonitorPolicyError(
                reason="unexpected_stage08k_artifact_sha256",
                field=field_name,
            )
    root = artifacts.artifact_root.expanduser().resolve(strict=False)
    if root != Path("/opt/roehub/state/rl_trading"):
        raise Stage08kMonitorPolicyError(reason="unexpected_artifact_root", field=str(root))
    candidate = _load_trusted_json(
        artifacts.candidate_manifest_path,
        artifacts.candidate_manifest_sha256,
        root=root,
        field="candidate_manifest",
    )
    evaluation = _load_trusted_json(
        artifacts.evaluation_manifest_path,
        artifacts.evaluation_manifest_sha256,
        root=root,
        field="evaluation_manifest",
    )
    normalization = _load_trusted_json(
        artifacts.normalization_stats_path,
        artifacts.normalization_stats_file_sha256,
        root=root,
        field="normalization_stats",
    )
    _validate_required_lineage(candidate=candidate, evaluation=evaluation)
    _validate_candidate_artifact_paths(candidate=candidate, artifacts=artifacts)
    alpha_payload = _mapping(
        _mapping(evaluation.get("config"), "config").get("alpha_config"),
        "alpha_config",
    )
    alpha = _alpha_config_from_payload(
        alpha_payload,
        torch_num_threads=torch_num_threads,
        torch_num_interop_threads=torch_num_interop_threads,
    )
    stats = _normalization_stats_from_payload(normalization)
    if stats.stats_hash() != str(evaluation.get("normalization_stats_hash", "")):
        raise Stage08kMonitorPolicyError(reason="normalization_stats_hash_mismatch")
    checkpoint_path = _trusted_path(
        artifacts.checkpoint_path,
        expected_sha256=artifacts.checkpoint_sha256,
        root=root,
        field="checkpoint",
    )
    agent = TorchD3qnPerAgent(config=alpha, device_policy="cpu_only_deterministic")
    payload = agent.torch.load(
        str(checkpoint_path),
        map_location=agent.device,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise Stage08kMonitorPolicyError(reason="checkpoint_payload_invalid")
    if payload.get("architecture_id") != UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1:
        raise Stage08kMonitorPolicyError(reason="checkpoint_architecture_mismatch")
    if payload.get("stage") != "08E":
        raise Stage08kMonitorPolicyError(reason="checkpoint_stage_mismatch")
    if payload.get("config_hash") != candidate.get("config_hash"):
        raise Stage08kMonitorPolicyError(reason="checkpoint_config_hash_mismatch")
    policy_state = payload.get("policy_state")
    target_state = payload.get("target_state")
    if not isinstance(policy_state, Mapping) or not isinstance(target_state, Mapping):
        raise Stage08kMonitorPolicyError(reason="checkpoint_model_state_missing")
    agent.policy_net.load_state_dict(policy_state)
    agent.target_net.load_state_dict(target_state)
    agent.policy_net.eval()
    agent.target_net.eval()
    return Stage08kPreloadedMonitorPolicy(
        agent=agent,
        alpha=alpha,
        normalization_stats=stats,
        policy_config=policy_config,
    )


def score_stage08k_live_signal_v1(
    candles: Sequence[RlFeatureCandle],
) -> Stage08kArticleSignal:
    policy = article_session_extraction_policy_v1()
    if len(candles) < policy.pre_signal_len:
        return Stage08kArticleSignal(False, 0.0, 0.0, 0.0, "insufficient_history")
    history = _feature_matrix(candles[-policy.pre_signal_len :])
    open_ = history[:, FEATURE_NAMES_V1.index("open")].astype(np.float64)
    close = history[:, FEATURE_NAMES_V1.index("close")].astype(np.float64)
    event_len = policy.article_event_window_minutes
    event_return = float((close[-1] / open_[-event_len]) - 1.0)
    volatility_score = abs(event_return)
    previous: list[float] = []
    current_start = history.shape[0] - event_len
    for start in range(current_start):
        end = start + event_len - 1
        previous.append(abs(float((close[end] / open_[start]) - 1.0)))
    contrast_lookback = policy.article_contrast_window_minutes - event_len
    contrast_max = max(previous[-contrast_lookback:], default=0.0)
    eligible = (
        volatility_score >= policy.article_event_move_threshold
        and contrast_max < policy.article_similar_impulse_threshold
    )
    reason = "article_signal_eligible" if eligible else (
        "event_move_below_threshold"
        if volatility_score < policy.article_event_move_threshold
        else "similar_prior_impulse_in_contrast_window"
    )
    return Stage08kArticleSignal(
        eligible=eligible,
        event_return=event_return,
        volatility_score=volatility_score,
        contrast_max_abs_return=contrast_max,
        reason=reason,
    )


def stage08k_monitor_contract_payload_v1(
    *, policy_config: Stage08kMonitorPolicyConfig,
) -> dict[str, object]:
    return {
        "candidate_manifest_sha256": STAGE08K_MONITOR_REQUIRED_CANDIDATE_SHA256_V1,
        "checkpoint_sha256": STAGE08K_MONITOR_REQUIRED_CHECKPOINT_SHA256_V1,
        "evaluation_manifest_sha256": STAGE08K_MONITOR_REQUIRED_EVALUATION_SHA256_V1,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "model_version_id": STAGE08K_MONITOR_MODEL_VERSION_V1,
        "normalization_file_sha256": STAGE08K_MONITOR_REQUIRED_NORMALIZATION_FILE_SHA256_V1,
        "policy_hash": policy_config.policy_hash(),
        "policy_id": policy_config.policy_id,
        "selector_id": SESSIONIZED_ARTICLE_POLICY_ID_V1,
        "status": "research_monitor_only",
    }


def _feature_matrix(candles: Sequence[RlFeatureCandle]) -> np.ndarray:
    from .feature_contract import build_article_feature_vector_v1

    matrix = np.asarray(
        [build_article_feature_vector_v1(candle) for candle in candles],
        dtype=np.float32,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES_V1):
        raise Stage08kMonitorPolicyError(reason="feature_matrix_shape_mismatch")
    if not np.all(np.isfinite(matrix)):
        raise Stage08kMonitorPolicyError(reason="non_finite_feature_matrix")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _monitor_feature_hash(
    candles: Sequence[RlFeatureCandle], *, signal: Stage08kArticleSignal
) -> str:
    return hash_json_payload_v1(
        {
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
            "row_count": len(candles),
            "signal": {
                "contrast_max_abs_return": signal.contrast_max_abs_return,
                "event_return": signal.event_return,
                "volatility_score": signal.volatility_score,
            },
        }
    )


def _load_trusted_json(
    path: Path,
    expected_sha256: str,
    *,
    root: Path,
    field: str,
) -> Mapping[str, Any]:
    resolved = _trusted_path(path, expected_sha256=expected_sha256, root=root, field=field)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise Stage08kMonitorPolicyError(reason="artifact_payload_not_mapping", field=field)
    return cast(Mapping[str, Any], payload)


def _trusted_path(path: Path, *, expected_sha256: str, root: Path, field: str) -> Path:
    if len(expected_sha256) != 64:
        raise Stage08kMonitorPolicyError(reason="invalid_expected_sha256", field=field)
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Stage08kMonitorPolicyError(reason="artifact_outside_root", field=field) from exc
    if not resolved.is_file():
        raise Stage08kMonitorPolicyError(reason="artifact_missing", field=str(resolved))
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise Stage08kMonitorPolicyError(reason="artifact_sha256_mismatch", field=field)
    return resolved


def _validate_required_lineage(
    *, candidate: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> None:
    if candidate.get("stage") != "08E" or candidate.get("status") != "completed":
        raise Stage08kMonitorPolicyError(reason="candidate_lineage_not_completed")
    if candidate.get("architecture_id") != UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1:
        raise Stage08kMonitorPolicyError(reason="candidate_architecture_mismatch")
    if (
        evaluation.get("stage") != "08F"
        or evaluation.get("status") != "accepted_for_research"
        or evaluation.get("research_candidate_save_allowed") is not True
    ):
        raise Stage08kMonitorPolicyError(reason="evaluation_research_lineage_mismatch")
    dependency = _mapping(evaluation.get("candidate_dependency"), "candidate_dependency")
    if dependency.get("manifest_sha256") != STAGE08K_MONITOR_REQUIRED_CANDIDATE_SHA256_V1:
        raise Stage08kMonitorPolicyError(reason="candidate_dependency_sha256_mismatch")
    if evaluation.get("feature_contract_hash") != FEATURE_CONTRACT_HASH_V1:
        raise Stage08kMonitorPolicyError(reason="feature_contract_hash_mismatch")


def _validate_candidate_artifact_paths(
    *, candidate: Mapping[str, Any], artifacts: Stage08kArtifactContract
) -> None:
    artifact_hashes = _mapping(candidate.get("artifact_hashes"), "artifact_hashes")
    expected = {
        "best_checkpoint": (artifacts.checkpoint_path, artifacts.checkpoint_sha256),
        "normalization_stats": (
            artifacts.normalization_stats_path,
            artifacts.normalization_stats_file_sha256,
        ),
    }
    for name, (path, sha256) in expected.items():
        row = _mapping(artifact_hashes.get(name), f"artifact_hashes.{name}")
        if str(row.get("path", "")) != str(path) or str(row.get("sha256", "")) != sha256:
            raise Stage08kMonitorPolicyError(
                reason="candidate_artifact_lineage_mismatch",
                field=name,
            )
    checkpoint_policy = _mapping(candidate.get("checkpoint_policy"), "checkpoint_policy")
    if checkpoint_policy.get("default_evaluation_checkpoint") != "best":
        raise Stage08kMonitorPolicyError(reason="candidate_default_checkpoint_not_best")


def _alpha_config_from_payload(
    payload: Mapping[str, Any], *, torch_num_threads: int, torch_num_interop_threads: int
) -> UpstreamAlphaConfig:
    allowed = {item.name for item in fields(UpstreamAlphaConfig)}
    ignored_derived = {
        "architecture_id",
        "cnn_input_shape",
        "config_id",
        "feature_contract_hash",
        "feature_names",
        "input_history_len",
        "methodology_parity_id",
        "state_dim",
        "upstream_source_sha",
    }
    unknown = set(payload) - allowed - ignored_derived
    if unknown:
        raise Stage08kMonitorPolicyError(
            reason="unknown_alpha_config_fields", field=",".join(sorted(unknown))
        )
    values = {key: payload[key] for key in allowed if key in payload}
    for key in ("cnn_maps", "cnn_kernels", "cnn_strides", "dense_val", "dense_adv"):
        if key in values:
            values[key] = tuple(int(value) for value in values[key])
    values["torch_num_threads"] = torch_num_threads
    values["torch_num_interop_threads"] = torch_num_interop_threads
    return UpstreamAlphaConfig(**values)


def _normalization_stats_from_payload(payload: Mapping[str, Any]) -> NormalizationStats:
    means = _mapping(payload.get("means"), "means")
    stds = _mapping(payload.get("stds"), "stds")
    return NormalizationStats(
        means={str(key): float(value) for key, value in means.items()},
        stds={str(key): float(value) for key, value in stds.items()},
        source_split=str(payload.get("source_split", "")),
        sequence_count=int(payload.get("sequence_count", 0)),
        feature_names=tuple(str(value) for value in payload.get("feature_names", ())),
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage08kMonitorPolicyError(reason="artifact_mapping_required", field=field)
    return cast(Mapping[str, Any], value)
