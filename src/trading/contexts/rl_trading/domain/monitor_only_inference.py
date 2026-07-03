from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

import numpy as np

from .feature_contract import (
    FEATURE_CONTRACT_HASH_V1,
    FEATURE_NAMES_V1,
    RlFeatureCandle,
    build_article_feature_vector_v1,
)
from .model_registry import (
    STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1,
    STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
)

STAGE13_SCHEMA_VERSION_V1 = 1
STAGE13_DECISION_KIND_V1 = "rl_trading_stage13_monitor_only_decision_v1"
STAGE13_PARITY_KIND_V1 = "rl_trading_stage13_train_live_feature_parity_v1"
STAGE13_SOURCE_EVENT_OUTCOME_V1 = "no_intent"
STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1 = "monitor_only_no_intent"
STAGE13_MODE_V1 = "monitor_only"
STAGE13_SOURCE_TYPE_V1 = "ml_agent_decision"
STAGE13_ALLOWED_ACTIONS_V1: tuple[str, ...] = ("hold", "open_long", "open_short")

Stage13MarketType = Literal["spot", "futures"]


class Stage13MonitorOnlyInferenceError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Stage13FeatureWindow:
    exchange: str
    market_type: Stage13MarketType
    symbol: str
    instrument_key: str
    ts_open_utc: datetime
    ts_close_utc: datetime
    candles: tuple[RlFeatureCandle, ...]
    source: str = "redis_streams_live_feed"

    def __post_init__(self) -> None:
        _validate_exchange(self.exchange)
        if self.market_type not in {"spot", "futures"}:
            raise Stage13MonitorOnlyInferenceError(
                reason="unsupported_market_type",
                field="market_type",
            )
        _validate_symbol(self.symbol)
        _non_empty_text(self.instrument_key, "instrument_key")
        if not self.candles:
            raise Stage13MonitorOnlyInferenceError(reason="feature_window_empty")
        if self.ts_open_utc.tzinfo is None or self.ts_close_utc.tzinfo is None:
            raise Stage13MonitorOnlyInferenceError(reason="window_timestamps_must_be_aware")
        if self.ts_close_utc <= self.ts_open_utc:
            raise Stage13MonitorOnlyInferenceError(reason="invalid_window_bounds")


@dataclass(frozen=True, slots=True)
class Stage13DecisionContext:
    owner_user_id: str
    strategy_id: str
    strategy_run_id: str
    exchange: str
    market_type: Stage13MarketType
    symbol: str
    instrument_key: str

    def __post_init__(self) -> None:
        for field_name in ("owner_user_id", "strategy_id", "strategy_run_id"):
            _validate_uuid_text(getattr(self, field_name), field_name)
        _validate_exchange(self.exchange)
        if self.market_type not in {"spot", "futures"}:
            raise Stage13MonitorOnlyInferenceError(
                reason="unsupported_market_type",
                field="market_type",
            )
        _validate_symbol(self.symbol)
        _non_empty_text(self.instrument_key, "instrument_key")


@dataclass(frozen=True, slots=True)
class Stage13InferenceDecision:
    decision_id: str
    model_version_id: str
    action_id: int
    action_name: str
    confidence: float
    feature_hash: str
    feature_contract_hash: str
    window_ts_close_utc: datetime
    mode: str = STAGE13_MODE_V1
    outcome: str = STAGE13_SOURCE_EVENT_OUTCOME_V1
    outcome_reason: str = STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1

    def as_payload(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "confidence": self.confidence,
            "decision_id": self.decision_id,
            "feature_contract_hash": self.feature_contract_hash,
            "feature_hash": self.feature_hash,
            "kind": STAGE13_DECISION_KIND_V1,
            "mode": self.mode,
            "model_version_id": self.model_version_id,
            "outcome": self.outcome,
            "outcome_reason": self.outcome_reason,
            "schema_version": STAGE13_SCHEMA_VERSION_V1,
            "source_type": STAGE13_SOURCE_TYPE_V1,
            "window_ts_close_utc": _format_utc(self.window_ts_close_utc),
        }


@dataclass(frozen=True, slots=True)
class Stage13SourceEventPayload:
    source_event_ref: str
    source_ref_json: Mapping[str, str]
    idempotency_key: str
    outcome: str = STAGE13_SOURCE_EVENT_OUTCOME_V1
    outcome_reason: str = STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1


@dataclass(frozen=True, slots=True)
class Stage13LatencyObservation:
    candle_close_to_feature_ready_s: float
    feature_to_decision_s: float
    decision_to_source_event_s: float

    def __post_init__(self) -> None:
        for field_name in (
            "candle_close_to_feature_ready_s",
            "feature_to_decision_s",
            "decision_to_source_event_s",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise Stage13MonitorOnlyInferenceError(
                    reason="invalid_latency_observation",
                    field=field_name,
                )


class Stage13PreloadedSupervisedPolicy:
    def __init__(
        self,
        *,
        model_version_id: str,
        model_state_hash: str,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        weights: np.ndarray,
        label_order: Mapping[int, str],
        candidate_manifest_sha256: str,
        loaded_at_utc: datetime,
    ) -> None:
        if loaded_at_utc.tzinfo is None:
            raise Stage13MonitorOnlyInferenceError(reason="loaded_at_utc_must_be_aware")
        _non_empty_text(model_version_id, "model_version_id")
        _validate_sha256(model_state_hash, "model_state_hash")
        _validate_sha256(candidate_manifest_sha256, "candidate_manifest_sha256")
        if scaler_mean.ndim != 1 or scaler_std.ndim != 1:
            raise Stage13MonitorOnlyInferenceError(reason="scaler_vectors_must_be_1d")
        if weights.ndim != 2:
            raise Stage13MonitorOnlyInferenceError(reason="weights_must_be_2d")
        if weights.shape[0] != scaler_mean.shape[0] or scaler_std.shape[0] != scaler_mean.shape[0]:
            raise Stage13MonitorOnlyInferenceError(reason="model_feature_count_mismatch")
        if weights.shape[1] < 1:
            raise Stage13MonitorOnlyInferenceError(reason="model_action_count_missing")
        if np.any(scaler_std <= 0.0):
            raise Stage13MonitorOnlyInferenceError(reason="non_positive_scaler_std")
        for action_name in label_order.values():
            if action_name not in STAGE13_ALLOWED_ACTIONS_V1:
                raise Stage13MonitorOnlyInferenceError(
                    reason="unsupported_policy_action",
                    field=action_name,
                )
        self.model_version_id = model_version_id
        self.model_state_hash = model_state_hash
        self.scaler_mean = scaler_mean.astype(np.float64, copy=True)
        self.scaler_std = scaler_std.astype(np.float64, copy=True)
        self.weights = weights.astype(np.float64, copy=True)
        self.label_order = dict(label_order)
        self.candidate_manifest_sha256 = candidate_manifest_sha256
        self.loaded_at_utc = loaded_at_utc
        self.feature_count = int(self.scaler_mean.shape[0])

    def decide(
        self,
        *,
        feature_matrix: np.ndarray,
        feature_hash: str,
        window_ts_close_utc: datetime,
    ) -> Stage13InferenceDecision:
        if window_ts_close_utc.tzinfo is None:
            raise Stage13MonitorOnlyInferenceError(reason="window_ts_close_utc_must_be_aware")
        _validate_sha256(feature_hash, "feature_hash")
        flat = np.asarray(feature_matrix, dtype=np.float64).reshape(-1)
        if flat.shape[0] != self.feature_count:
            raise Stage13MonitorOnlyInferenceError(
                reason="feature_count_mismatch",
                field=f"{flat.shape[0]} != {self.feature_count}",
            )
        normalized = (flat - self.scaler_mean) / self.scaler_std
        logits = normalized @ self.weights
        probabilities = _softmax(logits)
        label_id = int(np.argmax(probabilities))
        action_name = self.label_order.get(label_id)
        if action_name is None:
            raise Stage13MonitorOnlyInferenceError(
                reason="predicted_action_missing_from_label_order",
                field=str(label_id),
            )
        confidence = float(probabilities[label_id])
        body = {
            "action_id": label_id,
            "action_name": action_name,
            "confidence": _round_float(confidence),
            "feature_hash": feature_hash,
            "model_state_hash": self.model_state_hash,
            "model_version_id": self.model_version_id,
            "mode": STAGE13_MODE_V1,
        }
        return Stage13InferenceDecision(
            decision_id=hash_json_payload_v1(body),
            model_version_id=self.model_version_id,
            action_id=label_id,
            action_name=action_name,
            confidence=_round_float(confidence),
            feature_hash=feature_hash,
            feature_contract_hash=FEATURE_CONTRACT_HASH_V1,
            window_ts_close_utc=window_ts_close_utc.astimezone(UTC).replace(microsecond=0),
        )


def preload_stage13_policy_from_candidate_manifest_v1(
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_sha256: str,
    loaded_at_utc: datetime,
) -> Stage13PreloadedSupervisedPolicy:
    _validate_sha256(candidate_manifest_sha256, "candidate_manifest_sha256")
    if candidate_manifest.get("candidate_id") != STAGE09_ACCEPTED_CANDIDATE_ID_V1:
        raise Stage13MonitorOnlyInferenceError(reason="unexpected_stage13_candidate_id")
    if candidate_manifest_sha256 != STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1:
        raise Stage13MonitorOnlyInferenceError(
            reason="unexpected_stage13_candidate_manifest_sha256"
        )
    if (
        candidate_manifest.get("stage") != "08M"
        or candidate_manifest.get("stage09_allowed") is not True
    ):
        raise Stage13MonitorOnlyInferenceError(reason="candidate_manifest_not_accepted")
    if candidate_manifest.get("policy_name") != STAGE09_ACCEPTED_CANDIDATE_POLICY_V1:
        raise Stage13MonitorOnlyInferenceError(reason="candidate_policy_mismatch")

    model_state = _mapping(candidate_manifest.get("model_state"), "model_state")
    model_state_hash = str(candidate_manifest.get("model_state_hash", ""))
    _validate_sha256(model_state_hash, "model_state_hash")
    feature_count = _positive_int(model_state.get("feature_count"), "feature_count")
    scaler_mean = _float_vector(model_state.get("scaler_mean"), "scaler_mean")
    scaler_std = _float_vector(model_state.get("scaler_std"), "scaler_std")
    weights = _float_matrix(model_state.get("weights"), "weights")
    if len(scaler_mean) != feature_count or len(scaler_std) != feature_count:
        raise Stage13MonitorOnlyInferenceError(reason="model_state_feature_count_mismatch")
    label_order = {
        int(key): str(value)
        for key, value in _mapping(model_state.get("label_order"), "label_order").items()
    }
    return Stage13PreloadedSupervisedPolicy(
        model_version_id=str(candidate_manifest["candidate_id"]),
        model_state_hash=model_state_hash,
        scaler_mean=np.asarray(scaler_mean, dtype=np.float64),
        scaler_std=np.asarray(scaler_std, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        label_order=label_order,
        candidate_manifest_sha256=candidate_manifest_sha256,
        loaded_at_utc=loaded_at_utc,
    )


def feature_window_from_redis_payloads_v1(
    *,
    payloads: Sequence[Mapping[str, object]],
    exchange: str,
    market_type: Stage13MarketType,
    symbol: str,
    instrument_key: str,
) -> Stage13FeatureWindow:
    if not payloads:
        raise Stage13MonitorOnlyInferenceError(reason="redis_feature_window_empty")
    sorted_payloads = sorted(payloads, key=lambda item: _parse_utc(str(item["ts_open"])))
    candles = tuple(redis_payload_to_feature_candle_v1(payload) for payload in sorted_payloads)
    observed_keys = {
        str(payload.get("instrument_key", "")).strip()
        for payload in sorted_payloads
        if str(payload.get("instrument_key", "")).strip()
    }
    if observed_keys and observed_keys != {instrument_key}:
        raise Stage13MonitorOnlyInferenceError(
            reason="redis_window_instrument_mismatch",
            field="instrument_key",
        )
    return Stage13FeatureWindow(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        instrument_key=instrument_key,
        ts_open_utc=_parse_utc(str(sorted_payloads[0]["ts_open"])),
        ts_close_utc=_parse_utc(str(sorted_payloads[-1]["ts_close"])),
        candles=candles,
    )


def offline_feature_window_from_candles_v1(
    *,
    candles: Sequence[RlFeatureCandle],
    exchange: str,
    market_type: Stage13MarketType,
    symbol: str,
    instrument_key: str,
    ts_open_utc: datetime,
    ts_close_utc: datetime,
) -> Stage13FeatureWindow:
    return Stage13FeatureWindow(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        instrument_key=instrument_key,
        ts_open_utc=ts_open_utc,
        ts_close_utc=ts_close_utc,
        candles=tuple(candles),
        source="offline_dataset_fixture",
    )


def redis_payload_to_feature_candle_v1(payload: Mapping[str, object]) -> RlFeatureCandle:
    if str(payload.get("schema_version", "")).strip() != "1":
        raise Stage13MonitorOnlyInferenceError(reason="unsupported_redis_schema_version")
    return RlFeatureCandle(
        open=_float_field(payload, "open"),
        high=_float_field(payload, "high"),
        low=_float_field(payload, "low"),
        close=_float_field(payload, "close"),
        volume_base=_float_field(payload, "volume_base"),
        volume_quote=_optional_float_field(payload, "volume_quote"),
        trades_count=_optional_int_field(payload, "trades_count"),
    )


def build_stage13_feature_matrix_v1(window: Stage13FeatureWindow) -> tuple[np.ndarray, str]:
    rows = [build_article_feature_vector_v1(candle) for candle in window.candles]
    matrix = np.asarray(rows, dtype=np.float32)
    payload = {
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "instrument_key": window.instrument_key,
        "rows": [[_round_float(float(value)) for value in row] for row in matrix.tolist()],
        "schema_version": STAGE13_SCHEMA_VERSION_V1,
    }
    return matrix, hash_json_payload_v1(payload)


def compare_stage13_train_live_feature_parity_v1(
    *,
    live_window: Stage13FeatureWindow,
    offline_window: Stage13FeatureWindow,
) -> dict[str, object]:
    live_matrix, live_hash = build_stage13_feature_matrix_v1(live_window)
    offline_matrix, offline_hash = build_stage13_feature_matrix_v1(offline_window)
    if live_matrix.shape != offline_matrix.shape:
        return {
            "kind": STAGE13_PARITY_KIND_V1,
            "max_abs_diff": None,
            "offline_feature_hash": offline_hash,
            "live_feature_hash": live_hash,
            "reason": "feature_shape_mismatch",
            "status": "blocked",
        }
    max_abs_diff = float(np.max(np.abs(live_matrix - offline_matrix))) if live_matrix.size else 0.0
    status = "accepted" if max_abs_diff == 0.0 else "blocked"
    return {
        "kind": STAGE13_PARITY_KIND_V1,
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "live_feature_hash": live_hash,
        "max_abs_diff": max_abs_diff,
        "offline_feature_hash": offline_hash,
        "reason": "features_identical" if status == "accepted" else "feature_values_drift",
        "schema_version": STAGE13_SCHEMA_VERSION_V1,
        "status": status,
    }


def build_stage13_source_event_payload_v1(
    *,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> Stage13SourceEventPayload:
    source_ref_json = {
        "action": decision.action_name,
        "action_id": str(decision.action_id),
        "exchange": context.exchange,
        "feature_hash": decision.feature_hash,
        "instrument_key": context.instrument_key,
        "market_type": context.market_type,
        "mode": STAGE13_MODE_V1,
        "model_version_id": decision.model_version_id,
        "strategy_id": context.strategy_id,
        "strategy_run_id": context.strategy_run_id,
        "symbol": context.symbol,
    }
    return Stage13SourceEventPayload(
        source_event_ref=f"rl:{decision.decision_id}",
        source_ref_json=source_ref_json,
        idempotency_key="|".join(
            (
                STAGE13_SOURCE_TYPE_V1,
                context.strategy_id,
                context.strategy_run_id,
                context.instrument_key,
                decision.feature_hash,
                decision.model_version_id,
                STAGE13_MODE_V1,
            )
        ),
    )


def summarize_stage13_latency_observations_v1(
    observations: Sequence[Stage13LatencyObservation],
) -> dict[str, object]:
    if not observations:
        raise Stage13MonitorOnlyInferenceError(reason="latency_observations_empty")
    segments = {
        "candle_close_to_feature_ready": [
            item.candle_close_to_feature_ready_s for item in observations
        ],
        "feature_to_decision": [item.feature_to_decision_s for item in observations],
        "decision_to_source_event": [item.decision_to_source_event_s for item in observations],
    }
    return {
        "observations": len(observations),
        "p95_seconds": {name: _round_float(_p95(values)) for name, values in segments.items()},
        "schema_version": STAGE13_SCHEMA_VERSION_V1,
    }


def hash_json_payload_v1(payload: Mapping[str, object]) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = np.sum(exp)
    if not math.isfinite(float(total)) or total <= 0.0:
        raise Stage13MonitorOnlyInferenceError(reason="invalid_policy_logits")
    return exp / total


def _p95(values: Sequence[float]) -> float:
    sorted_values = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return sorted_values[index]


def _float_field(payload: Mapping[str, object], field: str) -> float:
    raw = payload.get(field)
    if raw is None or str(raw).strip() == "":
        raise Stage13MonitorOnlyInferenceError(reason="redis_field_required", field=field)
    value = float(str(raw))
    if not math.isfinite(value):
        raise Stage13MonitorOnlyInferenceError(reason="redis_field_non_finite", field=field)
    return value


def _optional_float_field(payload: Mapping[str, object], field: str) -> float | None:
    raw = payload.get(field)
    if raw is None or str(raw).strip() == "":
        return None
    value = float(str(raw))
    if not math.isfinite(value):
        raise Stage13MonitorOnlyInferenceError(reason="redis_field_non_finite", field=field)
    return value


def _optional_int_field(payload: Mapping[str, object], field: str) -> int | None:
    raw = payload.get(field)
    if raw is None or str(raw).strip() == "":
        return None
    return int(str(raw))


def _float_vector(value: object, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise Stage13MonitorOnlyInferenceError(reason="expected_float_vector", field=field)
    return [float(item) for item in value]


def _float_matrix(value: object, field: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise Stage13MonitorOnlyInferenceError(reason="expected_float_matrix", field=field)
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, str | bytes | bytearray):
            raise Stage13MonitorOnlyInferenceError(reason="expected_float_matrix", field=field)
        matrix.append([float(item) for item in row])
    return matrix


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage13MonitorOnlyInferenceError(reason="expected_mapping", field=field)
    return cast(Mapping[str, Any], value)


def _positive_int(value: object, field: str) -> int:
    parsed = int(str(value))
    if parsed <= 0:
        raise Stage13MonitorOnlyInferenceError(reason="positive_int_required", field=field)
    return parsed


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Stage13MonitorOnlyInferenceError(reason="invalid_sha256", field=field)


def _validate_uuid_text(value: str, field: str) -> None:
    import uuid

    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise Stage13MonitorOnlyInferenceError(reason="invalid_uuid", field=field) from exc


def _validate_exchange(value: str) -> None:
    if value.strip().lower() != value or not value:
        raise Stage13MonitorOnlyInferenceError(reason="exchange_must_be_lowercase")


def _validate_symbol(value: str) -> None:
    if value.strip().upper() != value or not value:
        raise Stage13MonitorOnlyInferenceError(reason="symbol_must_be_uppercase")


def _non_empty_text(value: str, field: str) -> None:
    if not value.strip():
        raise Stage13MonitorOnlyInferenceError(reason="missing_text", field=field)


def _parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise Stage13MonitorOnlyInferenceError(reason="timestamp_must_be_timezone_aware")
    return parsed.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round_float(value: float) -> float:
    return round(float(value), 12)
