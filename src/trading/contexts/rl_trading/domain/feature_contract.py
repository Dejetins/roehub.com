from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

FEATURE_CONTRACT_ID_V1 = "rl_trading.article_compatible.binance_futures"
FEATURE_CONTRACT_VERSION_V1 = 1
FEATURE_DTYPE_V1 = "float32"
FEATURE_NAMES_V1: tuple[str, ...] = (
    "open",
    "high",
    "volume_weighted_average",
    "low",
    "close",
    "volume",
    "num_trades",
)

TrainingSourceStatus = Literal[
    "trainable",
    "blocked",
    "research_only_approximation",
    "blocked_not_training_source_v1",
]
MetadataRequirementStatus = Literal[
    "available_current_snapshot_only",
    "missing_required_source",
    "assumption_required",
]


class FeatureContractViolation(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FeatureChannel:
    name: str
    source: str
    required: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required": self.required,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TrainingSourceBranch:
    exchange: str
    market_type: str
    status: TrainingSourceStatus
    reason: str

    def as_payload(self) -> dict[str, str]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "reason": self.reason,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class FuturesMetadataRequirement:
    name: str
    status: MetadataRequirementStatus
    gate_behavior: str

    def as_payload(self) -> dict[str, str]:
        return {
            "gate_behavior": self.gate_behavior,
            "name": self.name,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class RlFeatureCandle:
    open: float
    high: float
    low: float
    close: float
    volume_base: float
    volume_quote: float | None
    trades_count: int | None


_FEATURE_CHANNELS_V1: tuple[FeatureChannel, ...] = (
    FeatureChannel(name="open", source="canonical_candles_1m.open", required=True),
    FeatureChannel(name="high", source="canonical_candles_1m.high", required=True),
    FeatureChannel(
        name="volume_weighted_average",
        source="volume_quote / volume_base; close when both volumes are zero",
        required=True,
    ),
    FeatureChannel(name="low", source="canonical_candles_1m.low", required=True),
    FeatureChannel(name="close", source="canonical_candles_1m.close", required=True),
    FeatureChannel(name="volume", source="canonical_candles_1m.volume_base", required=True),
    FeatureChannel(name="num_trades", source="canonical_candles_1m.trades_count", required=True),
)

_TRAINING_SOURCE_MATRIX_V1: tuple[TrainingSourceBranch, ...] = (
    TrainingSourceBranch(
        exchange="binance",
        market_type="futures",
        status="trainable",
        reason=(
            "only v1 training branch; article-compatible candle fields are available, "
            "while futures metadata gate still blocks production-grade evaluation/activation"
        ),
    ),
    TrainingSourceBranch(
        exchange="binance",
        market_type="spot",
        status="blocked_not_training_source_v1",
        reason="spot branch is product/execution inventory only for this cycle",
    ),
    TrainingSourceBranch(
        exchange="bybit",
        market_type="spot",
        status="blocked_not_training_source_v1",
        reason="not a v1 training source; no Bybit trades_count enrich or feature-mask branch",
    ),
    TrainingSourceBranch(
        exchange="bybit",
        market_type="futures",
        status="blocked_not_training_source_v1",
        reason="not a v1 training source; no Bybit trades_count enrich or feature-mask branch",
    ),
)

_FUTURES_METADATA_REQUIREMENTS_V1: tuple[FuturesMetadataRequirement, ...] = (
    FuturesMetadataRequirement(
        name="funding_rate_history",
        status="missing_required_source",
        gate_behavior=(
            "block production-grade futures backtest/evaluation until sourced or "
            "explicitly approximated"
        ),
    ),
    FuturesMetadataRequirement(
        name="mark_price_history",
        status="missing_required_source",
        gate_behavior=(
            "block liquidation/mark-to-market claims until sourced or explicitly approximated"
        ),
    ),
    FuturesMetadataRequirement(
        name="index_price_history",
        status="missing_required_source",
        gate_behavior="block basis/mark-index assumptions until sourced or explicitly approximated",
    ),
    FuturesMetadataRequirement(
        name="point_in_time_filters",
        status="available_current_snapshot_only",
        gate_behavior=(
            "current ref_instruments filters are not enough for historical "
            "survivorship-bias proof"
        ),
    ),
    FuturesMetadataRequirement(
        name="leverage_tiers",
        status="missing_required_source",
        gate_behavior=(
            "block leverage/liquidation-sensitive evaluation until sourced or explicitly "
            "approximated"
        ),
    ),
    FuturesMetadataRequirement(
        name="fee_policy",
        status="assumption_required",
        gate_behavior=(
            "Stage 08 scorecard must declare maker/taker fee assumptions before "
            "candidate acceptance"
        ),
    ),
    FuturesMetadataRequirement(
        name="slippage_policy",
        status="assumption_required",
        gate_behavior=(
            "Stage 08 scorecard must declare slippage assumptions before candidate "
            "acceptance"
        ),
    ),
    FuturesMetadataRequirement(
        name="liquidation_policy",
        status="assumption_required",
        gate_behavior=(
            "Stage 08 scorecard must declare liquidation assumptions before candidate "
            "acceptance"
        ),
    ),
)


def feature_contract_canonical_payload_v1() -> dict[str, object]:
    return {
        "contract_id": FEATURE_CONTRACT_ID_V1,
        "feature_dtype": FEATURE_DTYPE_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "feature_schema_version": FEATURE_CONTRACT_VERSION_V1,
        "features": [channel.as_payload() for channel in _FEATURE_CHANNELS_V1],
        "live_feed_policy": {
            "hot_path_clickhouse_scan": "forbidden",
            "missing_trades_count_behavior": "fail_closed_degraded_or_blocked",
            "redis_required_for_rl": ["volume_quote", "trades_count"],
            "repair_scope": "gap_or_degraded_path_only",
        },
        "missing_field_policy": {
            "required_fields": [
                "open",
                "high",
                "low",
                "close",
                "volume_base",
                "volume_quote",
                "trades_count",
            ],
            "behavior": "fail_closed",
            "no_feature_mask_training_branch": True,
        },
        "normalization": {
            "fit_scope": "train_split_only_per_exchange_market_symbol",
            "inputs": ["mean", "std"],
            "kind": "z_score",
            "live_behavior": "use_accepted_training_stats_manifest_without_hot_path_refit",
            "min_std": "1e-12",
        },
        "source": {
            "exchange": "binance",
            "market_type": "futures",
            "timeframe": "1m_closed_candle",
        },
        "vwap_policy": {
            "formula": "volume_quote / volume_base when volume_base > 0",
            "missing_volume_quote": "fail_closed",
            "zero_base_zero_quote": "use_close",
            "zero_base_positive_quote": "fail_closed",
        },
    }


def feature_contract_canonical_json_v1() -> str:
    return json.dumps(
        feature_contract_canonical_payload_v1(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def feature_contract_hash_v1() -> str:
    return hashlib.sha256(feature_contract_canonical_json_v1().encode("utf-8")).hexdigest()


FEATURE_CONTRACT_HASH_V1 = feature_contract_hash_v1()


def training_source_matrix_payload_v1() -> tuple[dict[str, str], ...]:
    return tuple(branch.as_payload() for branch in _TRAINING_SOURCE_MATRIX_V1)


def futures_metadata_gate_payload_v1() -> dict[str, object]:
    return {
        "activation_behavior": "fail_closed_for_production_grade_futures_evaluation_until_resolved",
        "gate_id": "binance_futures_metadata_gate_v1",
        "requirements": [
            requirement.as_payload() for requirement in _FUTURES_METADATA_REQUIREMENTS_V1
        ],
    }


def build_article_feature_vector_v1(candle: RlFeatureCandle) -> tuple[float, ...]:
    open_price = _finite_float(value=candle.open, field="open")
    high = _finite_float(value=candle.high, field="high")
    low = _finite_float(value=candle.low, field="low")
    close = _finite_float(value=candle.close, field="close")
    volume = _non_negative_float(value=candle.volume_base, field="volume_base")
    vwap = derive_volume_weighted_average_v1(candle)
    trades_count = _required_trades_count(candle.trades_count)

    if high < max(open_price, close):
        raise FeatureContractViolation(reason="invalid_ohlc_high", field="high")
    if low > min(open_price, close):
        raise FeatureContractViolation(reason="invalid_ohlc_low", field="low")

    return (
        open_price,
        high,
        vwap,
        low,
        close,
        volume,
        float(trades_count),
    )


def derive_volume_weighted_average_v1(candle: RlFeatureCandle) -> float:
    volume_base = _non_negative_float(value=candle.volume_base, field="volume_base")
    if candle.volume_quote is None:
        raise FeatureContractViolation(reason="missing_volume_quote", field="volume_quote")
    volume_quote = _non_negative_float(value=candle.volume_quote, field="volume_quote")
    close = _finite_float(value=candle.close, field="close")

    if volume_base > 0.0:
        return volume_quote / volume_base
    if volume_quote == 0.0:
        return close
    raise FeatureContractViolation(
        reason="inconsistent_zero_base_positive_quote_volume",
        field="volume_quote",
    )


def _required_trades_count(value: int | None) -> int:
    if value is None:
        raise FeatureContractViolation(reason="missing_trades_count", field="trades_count")
    if isinstance(value, bool):
        raise FeatureContractViolation(reason="invalid_trades_count", field="trades_count")
    if value < 0:
        raise FeatureContractViolation(reason="negative_trades_count", field="trades_count")
    return int(value)


def _finite_float(*, value: float, field: str) -> float:
    if isinstance(value, bool):
        raise FeatureContractViolation(reason="invalid_numeric_field", field=field)
    out = float(value)
    if not math.isfinite(out):
        raise FeatureContractViolation(reason="non_finite_numeric_field", field=field)
    return out


def _non_negative_float(*, value: float, field: str) -> float:
    out = _finite_float(value=value, field=field)
    if out < 0.0:
        raise FeatureContractViolation(reason="negative_numeric_field", field=field)
    return out
