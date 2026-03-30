"""Contracts for deterministic backtest artifact store and signal rules v2."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, cast

import numpy as np

from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import TimeRange

ARTIFACT_STORE_V2_ROOT_LITERAL = "artifacts/backtest/v2"
CURRENT_ARTIFACT_POINTER_FILENAME_V2 = "current.yaml"
ARTIFACT_MANIFEST_FILENAME_V2 = "manifest.yaml"
PRICES_DIRECTORY_LITERAL_V2 = "prices"
SIGNALS_DIRECTORY_LITERAL_V2 = "signals"
MAPPINGS_DIRECTORY_LITERAL_V2 = "mappings"
HIT_TIMES_DIRECTORY_LITERAL_V2 = "hit_times"
ARTIFACT_SLOT_A_LITERAL_V2 = "slot_a"
ARTIFACT_SLOT_B_LITERAL_V2 = "slot_b"
HIT_TIMES_TIMEFRAME_LITERAL_V2 = "1m"
BAR_OPEN_MAPPING_FILENAME_V2 = "bar_open_1m_idx.u32.npy"
BAR_CLOSE_MAPPING_FILENAME_V2 = "bar_close_1m_idx.u32.npy"
OPEN_TIME_FILENAME_V2 = "open_time.i64.npy"
CLOSE_TIME_FILENAME_V2 = "close_time.i64.npy"
OHLCV_FILENAME_V2 = "ohlcv.f32.npy"
SIGNALS_FILENAME_V2 = "signals.i8.npy"
TP_VALUES_FILENAME_V2 = "tp_values.f32.npy"
SL_VALUES_FILENAME_V2 = "sl_values.f32.npy"
LONG_TP_FILENAME_V2 = "long_tp.u32.npy"
LONG_SL_FILENAME_V2 = "long_sl.u32.npy"
SHORT_TP_FILENAME_V2 = "short_tp.u32.npy"
SHORT_SL_FILENAME_V2 = "short_sl.u32.npy"
CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2 = 1
ROOT_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2 = 1
SIGNAL_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2 = 1
HIT_TIMES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2 = 1
ROOT_ARTIFACT_MANIFEST_KIND_V2 = "slot_root"
SIGNAL_ARTIFACT_MANIFEST_KIND_V2 = "signal"
HIT_TIMES_ARTIFACT_MANIFEST_KIND_V2 = "hit_times_1m"
ARTIFACT_SIGNAL_DTYPE_LITERAL_V2 = "int8"
ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2 = "int64"
ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2 = "float32"
ARTIFACT_MAPPING_DTYPE_LITERAL_V2 = "uint32"
ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2 = "float32"
ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2 = "uint32"
ARTIFACT_SIGNAL_VALUE_SET_V2: tuple[int, int, int] = (-1, 0, 1)
ARTIFACT_SIGNAL_AXIS_ORDER_V2: tuple[str, str] = ("variant", "time")
ARTIFACT_TIME_AXIS_ORDER_V2: tuple[str, ...] = ("time",)
ARTIFACT_PRICE_OHLCV_AXIS_ORDER_V2: tuple[str, str] = ("time", "field")
ARTIFACT_HIT_TIMES_LEVEL_AXIS_ORDER_V2: tuple[str, ...] = ("level",)
ARTIFACT_HIT_TIMES_TABLE_AXIS_ORDER_V2: tuple[str, str] = ("level", "time")
ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2 = "non_decreasing_by_level"
ARTIFACT_PLACEHOLDER_SHA256_V2 = "0" * 64
ARTIFACT_PUBLISH_BLOCKING_JOB_STATES_V2: tuple[str, ...] = ("queued", "running")
ARTIFACT_PUBLISH_BLOCKING_EXECUTION_MODES_V2: tuple[str, ...] = (
    "background_auto",
    "background_manual_legacy",
)
ARTIFACT_PUBLISH_FAILURE_CODE_INACTIVE_SLOT_PINNED_V2 = "inactive_slot_pinned"
SUPPORTED_CURRENT_ARTIFACT_POINTER_SCHEMA_VERSIONS_V2: tuple[int, ...] = (
    CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
)
CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2: tuple[str, ...] = (
    "schema_version",
    "active_slot",
    "slot_generation",
    "asof_date",
    "manifest_sha256",
    "published_at_utc",
)
ROOT_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2: tuple[str, ...] = (
    "schema_version",
    "manifest_kind",
    "slot",
    "slot_generation",
    "asof_date",
    "identity",
    "prices",
    "mappings",
    "signals",
    "hit_times",
    "signal_encoding",
    "provenance",
)
SIGNAL_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2: tuple[str, ...] = (
    "schema_version",
    "manifest_kind",
    "slot",
    "slot_generation",
    "asof_date",
    "indicator_id",
    "timeframe",
    "signals",
    "rows_count",
    "timeline",
    "signal_value_set",
    "grid",
    "provenance",
)
HIT_TIMES_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2: tuple[str, ...] = (
    "schema_version",
    "manifest_kind",
    "slot",
    "slot_generation",
    "asof_date",
    "timeframe",
    "timeline_bar_count",
    "sentinel_index",
    "tp_values",
    "sl_values",
    "tables",
    "provenance",
)
SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2: Mapping[int, tuple[str, str]] = MappingProxyType(
    {
        1: ("binance", "spot"),
        2: ("binance", "futures"),
        3: ("bybit", "spot"),
        4: ("bybit", "futures"),
    }
)

_STRICT_DATE_LITERAL_PATTERN_V2 = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STRICT_UTC_TIMESTAMP_LITERAL_PATTERN_V2 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256_HEX_PATTERN_V2 = re.compile(r"^[0-9a-f]{64}$")

type ArtifactSlotLiteralV2 = Literal["slot_a", "slot_b"]

ALLOWED_ARTIFACT_SLOTS_V2: tuple[ArtifactSlotLiteralV2, ...] = (
    ARTIFACT_SLOT_A_LITERAL_V2,
    ARTIFACT_SLOT_B_LITERAL_V2,
)
ARTIFACT_PRICE_TIMEFRAMES_V2: tuple[str, ...] = (
    "1m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "1d",
    "2d",
    "3d",
)
ARTIFACT_SIGNAL_TIMEFRAMES_V2: tuple[str, ...] = (
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "1d",
    "2d",
    "3d",
)
ARTIFACT_MAPPING_TIMEFRAMES_V2: tuple[str, ...] = ARTIFACT_SIGNAL_TIMEFRAMES_V2
ARTIFACT_HIT_TIMES_TIMEFRAMES_V2: tuple[str, ...] = (HIT_TIMES_TIMEFRAME_LITERAL_V2,)

type SignalRuleFamilyLiteralV2 = Literal[
    "compare_price_to_output",
    "threshold_band",
    "sign",
    "delta_sign",
    "compare_volume_to_output",
    "candle_body_direction",
    "pivot_events",
    "threshold_centered",
]
type SignalSourceLiteralV2 = Literal[
    "close",
    "open",
    "high",
    "low",
    "hl2",
    "hlc3",
    "ohlc4",
]
type StageADirectionModeLiteralV2 = Literal["long-only", "short-only", "long-short"]
type StageBExitReasonLiteralV2 = Literal[
    "signal_exit",
    "tp",
    "sl",
    "close_on_end",
    "unclosed",
]
type SignalRuleScalarV2 = int | float | str | bool | None

SIGNALS_V1_PARAMS_PATH_LITERAL_V2 = "signals.v1.params"
SIGNALS_V1_PARAMS_POLICY_LITERAL_V2 = "default-only"
SIGNAL_CODE_NEUTRAL_V2 = 0
SIGNAL_CODE_LONG_V2 = 1
SIGNAL_CODE_SHORT_V2 = -1
SIGNAL_CODE_DTYPE_LITERAL_V2 = "int8"
SIGNAL_CODE_VALUE_SET_V2: tuple[int, int, int] = (
    SIGNAL_CODE_SHORT_V2,
    SIGNAL_CODE_NEUTRAL_V2,
    SIGNAL_CODE_LONG_V2,
)
SUPPORTED_SIGNAL_RULE_FAMILIES_V2: tuple[SignalRuleFamilyLiteralV2, ...] = (
    "compare_price_to_output",
    "threshold_band",
    "sign",
    "delta_sign",
    "compare_volume_to_output",
    "candle_body_direction",
    "pivot_events",
    "threshold_centered",
)
SUPPORTED_SIGNAL_INPUT_SOURCES_V2: tuple[SignalSourceLiteralV2, ...] = (
    "close",
    "open",
    "high",
    "low",
    "hl2",
    "hlc3",
    "ohlc4",
)


def validate_signal_rule_family_v2(value: str) -> SignalRuleFamilyLiteralV2:
    """
    Validate one signal-rule family literal used by the v2 rules engine contract.

    Args:
        value: Candidate rule-family literal.
    Returns:
        SignalRuleFamilyLiteralV2: Canonical lower-case family literal.
    Assumptions:
        Rule-family set is fixed by R4-01 and must stay explicit.
    Raises:
        ValueError: If the literal is blank or outside the supported family set.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - docs/architecture/indicators/indicators_formula.yaml
    """
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SIGNAL_RULE_FAMILIES_V2:
        raise ValueError(
            "signal rule family must be one of "
            f"{SUPPORTED_SIGNAL_RULE_FAMILIES_V2}, got {value!r}"
        )
    return cast(SignalRuleFamilyLiteralV2, normalized)


def validate_signal_input_source_v2(value: str) -> SignalSourceLiteralV2:
    """
    Validate one `inputs.source` literal used by the v2 signal-rules engine.

    Args:
        value: Candidate `inputs.source` literal.
    Returns:
        SignalSourceLiteralV2: Canonical lower-case source literal.
    Assumptions:
        Source-axis semantics are limited to explicit candle-derived literals.
    Raises:
        ValueError: If the source literal is blank or unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - configs/prod/indicators.yaml
    """
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SIGNAL_INPUT_SOURCES_V2:
        raise ValueError(
            "inputs.source must be one of "
            f"{SUPPORTED_SIGNAL_INPUT_SOURCES_V2}, got {value!r}"
        )
    return cast(SignalSourceLiteralV2, normalized)


def _normalize_non_empty_literal_tuple_v2(
    *,
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    """
    Normalize and deterministically sort one tuple of non-empty string literals.

    Args:
        values: Candidate string tuple.
        field_name: Field label used in deterministic error messages.
    Returns:
        tuple[str, ...]: Lower-case unique tuple sorted lexicographically.
    Assumptions:
        String literals are small metadata tuples and deterministic sorting is acceptable.
    Raises:
        ValueError: If one literal is blank after normalization.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
    """
    normalized: set[str] = set()
    for raw_value in values:
        literal = raw_value.strip().lower()
        if not literal:
            raise ValueError(f"{field_name} must not contain blank values")
        normalized.add(literal)
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class SignalRuleSpecV2:
    """
    Explicit v2-aligned rule binding for one supported backtest indicator id.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - docs/architecture/indicators/indicators_formula.yaml
      - configs/prod/indicators.yaml
    """

    indicator_id: str
    rule_family: SignalRuleFamilyLiteralV2
    required_signal_params: tuple[str, ...] = ()
    required_dependency_ids: tuple[str, ...] = ()
    uses_inputs_source: bool = False
    threshold_center: float | None = None
    candle_body_min_param_name: str | None = None

    def __post_init__(self) -> None:
        """
        Validate stable v2 rule-spec invariants for one indicator binding.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Rule metadata is constructed once at import/startup time and stays immutable.
        Raises:
            ValueError: If the indicator id, rule family, or parameter metadata is invalid.
        Side Effects:
            Normalizes identifier and tuple fields into canonical lower-case ordering.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
          - configs/prod/indicators.yaml
        """
        normalized_indicator_id = validate_indicator_id_v2(self.indicator_id)
        object.__setattr__(self, "indicator_id", normalized_indicator_id)
        object.__setattr__(
            self,
            "rule_family",
            validate_signal_rule_family_v2(self.rule_family),
        )
        object.__setattr__(
            self,
            "required_signal_params",
            _normalize_non_empty_literal_tuple_v2(
                values=self.required_signal_params,
                field_name="required_signal_params",
            ),
        )
        object.__setattr__(
            self,
            "required_dependency_ids",
            _normalize_non_empty_literal_tuple_v2(
                values=self.required_dependency_ids,
                field_name="required_dependency_ids",
            ),
        )
        if self.threshold_center is not None and not np.isfinite(self.threshold_center):
            raise ValueError("SignalRuleSpecV2.threshold_center must be finite when provided")
        if self.candle_body_min_param_name is None:
            return
        normalized_min_param_name = self.candle_body_min_param_name.strip().lower()
        if not normalized_min_param_name:
            raise ValueError(
                "SignalRuleSpecV2.candle_body_min_param_name must be non-empty when provided"
            )
        object.__setattr__(
            self,
            "candle_body_min_param_name",
            normalized_min_param_name,
        )


@dataclass(frozen=True, slots=True)
class SignalRuleEvaluationRequestV2:
    """
    Typed pure-input envelope for one v2 signal-rules evaluation call.

    `signal_params` mirrors `signals.v1.params` and remains `default-only` in R4-01.
    `inputs_source` models explicit `inputs.source` semantics for indicators that carry a source
    axis in defaults/config or in compare-price rule evaluation.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
      - configs/prod/indicators.yaml
    """

    indicator_id: str
    candles: CandleArrays
    primary_output: np.ndarray
    inputs_source: str | None = None
    signal_params: Mapping[str, SignalRuleScalarV2] = field(default_factory=dict)
    dependency_outputs: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SignalRuleEvaluationResultV2:
    """
    Typed deterministic result of one v2 signal-rules evaluation call.

    `signal_codes` always uses compact `int8` encoding with `NEUTRAL = 0`, `LONG = 1`,
    `SHORT = -1`, and the value set `{-1,0,1}`.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    """

    indicator_id: str
    rule_family: SignalRuleFamilyLiteralV2
    inputs_source: str | None
    signal_params: Mapping[str, SignalRuleScalarV2]
    signal_codes: np.ndarray


def ordered_artifact_slots_v2() -> tuple[ArtifactSlotLiteralV2, ...]:
    """
    Return the canonical artifact slot order for R2-01.

    Args:
        None.
    Returns:
        tuple[ArtifactSlotLiteralV2, ...]: Stable ordered slot literals.
    Assumptions:
        The active dataset always lives in one of two fixed slots.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return ALLOWED_ARTIFACT_SLOTS_V2


def inactive_artifact_slot_v2(active_slot: str) -> ArtifactSlotLiteralV2:
    """
    Resolve the deterministic inactive slot opposite to the current active slot.

    Args:
        active_slot: Current active slot literal.
    Returns:
        ArtifactSlotLiteralV2: The opposite fixed slot literal.
    Assumptions:
        Milestone R2 uses exactly two slots and publish always targets the inactive one.
    Raises:
        ValueError: If the active slot literal is outside the fixed slot contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    validated_active_slot = validate_artifact_slot_v2(active_slot)
    if validated_active_slot == ARTIFACT_SLOT_A_LITERAL_V2:
        return ARTIFACT_SLOT_B_LITERAL_V2
    return ARTIFACT_SLOT_A_LITERAL_V2


def validate_artifact_coordinate_token_v2(token: str, *, field_name: str) -> str:
    """
    Validate one artifact coordinate token with fail-fast filesystem-safe rules.

    Args:
        token: Candidate coordinate literal for exchange, market_type, or symbol.
        field_name: Human-readable coordinate field name used in error messages.
    Returns:
        str: The original token when it satisfies the R2-01 contract.
    Assumptions:
        Coordinates are single path components and must never require normalization.
    Raises:
        ValueError: If the token is empty, contains whitespace, contains path separators,
            or includes traversal patterns.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    return _validate_safe_path_token_v2(token=token, field_name=f"coordinate {field_name}")


def validate_indicator_id_v2(indicator_id: str) -> str:
    """
    Validate one indicator identifier token used inside `signals/<tf>/<indicator_id>/`.

    Args:
        indicator_id: Candidate indicator identifier literal.
    Returns:
        str: The original indicator identifier when valid.
    Assumptions:
        Indicator ids may contain dots such as `ma.ema`, but remain one safe path token.
    Raises:
        ValueError: If the identifier is empty, contains whitespace, separators, or traversal.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    return _validate_safe_path_token_v2(token=indicator_id, field_name="indicator_id")


def validate_artifact_slot_v2(slot: str) -> ArtifactSlotLiteralV2:
    """
    Validate one artifact slot literal against the fixed R2-01 slot set.

    Args:
        slot: Candidate slot literal.
    Returns:
        ArtifactSlotLiteralV2: Canonical slot literal.
    Assumptions:
        Only `slot_a` and `slot_b` are valid during Milestone R2.
    Raises:
        ValueError: If the slot is not one of the fixed allowed literals.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    _validate_allowed_literal_v2(
        value=slot,
        field_name="slot",
        allowed_literals=ALLOWED_ARTIFACT_SLOTS_V2,
    )
    if slot == ARTIFACT_SLOT_A_LITERAL_V2:
        return ARTIFACT_SLOT_A_LITERAL_V2
    return ARTIFACT_SLOT_B_LITERAL_V2


def validate_current_pointer_schema_version_v2(schema_version: int) -> int:
    """
    Validate `current.yaml.schema_version` against the supported R2-02 set.

    Args:
        schema_version: Candidate pointer schema version value.
    Returns:
        int: Supported schema version literal.
    Assumptions:
        R2-02 supports only one strict pointer schema version.
    Raises:
        ValueError: If the value is not an integer schema version supported by runtime.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("current.yaml field 'schema_version' must be int")
    if schema_version not in SUPPORTED_CURRENT_ARTIFACT_POINTER_SCHEMA_VERSIONS_V2:
        raise ValueError(
            "current.yaml field 'schema_version' must be one of "
            f"{SUPPORTED_CURRENT_ARTIFACT_POINTER_SCHEMA_VERSIONS_V2}; "
            f"got {schema_version!r}"
        )
    return schema_version


def validate_current_pointer_slot_generation_v2(slot_generation: int) -> int:
    """
    Validate `current.yaml.slot_generation` as a positive integer.

    Args:
        slot_generation: Candidate slot generation scalar.
    Returns:
        int: Validated positive slot generation.
    Assumptions:
        Slot generation increments monotonically on each successful publish switch.
    Raises:
        ValueError: If the value is not a positive integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if isinstance(slot_generation, bool) or not isinstance(slot_generation, int):
        raise ValueError("current.yaml field 'slot_generation' must be int")
    if slot_generation <= 0:
        raise ValueError("current.yaml field 'slot_generation' must be > 0")
    return slot_generation


def validate_current_pointer_asof_date_v2(asof_date: str) -> str:
    """
    Validate `current.yaml.asof_date` as a strict `YYYY-MM-DD` literal.

    Args:
        asof_date: Candidate as-of date literal.
    Returns:
        str: Canonical date literal.
    Assumptions:
        R2-02 serializes pointer identity with exact date-only precision.
    Raises:
        ValueError: If the literal is not a valid strict calendar date.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not isinstance(asof_date, str):
        raise ValueError("current.yaml field 'asof_date' must be str")
    if _STRICT_DATE_LITERAL_PATTERN_V2.fullmatch(asof_date) is None:
        raise ValueError("current.yaml field 'asof_date' must be YYYY-MM-DD")
    try:
        date.fromisoformat(asof_date)
    except ValueError as error:
        raise ValueError("current.yaml field 'asof_date' must be valid YYYY-MM-DD") from error
    return asof_date


def validate_current_pointer_manifest_sha256_v2(manifest_sha256: str) -> str:
    """
    Validate `current.yaml.manifest_sha256` as a strict lowercase SHA-256 literal.

    Args:
        manifest_sha256: Candidate manifest hash literal.
    Returns:
        str: Canonical lowercase SHA-256 literal.
    Assumptions:
        Pointer identity stores manifest hashes as 64 lowercase hexadecimal characters.
    Raises:
        ValueError: If the hash is not 64 lowercase hexadecimal characters.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not isinstance(manifest_sha256, str):
        raise ValueError("current.yaml field 'manifest_sha256' must be str")
    if _SHA256_HEX_PATTERN_V2.fullmatch(manifest_sha256) is None:
        raise ValueError("current.yaml field 'manifest_sha256' must be 64 lowercase hex chars")
    return manifest_sha256


def validate_current_pointer_published_at_utc_v2(published_at_utc: str) -> str:
    """
    Validate `current.yaml.published_at_utc` as a strict UTC timestamp literal.

    Args:
        published_at_utc: Candidate UTC timestamp literal.
    Returns:
        str: Canonical UTC timestamp literal with `Z` suffix.
    Assumptions:
        Pointer timestamps are serialized with second precision and explicit UTC marker.
    Raises:
        ValueError: If the literal is not `YYYY-MM-DDTHH:MM:SSZ` in UTC.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not isinstance(published_at_utc, str):
        raise ValueError("current.yaml field 'published_at_utc' must be str")
    if _STRICT_UTC_TIMESTAMP_LITERAL_PATTERN_V2.fullmatch(published_at_utc) is None:
        raise ValueError("current.yaml field 'published_at_utc' must be YYYY-MM-DDTHH:MM:SSZ")
    parsed = datetime.fromisoformat(published_at_utc.replace("Z", "+00:00"))
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("current.yaml field 'published_at_utc' must be UTC")
    return published_at_utc


def validate_price_timeframe_v2(timeframe: str) -> str:
    """
    Validate one price artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `prices/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        Price artifacts exist for base `1m` and every supported request timeframe.
    Raises:
        ValueError: If the timeframe is outside the documented price artifact contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="price timeframe",
        allowed_literals=ARTIFACT_PRICE_TIMEFRAMES_V2,
    )
    return timeframe


def validate_signal_timeframe_v2(timeframe: str) -> str:
    """
    Validate one signal artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `signals/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        Signal artifacts are generated only for supported request timeframes, not for `1m`.
    Raises:
        ValueError: If the timeframe is outside the documented signal artifact contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="signal timeframe",
        allowed_literals=ARTIFACT_SIGNAL_TIMEFRAMES_V2,
    )
    return timeframe


def validate_mapping_timeframe_v2(timeframe: str) -> str:
    """
    Validate one mapping artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `mappings/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        Mapping artifacts are generated for every supported request timeframe.
    Raises:
        ValueError: If the timeframe is outside the documented mapping artifact contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="mapping timeframe",
        allowed_literals=ARTIFACT_MAPPING_TIMEFRAMES_V2,
    )
    return timeframe


def validate_hit_times_timeframe_v2(timeframe: str) -> str:
    """
    Validate one hit-times artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `hit_times/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        R2-01 fixes hit-times manifests under `hit_times/1m/`.
    Raises:
        ValueError: If the timeframe differs from the fixed `1m` contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="hit-times timeframe",
        allowed_literals=ARTIFACT_HIT_TIMES_TIMEFRAMES_V2,
    )
    return timeframe


def freeze_artifact_payload_mapping_v2(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Freeze one YAML payload into a stable key-sorted read-only mapping.

    Args:
        payload: Parsed YAML mapping with string keys.
    Returns:
        Mapping[str, Any]: Shallow immutable mapping with deterministic key order.
    Assumptions:
        Nested YAML values are preserved as loaded because R2-01 does not yet impose schema
        coercion for manifests.
    Raises:
        ValueError: If a payload key is not a string.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    normalized_keys: list[str] = []
    for key in payload.keys():
        if not isinstance(key, str):
            raise ValueError("artifact YAML payload keys must be strings")
        normalized_keys.append(key)
    normalized_payload: dict[str, Any] = {}
    for key in sorted(normalized_keys):
        normalized_payload[key] = payload[key]
    return MappingProxyType(normalized_payload)


def artifact_coordinates_from_market_id_v2(*, market_id: int, symbol: str) -> ArtifactCoordinatesV2:
    """
    Resolve artifact coordinates from canonical `ref_market.market_id` and symbol.

    Args:
        market_id: Stable market identifier from request/spec payload.
        symbol: Instrument symbol literal.
    Returns:
        ArtifactCoordinatesV2: Deterministic artifact coordinates for symbol-root resolution.
    Assumptions:
        R2-02 bridges `market_id` to `(exchange, market_type)` via the canonical seeded
        `ref_market` ids until R2-04 introduces dedicated artifact config loading.
    Raises:
        ValueError: If the market id is unsupported by the current fixed bridge mapping.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/market_data/application/use_cases/seed_ref_market.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    resolved_scope = SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2.get(market_id)
    if resolved_scope is None:
        raise ValueError(
            "artifact market bridge does not support market_id "
            f"{market_id!r}; expected one of {tuple(sorted(SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2))}"
        )
    exchange, market_type = resolved_scope
    return ArtifactCoordinatesV2(exchange=exchange, market_type=market_type, symbol=symbol)


def artifact_market_id_from_coordinates_v2(coordinates: ArtifactCoordinatesV2) -> int:
    """
    Resolve canonical `market_id` from artifact coordinates using the fixed R2-02 bridge.

    Args:
        coordinates: Deterministic artifact coordinates.
    Returns:
        int: Canonical market id matching the artifact symbol-root market scope.
    Assumptions:
        Coordinate-to-market resolution stays aligned with `seed_ref_market` during R2-02.
    Raises:
        ValueError: If the coordinate scope has no canonical market id mapping.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/market_data/application/use_cases/seed_ref_market.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    for market_id, scope in SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2.items():
        if scope == (coordinates.exchange, coordinates.market_type):
            return market_id
    raise ValueError(
        "artifact market bridge does not support coordinates "
        f"{coordinates.exchange!r}/{coordinates.market_type!r}"
    )


@dataclass(frozen=True, slots=True)
class ArtifactCoordinatesV2:
    """
    Deterministic artifact coordinates that select one backtest dataset namespace.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    exchange: str
    market_type: str
    symbol: str

    def __post_init__(self) -> None:
        """
        Validate coordinate tokens for deterministic path composition.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Coordinates are passed directly into filesystem path builders without normalization.
        Raises:
            ValueError: If one coordinate violates the filesystem-safe token contract.
        Side Effects:
            Normalizes the stored coordinates to validated canonical literals.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        object.__setattr__(
            self,
            "exchange",
            validate_artifact_coordinate_token_v2(self.exchange, field_name="exchange"),
        )
        object.__setattr__(
            self,
            "market_type",
            validate_artifact_coordinate_token_v2(self.market_type, field_name="market_type"),
        )
        object.__setattr__(
            self,
            "symbol",
            validate_artifact_coordinate_token_v2(self.symbol, field_name="symbol"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPricePathsV2:
    """
    Explicit paths for one `prices/<tf>/` artifact directory.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    open_time: Path
    close_time: Path
    ohlcv: Path


@dataclass(frozen=True, slots=True)
class ArtifactSignalPathsV2:
    """
    Explicit paths for one `signals/<tf>/<indicator_id>/` artifact directory.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    manifest: Path
    signals: Path


@dataclass(frozen=True, slots=True)
class ArtifactMappingPathsV2:
    """
    Explicit paths for one `mappings/<tf>/` artifact directory.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    bar_open_1m_idx: Path
    bar_close_1m_idx: Path


@dataclass(frozen=True, slots=True)
class ArtifactHitTimesPathsV2:
    """
    Explicit paths for the fixed `hit_times/1m/` artifact directory.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    manifest: Path
    tp_values: Path
    sl_values: Path
    long_tp: Path
    long_sl: Path
    short_tp: Path
    short_sl: Path


@dataclass(frozen=True, slots=True)
class ArtifactArrayMetadataV2:
    """
    Strict metadata contract for one artifact array referenced from a manifest.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    path: str
    dtype: str
    shape: tuple[int, ...]
    axis_order: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        """
        Validate immutable array metadata used by strict manifest contracts.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Paths are stored as slot-relative literals and shapes are explicit positive integers.
        Raises:
            ValueError: If path, dtype, shape, axis order, or hash fields are malformed.
        Side Effects:
            Normalizes metadata fields to validated canonical literals.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(self, "path", validate_relative_artifact_path_v2(self.path))
        object.__setattr__(self, "dtype", validate_artifact_dtype_literal_v2(self.dtype))
        object.__setattr__(self, "shape", validate_artifact_shape_v2(self.shape))
        object.__setattr__(
            self,
            "axis_order",
            validate_artifact_axis_order_v2(self.axis_order),
        )
        object.__setattr__(
            self,
            "sha256",
            validate_current_pointer_manifest_sha256_v2(self.sha256),
        )


@dataclass(frozen=True, slots=True)
class ArtifactTimelineCoverageV2:
    """
    Fixed timeline coverage metadata reused by price and signal manifests.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    bar_count: int
    open_time_start: int
    open_time_end: int
    close_time_start: int
    close_time_end: int

    def __post_init__(self) -> None:
        """
        Validate timeline coverage scalars against monotone positive-count expectations.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Coverage is written from already-materialized numeric timelines without coercion.
        Raises:
            ValueError: If counts are non-positive or time boundary literals are not integers.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(self, "bar_count", validate_positive_manifest_int_v2(self.bar_count))
        object.__setattr__(
            self,
            "open_time_start",
            validate_manifest_integer_literal_v2(
                self.open_time_start,
                field_name="timeline.open_time_start",
            ),
        )
        object.__setattr__(
            self,
            "open_time_end",
            validate_manifest_integer_literal_v2(
                self.open_time_end,
                field_name="timeline.open_time_end",
            ),
        )
        object.__setattr__(
            self,
            "close_time_start",
            validate_manifest_integer_literal_v2(
                self.close_time_start,
                field_name="timeline.close_time_start",
            ),
        )
        object.__setattr__(
            self,
            "close_time_end",
            validate_manifest_integer_literal_v2(
                self.close_time_end,
                field_name="timeline.close_time_end",
            ),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPriceTimeframeManifestV2:
    """
    Strict root-manifest section for one `prices/<tf>/` artifact family.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    timeframe: str
    open_time: ArtifactArrayMetadataV2
    close_time: ArtifactArrayMetadataV2
    ohlcv: ArtifactArrayMetadataV2
    coverage: ArtifactTimelineCoverageV2

    def __post_init__(self) -> None:
        """
        Validate one strict price-manifest timeframe section.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Price manifests exist only for supported artifact price timeframes.
        Raises:
            ValueError: If the timeframe literal violates the fixed artifact contract.
        Side Effects:
            Normalizes the stored timeframe literal.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(self, "timeframe", validate_price_timeframe_v2(self.timeframe))


@dataclass(frozen=True, slots=True)
class ArtifactMappingTimeframeManifestV2:
    """
    Strict root-manifest section for one `mappings/<tf>/` artifact family.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    timeframe: str
    bar_open_1m_idx: ArtifactArrayMetadataV2
    bar_close_1m_idx: ArtifactArrayMetadataV2

    def __post_init__(self) -> None:
        """
        Validate one strict mapping-manifest timeframe section.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Mapping artifacts exist only for supported request timeframes.
        Raises:
            ValueError: If the timeframe literal violates the fixed artifact contract.
        Side Effects:
            Normalizes the stored timeframe literal.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(self, "timeframe", validate_mapping_timeframe_v2(self.timeframe))


@dataclass(frozen=True, slots=True)
class ArtifactSignalCatalogEntryV2:
    """
    Root-manifest reference to one strict per-indicator signal manifest.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    timeframe: str
    indicator_id: str
    manifest_path: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        """
        Validate one root-manifest signal reference entry.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Signal manifest references stay slot-relative and deterministic.
        Raises:
            ValueError: If timeframe, indicator id, relative path, or hash are invalid.
        Side Effects:
            Normalizes stored literals to canonical validated values.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(self, "timeframe", validate_signal_timeframe_v2(self.timeframe))
        object.__setattr__(self, "indicator_id", validate_indicator_id_v2(self.indicator_id))
        object.__setattr__(
            self,
            "manifest_path",
            validate_relative_artifact_path_v2(self.manifest_path),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            validate_current_pointer_manifest_sha256_v2(self.manifest_sha256),
        )


@dataclass(frozen=True, slots=True)
class ArtifactSignalCatalogV2:
    """
    Root-manifest catalog of signal manifests and supported signal dimensions.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    supported_timeframes: tuple[str, ...]
    supported_indicator_ids: tuple[str, ...]
    manifests: tuple[ArtifactSignalCatalogEntryV2, ...]

    def __post_init__(self) -> None:
        """
        Validate catalog ordering and supported literal sets for signal manifests.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Signal manifest catalogs are serialized in deterministic order without duplicates.
        Raises:
            ValueError: If one timeframe, indicator id, or manifest entry is duplicated.
        Side Effects:
            Replaces tuples with deterministic canonical ordering.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(
            self,
            "supported_timeframes",
            _sorted_unique_timeframes_v2(
                values=self.supported_timeframes,
                allowed_literals=ARTIFACT_SIGNAL_TIMEFRAMES_V2,
                field_name="signals.supported_timeframes",
                validator=validate_signal_timeframe_v2,
            ),
        )
        object.__setattr__(
            self,
            "supported_indicator_ids",
            _sorted_unique_indicator_ids_v2(
                values=self.supported_indicator_ids,
                field_name="signals.supported_indicator_ids",
            ),
        )
        object.__setattr__(
            self,
            "manifests",
            _sorted_signal_catalog_entries_v2(self.manifests),
        )


@dataclass(frozen=True, slots=True)
class ArtifactHitTimesReferenceV2:
    """
    Root-manifest reference to the strict `hit_times/1m/manifest.yaml` document.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    timeframe: str
    manifest_path: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        """
        Validate the root-manifest hit-times reference entry.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            R2 fixes hit-times to a single `1m` manifest path.
        Raises:
            ValueError: If timeframe, relative path, or hash are invalid.
        Side Effects:
            Normalizes stored literals to canonical validated values.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(self, "timeframe", validate_hit_times_timeframe_v2(self.timeframe))
        object.__setattr__(
            self,
            "manifest_path",
            validate_relative_artifact_path_v2(self.manifest_path),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            validate_current_pointer_manifest_sha256_v2(self.manifest_sha256),
        )


@dataclass(frozen=True, slots=True)
class ArtifactSignalEncodingContractV2:
    """
    Root-manifest runtime contract for signal dtype, axis order, and allowed values.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    dtype: str
    axis_order: tuple[str, ...]
    value_set: tuple[int, ...]

    def __post_init__(self) -> None:
        """
        Validate the fixed runtime signal-encoding contract read from root manifest.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Engine v2 stores signals as `int8` with deterministic `[variant, time]` layout.
        Raises:
            ValueError: If dtype, axis order, or signal value set violate the fixed contract.
        Side Effects:
            Normalizes stored values to canonical tuples.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(self, "dtype", validate_artifact_dtype_literal_v2(self.dtype))
        object.__setattr__(
            self,
            "axis_order",
            validate_artifact_axis_order_v2(self.axis_order),
        )
        object.__setattr__(
            self,
            "value_set",
            validate_signal_value_set_v2(self.value_set),
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifestProvenanceV2:
    """
    Strict provenance payload carried by root/signal/hit-times manifests.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    generator: str
    generator_version: str
    generated_at_utc: str
    config_sha256: str
    inputs_sha256: str

    def __post_init__(self) -> None:
        """
        Validate provenance fields that identify the manifest producer and source inputs.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Provenance fields are immutable publish metadata and must remain explicit.
        Raises:
            ValueError: If generator literals are empty or hash/timestamp literals are invalid.
        Side Effects:
            Normalizes stored literals to validated canonical values.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(
            self,
            "generator",
            validate_manifest_text_literal_v2(self.generator, field_name="provenance.generator"),
        )
        object.__setattr__(
            self,
            "generator_version",
            validate_manifest_text_literal_v2(
                self.generator_version,
                field_name="provenance.generator_version",
            ),
        )
        object.__setattr__(
            self,
            "generated_at_utc",
            validate_current_pointer_published_at_utc_v2(self.generated_at_utc),
        )
        object.__setattr__(
            self,
            "config_sha256",
            validate_current_pointer_manifest_sha256_v2(self.config_sha256),
        )
        object.__setattr__(
            self,
            "inputs_sha256",
            validate_current_pointer_manifest_sha256_v2(self.inputs_sha256),
        )


@dataclass(frozen=True, slots=True)
class ArtifactSignalGridContractV2:
    """
    Strict metadata describing deterministic signal-row ordering and defaults.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    variant_key_version: int
    variant_keys_sha256: str
    signals_v1_params_defaults: Mapping[str, Any]

    def __post_init__(self) -> None:
        """
        Validate grid metadata carried by strict per-indicator signal manifests.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variant key semantics stay aligned with v1 and defaults are serialized as a mapping.
        Raises:
            ValueError: If variant-key version, hash, or defaults payload are invalid.
        Side Effects:
            Freezes the defaults mapping into a deterministic read-only payload.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(
            self,
            "variant_key_version",
            validate_positive_manifest_int_v2(self.variant_key_version),
        )
        object.__setattr__(
            self,
            "variant_keys_sha256",
            validate_current_pointer_manifest_sha256_v2(self.variant_keys_sha256),
        )
        object.__setattr__(
            self,
            "signals_v1_params_defaults",
            freeze_artifact_payload_mapping_v2(self.signals_v1_params_defaults),
        )


@dataclass(frozen=True, slots=True)
class ArtifactHitTimesTableManifestV2:
    """
    Strict metadata contract for one hit-times lookup table.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    array: ArtifactArrayMetadataV2
    monotonicity: str

    def __post_init__(self) -> None:
        """
        Validate hit-times table metadata and declared monotonicity contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            All runtime hit-times tables use the same non-decreasing-by-level invariant.
        Raises:
            ValueError: If monotonicity literal is unsupported.
        Side Effects:
            Normalizes the monotonicity literal.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        object.__setattr__(
            self,
            "monotonicity",
            validate_hit_times_monotonicity_literal_v2(self.monotonicity),
        )


@dataclass(frozen=True, slots=True)
class ArtifactCurrentPointerV2:
    """
    Parsed strict `current.yaml` payload with the typed identity fields required by R2-02.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    path: Path
    active_slot: ArtifactSlotLiteralV2
    raw_payload: Mapping[str, Any]
    schema_version: int
    slot_generation: int
    asof_date: str
    manifest_sha256: str
    published_at_utc: str

    def __post_init__(self) -> None:
        """
        Re-validate the strict pointer identity contract and freeze the raw payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `current.yaml` contains exactly the required R2-02 fields with no extras.
        Raises:
            ValueError: If the slot literal or payload shape violates the contract.
        Side Effects:
            Replaces `raw_payload` with a stable read-only mapping.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        raw_keys = tuple(sorted(self.raw_payload.keys()))
        required_keys = tuple(sorted(CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2))
        if raw_keys != required_keys:
            missing_keys = tuple(
                key
                for key in CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2
                if key not in self.raw_payload
            )
            extra_keys = tuple(
                key for key in raw_keys if key not in CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2
            )
            details: list[str] = []
            if len(missing_keys) > 0:
                details.append(f"missing keys {missing_keys}")
            if len(extra_keys) > 0:
                details.append(f"unexpected keys {extra_keys}")
            raise ValueError(
                f"{self.path} must contain exactly keys "
                f"{CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2}"
                + (f"; {'; '.join(details)}" if len(details) > 0 else "")
            )
        object.__setattr__(self, "active_slot", validate_artifact_slot_v2(self.active_slot))
        object.__setattr__(
            self,
            "schema_version",
            validate_current_pointer_schema_version_v2(self.schema_version),
        )
        object.__setattr__(
            self,
            "slot_generation",
            validate_current_pointer_slot_generation_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "asof_date",
            validate_current_pointer_asof_date_v2(self.asof_date),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            validate_current_pointer_manifest_sha256_v2(self.manifest_sha256),
        )
        object.__setattr__(
            self,
            "published_at_utc",
            validate_current_pointer_published_at_utc_v2(self.published_at_utc),
        )
        object.__setattr__(
            self, "raw_payload", freeze_artifact_payload_mapping_v2(self.raw_payload)
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifestDocumentV2:
    """
    Parsed strict root `manifest.yaml` document returned by explicit-path loaders.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    path: Path
    raw_payload: Mapping[str, Any]
    slot: ArtifactSlotLiteralV2
    schema_version: int
    manifest_kind: str
    slot_generation: int
    asof_date: str
    identity: ArtifactCoordinatesV2
    prices: tuple[ArtifactPriceTimeframeManifestV2, ...]
    mappings: tuple[ArtifactMappingTimeframeManifestV2, ...]
    signals: ArtifactSignalCatalogV2
    hit_times: ArtifactHitTimesReferenceV2
    signal_encoding: ArtifactSignalEncodingContractV2
    provenance: ArtifactManifestProvenanceV2

    def __post_init__(self) -> None:
        """
        Re-validate the strict root-manifest contract and freeze raw payload ordering.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Root manifests contain only the fixed R2-03 schema and explicit runtime metadata.
        Raises:
            ValueError: If root-manifest slot, version, kind, keys, or payload shape are invalid.
        Side Effects:
            Normalizes validated literals and freezes `raw_payload`.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        _validate_exact_mapping_keys_v2(
            payload=self.raw_payload,
            required_keys=ROOT_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
            path=self.path,
        )
        object.__setattr__(self, "slot", validate_artifact_slot_v2(self.slot))
        object.__setattr__(
            self,
            "schema_version",
            validate_manifest_schema_version_v2(
                self.schema_version,
                field_name="root manifest schema_version",
                expected_schema_version=ROOT_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
            ),
        )
        object.__setattr__(
            self,
            "manifest_kind",
            validate_manifest_kind_v2(
                self.manifest_kind,
                expected_kind=ROOT_ARTIFACT_MANIFEST_KIND_V2,
            ),
        )
        object.__setattr__(
            self,
            "slot_generation",
            validate_positive_manifest_int_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "asof_date",
            validate_current_pointer_asof_date_v2(self.asof_date),
        )
        object.__setattr__(
            self, "raw_payload", freeze_artifact_payload_mapping_v2(self.raw_payload)
        )


@dataclass(frozen=True, slots=True)
class ArtifactSignalManifestDocumentV2:
    """
    Parsed strict per-indicator signal `manifest.yaml` document.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    path: Path
    raw_payload: Mapping[str, Any]
    slot: ArtifactSlotLiteralV2
    schema_version: int
    manifest_kind: str
    slot_generation: int
    asof_date: str
    indicator_id: str
    timeframe: str
    signals: ArtifactArrayMetadataV2
    rows_count: int
    timeline: ArtifactTimelineCoverageV2
    signal_value_set: tuple[int, ...]
    grid: ArtifactSignalGridContractV2
    provenance: ArtifactManifestProvenanceV2

    def __post_init__(self) -> None:
        """
        Re-validate the strict signal-manifest contract and freeze raw payload ordering.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Signal manifests carry fully fixed runtime metadata and no unsupported drift.
        Raises:
            ValueError: If signal-manifest keys or typed identity fields are invalid.
        Side Effects:
            Normalizes validated literals and freezes `raw_payload`.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        _validate_exact_mapping_keys_v2(
            payload=self.raw_payload,
            required_keys=SIGNAL_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
            path=self.path,
        )
        object.__setattr__(self, "slot", validate_artifact_slot_v2(self.slot))
        object.__setattr__(
            self,
            "schema_version",
            validate_manifest_schema_version_v2(
                self.schema_version,
                field_name="signal manifest schema_version",
                expected_schema_version=SIGNAL_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
            ),
        )
        object.__setattr__(
            self,
            "manifest_kind",
            validate_manifest_kind_v2(
                self.manifest_kind,
                expected_kind=SIGNAL_ARTIFACT_MANIFEST_KIND_V2,
            ),
        )
        object.__setattr__(
            self,
            "slot_generation",
            validate_positive_manifest_int_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "asof_date",
            validate_current_pointer_asof_date_v2(self.asof_date),
        )
        object.__setattr__(self, "indicator_id", validate_indicator_id_v2(self.indicator_id))
        object.__setattr__(self, "timeframe", validate_signal_timeframe_v2(self.timeframe))
        object.__setattr__(
            self,
            "rows_count",
            validate_positive_manifest_int_v2(self.rows_count),
        )
        object.__setattr__(
            self,
            "signal_value_set",
            validate_signal_value_set_v2(self.signal_value_set),
        )
        object.__setattr__(
            self, "raw_payload", freeze_artifact_payload_mapping_v2(self.raw_payload)
        )


@dataclass(frozen=True, slots=True)
class ArtifactHitTimesManifestDocumentV2:
    """
    Parsed strict `hit_times/1m/manifest.yaml` document.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """

    path: Path
    raw_payload: Mapping[str, Any]
    slot: ArtifactSlotLiteralV2
    schema_version: int
    manifest_kind: str
    slot_generation: int
    asof_date: str
    timeframe: str
    timeline_bar_count: int
    sentinel_index: int
    tp_values: ArtifactArrayMetadataV2
    sl_values: ArtifactArrayMetadataV2
    long_tp: ArtifactHitTimesTableManifestV2
    long_sl: ArtifactHitTimesTableManifestV2
    short_tp: ArtifactHitTimesTableManifestV2
    short_sl: ArtifactHitTimesTableManifestV2
    provenance: ArtifactManifestProvenanceV2

    def __post_init__(self) -> None:
        """
        Re-validate the strict hit-times-manifest contract and freeze raw payload ordering.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Hit-times manifests define a single fixed `1m` runtime contract.
        Raises:
            ValueError: If hit-times-manifest keys or typed identity fields are invalid.
        Side Effects:
            Normalizes validated literals and freezes `raw_payload`.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
        """
        _validate_exact_mapping_keys_v2(
            payload=self.raw_payload,
            required_keys=HIT_TIMES_ARTIFACT_MANIFEST_REQUIRED_KEYS_V2,
            path=self.path,
        )
        object.__setattr__(self, "slot", validate_artifact_slot_v2(self.slot))
        object.__setattr__(
            self,
            "schema_version",
            validate_manifest_schema_version_v2(
                self.schema_version,
                field_name="hit_times manifest schema_version",
                expected_schema_version=HIT_TIMES_ARTIFACT_MANIFEST_SCHEMA_VERSION_V2,
            ),
        )
        object.__setattr__(
            self,
            "manifest_kind",
            validate_manifest_kind_v2(
                self.manifest_kind,
                expected_kind=HIT_TIMES_ARTIFACT_MANIFEST_KIND_V2,
            ),
        )
        object.__setattr__(
            self,
            "slot_generation",
            validate_positive_manifest_int_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "asof_date",
            validate_current_pointer_asof_date_v2(self.asof_date),
        )
        object.__setattr__(self, "timeframe", validate_hit_times_timeframe_v2(self.timeframe))
        object.__setattr__(
            self,
            "timeline_bar_count",
            validate_positive_manifest_int_v2(self.timeline_bar_count),
        )
        object.__setattr__(
            self,
            "sentinel_index",
            validate_non_negative_manifest_int_v2(self.sentinel_index),
        )
        object.__setattr__(
            self, "raw_payload", freeze_artifact_payload_mapping_v2(self.raw_payload)
        )


@dataclass(frozen=True, slots=True)
class ArtifactValidationDiagnosticV2:
    """
    Stable structured validation error emitted by strict manifest validators.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    code: str
    message: str
    location: str
    manifest_path: Path
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        """
        Validate stable diagnostic fields used by publish guard and operator logs.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Codes and locations remain short deterministic literals.
        Raises:
            ValueError: If code, message, or location are empty.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        object.__setattr__(
            self,
            "code",
            validate_manifest_text_literal_v2(self.code, field_name="diagnostic.code"),
        )
        object.__setattr__(
            self,
            "message",
            validate_manifest_text_literal_v2(self.message, field_name="diagnostic.message"),
        )
        object.__setattr__(
            self,
            "location",
            validate_manifest_text_literal_v2(self.location, field_name="diagnostic.location"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactSignalValidationSpecV2:
    """
    Explicit one-indicator validation target used by R2-02 slot publishing checks.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    timeframe: str
    indicator_id: str

    def __post_init__(self) -> None:
        """
        Validate one explicit signal artifact validation target.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Signal validation targets remain explicit even when they are translated from
            `backtest_artifacts.validation_plan.signal_artifacts`.
        Raises:
            ValueError: If timeframe or indicator id violates the deterministic path contract.
        Side Effects:
            Normalizes stored literals to validated canonical values.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        object.__setattr__(self, "timeframe", validate_signal_timeframe_v2(self.timeframe))
        object.__setattr__(self, "indicator_id", validate_indicator_id_v2(self.indicator_id))


@dataclass(frozen=True, slots=True)
class ArtifactSlotValidationSpecV2:
    """
    Explicit validation plan for an already-built inactive slot in R2 publish flow.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    price_timeframes: tuple[str, ...] = ()
    mapping_timeframes: tuple[str, ...] = ()
    signal_artifacts: tuple[ArtifactSignalValidationSpecV2, ...] = ()
    require_hit_times_manifest: bool = True

    def __post_init__(self) -> None:
        """
        Validate and deterministically order the explicit slot validation plan.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Validation order must stay stable regardless of caller tuple ordering or
            R2-04 config author ordering.
        Raises:
            ValueError: If one timeframe or signal artifact violates the path contract.
        Side Effects:
            Replaces stored tuples with deterministic canonical ordering.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        object.__setattr__(
            self,
            "price_timeframes",
            _sorted_unique_timeframes_v2(
                values=self.price_timeframes,
                allowed_literals=ARTIFACT_PRICE_TIMEFRAMES_V2,
                field_name="price_timeframes",
                validator=validate_price_timeframe_v2,
            ),
        )
        object.__setattr__(
            self,
            "mapping_timeframes",
            _sorted_unique_timeframes_v2(
                values=self.mapping_timeframes,
                allowed_literals=ARTIFACT_MAPPING_TIMEFRAMES_V2,
                field_name="mapping_timeframes",
                validator=validate_mapping_timeframe_v2,
            ),
        )
        object.__setattr__(
            self,
            "signal_artifacts",
            _sorted_signal_validation_specs_v2(self.signal_artifacts),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPublishPrecheckV2:
    """
    Deterministic precheck diagnostics for `build inactive slot` before publish switch.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    coordinates: ArtifactCoordinatesV2
    current_pointer_path: Path
    current_pointer: ArtifactCurrentPointerV2 | None
    inactive_slot: ArtifactSlotLiteralV2
    target_slot_generation: int
    inactive_manifest_path: Path
    inactive_manifest_hash: str | None
    blocking_active_run_count: int
    ready: bool
    bootstrap: bool = False
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        """
        Validate bootstrap and steady-state precheck invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Bootstrap prechecks target `slot_a` generation `1`, while steady-state prechecks
            retain the resolved strict `current.yaml` identity.
        Raises:
            ValueError: If slot/generation fields are invalid or bootstrap/current-pointer state
                is contradictory.
        Side Effects:
            Normalizes slot and generation literals.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        object.__setattr__(self, "inactive_slot", validate_artifact_slot_v2(self.inactive_slot))
        object.__setattr__(
            self,
            "target_slot_generation",
            validate_current_pointer_slot_generation_v2(self.target_slot_generation),
        )
        object.__setattr__(
            self,
            "blocking_active_run_count",
            validate_non_negative_manifest_int_v2(self.blocking_active_run_count),
        )
        if self.current_pointer is None:
            if not self.bootstrap:
                raise ValueError(
                    "ArtifactPublishPrecheckV2.current_pointer is required outside bootstrap mode"
                )
            return
        if self.bootstrap:
            raise ValueError(
                "ArtifactPublishPrecheckV2.bootstrap cannot be true when current_pointer exists"
            )
        if self.current_pointer.path != self.current_pointer_path:
            raise ValueError(
                "ArtifactPublishPrecheckV2.current_pointer_path must match current_pointer.path"
            )
        if self.target_slot_generation != self.current_pointer.slot_generation + 1:
            raise ValueError(
                "ArtifactPublishPrecheckV2.target_slot_generation must equal "
                "current_pointer.slot_generation + 1"
            )


@dataclass(frozen=True, slots=True)
class ArtifactSlotValidationResultV2:
    """
    Validation output for a prepared inactive slot just before pointer switch.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    slot: ArtifactSlotLiteralV2
    slot_manifest: ArtifactManifestDocumentV2 | None
    signal_manifests: tuple[ArtifactSignalManifestDocumentV2, ...]
    hit_times_manifest: ArtifactHitTimesManifestDocumentV2 | None
    manifest_sha256: str | None
    validation_spec: ArtifactSlotValidationSpecV2
    diagnostics: tuple[ArtifactValidationDiagnosticV2, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactPublishResultV2:
    """
    Structured result payload for successful R2-02 current-pointer publish switch.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    coordinates: ArtifactCoordinatesV2
    previous_pointer: ArtifactCurrentPointerV2 | None
    published_pointer: ArtifactCurrentPointerV2
    precheck: ArtifactPublishPrecheckV2
    validation: ArtifactSlotValidationResultV2


@dataclass(frozen=True, slots=True)
class ArtifactCanonicalPriceExportRequestV2:
    """
    Explicit request DTO for canonical `1m`-rooted price export into the inactive artifact slot.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/shared_kernel/primitives/time_range.py
    """

    coordinates: ArtifactCoordinatesV2
    time_range: TimeRange
    asof_date: str
    generated_at_utc: str
    target_slot: ArtifactSlotLiteralV2 | None = None
    target_slot_generation: int | None = None
    force_full_rebuild: bool = False

    def __post_init__(self) -> None:
        """
        Validate stable request identity fields for deterministic R3-02 price export.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Request scope is one symbol root and one source `TimeRange [start, end)`.
        Raises:
            ValueError: If coordinates, as-of date, or generated timestamp violate strict
                artifact contracts.
        Side Effects:
            Normalizes validated date and timestamp literals.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        if self.coordinates is None:  # type: ignore[truthy-bool]
            raise ValueError("ArtifactCanonicalPriceExportRequestV2.coordinates is required")
        if self.time_range is None:  # type: ignore[truthy-bool]
            raise ValueError("ArtifactCanonicalPriceExportRequestV2.time_range is required")
        object.__setattr__(
            self,
            "asof_date",
            validate_current_pointer_asof_date_v2(self.asof_date),
        )
        object.__setattr__(
            self,
            "generated_at_utc",
            validate_current_pointer_published_at_utc_v2(self.generated_at_utc),
        )
        if self.target_slot is None and self.target_slot_generation is not None:
            raise ValueError(
                "ArtifactCanonicalPriceExportRequestV2.target_slot_generation requires target_slot"
            )
        if self.target_slot is not None and self.target_slot_generation is None:
            raise ValueError(
                "ArtifactCanonicalPriceExportRequestV2.target_slot requires "
                "target_slot_generation"
            )
        if self.target_slot is None:
            return
        target_slot_generation = self.target_slot_generation
        if target_slot_generation is None:
            raise ValueError(
                "ArtifactCanonicalPriceExportRequestV2.target_slot_generation is required"
            )
        object.__setattr__(self, "target_slot", validate_artifact_slot_v2(self.target_slot))
        object.__setattr__(
            self,
            "target_slot_generation",
            validate_current_pointer_slot_generation_v2(target_slot_generation),
        )


@dataclass(frozen=True, slots=True)
class ArtifactTailRebuildBarsV2:
    """
    Stage-level bounded tail rewrite counters emitted by the shared artifact precompute flow.

    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """

    prices: int = 0
    mappings: int = 0
    signals: int = 0
    hit_times: int = 0

    def __post_init__(self) -> None:
        """
        Validate non-negative bounded tail counters for every artifact stage.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Each counter represents a deterministic number of rewritten bars/items for the stage.
        Raises:
            ValueError: If one stage counter is negative.
        Side Effects:
            Normalizes counters through strict non-negative integer validation.
        Docs:
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        object.__setattr__(self, "prices", validate_non_negative_manifest_int_v2(self.prices))
        object.__setattr__(self, "mappings", validate_non_negative_manifest_int_v2(self.mappings))
        object.__setattr__(self, "signals", validate_non_negative_manifest_int_v2(self.signals))
        object.__setattr__(
            self,
            "hit_times",
            validate_non_negative_manifest_int_v2(self.hit_times),
        )

    def as_dict(self) -> dict[str, int]:
        """
        Serialize stage-level tail counters into a stable JSON-friendly mapping.

        Args:
            None.
        Returns:
            dict[str, int]: Deterministic stage-to-counter mapping.
        Assumptions:
            Scheduler metrics and CLI diagnostics consume the same stage names.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
        """
        return {
            "prices": self.prices,
            "mappings": self.mappings,
            "signals": self.signals,
            "hit_times": self.hit_times,
        }


@dataclass(frozen=True, slots=True)
class ArtifactCanonicalPriceExportResultV2:
    """
    Structured result payload for R3-02 price export into the inactive slot.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    coordinates: ArtifactCoordinatesV2
    slot: ArtifactSlotLiteralV2
    slot_generation: int
    asof_date: str
    manifest_path: Path
    manifest_sha256: str
    price_paths: ArtifactPricePathsV2
    coverage: ArtifactTimelineCoverageV2
    source_time_range: TimeRange
    source_candle_count: int
    reused_prefix_bars: int
    rewritten_tail_bars: int
    tail_rebuild_bars: ArtifactTailRebuildBarsV2 = field(
        default_factory=ArtifactTailRebuildBarsV2
    )

    def __post_init__(self) -> None:
        """
        Validate immutable export result fields exposed by the precompute runner.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `price_paths` still points to the canonical `prices/1m` family while the root manifest
            now also references rolled `prices/<tf>` artifacts written in the same build.
        Raises:
            ValueError: If slot identity, slot generation, or count fields are invalid.
        Side Effects:
            Normalizes strict slot and date/hash literals.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        object.__setattr__(self, "slot", validate_artifact_slot_v2(self.slot))
        object.__setattr__(
            self,
            "slot_generation",
            validate_current_pointer_slot_generation_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "asof_date",
            validate_current_pointer_asof_date_v2(self.asof_date),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            validate_current_pointer_manifest_sha256_v2(self.manifest_sha256),
        )
        object.__setattr__(
            self,
            "source_candle_count",
            validate_positive_manifest_int_v2(self.source_candle_count),
        )
        object.__setattr__(
            self,
            "reused_prefix_bars",
            validate_non_negative_manifest_int_v2(self.reused_prefix_bars),
        )
        object.__setattr__(
            self,
            "rewritten_tail_bars",
            validate_positive_manifest_int_v2(self.rewritten_tail_bars),
        )
        if self.tail_rebuild_bars is None:  # type: ignore[truthy-bool]
            raise ValueError("ArtifactCanonicalPriceExportResultV2.tail_rebuild_bars is required")


@dataclass(frozen=True, slots=True)
class ArtifactPricesMappingsPublishResultV2:
    """
    Structured R3-04 flow result for `precheck -> build inactive slot -> validate -> publish`.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    validation_spec: ArtifactSlotValidationSpecV2
    precheck: ArtifactPublishPrecheckV2
    build_result: ArtifactCanonicalPriceExportResultV2
    publish_result: ArtifactPublishResultV2

    def __post_init__(self) -> None:
        """
        Validate that R3-04 stage artifacts and published pointer identities stay aligned.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            R3-04 publishes only the inactive slot rebuilt by the immediately preceding
            `prices + mappings` build and uses the same root `manifest.yaml` hash for
            `current.yaml`.
        Raises:
            ValueError: If the precheck, build result, and published pointer disagree on slot,
                generation, `asof_date`, coordinates, or manifest hash.
        Side Effects:
            None.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
          - docs/runbooks/backtest-artifacts-rebuild.md
        """
        if self.validation_spec.signal_artifacts != ():
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2.validation_spec must keep "
                "signal_artifacts=() for the R3-04 prices+mappings stage"
            )
        if self.validation_spec.require_hit_times_manifest:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2.validation_spec must keep "
                "require_hit_times_manifest=False for the R3-04 prices+mappings stage"
            )
        if self.precheck.coordinates != self.build_result.coordinates:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2 precheck/build coordinates must match; "
                f"got {self.precheck.coordinates!r} and {self.build_result.coordinates!r}"
            )
        if self.publish_result.coordinates != self.build_result.coordinates:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2 build/publish coordinates must match; "
                f"got {self.build_result.coordinates!r} and {self.publish_result.coordinates!r}"
            )
        if self.precheck != self.publish_result.precheck:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2.publish_result.precheck must equal the "
                "recorded R3-04 precheck snapshot"
            )
        if self.precheck.inactive_slot != self.build_result.slot:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2 build_result.slot must match the "
                f"prechecked inactive slot; got {self.build_result.slot!r}, expected "
                f"{self.precheck.inactive_slot!r}"
            )
        published_pointer = self.publish_result.published_pointer
        if self.build_result.slot != published_pointer.active_slot:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2 published pointer must activate the "
                f"rebuilt slot; got {published_pointer.active_slot!r}, expected "
                f"{self.build_result.slot!r}"
            )
        if self.build_result.slot_generation != published_pointer.slot_generation:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2 slot_generation must match between "
                f"build result and published pointer; got {self.build_result.slot_generation!r} "
                f"and {published_pointer.slot_generation!r}"
            )
        if self.build_result.asof_date != published_pointer.asof_date:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2 asof_date must match between build "
                f"result and published pointer; got {self.build_result.asof_date!r} and "
                f"{published_pointer.asof_date!r}"
            )
        if self.build_result.manifest_sha256 != published_pointer.manifest_sha256:
            raise ValueError(
                "ArtifactPricesMappingsPublishResultV2 manifest_sha256 must match between build "
                "result and published pointer"
            )


@dataclass(frozen=True, slots=True)
class ArtifactPinnedIdentityV2:
    """
    Immutable persisted artifact identity used to reopen one published slot deterministically.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    """

    artifact_slot: ArtifactSlotLiteralV2
    slot_generation: int
    artifact_asof_date: str
    artifact_manifest_hash: str

    def __post_init__(self) -> None:
        """
        Validate one persisted slot identity without touching runtime files.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Background runs persist these fields at create time and later reuse them as immutable
            slot identity without hash recomputation.
        Raises:
            ValueError: If slot, generation, date, or manifest hash literals are invalid.
        Side Effects:
            Normalizes validated literals to canonical values.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
          - src/trading/contexts/backtest/domain/entities/backtest_job.py
        """
        object.__setattr__(self, "artifact_slot", validate_artifact_slot_v2(self.artifact_slot))
        object.__setattr__(
            self,
            "slot_generation",
            validate_current_pointer_slot_generation_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "artifact_asof_date",
            validate_current_pointer_asof_date_v2(self.artifact_asof_date),
        )
        object.__setattr__(
            self,
            "artifact_manifest_hash",
            validate_current_pointer_manifest_sha256_v2(self.artifact_manifest_hash),
        )


@dataclass(frozen=True, slots=True)
class ArtifactSlotPinnedRuntimeContextV2:
    """
    Shared immutable slot-pinned context used at sync and background runtime start.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    coordinates: ArtifactCoordinatesV2
    artifact_slot: ArtifactSlotLiteralV2
    slot_generation: int
    artifact_asof_date: str
    artifact_manifest_hash: str
    slot_root_path: Path
    slot_manifest_path: Path
    slot_manifest: ArtifactManifestDocumentV2

    def __post_init__(self) -> None:
        """
        Validate that the resolved slot-pinned context stays internally aligned.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Runtime bootstrap must pin slot identity once and then reuse explicit manifest-driven
            paths without directory scanning or hot-path hash recomputation.
        Raises:
            ValueError: If slot identity or explicit manifest/root paths drift from each other.
        Side Effects:
            Normalizes validated literals to canonical values.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
          - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
        """
        object.__setattr__(self, "artifact_slot", validate_artifact_slot_v2(self.artifact_slot))
        object.__setattr__(
            self,
            "slot_generation",
            validate_current_pointer_slot_generation_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "artifact_asof_date",
            validate_current_pointer_asof_date_v2(self.artifact_asof_date),
        )
        object.__setattr__(
            self,
            "artifact_manifest_hash",
            validate_current_pointer_manifest_sha256_v2(self.artifact_manifest_hash),
        )
        if self.slot_manifest_path != self.slot_manifest.path:
            raise ValueError(
                "ArtifactSlotPinnedRuntimeContextV2.slot_manifest_path must equal "
                f"slot_manifest.path; got {self.slot_manifest_path!r} and "
                f"{self.slot_manifest.path!r}"
            )
        if self.slot_root_path != self.slot_manifest.path.parent:
            raise ValueError(
                "ArtifactSlotPinnedRuntimeContextV2.slot_root_path must equal "
                f"slot_manifest.path.parent; got {self.slot_root_path!r} and "
                f"{self.slot_manifest.path.parent!r}"
            )
        if self.slot_manifest.identity != self.coordinates:
            raise ValueError(
                "ArtifactSlotPinnedRuntimeContextV2.slot_manifest.identity must match "
                f"coordinates; got {self.slot_manifest.identity!r}, expected "
                f"{self.coordinates!r}"
            )
        if self.slot_manifest.slot != self.artifact_slot:
            raise ValueError(
                "ArtifactSlotPinnedRuntimeContextV2.slot_manifest.slot must match "
                f"artifact_slot; got {self.slot_manifest.slot!r}, expected "
                f"{self.artifact_slot!r}"
            )
        if self.slot_manifest.slot_generation != self.slot_generation:
            raise ValueError(
                "ArtifactSlotPinnedRuntimeContextV2.slot_manifest.slot_generation must match "
                f"slot_generation; got {self.slot_manifest.slot_generation!r}, expected "
                f"{self.slot_generation!r}"
            )
        if self.slot_manifest.asof_date != self.artifact_asof_date:
            raise ValueError(
                "ArtifactSlotPinnedRuntimeContextV2.slot_manifest.asof_date must match "
                f"artifact_asof_date; got {self.slot_manifest.asof_date!r}, expected "
                f"{self.artifact_asof_date!r}"
            )


@dataclass(frozen=True, slots=True)
class ArtifactPriceArraysV2:
    """
    Memory-mapped price family loaded from one explicit `prices/<tf>` contract.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    timeframe: str
    manifest: ArtifactPriceTimeframeManifestV2
    open_time: np.ndarray
    close_time: np.ndarray
    ohlcv: np.ndarray


@dataclass(frozen=True, slots=True)
class ArtifactMappingArraysV2:
    """
    Memory-mapped timeframe mapping family loaded from one explicit `mappings/<tf>` contract.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    timeframe: str
    manifest: ArtifactMappingTimeframeManifestV2
    bar_open_1m_idx: np.ndarray
    bar_close_1m_idx: np.ndarray


@dataclass(frozen=True, slots=True)
class ArtifactHitTimesArraysV2:
    """
    Memory-mapped strict `hit_times/1m` arrays reused by future runtime kernels.

    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    manifest: ArtifactHitTimesManifestDocumentV2
    tp_values: np.ndarray
    sl_values: np.ndarray
    long_tp: np.ndarray
    long_sl: np.ndarray
    short_tp: np.ndarray
    short_sl: np.ndarray


@dataclass(frozen=True, slots=True)
class ArtifactSignalMatrixV2:
    """
    Memory-mapped signal matrix loaded from one explicit `signals/<tf>/<indicator_id>` family.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """

    timeframe: str
    indicator_id: str
    manifest: ArtifactSignalManifestDocumentV2
    matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class StageACompactTradeV2:
    """
    Deterministic Stage A compact trade entry built without Stage B risk exits.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py
    """

    entry_signal_idx: int
    entry_exec_idx: int
    direction: int
    sig_exit_signal_idx: int | None
    sig_exit_exec_idx: int

    def __post_init__(self) -> None:
        """
        Validate one compact Stage A trade payload against deterministic kernel invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `entry_exec_idx` and `sig_exit_exec_idx` are rebased to the local execution timeline
            of the current run, while `sig_exit_exec_idx == sentinel_index` denotes no signal exit.
        Raises:
            ValueError: If indexes are negative, direction is unsupported, or signal-exit order
                is invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
          - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
        """
        if self.entry_signal_idx < 0:
            raise ValueError("StageACompactTradeV2.entry_signal_idx must be >= 0")
        if self.entry_exec_idx < 0:
            raise ValueError("StageACompactTradeV2.entry_exec_idx must be >= 0")
        if self.direction not in (-1, 1):
            raise ValueError("StageACompactTradeV2.direction must be -1 or 1")
        if (
            self.sig_exit_signal_idx is not None
            and self.sig_exit_signal_idx < self.entry_signal_idx
        ):
            raise ValueError(
                "StageACompactTradeV2.sig_exit_signal_idx must be >= entry_signal_idx"
            )
        if self.sig_exit_exec_idx < self.entry_exec_idx:
            raise ValueError("StageACompactTradeV2.sig_exit_exec_idx must be >= entry_exec_idx")


@dataclass(frozen=True, slots=True)
class StageANoRiskMetricsV2:
    """
    Deterministic no-risk Stage A metrics used for shortlist ranking and chunked processing.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
    """

    total_return_pct: float
    max_drawdown_pct: float
    return_over_max_drawdown: float
    profit_factor: float
    trade_count: int
    sharpe_trades: float
    win_rate_pct: float
    avg_trade_ret_pct: float
    avg_trade_exec_bars: float
    exposure_pct: float

    def __post_init__(self) -> None:
        """
        Validate Stage A no-risk metric scalars produced by shortlist kernels.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Metrics are deterministic numeric scalars and may use `inf` for no-loss /
            no-drawdown edge cases needed by ranking.
        Raises:
            ValueError: If one metric is non-numeric or `trade_count` is negative.
        Side Effects:
            Normalizes numeric fields to builtin `float`/`int`.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-v2-benchmarks.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py
          - tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py
        """
        numeric_fields = (
            "total_return_pct",
            "max_drawdown_pct",
            "return_over_max_drawdown",
            "profit_factor",
            "sharpe_trades",
            "win_rate_pct",
            "avg_trade_ret_pct",
            "avg_trade_exec_bars",
            "exposure_pct",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"StageANoRiskMetricsV2.{field_name} must be numeric")
            object.__setattr__(self, field_name, float(value))
        if self.trade_count < 0:
            raise ValueError("StageANoRiskMetricsV2.trade_count must be >= 0")


@dataclass(frozen=True, slots=True)
class StageBHitTimesSliceV2:
    """
    Local execution-window slice of strict `1m hit-times` tables used by Stage B kernels.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """

    tp_values: np.ndarray
    sl_values: np.ndarray
    long_tp: np.ndarray
    long_sl: np.ndarray
    short_tp: np.ndarray
    short_sl: np.ndarray
    sentinel_index: int

    def __post_init__(self) -> None:
        """
        Validate local Stage B hit-times slice shapes against the shared sentinel contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Arrays are already rebased into local execution coordinates by runtime helpers.
        Raises:
            ValueError: If one array shape or sentinel-bound invariant is violated.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        if self.sentinel_index < 0:
            raise ValueError("StageBHitTimesSliceV2.sentinel_index must be >= 0")
        expected_time = self.sentinel_index
        if self.long_tp.shape != (self.tp_values.shape[0], expected_time):
            raise ValueError(
                "StageBHitTimesSliceV2.long_tp shape must match tp_values and sentinel"
            )
        if self.short_tp.shape != (self.tp_values.shape[0], expected_time):
            raise ValueError(
                "StageBHitTimesSliceV2.short_tp shape must match tp_values and sentinel"
            )
        if self.long_sl.shape != (self.sl_values.shape[0], expected_time):
            raise ValueError(
                "StageBHitTimesSliceV2.long_sl shape must match sl_values and sentinel"
            )
        if self.short_sl.shape != (self.sl_values.shape[0], expected_time):
            raise ValueError(
                "StageBHitTimesSliceV2.short_sl shape must match sl_values and sentinel"
            )


@dataclass(frozen=True, slots=True)
class StageBTradeExitV2:
    """
    Deterministic exact exit fact for one compact trade in Stage B `signal_tf + 1m_risk`.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """

    trade_index: int
    entry_exec_idx: int
    direction: int
    sig_exit_exec_idx: int
    exit_exec_idx: int
    exit_reason: StageBExitReasonLiteralV2
    gross_factor: float
    closed: bool

    def __post_init__(self) -> None:
        """
        Validate one exact Stage B exit payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `gross_factor` is pre-fee and `closed=False` implies `exit_reason='unclosed'`.
        Raises:
            ValueError: If indexes, direction, factor, or reason invariants are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
          - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
        """
        if self.trade_index < 0:
            raise ValueError("StageBTradeExitV2.trade_index must be >= 0")
        if self.entry_exec_idx < 0:
            raise ValueError("StageBTradeExitV2.entry_exec_idx must be >= 0")
        if self.sig_exit_exec_idx < self.entry_exec_idx:
            raise ValueError("StageBTradeExitV2.sig_exit_exec_idx must be >= entry_exec_idx")
        if self.exit_exec_idx < self.entry_exec_idx:
            raise ValueError("StageBTradeExitV2.exit_exec_idx must be >= entry_exec_idx")
        if self.direction not in (-1, 1):
            raise ValueError("StageBTradeExitV2.direction must be -1 or 1")
        if isinstance(self.gross_factor, bool) or not isinstance(self.gross_factor, int | float):
            raise ValueError("StageBTradeExitV2.gross_factor must be numeric")
        object.__setattr__(self, "gross_factor", float(self.gross_factor))
        if not self.closed and self.exit_reason != "unclosed":
            raise ValueError("StageBTradeExitV2.closed=False requires exit_reason='unclosed'")


@dataclass(frozen=True, slots=True)
class StageBFastSearchResultV2:
    """
    Fast TP/SL search output over shipped `1m hit-times` for one compact trade list.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
    """

    total_return_pct: np.ndarray
    best_tp_index: int
    best_sl_index: int
    best_total_return_pct: float

    def __post_init__(self) -> None:
        """
        Validate one fast-search return matrix and the selected best-cell coordinates.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Matrix shape is `[n_tp, n_sl]` and best indexes point inside that matrix.
        Raises:
            ValueError: If the matrix is not 2D or one best-cell coordinate is out of range.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
          - tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
        """
        if self.total_return_pct.ndim != 2:
            raise ValueError("StageBFastSearchResultV2.total_return_pct must be a 2D array")
        rows, cols = self.total_return_pct.shape
        if rows == 0 or cols == 0:
            raise ValueError("StageBFastSearchResultV2.total_return_pct must not be empty")
        if self.best_tp_index < 0 or self.best_tp_index >= rows:
            raise ValueError("StageBFastSearchResultV2.best_tp_index is out of range")
        if self.best_sl_index < 0 or self.best_sl_index >= cols:
            raise ValueError("StageBFastSearchResultV2.best_sl_index is out of range")
        if isinstance(self.best_total_return_pct, bool) or not isinstance(
            self.best_total_return_pct,
            int | float,
        ):
            raise ValueError("StageBFastSearchResultV2.best_total_return_pct must be numeric")
        object.__setattr__(self, "best_total_return_pct", float(self.best_total_return_pct))


@dataclass(frozen=True, slots=True)
class StageBReplayPayloadV2:
    """
    Exact replay payload for one selected Stage B TP/SL cell over compact trades.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
    """

    tp_index: int | None
    sl_index: int | None
    sentinel_index: int
    close_on_end: bool
    trade_exits: tuple[StageBTradeExitV2, ...]

    def __post_init__(self) -> None:
        """
        Validate the exact replay payload for one selected risk cell.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Disabled TP/SL axes are represented as `None` indexes.
        Raises:
            ValueError: If the sentinel or selected cell indexes are invalid.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py
          - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
        """
        if self.sentinel_index < 0:
            raise ValueError("StageBReplayPayloadV2.sentinel_index must be >= 0")
        if self.tp_index is not None and self.tp_index < 0:
            raise ValueError("StageBReplayPayloadV2.tp_index must be >= 0 when set")
        if self.sl_index is not None and self.sl_index < 0:
            raise ValueError("StageBReplayPayloadV2.sl_index must be >= 0 when set")


@dataclass(frozen=True, slots=True)
class StageBMetricsV2:
    """
    Deterministic Stage B metrics computed from one exact replay payload.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-v2-benchmarks.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
      - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
    """

    total_return_pct: float
    max_drawdown_pct: float
    return_over_max_drawdown: float
    profit_factor: float
    trade_count: int
    win_rate_pct: float
    avg_trade_ret_pct: float
    avg_trade_exec_bars: float
    exposure_pct: float
    sharpe_trades: float

    def __post_init__(self) -> None:
        """
        Validate Stage B exact metrics used by ranking and summary payloads.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Metrics are numeric scalars and may legitimately contain `inf`.
        Raises:
            ValueError: If one field is non-numeric or `trade_count` is negative.
        Side Effects:
            Normalizes numeric fields to builtin `float`/`int`.
        Docs:
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
          - docs/architecture/backtest/backtest-v2-benchmarks.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/metrics_kernel.py
          - tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py
        """
        numeric_fields = (
            "total_return_pct",
            "max_drawdown_pct",
            "return_over_max_drawdown",
            "profit_factor",
            "win_rate_pct",
            "avg_trade_ret_pct",
            "avg_trade_exec_bars",
            "exposure_pct",
            "sharpe_trades",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"StageBMetricsV2.{field_name} must be numeric")
            object.__setattr__(self, field_name, float(value))
        if self.trade_count < 0:
            raise ValueError("StageBMetricsV2.trade_count must be >= 0")


@dataclass(frozen=True, slots=True)
class ArtifactPrecomputeRuntimeSettingsV2:
    """
    Minimal service-layer runtime settings required by R3-03/R4-03/R5-01 precompute orchestration.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """

    price_tail_bars_1m: int
    mapping_tail_bars_1m: int
    signal_tail_bars_1m: int
    hit_times_tp_levels_pct: tuple[float, ...]
    hit_times_sl_levels_pct: tuple[float, ...]
    config_sha256: str
    signal_artifacts: tuple[ArtifactSignalValidationSpecV2, ...] = ()
    max_signal_rows_per_artifact: int = 1_000_000
    max_hit_times_cells: int = 1_000_000

    def __post_init__(self) -> None:
        """
        Validate strict precompute settings derived from artifact runtime config.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Adapter/wiring code translates the full artifact runtime config into this minimal
            service DTO for price+mapping+signals+hit-times orchestration.
        Raises:
            ValueError: If the tail lookback or config hash violates strict publish contracts.
        Side Effects:
            Replaces explicit signal targets with deterministic canonical ordering.
        Docs:
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
          - docs/architecture/backtest/backtest-artifact-store-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        object.__setattr__(
            self,
            "price_tail_bars_1m",
            validate_positive_manifest_int_v2(self.price_tail_bars_1m),
        )
        object.__setattr__(
            self,
            "mapping_tail_bars_1m",
            validate_positive_manifest_int_v2(self.mapping_tail_bars_1m),
        )
        object.__setattr__(
            self,
            "signal_tail_bars_1m",
            validate_positive_manifest_int_v2(self.signal_tail_bars_1m),
        )
        object.__setattr__(
            self,
            "hit_times_tp_levels_pct",
            _normalize_positive_float_grid_v2(
                values=self.hit_times_tp_levels_pct,
                field_name="ArtifactPrecomputeRuntimeSettingsV2.hit_times_tp_levels_pct",
            ),
        )
        object.__setattr__(
            self,
            "hit_times_sl_levels_pct",
            _normalize_positive_float_grid_v2(
                values=self.hit_times_sl_levels_pct,
                field_name="ArtifactPrecomputeRuntimeSettingsV2.hit_times_sl_levels_pct",
            ),
        )
        object.__setattr__(
            self,
            "config_sha256",
            validate_current_pointer_manifest_sha256_v2(self.config_sha256),
        )
        object.__setattr__(
            self,
            "signal_artifacts",
            _sorted_signal_validation_specs_v2(self.signal_artifacts),
        )
        object.__setattr__(
            self,
            "max_signal_rows_per_artifact",
            validate_positive_manifest_int_v2(self.max_signal_rows_per_artifact),
        )
        object.__setattr__(
            self,
            "max_hit_times_cells",
            validate_positive_manifest_int_v2(self.max_hit_times_cells),
        )


def _normalize_positive_float_grid_v2(
    *,
    values: tuple[float, ...],
    field_name: str,
) -> tuple[float, ...]:
    """
    Normalize one positive ascending float grid used by hit-times materialization.

    Args:
        values: Candidate percentage-grid values.
        field_name: Stable field label used in fail-fast errors.
    Returns:
        tuple[float, ...]: Ascending unique positive float grid.
    Assumptions:
        Runtime settings carry human-percent values and must stay deterministic.
    Raises:
        ValueError: If the grid is empty, non-positive, or contains duplicates.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
    """
    if len(values) == 0:
        raise ValueError(f"{field_name} must contain at least one value")
    normalized = tuple(sorted(float(value) for value in values))
    if normalized[0] <= 0.0:
        raise ValueError(f"{field_name} values must be > 0, got {normalized!r}")
    if any(left == right for left, right in zip(normalized, normalized[1:])):
        raise ValueError(f"{field_name} must not contain duplicate values; got {normalized!r}")
    return normalized


class BacktestArtifactPathResolverV2(Protocol):
    """
    Port for deterministic filesystem path resolution in artifact store v2.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    def ordered_slots(self) -> tuple[ArtifactSlotLiteralV2, ...]:
        """Return the fixed slot order used by runtime-facing callers."""
        ...

    def symbol_root(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """Resolve the `<exchange>/<market_type>/<symbol>/` root."""
        ...

    def current_pointer_path(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """Resolve the `current.yaml` path for one symbol root."""
        ...

    def slot_root(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve the path of one fixed artifact slot root."""
        ...

    def slot_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve the `manifest.yaml` path for one fixed artifact slot."""
        ...

    def price_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactPricePathsV2:
        """Resolve explicit price artifact paths for one timeframe."""
        ...

    def signal_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalPathsV2:
        """Resolve explicit signal artifact paths for one indicator and timeframe."""
        ...

    def mapping_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactMappingPathsV2:
        """Resolve explicit bar mapping artifact paths for one timeframe."""
        ...

    def hit_times_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve the fixed `hit_times/1m/manifest.yaml` path."""
        ...

    def hit_times_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactHitTimesPathsV2:
        """Resolve explicit hit-times artifact paths without touching disk."""
        ...


class BacktestArtifactLoaderV2(Protocol):
    """
    Port for explicit-path metadata reads in artifact store v2.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    def load_current_pointer(self, coordinates: ArtifactCoordinatesV2) -> ArtifactCurrentPointerV2:
        """Read one `current.yaml` document by deterministic coordinates."""
        ...

    def load_current_pointer_from_path(self, path: Path) -> ArtifactCurrentPointerV2:
        """Read one `current.yaml` document from an explicit path."""
        ...

    def load_slot_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactManifestDocumentV2:
        """Read one slot `manifest.yaml` document by deterministic coordinates."""
        ...

    def load_manifest_from_path(
        self,
        path: Path,
        *,
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactManifestDocumentV2:
        """Read one slot `manifest.yaml` document from an explicit path."""
        ...

    def load_signal_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalManifestDocumentV2:
        """Read one per-indicator signal manifest by deterministic coordinates."""
        ...

    def load_signal_manifest_from_path(
        self,
        path: Path,
        *,
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactSignalManifestDocumentV2:
        """Read one per-indicator signal manifest from an explicit path."""
        ...

    def load_hit_times_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactHitTimesManifestDocumentV2:
        """Read the fixed `hit_times/1m/manifest.yaml` by deterministic coordinates."""
        ...

    def load_hit_times_manifest_from_path(
        self,
        path: Path,
        *,
        slot: ArtifactSlotLiteralV2,
    ) -> ArtifactHitTimesManifestDocumentV2:
        """Read one `hit_times/1m/manifest.yaml` document from an explicit path."""
        ...

    def load_active_slot_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
    ) -> ArtifactManifestDocumentV2:
        """Read the active slot manifest by first resolving `current.yaml`."""
        ...

    def resolve_current_pointer_path(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """Resolve the `current.yaml` path without touching disk."""
        ...

    def resolve_slot_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve one slot `manifest.yaml` path without touching disk."""
        ...

    def resolve_price_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactPricePathsV2:
        """Resolve price artifact paths without touching disk."""
        ...

    def resolve_signal_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalPathsV2:
        """Resolve signal artifact paths without touching disk."""
        ...

    def resolve_mapping_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactMappingPathsV2:
        """Resolve mapping artifact paths without touching disk."""
        ...

    def resolve_hit_times_manifest_path(
        self, coordinates: ArtifactCoordinatesV2, slot: str
    ) -> Path:
        """Resolve the `hit_times/1m/manifest.yaml` path without touching disk."""
        ...

    def resolve_hit_times_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactHitTimesPathsV2:
        """Resolve hit-times artifact paths without touching disk."""
        ...


class BacktestArtifactSlotResolverV2(Protocol):
    """
    Port for shared slot-pinned runtime bootstrap over strict artifact identities.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    def resolve_active_context(
        self,
        coordinates: ArtifactCoordinatesV2,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        """Resolve the active slot-pinned context from strict `current.yaml`."""
        ...

    def resolve_pinned_context(
        self,
        coordinates: ArtifactCoordinatesV2,
        pinned_identity: ArtifactPinnedIdentityV2,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        """Resolve one slot-pinned context from persisted pin metadata only."""
        ...


class BacktestPriceArraysLoaderV2(Protocol):
    """
    Port for strict mmap-based price, mapping, and `hit_times/1m` runtime loading.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """

    def load_price_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactPriceArraysV2:
        """Load one explicit `prices/<tf>` family via `np.load(..., mmap_mode='r')`."""
        ...

    def load_mapping_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactMappingArraysV2:
        """Load one explicit `mappings/<tf>` family via `np.load(..., mmap_mode='r')`."""
        ...

    def load_hit_times_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
    ) -> ArtifactHitTimesArraysV2:
        """Load strict `hit_times/1m` arrays via explicit manifest-driven paths."""
        ...


class BacktestSignalMatrixLoaderV2(Protocol):
    """
    Port for strict mmap-based signal-matrix loading and deterministic subset row reads.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
    """

    def load_signal_matrix(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalMatrixV2:
        """Load one explicit signal matrix via `np.load(..., mmap_mode='r')`."""
        ...

    def load_signal_rows(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
        row_selection: slice | tuple[int, ...],
    ) -> np.ndarray:
        """Load one deterministic subset of signal rows without runtime discovery."""
        ...


class BacktestArtifactCurrentPointerWriterV2(Protocol):
    """
    Port for deterministic atomic `current.yaml` replacement in artifact store v2.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """

    def write_current_pointer_atomically(
        self,
        coordinates: ArtifactCoordinatesV2,
        pointer: ArtifactCurrentPointerV2,
    ) -> Path:
        """Atomically replace one symbol-root `current.yaml` with deterministic payload bytes."""
        ...


def _validate_safe_path_token_v2(token: str, *, field_name: str) -> str:
    """
    Enforce the shared filesystem-safe token rules used by R2-01 path builders.

    Args:
        token: Candidate path token.
        field_name: Human-readable field name used in stable error messages.
    Returns:
        str: The original token when valid.
    Assumptions:
        Tokens are stored and reused verbatim, so implicit normalization is forbidden.
    Raises:
        ValueError: If the token is empty, contains whitespace, separators, or traversal.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    if token == "" or token.strip() == "":
        raise ValueError(
            f"artifact {field_name} must be a non-empty safe token without whitespace, "
            f"separators, or '..'; got {token!r}"
        )
    if any(character.isspace() for character in token):
        raise ValueError(
            f"artifact {field_name} must be a non-empty safe token without whitespace, "
            f"separators, or '..'; got {token!r}"
        )
    if token in {".", ".."} or ".." in token or "/" in token or "\\" in token or "\x00" in token:
        raise ValueError(
            f"artifact {field_name} must be a non-empty safe token without whitespace, "
            f"separators, or '..'; got {token!r}"
        )
    return token


def _validate_allowed_literal_v2(
    *,
    value: str,
    field_name: str,
    allowed_literals: tuple[str, ...],
) -> None:
    """
    Enforce one fixed literal set with deterministic error messages.

    Args:
        value: Candidate literal to validate.
        field_name: Human-readable field name used in error messages.
        allowed_literals: Canonical ordered literal set.
    Returns:
        None.
    Assumptions:
        Allowed literals are already ordered for deterministic diagnostics.
    Raises:
        ValueError: If the candidate literal is not present in the allowed set.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    if value not in allowed_literals:
        raise ValueError(f"artifact {field_name} must be one of {allowed_literals}; got {value!r}")


def _sorted_unique_timeframes_v2(
    *,
    values: tuple[str, ...],
    allowed_literals: tuple[str, ...],
    field_name: str,
    validator: Callable[[str], str],
) -> tuple[str, ...]:
    """
    Validate and deterministically order one timeframe tuple against the canonical contract.

    Args:
        values: Candidate timeframe tuple.
        allowed_literals: Canonical ordered timeframe literals for this scope.
        field_name: Human-readable field name used in error messages.
        validator: Scope-specific timeframe validator callable.
    Returns:
        tuple[str, ...]: Deterministically ordered unique timeframe tuple.
    Assumptions:
        Validation plans must use canonical ordering even if callers provide arbitrary tuples.
    Raises:
        ValueError: If one timeframe is invalid or appears more than once.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    seen: set[str] = set()
    validated_values: list[str] = []
    for raw_value in values:
        validated_value = validator(raw_value)
        if validated_value in seen:
            raise ValueError(
                f"artifact validation field '{field_name}' contains duplicate "
                f"{validated_value!r}"
            )
        seen.add(validated_value)
        validated_values.append(validated_value)
    ordered_values: list[str] = []
    for allowed_literal in allowed_literals:
        if allowed_literal in seen:
            ordered_values.append(allowed_literal)
    return tuple(ordered_values)


def validate_manifest_schema_version_v2(
    schema_version: int,
    *,
    field_name: str,
    expected_schema_version: int,
) -> int:
    """
    Validate one strict manifest schema version against a fixed explicit literal.

    Args:
        schema_version: Candidate schema version scalar.
        field_name: Stable field label used in deterministic error messages.
        expected_schema_version: Single supported schema version for this manifest type.
    Returns:
        int: Validated schema version literal.
    Assumptions:
        R2-03 forbids dynamic schema discovery and supports one explicit version per manifest type.
    Raises:
        ValueError: If the schema version is not the expected integer literal.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"{field_name} must be int")
    if schema_version != expected_schema_version:
        raise ValueError(f"{field_name} must be {expected_schema_version}; got {schema_version!r}")
    return schema_version


def validate_manifest_kind_v2(kind: str, *, expected_kind: str) -> str:
    """
    Validate one strict manifest-kind literal against the fixed contract.

    Args:
        kind: Candidate manifest kind literal.
        expected_kind: Single supported literal for the manifest type.
    Returns:
        str: Validated manifest kind literal.
    Assumptions:
        Manifest type is explicit and must never be inferred dynamically from payload shape.
    Raises:
        ValueError: If the manifest kind differs from the expected literal.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    validate_manifest_text_literal_v2(kind, field_name="manifest_kind")
    if kind != expected_kind:
        raise ValueError(f"manifest_kind must be {expected_kind!r}; got {kind!r}")
    return kind


def validate_manifest_text_literal_v2(value: str, *, field_name: str) -> str:
    """
    Validate one non-empty manifest text literal without implicit normalization.

    Args:
        value: Candidate text literal.
        field_name: Stable field label used in deterministic error messages.
    Returns:
        str: Original validated text literal.
    Assumptions:
        Strict manifest contracts reject empty/whitespace-only text fields.
    Raises:
        ValueError: If the text literal is empty or contains only whitespace.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    if value.strip() == "":
        raise ValueError(f"{field_name} must be non-empty")
    return value


def validate_manifest_integer_literal_v2(value: int, *, field_name: str) -> int:
    """
    Validate one manifest integer literal without coercion.

    Args:
        value: Candidate integer scalar.
        field_name: Stable field label used in deterministic error messages.
    Returns:
        int: Original validated integer.
    Assumptions:
        Manifest numeric fields are already materialized as integers before serialization.
    Raises:
        ValueError: If the value is not an integer literal.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be int")
    return value


def validate_positive_manifest_int_v2(value: int) -> int:
    """
    Validate one manifest integer field as strictly positive.

    Args:
        value: Candidate integer scalar.
    Returns:
        int: Validated positive integer.
    Assumptions:
        Counts and slot generations in strict manifests are positive integers.
    Raises:
        ValueError: If the value is not a positive integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    validated_value = validate_manifest_integer_literal_v2(value, field_name="manifest integer")
    if validated_value <= 0:
        raise ValueError("manifest integer must be > 0")
    return validated_value


def validate_non_negative_manifest_int_v2(value: int) -> int:
    """
    Validate one manifest integer field as zero-or-positive.

    Args:
        value: Candidate integer scalar.
    Returns:
        int: Validated non-negative integer.
    Assumptions:
        Sentinel indexes may equal zero for degenerate test fixtures but never be negative.
    Raises:
        ValueError: If the value is negative or not an integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    validated_value = validate_manifest_integer_literal_v2(value, field_name="manifest integer")
    if validated_value < 0:
        raise ValueError("manifest integer must be >= 0")
    return validated_value


def validate_relative_artifact_path_v2(path_literal: str) -> str:
    """
    Validate one slot-relative artifact path literal stored inside strict manifests.

    Args:
        path_literal: Candidate slot-relative path string.
    Returns:
        str: Original validated relative path literal.
    Assumptions:
        Manifest paths are serialized relative to the slot root and must not require cleanup.
    Raises:
        ValueError: If the path is empty, absolute, or contains traversal segments.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    validated_literal = validate_manifest_text_literal_v2(
        path_literal,
        field_name="manifest path",
    )
    if validated_literal.startswith("/") or validated_literal.startswith("\\"):
        raise ValueError("manifest path must be slot-relative")
    if "\\" in validated_literal or "\x00" in validated_literal:
        raise ValueError("manifest path must not contain backslashes or NUL")
    parts = tuple(part for part in Path(validated_literal).parts if part not in {"", "."})
    if len(parts) == 0:
        raise ValueError("manifest path must not be empty")
    for part in parts:
        if part == "..":
            raise ValueError("manifest path must not contain '..'")
        _validate_safe_path_token_v2(token=part, field_name="manifest path segment")
    return "/".join(parts)


def validate_artifact_dtype_literal_v2(dtype_literal: str) -> str:
    """
    Validate one explicit dtype literal used by strict manifest array metadata.

    Args:
        dtype_literal: Candidate dtype literal.
    Returns:
        str: Validated dtype literal.
    Assumptions:
        R2-03 uses a fixed small set of explicit numeric dtype literals.
    Raises:
        ValueError: If the dtype literal is not supported by manifest contracts.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    allowed_dtypes = (
        ARTIFACT_SIGNAL_DTYPE_LITERAL_V2,
        ARTIFACT_PRICE_TIME_DTYPE_LITERAL_V2,
        ARTIFACT_PRICE_OHLCV_DTYPE_LITERAL_V2,
        ARTIFACT_MAPPING_DTYPE_LITERAL_V2,
        ARTIFACT_HIT_TIMES_GRID_DTYPE_LITERAL_V2,
        ARTIFACT_HIT_TIMES_TABLE_DTYPE_LITERAL_V2,
    )
    validate_manifest_text_literal_v2(dtype_literal, field_name="array dtype")
    _validate_allowed_literal_v2(
        value=dtype_literal,
        field_name="array dtype",
        allowed_literals=allowed_dtypes,
    )
    return dtype_literal


def validate_artifact_shape_v2(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Validate one explicit shape tuple used by strict manifest array metadata.

    Args:
        shape: Candidate shape tuple.
    Returns:
        tuple[int, ...]: Validated shape tuple with strictly positive dimensions.
    Assumptions:
        Published artifact arrays are non-empty and declare every dimension explicitly.
    Raises:
        ValueError: If shape is not a non-empty tuple of positive integers.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if not isinstance(shape, tuple):
        raise ValueError("array shape must be tuple")
    if len(shape) == 0:
        raise ValueError("array shape must not be empty")
    validated_dimensions: list[int] = []
    for index, value in enumerate(shape):
        dimension = validate_manifest_integer_literal_v2(value, field_name=f"shape[{index}]")
        if dimension <= 0:
            raise ValueError(f"shape[{index}] must be > 0")
        validated_dimensions.append(dimension)
    return tuple(validated_dimensions)


def validate_artifact_axis_order_v2(axis_order: tuple[str, ...]) -> tuple[str, ...]:
    """
    Validate one explicit axis-order tuple used by strict manifest array metadata.

    Args:
        axis_order: Candidate axis-order tuple.
    Returns:
        tuple[str, ...]: Validated axis-order tuple without duplicates.
    Assumptions:
        Axis-order literals are explicit and deterministic for every artifact family.
    Raises:
        ValueError: If axis order is empty, non-string, or contains duplicates.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if not isinstance(axis_order, tuple):
        raise ValueError("array axis_order must be tuple")
    if len(axis_order) == 0:
        raise ValueError("array axis_order must not be empty")
    seen: set[str] = set()
    validated_axes: list[str] = []
    for axis in axis_order:
        validated_axis = validate_manifest_text_literal_v2(axis, field_name="axis_order value")
        if validated_axis in seen:
            raise ValueError(f"array axis_order contains duplicate {validated_axis!r}")
        seen.add(validated_axis)
        validated_axes.append(validated_axis)
    return tuple(validated_axes)


def validate_signal_value_set_v2(value_set: tuple[int, ...]) -> tuple[int, int, int]:
    """
    Validate the fixed signal value set contract `{-1, 0, 1}`.

    Args:
        value_set: Candidate ordered signal value tuple.
    Returns:
        tuple[int, int, int]: Validated canonical signal value set.
    Assumptions:
        Engine v2 stores signals only as `SHORT=-1`, `NEUTRAL=0`, `LONG=1`.
    Raises:
        ValueError: If the value set differs from the fixed ordered contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    if not isinstance(value_set, tuple):
        raise ValueError("signal value set must be tuple")
    validated_values: list[int] = []
    for index, value in enumerate(value_set):
        validated_values.append(
            validate_manifest_integer_literal_v2(value, field_name=f"signal_value_set[{index}]")
        )
    validated_tuple = tuple(validated_values)
    if validated_tuple != ARTIFACT_SIGNAL_VALUE_SET_V2:
        raise ValueError(
            "signal value set must be exactly "
            f"{ARTIFACT_SIGNAL_VALUE_SET_V2}; got {validated_tuple!r}"
        )
    return ARTIFACT_SIGNAL_VALUE_SET_V2


def validate_hit_times_monotonicity_literal_v2(monotonicity: str) -> str:
    """
    Validate the declared monotonicity literal for hit-times lookup tables.

    Args:
        monotonicity: Candidate monotonicity literal.
    Returns:
        str: Validated monotonicity literal.
    Assumptions:
        R2-03 supports one explicit hit-times monotonicity contract.
    Raises:
        ValueError: If the monotonicity literal is unsupported.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    validate_manifest_text_literal_v2(monotonicity, field_name="hit_times monotonicity")
    if monotonicity != ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2:
        raise ValueError(
            "hit_times monotonicity must be "
            f"{ARTIFACT_HIT_TIMES_TABLE_MONOTONICITY_LITERAL_V2!r}; got {monotonicity!r}"
        )
    return monotonicity


def _validate_exact_mapping_keys_v2(
    *,
    payload: Mapping[str, Any],
    required_keys: tuple[str, ...],
    path: Path,
) -> None:
    """
    Validate that one manifest payload contains exactly the required keys and no drift.

    Args:
        payload: Parsed YAML mapping payload.
        required_keys: Canonical ordered required key tuple.
        path: Source manifest path used in deterministic error messages.
    Returns:
        None.
    Assumptions:
        Strict manifest schemas reject both missing keys and unsupported extra keys.
    Raises:
        ValueError: If payload keys differ from the fixed schema.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    raw_keys = tuple(sorted(payload.keys()))
    expected_keys = tuple(sorted(required_keys))
    if raw_keys != expected_keys:
        missing_keys = tuple(key for key in required_keys if key not in payload)
        extra_keys = tuple(key for key in raw_keys if key not in required_keys)
        details: list[str] = []
        if len(missing_keys) > 0:
            details.append(f"missing keys {missing_keys}")
        if len(extra_keys) > 0:
            details.append(f"unexpected keys {extra_keys}")
        raise ValueError(
            f"{path} must contain exactly keys {required_keys}"
            + (f"; {'; '.join(details)}" if len(details) > 0 else "")
        )


def _sorted_unique_indicator_ids_v2(
    *,
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    """
    Validate and deterministically order an indicator-id tuple.

    Args:
        values: Candidate indicator-id tuple.
        field_name: Stable field label used in deterministic error messages.
    Returns:
        tuple[str, ...]: Lexicographically ordered unique validated indicator ids.
    Assumptions:
        Indicator-id ordering must be stable regardless of serialization order in YAML.
    Raises:
        ValueError: If one indicator id is invalid or duplicated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    seen: set[str] = set()
    validated_values: list[str] = []
    for raw_value in values:
        validated_value = validate_indicator_id_v2(raw_value)
        if validated_value in seen:
            raise ValueError(
                f"artifact validation field '{field_name}' contains duplicate "
                f"{validated_value!r}"
            )
        seen.add(validated_value)
        validated_values.append(validated_value)
    return tuple(sorted(validated_values))


def _sorted_signal_catalog_entries_v2(
    values: tuple[ArtifactSignalCatalogEntryV2, ...],
) -> tuple[ArtifactSignalCatalogEntryV2, ...]:
    """
    Validate and deterministically order root-manifest signal catalog entries.

    Args:
        values: Candidate signal catalog entry tuple.
    Returns:
        tuple[ArtifactSignalCatalogEntryV2, ...]: Canonically ordered unique catalog entries.
    Assumptions:
        Catalog ordering is deterministic by timeframe contract and indicator id.
    Raises:
        ValueError: If one `(timeframe, indicator_id)` pair is duplicated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    seen: set[tuple[str, str]] = set()
    validated_values: list[ArtifactSignalCatalogEntryV2] = []
    for item in values:
        validated_item = ArtifactSignalCatalogEntryV2(
            timeframe=item.timeframe,
            indicator_id=item.indicator_id,
            manifest_path=item.manifest_path,
            manifest_sha256=item.manifest_sha256,
        )
        identity = (validated_item.timeframe, validated_item.indicator_id)
        if identity in seen:
            raise ValueError(
                "root manifest field 'signals.manifests' contains duplicate " f"{identity!r}"
            )
        seen.add(identity)
        validated_values.append(validated_item)

    timeframe_order = {
        literal: index for index, literal in enumerate(ARTIFACT_SIGNAL_TIMEFRAMES_V2)
    }
    ordered_values = sorted(
        validated_values,
        key=lambda item: (timeframe_order[item.timeframe], item.indicator_id),
    )
    return tuple(ordered_values)


def _sorted_signal_validation_specs_v2(
    values: tuple[ArtifactSignalValidationSpecV2, ...],
) -> tuple[ArtifactSignalValidationSpecV2, ...]:
    """
    Validate and deterministically order explicit signal validation targets.

    Args:
        values: Candidate signal validation tuple.
    Returns:
        tuple[ArtifactSignalValidationSpecV2, ...]: Canonically ordered unique signal targets.
    Assumptions:
        Signal validation order is deterministic by timeframe contract and indicator id.
    Raises:
        ValueError: If one `(timeframe, indicator_id)` pair is duplicated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    seen: set[tuple[str, str]] = set()
    validated_values: list[ArtifactSignalValidationSpecV2] = []
    for item in values:
        validated_item = ArtifactSignalValidationSpecV2(
            timeframe=item.timeframe,
            indicator_id=item.indicator_id,
        )
        identity = (validated_item.timeframe, validated_item.indicator_id)
        if identity in seen:
            raise ValueError(
                "artifact validation field 'signal_artifacts' contains duplicate " f"{identity!r}"
            )
        seen.add(identity)
        validated_values.append(validated_item)

    timeframe_order = {
        literal: index for index, literal in enumerate(ARTIFACT_SIGNAL_TIMEFRAMES_V2)
    }
    ordered_values = sorted(
        validated_values,
        key=lambda item: (timeframe_order[item.timeframe], item.indicator_id),
    )
    return tuple(ordered_values)
