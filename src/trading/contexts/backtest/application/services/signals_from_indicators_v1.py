from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from trading.contexts.backtest.domain.value_objects import (
    AggregatedSignalsV1,
    IndicatorSignalsV1,
    SignalV1,
)
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorTensor
from trading.contexts.indicators.domain.entities import IndicatorId, Layout
from trading.contexts.indicators.domain.specifications import GridSpec

SignalRuleScalar = int | float | str | bool | None

_RULE_COMPARE_PRICE_TO_OUTPUT = "compare_price_to_output"
_RULE_THRESHOLD_BAND = "threshold_band"
_RULE_SIGN = "sign"
_RULE_DELTA_SIGN = "delta_sign"
_RULE_COMPARE_VOLUME_TO_OUTPUT = "compare_volume_to_output"
_RULE_CANDLE_BODY_DIRECTION = "candle_body_direction"
_RULE_PIVOT_EVENTS = "pivot_events"
_RULE_THRESHOLD_CENTERED = "threshold_centered"

SIGNAL_CODE_NEUTRAL_V1 = np.int8(0)
SIGNAL_CODE_LONG_V1 = np.int8(1)
SIGNAL_CODE_SHORT_V1 = np.int8(-1)


def _normalize_string_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    """
    Normalize tuple of identifiers into deduplicated lowercase tuple.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        values: Tuple with raw identifiers.
    Returns:
        tuple[str, ...]: Deduplicated normalized identifiers.
    Assumptions:
        Input tuple size is small and deterministic sorting is acceptable.
    Raises:
        ValueError: If one identifier is blank after normalization.
    Side Effects:
        None.
    """
    normalized: set[str] = set()
    for raw in values:
        normalized_value = raw.strip().lower()
        if not normalized_value:
            raise ValueError("Identifier tuples must not contain blank values")
        normalized.add(normalized_value)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class IndicatorSignalEvaluationInputV1:
    """
    Normalized input envelope for one indicator signal-evaluation call.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - src/trading/contexts/indicators/application/dto/indicator_tensor.py
    """

    indicator_id: str
    primary_output: np.ndarray
    indicator_inputs: Mapping[str, SignalRuleScalar] = field(default_factory=dict)
    signal_params: Mapping[str, SignalRuleScalar] = field(default_factory=dict)
    dependency_outputs: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and normalize one indicator signal-evaluation input payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `primary_output` and dependency arrays are one-dimensional and bar-aligned.
        Raises:
            ValueError: If identifier is blank, arrays are invalid, or dependency ids are blank.
        Side Effects:
            Normalizes identifier/mapping keys and converts arrays to contiguous float32.
        """
        normalized_indicator_id = self.indicator_id.strip().lower()
        if not normalized_indicator_id:
            raise ValueError("IndicatorSignalEvaluationInputV1.indicator_id must be non-empty")
        object.__setattr__(self, "indicator_id", normalized_indicator_id)
        object.__setattr__(
            self,
            "primary_output",
            _normalize_series_array(
                name="IndicatorSignalEvaluationInputV1.primary_output",
                values=self.primary_output,
            ),
        )
        object.__setattr__(
            self,
            "indicator_inputs",
            MappingProxyType(
                _normalize_scalar_mapping(
                    values=self.indicator_inputs,
                    field_name="IndicatorSignalEvaluationInputV1.indicator_inputs",
                )
            ),
        )
        object.__setattr__(
            self,
            "signal_params",
            MappingProxyType(
                _normalize_scalar_mapping(
                    values=self.signal_params,
                    field_name="IndicatorSignalEvaluationInputV1.signal_params",
                )
            ),
        )
        object.__setattr__(
            self,
            "dependency_outputs",
            MappingProxyType(
                _normalize_dependency_mapping(
                    values=self.dependency_outputs,
                    field_name="IndicatorSignalEvaluationInputV1.dependency_outputs",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class SignalRuleSpecV1:
    """
    Declarative rule-family binding for one indicator id in signal-engine v1.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - docs/architecture/indicators/indicators_formula.yaml
    """

    rule_family: str
    required_signal_params: tuple[str, ...] = ()
    required_dependency_ids: tuple[str, ...] = ()
    invert_price_comparison: bool = False
    threshold_center: float | None = None
    candle_body_min_param_name: str | None = None

    def __post_init__(self) -> None:
        """
        Validate declarative rule metadata invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Rule family names and parameter identifiers are stable string literals.
        Raises:
            ValueError: If family is blank or parameter/dependency names are invalid.
        Side Effects:
            Normalizes family and related identifiers into lowercase deterministic order.
        """
        normalized_family = self.rule_family.strip().lower()
        if not normalized_family:
            raise ValueError("SignalRuleSpecV1.rule_family must be non-empty")
        object.__setattr__(self, "rule_family", normalized_family)
        object.__setattr__(
            self,
            "required_signal_params",
            tuple(sorted(_normalize_string_tuple(self.required_signal_params))),
        )
        object.__setattr__(
            self,
            "required_dependency_ids",
            tuple(sorted(_normalize_string_tuple(self.required_dependency_ids))),
        )

        if self.threshold_center is not None and not np.isfinite(self.threshold_center):
            raise ValueError("SignalRuleSpecV1.threshold_center must be finite when provided")
        if self.candle_body_min_param_name is None:
            return
        normalized_body_param_name = self.candle_body_min_param_name.strip().lower()
        if not normalized_body_param_name:
            raise ValueError(
                "SignalRuleSpecV1.candle_body_min_param_name must be non-empty when provided"
            )
        object.__setattr__(self, "candle_body_min_param_name", normalized_body_param_name)


def _build_signal_rule_registry_v1() -> dict[str, SignalRuleSpecV1]:
    """
    Build indicator_id -> rule-family registry for signals-from-indicators v1.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        None.
    Returns:
        dict[str, SignalRuleSpecV1]: Deterministic sorted registry payload.
    Assumptions:
        Registry values mirror BKT-EPIC-03 rule-catalog semantics.
    Raises:
        ValueError: If one rule specification is invalid.
    Side Effects:
        None.
    """
    rule_threshold_band = SignalRuleSpecV1(
        rule_family=_RULE_THRESHOLD_BAND,
        required_signal_params=("long_threshold", "short_threshold"),
    )
    rule_sign = SignalRuleSpecV1(rule_family=_RULE_SIGN)
    rule_delta_sign = SignalRuleSpecV1(
        rule_family=_RULE_DELTA_SIGN,
        required_signal_params=("long_delta_periods", "short_delta_periods"),
    )
    rule_compare_price = SignalRuleSpecV1(rule_family=_RULE_COMPARE_PRICE_TO_OUTPUT)
    rule_compare_volume = SignalRuleSpecV1(rule_family=_RULE_COMPARE_VOLUME_TO_OUTPUT)
    rule_pivot_events = SignalRuleSpecV1(
        rule_family=_RULE_PIVOT_EVENTS,
        required_dependency_ids=("structure.pivot_high", "structure.pivot_low"),
    )

    entries: dict[str, SignalRuleSpecV1] = {
        "ma.dema": rule_compare_price,
        "ma.ema": rule_compare_price,
        "ma.hma": rule_compare_price,
        "ma.lwma": rule_compare_price,
        "ma.rma": rule_compare_price,
        "ma.sma": rule_compare_price,
        "ma.tema": rule_compare_price,
        "ma.vwma": rule_compare_price,
        "ma.wma": rule_compare_price,
        "ma.zlema": rule_compare_price,
        "momentum.cci": rule_threshold_band,
        "momentum.fisher": rule_sign,
        "momentum.macd": rule_sign,
        "momentum.ppo": rule_sign,
        "momentum.roc": rule_sign,
        "momentum.rsi": rule_threshold_band,
        "momentum.stoch": rule_threshold_band,
        "momentum.stoch_rsi": rule_threshold_band,
        "momentum.trix": rule_sign,
        "momentum.williams_r": rule_threshold_band,
        "structure.candle_stats": SignalRuleSpecV1(
            rule_family=_RULE_CANDLE_BODY_DIRECTION,
            required_signal_params=("min_body_pct",),
            candle_body_min_param_name="min_body_pct",
        ),
        "structure.candle_stats_atr_norm": SignalRuleSpecV1(
            rule_family=_RULE_CANDLE_BODY_DIRECTION,
            required_signal_params=("min_body_atr",),
            candle_body_min_param_name="min_body_atr",
        ),
        "structure.distance_to_ma_norm": rule_threshold_band,
        "structure.percent_rank": rule_threshold_band,
        "structure.pivots": rule_pivot_events,
        "structure.zscore": rule_threshold_band,
        "trend.adx": rule_delta_sign,
        "trend.aroon": rule_threshold_band,
        "trend.chandelier_exit": rule_compare_price,
        "trend.donchian": rule_compare_price,
        "trend.ichimoku": rule_compare_price,
        "trend.keltner": rule_compare_price,
        "trend.linreg_slope": rule_sign,
        "trend.psar": rule_compare_price,
        "trend.supertrend": rule_compare_price,
        "trend.vortex": SignalRuleSpecV1(
            rule_family=_RULE_THRESHOLD_CENTERED,
            required_signal_params=("abs_threshold",),
            threshold_center=1.0,
        ),
        "volatility.atr": rule_delta_sign,
        "volatility.bbands": rule_compare_price,
        "volatility.bbands_bandwidth": rule_delta_sign,
        "volatility.bbands_percent_b": rule_threshold_band,
        "volatility.hv": rule_delta_sign,
        "volatility.stddev": rule_delta_sign,
        "volatility.tr": rule_delta_sign,
        "volatility.variance": rule_delta_sign,
        "volume.ad_line": rule_delta_sign,
        "volume.cmf": rule_threshold_band,
        "volume.mfi": rule_threshold_band,
        "volume.obv": rule_delta_sign,
        "volume.volume_sma": rule_compare_volume,
        "volume.vwap": rule_compare_price,
        "volume.vwap_deviation": SignalRuleSpecV1(
            rule_family=_RULE_COMPARE_PRICE_TO_OUTPUT,
            invert_price_comparison=True,
        ),
    }
    return dict(sorted(entries.items(), key=lambda item: item[0]))


_SIGNAL_RULE_REGISTRY_V1 = MappingProxyType(_build_signal_rule_registry_v1())


def supported_indicator_ids_for_signals_v1() -> tuple[str, ...]:
    """
    Return deterministic ordered list of indicator ids supported by signal rules v1.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        None.
    Returns:
        tuple[str, ...]: Stable sorted indicator ids.
    Assumptions:
        Registry is initialized at module import and remains immutable.
    Raises:
        None.
    Side Effects:
        None.
    """
    return tuple(_SIGNAL_RULE_REGISTRY_V1.keys())


def list_signal_rule_registry_v1() -> tuple[tuple[str, str], ...]:
    """
    Return `(indicator_id, rule_family)` pairs for debugging and deterministic introspection.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        None.
    Returns:
        tuple[tuple[str, str], ...]: Sorted rule registry projection.
    Assumptions:
        Registry values are immutable `SignalRuleSpecV1` objects.
    Raises:
        None.
    Side Effects:
        None.
    """
    pairs: list[tuple[str, str]] = []
    for indicator_id in supported_indicator_ids_for_signals_v1():
        pairs.append((indicator_id, _SIGNAL_RULE_REGISTRY_V1[indicator_id].rule_family))
    return tuple(pairs)


def signal_rule_spec_v1(*, indicator_id: str) -> SignalRuleSpecV1:
    """
    Resolve one rule specification by indicator id.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        indicator_id: Requested indicator identifier.
    Returns:
        SignalRuleSpecV1: Rule spec for the indicator.
    Assumptions:
        Identifier case/spacing are normalized at lookup.
    Raises:
        ValueError: If indicator id is blank or unsupported by rule registry v1.
    Side Effects:
        None.
    """
    normalized_indicator_id = indicator_id.strip().lower()
    if not normalized_indicator_id:
        raise ValueError("indicator_id must be non-empty")
    spec = _SIGNAL_RULE_REGISTRY_V1.get(normalized_indicator_id)
    if spec is None:
        raise ValueError(f"Unsupported indicator_id for signals v1: {normalized_indicator_id}")
    return spec


def expand_indicator_grids_with_signal_dependencies_v1(
    *,
    indicator_grids: Sequence[GridSpec],
) -> tuple[GridSpec, ...]:
    """
    Expand compute grid list with signal-rule dependencies in deterministic order.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        indicator_grids: Base indicator compute grids from request/template.
    Returns:
        tuple[GridSpec, ...]: Expanded deduplicated deterministic compute plan.
    Assumptions:
        `structure.pivots` rule requires wrappers `structure.pivot_low` and
            `structure.pivot_high` with same parameterization.
    Raises:
        ValueError: If one grid cannot be normalized into deterministic key.
    Side Effects:
        None.
    """
    expanded_candidates: list[GridSpec] = []
    for grid in _sort_indicator_grids_deterministically(indicator_grids=indicator_grids):
        expanded_candidates.append(grid)
        spec = _SIGNAL_RULE_REGISTRY_V1.get(grid.indicator_id.value)
        if spec is None:
            continue
        for dependency_id in spec.required_dependency_ids:
            expanded_candidates.append(
                _clone_grid_with_indicator_id(
                    grid=grid,
                    indicator_id=dependency_id,
                )
            )

    deduplicated: list[GridSpec] = []
    seen_keys: set[tuple[object, ...]] = set()
    for grid in expanded_candidates:
        key = _grid_deterministic_key(grid=grid)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated.append(grid)
    return tuple(deduplicated)


def indicator_primary_output_series_from_tensor_v1(
    *,
    tensor: IndicatorTensor,
    variant_index: int = 0,
) -> np.ndarray:
    """
    Extract one primary-output time series from indicator tensor for chosen variant index.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/indicators/application/dto/indicator_tensor.py
      - src/trading/contexts/indicators/domain/entities/layout.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py

    Args:
        tensor: Indicator tensor returned by `IndicatorCompute.compute`.
        variant_index: Flattened variant index in tensor layout.
    Returns:
        np.ndarray: Contiguous float32 1D primary-output series.
    Assumptions:
        Indicators context exposes one primary output per `indicator_id` in v1 contracts.
    Raises:
        ValueError: If variant index is out of range or layout is unsupported.
    Side Effects:
        None.
    """
    if variant_index < 0 or variant_index >= tensor.meta.variants:
        raise ValueError("variant_index must be in [0, meta.variants)")

    if tensor.layout is Layout.TIME_MAJOR:
        flattened = tensor.values.reshape(tensor.meta.t, tensor.meta.variants)
        return np.ascontiguousarray(flattened[:, variant_index], dtype=np.float32)

    if tensor.layout is Layout.VARIANT_MAJOR:
        flattened = tensor.values.reshape(tensor.meta.variants, tensor.meta.t)
        # HOT PATH: variant-major row extraction stays a view when dtype/layout already match.
        variant_view = flattened[variant_index, :]
        if variant_view.dtype == np.float32 and variant_view.flags.c_contiguous:
            return variant_view
        return np.ascontiguousarray(variant_view, dtype=np.float32)

    raise ValueError(f"Unsupported tensor layout: {tensor.layout}")


def build_indicator_signal_inputs_from_tensors_v1(
    *,
    tensors: Mapping[str, IndicatorTensor],
    indicator_inputs: Mapping[str, Mapping[str, SignalRuleScalar]] | None = None,
    signal_params: Mapping[str, Mapping[str, SignalRuleScalar]] | None = None,
    dependency_outputs: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    variant_index_by_indicator: Mapping[str, int] | None = None,
) -> tuple[IndicatorSignalEvaluationInputV1, ...]:
    """
    Build pure-evaluator inputs from compute tensors using primary output only.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
      - src/trading/contexts/indicators/application/dto/indicator_tensor.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py

    Args:
        tensors: Mapping from indicator id to compute tensor.
        indicator_inputs: Optional map with indicator input literals (e.g., `source`).
        signal_params: Optional map with signal-rule params per indicator.
        dependency_outputs: Optional dependency outputs for composite rules (e.g., pivots).
        variant_index_by_indicator: Optional variant index override per indicator id.
    Returns:
        tuple[IndicatorSignalEvaluationInputV1, ...]: Deterministic sorted evaluator inputs.
    Assumptions:
        All mappings use indicator ids as top-level keys.
    Raises:
        ValueError: If variant index map contains invalid value for an indicator.
    Side Effects:
        None.
    """
    resolved_indicator_inputs = indicator_inputs or {}
    resolved_signal_params = signal_params or {}
    resolved_dependency_outputs = dependency_outputs or {}
    resolved_variant_indexes = variant_index_by_indicator or {}

    items: list[IndicatorSignalEvaluationInputV1] = []
    for indicator_id in sorted(tensors.keys()):
        tensor = tensors[indicator_id]
        variant_index_raw = resolved_variant_indexes.get(indicator_id, 0)
        if isinstance(variant_index_raw, bool) or not isinstance(variant_index_raw, int):
            raise ValueError(
                "variant_index_by_indicator values must be integers when provided"
            )
        items.append(
            IndicatorSignalEvaluationInputV1(
                indicator_id=indicator_id,
                primary_output=indicator_primary_output_series_from_tensor_v1(
                    tensor=tensor,
                    variant_index=variant_index_raw,
                ),
                indicator_inputs=resolved_indicator_inputs.get(indicator_id, {}),
                signal_params=resolved_signal_params.get(indicator_id, {}),
                dependency_outputs=resolved_dependency_outputs.get(indicator_id, {}),
            )
        )
    return tuple(items)


def encode_signal_array_v1(*, signals: np.ndarray) -> np.ndarray:
    """
    Encode legacy signal labels into compact `np.int8` representation.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        signals: One-dimensional signal array containing legacy labels or compact codes.
    Returns:
        np.ndarray: Contiguous `np.int8` array with values `{-1, 0, 1}`.
    Assumptions:
        Input can contain `SignalV1`, string literals, or integer codes.
    Raises:
        ValueError: If array shape is invalid or one value is not recognized.
    Side Effects:
        None.
    """
    raw = np.asarray(signals)
    if raw.ndim != 1:
        raise ValueError("signals must be a 1D array")
    if is_int8_c_contiguous_signal_array_v1(signals=raw):
        _ensure_supported_signal_codes_v1(signal_codes=raw, field_name="signals")
        return raw
    if np.issubdtype(raw.dtype, np.integer):
        encoded = np.ascontiguousarray(raw, dtype=np.int8)
        _ensure_supported_signal_codes_v1(signal_codes=encoded, field_name="signals")
        return encoded

    encoded = np.full(raw.shape[0], SIGNAL_CODE_NEUTRAL_V1, dtype=np.int8)
    for index, value in enumerate(raw.tolist()):
        encoded[index] = _signal_value_to_code_v1(value=value, field_name="signals")
    return encoded


def is_int8_c_contiguous_signal_array_v1(*, signals: np.ndarray) -> bool:
    """
    Check whether signal array already matches canonical compact hot-path representation.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py

    Args:
        signals: Candidate signal array.
    Returns:
        bool: `True` when array is 1D, `np.int8`, and C-contiguous.
    Assumptions:
        Canonical shape/dtype/layout check is enough for fast-path copy avoidance.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (
        signals.ndim == 1
        and signals.dtype == np.int8
        and bool(signals.flags.c_contiguous)
    )


def decode_signal_codes_v1(*, signal_codes: np.ndarray) -> np.ndarray:
    """
    Decode compact `np.int8` signal codes into canonical `LONG|SHORT|NEUTRAL` labels.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        signal_codes: One-dimensional compact signal-code array.
    Returns:
        np.ndarray: Unicode label array with canonical `LONG|SHORT|NEUTRAL` values.
    Assumptions:
        Valid compact values are `NEUTRAL=0`, `LONG=1`, `SHORT=-1`.
    Raises:
        ValueError: If shape is invalid or contains unsupported codes.
    Side Effects:
        None.
    """
    encoded = encode_signal_array_v1(signals=signal_codes)
    decoded = _neutral_signal_array(length=int(encoded.shape[0]))
    decoded[encoded == SIGNAL_CODE_LONG_V1] = SignalV1.LONG.value
    decoded[encoded == SIGNAL_CODE_SHORT_V1] = SignalV1.SHORT.value
    return decoded


def evaluate_indicator_signal_encoded_v1(
    *,
    candles: CandleArrays,
    indicator_input: IndicatorSignalEvaluationInputV1,
) -> np.ndarray:
    """
    Evaluate one indicator signal series into compact `np.int8` codes.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        candles: Candle arrays aligned to target backtest timeline.
        indicator_input: Rule-evaluation input payload for one indicator.
    Returns:
        np.ndarray: Compact `np.int8` signal codes for one indicator.
    Assumptions:
        `indicator_input.primary_output` length equals candle timeline length.
    Raises:
        ValueError: If array lengths mismatch or rule parameters/dependencies are missing.
    Side Effects:
        None.
    """
    bar_count = int(candles.close.shape[0])
    if indicator_input.primary_output.shape[0] != bar_count:
        raise ValueError(
            f"{indicator_input.indicator_id}: primary_output length must equal candles length"
        )

    spec = signal_rule_spec_v1(indicator_id=indicator_input.indicator_id)
    _ensure_required_signal_params(
        indicator_id=indicator_input.indicator_id,
        spec=spec,
        signal_params=indicator_input.signal_params,
    )
    _ensure_required_dependencies(
        indicator_id=indicator_input.indicator_id,
        spec=spec,
        dependency_outputs=indicator_input.dependency_outputs,
        expected_length=bar_count,
    )

    if spec.rule_family == _RULE_COMPARE_PRICE_TO_OUTPUT:
        signal_codes = _rule_compare_price_to_output(
            candles=candles,
            primary_output=indicator_input.primary_output,
            indicator_inputs=indicator_input.indicator_inputs,
            invert=spec.invert_price_comparison,
        )
    elif spec.rule_family == _RULE_THRESHOLD_BAND:
        signal_codes = _rule_threshold_band(
            primary_output=indicator_input.primary_output,
            signal_params=indicator_input.signal_params,
        )
    elif spec.rule_family == _RULE_SIGN:
        signal_codes = _rule_sign(primary_output=indicator_input.primary_output)
    elif spec.rule_family == _RULE_DELTA_SIGN:
        signal_codes = _rule_delta_sign(
            primary_output=indicator_input.primary_output,
            signal_params=indicator_input.signal_params,
        )
    elif spec.rule_family == _RULE_COMPARE_VOLUME_TO_OUTPUT:
        signal_codes = _rule_compare_volume_to_output(
            candles=candles,
            primary_output=indicator_input.primary_output,
        )
    elif spec.rule_family == _RULE_CANDLE_BODY_DIRECTION:
        signal_codes = _rule_candle_body_direction(
            candles=candles,
            primary_output=indicator_input.primary_output,
            signal_params=indicator_input.signal_params,
            min_param_name=spec.candle_body_min_param_name,
        )
    elif spec.rule_family == _RULE_PIVOT_EVENTS:
        signal_codes = _rule_pivot_events(
            dependency_outputs=indicator_input.dependency_outputs,
            expected_length=bar_count,
        )
    elif spec.rule_family == _RULE_THRESHOLD_CENTERED:
        signal_codes = _rule_threshold_centered(
            primary_output=indicator_input.primary_output,
            signal_params=indicator_input.signal_params,
            center=spec.threshold_center,
        )
    else:
        raise ValueError(f"Unsupported rule family: {spec.rule_family}")
    return signal_codes


def evaluate_indicator_signal_v1(
    *,
    candles: CandleArrays,
    indicator_input: IndicatorSignalEvaluationInputV1,
) -> IndicatorSignalsV1:
    """
    Evaluate one indicator signal series (`LONG|SHORT|NEUTRAL`) on each bar.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py

    Args:
        candles: Candle arrays aligned to target backtest timeline.
        indicator_input: Rule-evaluation input payload for one indicator.
    Returns:
        IndicatorSignalsV1: Normalized signal series for the indicator.
    Assumptions:
        `indicator_input.primary_output` length equals candle timeline length.
    Raises:
        ValueError: If array lengths mismatch or rule parameters/dependencies are missing.
    Side Effects:
        None.
    """
    signal_codes = evaluate_indicator_signal_encoded_v1(
        candles=candles,
        indicator_input=indicator_input,
    )
    signal_values = decode_signal_codes_v1(signal_codes=signal_codes)
    return IndicatorSignalsV1(indicator_id=indicator_input.indicator_id, signals=signal_values)


def evaluate_and_aggregate_signals_encoded_v1(
    *,
    candles: CandleArrays,
    indicator_inputs: Sequence[IndicatorSignalEvaluationInputV1],
) -> np.ndarray:
    """
    Evaluate indicator rules and return aggregated compact `np.int8` final signal series.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        candles: Candle arrays aligned to target backtest timeline.
        indicator_inputs: Inputs for indicator-level rule evaluation.
    Returns:
        np.ndarray: Compact aggregated final signal array (`np.int8`).
    Assumptions:
        Aggregation order is deterministic by normalized `indicator_id`.
    Raises:
        ValueError: If one indicator payload is invalid or has mismatched timeline length.
    Side Effects:
        None.
    """
    per_indicator_codes: list[tuple[str, np.ndarray]] = []
    for indicator_input in sorted(indicator_inputs, key=lambda item: item.indicator_id):
        # HOT PATH: keep per-indicator signals in compact int8 form without unicode arrays.
        per_indicator_codes.append(
            (
                indicator_input.indicator_id,
                evaluate_indicator_signal_encoded_v1(
                    candles=candles,
                    indicator_input=indicator_input,
                ),
            )
        )
    return aggregate_indicator_signal_codes_v1(
        candles=candles,
        indicator_signal_codes=per_indicator_codes,
    )


def evaluate_and_aggregate_signals_v1(
    *,
    candles: CandleArrays,
    indicator_inputs: Sequence[IndicatorSignalEvaluationInputV1],
) -> AggregatedSignalsV1:
    """
    Evaluate indicator signal rules and aggregate them by v1 AND policy.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        candles: Candle arrays aligned to target backtest timeline.
        indicator_inputs: Inputs for indicator-level rule evaluation.
    Returns:
        AggregatedSignalsV1: Aggregated strategy-level v1 signal payload.
    Assumptions:
        Aggregation order is deterministic by normalized `indicator_id`.
    Raises:
        ValueError: If one indicator payload is invalid or has mismatched timeline length.
    Side Effects:
        None.
    """
    evaluated: list[IndicatorSignalsV1] = []
    for indicator_input in sorted(indicator_inputs, key=lambda item: item.indicator_id):
        evaluated.append(
            evaluate_indicator_signal_v1(
                candles=candles,
                indicator_input=indicator_input,
            )
        )
    return aggregate_indicator_signals_v1(candles=candles, indicator_signals=evaluated)


def aggregate_indicator_signal_codes_v1(
    *,
    candles: CandleArrays,
    indicator_signal_codes: Sequence[tuple[str, np.ndarray]],
) -> np.ndarray:
    """
    Aggregate compact per-indicator signal codes with v1 AND policy.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        candles: Candle arrays defining timeline length for output vectors.
        indicator_signal_codes: Per-indicator compact signal codes.
    Returns:
        np.ndarray: Aggregated compact final signal codes (`np.int8`).
    Assumptions:
        Empty indicator set resolves to all-neutral final signal with conflict neutralization.
    Raises:
        ValueError: If one signal series length mismatches the candle timeline.
    Side Effects:
        None.
    """
    bar_count = int(candles.close.shape[0])
    ordered_signal_codes = tuple(sorted(indicator_signal_codes, key=lambda item: item[0]))
    final_long = np.ones(bar_count, dtype=np.bool_)
    final_short = np.ones(bar_count, dtype=np.bool_)

    for indicator_id, signal_codes in ordered_signal_codes:
        normalized_codes = encode_signal_array_v1(signals=signal_codes)
        if normalized_codes.shape[0] != bar_count:
            raise ValueError(
                f"{indicator_id}: signal length must equal candles length"
            )
        final_long &= normalized_codes == SIGNAL_CODE_LONG_V1
        final_short &= normalized_codes == SIGNAL_CODE_SHORT_V1

    conflict_mask = final_long & final_short
    resolved_final_long = final_long & ~conflict_mask
    resolved_final_short = final_short & ~conflict_mask
    final_signal = _neutral_signal_code_array(length=bar_count)
    final_signal[resolved_final_long] = SIGNAL_CODE_LONG_V1
    final_signal[resolved_final_short] = SIGNAL_CODE_SHORT_V1
    return final_signal


def aggregate_indicator_signals_v1(
    *,
    candles: CandleArrays,
    indicator_signals: Sequence[IndicatorSignalsV1],
) -> AggregatedSignalsV1:
    """
    Aggregate per-indicator signal series with AND policy and deterministic conflicts handling.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        candles: Candle arrays defining timeline length for output vectors.
        indicator_signals: Per-indicator signal series.
    Returns:
        AggregatedSignalsV1: Strategy-level final signal and metadata vectors.
    Assumptions:
        `final_long = all(indicator_signal == LONG)` and
            `final_short = all(indicator_signal == SHORT)` for each bar.
    Raises:
        ValueError: If one signal series length mismatches the candle timeline.
    Side Effects:
        None.
    """
    bar_count = int(candles.close.shape[0])
    ordered_indicator_signals = tuple(
        sorted(indicator_signals, key=lambda item: item.indicator_id)
    )
    final_long = np.ones(bar_count, dtype=np.bool_)
    final_short = np.ones(bar_count, dtype=np.bool_)

    for item in ordered_indicator_signals:
        if item.signals.shape[0] != bar_count:
            raise ValueError(
                f"{item.indicator_id}: signal length must equal candles length"
            )
        final_long &= item.signals == SignalV1.LONG.value
        final_short &= item.signals == SignalV1.SHORT.value

    conflict_mask = final_long & final_short
    resolved_final_long = final_long & ~conflict_mask
    resolved_final_short = final_short & ~conflict_mask
    final_signal = _neutral_signal_array(length=bar_count)
    final_signal[resolved_final_long] = SignalV1.LONG.value
    final_signal[resolved_final_short] = SignalV1.SHORT.value

    return AggregatedSignalsV1(
        per_indicator_signals=ordered_indicator_signals,
        final_signal=final_signal,
        final_long=resolved_final_long,
        final_short=resolved_final_short,
        conflicting_signals=int(np.count_nonzero(conflict_mask)),
    )


def _rule_compare_price_to_output(
    *,
    candles: CandleArrays,
    primary_output: np.ndarray,
    indicator_inputs: Mapping[str, SignalRuleScalar],
    invert: bool,
) -> np.ndarray:
    """
    Evaluate `compare_price_to_output` rule family.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - src/trading/contexts/indicators/application/dto/candle_arrays.py

    Args:
        candles: Candle arrays used to resolve selected source series.
        primary_output: Indicator primary output.
        indicator_inputs: Optional indicator inputs (e.g., `source`).
        invert: Invert comparison polarity for mean-reversion variants.
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        NaN at source/output produces `NEUTRAL` on the same bar.
    Raises:
        ValueError: If source literal is unsupported.
    Side Effects:
        None.
    """
    price_series = _resolve_price_series_from_inputs(
        candles=candles,
        indicator_inputs=indicator_inputs,
    )
    finite_mask = np.isfinite(price_series) & np.isfinite(primary_output)
    long_mask = finite_mask & (price_series > primary_output)
    short_mask = finite_mask & (price_series < primary_output)
    if invert:
        return _compose_signal_from_masks(long_mask=short_mask, short_mask=long_mask)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _rule_threshold_band(
    *,
    primary_output: np.ndarray,
    signal_params: Mapping[str, SignalRuleScalar],
) -> np.ndarray:
    """
    Evaluate `threshold_band` rule family with deterministic orientation by threshold order.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        primary_output: Indicator primary output.
        signal_params: Rule params with `long_threshold` and `short_threshold`.
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        When `long_threshold <= short_threshold`, lower values are LONG and higher values
            are SHORT (mean-reversion orientation). Otherwise orientation is inverted
            (trend-following orientation).
    Raises:
        ValueError: If required threshold params are missing or non-numeric.
    Side Effects:
        None.
    """
    long_threshold = _require_float_param(signal_params=signal_params, name="long_threshold")
    short_threshold = _require_float_param(signal_params=signal_params, name="short_threshold")
    finite_mask = np.isfinite(primary_output)
    if long_threshold <= short_threshold:
        long_mask = finite_mask & (primary_output <= long_threshold)
        short_mask = finite_mask & (primary_output >= short_threshold)
        return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)

    long_mask = finite_mask & (primary_output >= long_threshold)
    short_mask = finite_mask & (primary_output <= short_threshold)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _rule_sign(*, primary_output: np.ndarray) -> np.ndarray:
    """
    Evaluate `sign` rule family (`>0 => LONG`, `<0 => SHORT`).

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py
      - configs/prod/indicators.yaml

    Args:
        primary_output: Indicator primary output.
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        NaN values are converted to `NEUTRAL`.
    Raises:
        None.
    Side Effects:
        None.
    """
    finite_mask = np.isfinite(primary_output)
    long_mask = finite_mask & (primary_output > 0.0)
    short_mask = finite_mask & (primary_output < 0.0)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _rule_delta_sign(
    *,
    primary_output: np.ndarray,
    signal_params: Mapping[str, SignalRuleScalar],
) -> np.ndarray:
    """
    Evaluate `delta_sign` with per-direction lag parameters from UI/config form.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        primary_output: Indicator primary output.
        signal_params: Rule params with `long_delta_periods` and `short_delta_periods`.
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        Input periods can be negative in UI/config; evaluator uses `abs(value)` lookback.
    Raises:
        ValueError: If period params are missing, not integers, or normalize to zero.
    Side Effects:
        None.
    """
    long_periods = _require_positive_abs_int_param(
        signal_params=signal_params,
        name="long_delta_periods",
    )
    short_periods = _require_positive_abs_int_param(
        signal_params=signal_params,
        name="short_delta_periods",
    )

    long_delta = np.full(primary_output.shape[0], np.nan, dtype=np.float32)
    short_delta = np.full(primary_output.shape[0], np.nan, dtype=np.float32)
    if long_periods < primary_output.shape[0]:
        long_delta[long_periods:] = (
            primary_output[long_periods:] - primary_output[:-long_periods]
        )
    if short_periods < primary_output.shape[0]:
        short_delta[short_periods:] = (
            primary_output[short_periods:] - primary_output[:-short_periods]
        )

    long_mask = np.isfinite(long_delta) & (long_delta > 0.0)
    short_mask = np.isfinite(short_delta) & (short_delta < 0.0)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _rule_compare_volume_to_output(
    *,
    candles: CandleArrays,
    primary_output: np.ndarray,
) -> np.ndarray:
    """
    Evaluate `compare_volume_to_output` rule family.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        candles: Candle arrays with volume series.
        primary_output: Indicator primary output.
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        NaN volume or output values are converted to `NEUTRAL`.
    Raises:
        None.
    Side Effects:
        None.
    """
    finite_mask = np.isfinite(candles.volume) & np.isfinite(primary_output)
    long_mask = finite_mask & (candles.volume > primary_output)
    short_mask = finite_mask & (candles.volume < primary_output)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _rule_candle_body_direction(
    *,
    candles: CandleArrays,
    primary_output: np.ndarray,
    signal_params: Mapping[str, SignalRuleScalar],
    min_param_name: str | None,
) -> np.ndarray:
    """
    Evaluate `candle_body_direction` rule family.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - src/trading/contexts/indicators/application/dto/candle_arrays.py

    Args:
        candles: Candle arrays with `open` and `close` series.
        primary_output: Indicator primary output (`body_pct` or `body_atr`).
        signal_params: Rule params with minimal body magnitude threshold.
        min_param_name: Name of threshold param (`min_body_pct` or `min_body_atr`).
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        Candle direction is derived from same-bar `close` vs `open`.
    Raises:
        ValueError: If threshold parameter name/value is missing or invalid.
    Side Effects:
        None.
    """
    if min_param_name is None:
        raise ValueError("candle_body_direction requires a min threshold parameter name")
    threshold = _require_float_param(signal_params=signal_params, name=min_param_name)

    finite_mask = (
        np.isfinite(primary_output) & np.isfinite(candles.open) & np.isfinite(candles.close)
    )
    has_body = finite_mask & (primary_output >= threshold)
    long_mask = has_body & (candles.close > candles.open)
    short_mask = has_body & (candles.close < candles.open)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _rule_pivot_events(
    *,
    dependency_outputs: Mapping[str, np.ndarray],
    expected_length: int,
) -> np.ndarray:
    """
    Evaluate `pivot_events` rule family using wrapper dependency outputs.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/indicators/domain/definitions/structure.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        dependency_outputs: Dependency output map by wrapper indicator id.
        expected_length: Expected bars count for dependency vectors.
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        Finite value in `pivot_low` means LONG event, finite value in `pivot_high` means SHORT.
    Raises:
        ValueError: If required wrapper series are missing or length-mismatched.
    Side Effects:
        None.
    """
    pivot_low = _require_dependency_series(
        dependency_outputs=dependency_outputs,
        dependency_id="structure.pivot_low",
        expected_length=expected_length,
    )
    pivot_high = _require_dependency_series(
        dependency_outputs=dependency_outputs,
        dependency_id="structure.pivot_high",
        expected_length=expected_length,
    )
    long_mask = np.isfinite(pivot_low)
    short_mask = np.isfinite(pivot_high)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _rule_threshold_centered(
    *,
    primary_output: np.ndarray,
    signal_params: Mapping[str, SignalRuleScalar],
    center: float | None,
) -> np.ndarray:
    """
    Evaluate centered-threshold rule (`center +/- abs_threshold`) for trend.vortex.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        primary_output: Indicator primary output.
        signal_params: Rule params with `abs_threshold`.
        center: Center value around which the threshold band is built.
    Returns:
        np.ndarray: Compact signal codes by bar.
    Assumptions:
        NaN values are mapped to `NEUTRAL`.
    Raises:
        ValueError: If center or threshold parameters are invalid.
    Side Effects:
        None.
    """
    if center is None:
        raise ValueError("threshold_centered rule requires a finite center value")
    abs_threshold = _require_float_param(signal_params=signal_params, name="abs_threshold")
    finite_mask = np.isfinite(primary_output)
    long_mask = finite_mask & (primary_output > center + abs_threshold)
    short_mask = finite_mask & (primary_output < center - abs_threshold)
    return _compose_signal_from_masks(long_mask=long_mask, short_mask=short_mask)


def _resolve_price_series_from_inputs(
    *,
    candles: CandleArrays,
    indicator_inputs: Mapping[str, SignalRuleScalar],
) -> np.ndarray:
    """
    Resolve price-like source series according to indicator `inputs.source` semantics.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - configs/prod/indicators.yaml

    Args:
        candles: Candle arrays with base OHLCV series.
        indicator_inputs: Indicator inputs mapping.
    Returns:
        np.ndarray: Resolved source series.
    Assumptions:
        Missing source defaults to `close`.
    Raises:
        ValueError: If `source` is unknown.
    Side Effects:
        None.
    """
    raw_source = indicator_inputs.get("source")
    normalized_source = "close"
    if raw_source is not None:
        source_literal = str(raw_source).strip().lower()
        if source_literal:
            normalized_source = source_literal

    if normalized_source == "open":
        return candles.open
    if normalized_source == "high":
        return candles.high
    if normalized_source == "low":
        return candles.low
    if normalized_source == "close":
        return candles.close
    if normalized_source == "hl2":
        return np.ascontiguousarray((candles.high + candles.low) / 2.0, dtype=np.float32)
    if normalized_source == "hlc3":
        return np.ascontiguousarray(
            (candles.high + candles.low + candles.close) / 3.0,
            dtype=np.float32,
        )
    if normalized_source == "ohlc4":
        return np.ascontiguousarray(
            (candles.open + candles.high + candles.low + candles.close) / 4.0,
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported inputs.source value: {normalized_source}")


def _compose_signal_from_masks(*, long_mask: np.ndarray, short_mask: np.ndarray) -> np.ndarray:
    """
    Compose compact signal-code series from long/short masks with conflict-neutralization.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        long_mask: Boolean mask of candidate LONG bars.
        short_mask: Boolean mask of candidate SHORT bars.
    Returns:
        np.ndarray: Compact signal codes by bar (`NEUTRAL=0`, `LONG=1`, `SHORT=-1`).
    Assumptions:
        Both masks are one-dimensional and bar-aligned.
    Raises:
        ValueError: If mask shapes mismatch.
    Side Effects:
        None.
    """
    if long_mask.shape != short_mask.shape:
        raise ValueError("long_mask and short_mask must have equal shape")
    conflict_mask = long_mask & short_mask
    resolved_long = long_mask & ~conflict_mask
    resolved_short = short_mask & ~conflict_mask
    # HOT PATH: use int8 composition to avoid unicode allocations in scorer path.
    signal_codes = _neutral_signal_code_array(length=int(long_mask.shape[0]))
    signal_codes[resolved_long] = SIGNAL_CODE_LONG_V1
    signal_codes[resolved_short] = SIGNAL_CODE_SHORT_V1
    return signal_codes


def _neutral_signal_array(*, length: int) -> np.ndarray:
    """
    Build deterministic neutral-filled signal array.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        length: Number of bars.
    Returns:
        np.ndarray: Unicode signal labels initialized to `NEUTRAL`.
    Assumptions:
        Length is non-negative.
    Raises:
        ValueError: If length is negative.
    Side Effects:
        None.
    """
    if length < 0:
        raise ValueError("length must be >= 0")
    return np.full(length, SignalV1.NEUTRAL.value, dtype="U7")


def _neutral_signal_code_array(*, length: int) -> np.ndarray:
    """
    Build deterministic compact neutral-filled signal-code array.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        length: Number of bars.
    Returns:
        np.ndarray: Compact `np.int8` neutral signal array.
    Assumptions:
        Length is non-negative.
    Raises:
        ValueError: If length is negative.
    Side Effects:
        None.
    """
    if length < 0:
        raise ValueError("length must be >= 0")
    return np.full(length, SIGNAL_CODE_NEUTRAL_V1, dtype=np.int8)


def _signal_value_to_code_v1(*, value: object, field_name: str) -> np.int8:
    """
    Normalize one raw signal value into canonical compact code.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py

    Args:
        value: Raw signal value.
        field_name: Field label used in validation errors.
    Returns:
        np.int8: Canonical compact code.
    Assumptions:
        Supported values are integer codes or `SignalV1`/string literals.
    Raises:
        ValueError: If value cannot be mapped to one of canonical codes.
    Side Effects:
        None.
    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        code_value = int(value)
        if code_value == int(SIGNAL_CODE_NEUTRAL_V1):
            return SIGNAL_CODE_NEUTRAL_V1
        if code_value == int(SIGNAL_CODE_LONG_V1):
            return SIGNAL_CODE_LONG_V1
        if code_value == int(SIGNAL_CODE_SHORT_V1):
            return SIGNAL_CODE_SHORT_V1
        raise ValueError(
            f"{field_name} values must be LONG, SHORT, NEUTRAL or compact codes -1, 0, 1"
        )

    normalized = str(value).strip().upper()
    if normalized == SignalV1.NEUTRAL.value:
        return SIGNAL_CODE_NEUTRAL_V1
    if normalized == SignalV1.LONG.value:
        return SIGNAL_CODE_LONG_V1
    if normalized == SignalV1.SHORT.value:
        return SIGNAL_CODE_SHORT_V1
    raise ValueError(
        f"{field_name} values must be LONG, SHORT, NEUTRAL or compact codes -1, 0, 1"
    )


def _ensure_supported_signal_codes_v1(*, signal_codes: np.ndarray, field_name: str) -> None:
    """
    Validate that compact signal array contains only canonical code values.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py

    Args:
        signal_codes: Compact signal-code array.
        field_name: Field label used in deterministic validation errors.
    Returns:
        None.
    Assumptions:
        Signal array is one-dimensional and already converted to `np.int8`.
    Raises:
        ValueError: If one element is outside canonical `{-1, 0, 1}` set.
    Side Effects:
        None.
    """
    invalid_mask = (
        (signal_codes != SIGNAL_CODE_NEUTRAL_V1)
        & (signal_codes != SIGNAL_CODE_LONG_V1)
        & (signal_codes != SIGNAL_CODE_SHORT_V1)
    )
    if np.any(invalid_mask):
        raise ValueError(
            f"{field_name} values must be LONG, SHORT, NEUTRAL or compact codes -1, 0, 1"
        )


def _ensure_required_signal_params(
    *,
    indicator_id: str,
    spec: SignalRuleSpecV1,
    signal_params: Mapping[str, SignalRuleScalar],
) -> None:
    """
    Validate that all rule-required signal parameters are provided.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        indicator_id: Indicator identifier for deterministic error messages.
        spec: Rule specification with required signal-parameter names.
        signal_params: Provided signal parameters mapping.
    Returns:
        None.
    Assumptions:
        Presence check is done before family-specific type/range validation.
    Raises:
        ValueError: If required parameter is absent from mapping.
    Side Effects:
        None.
    """
    for name in spec.required_signal_params:
        if name not in signal_params:
            raise ValueError(f"{indicator_id}: missing required signal param '{name}'")


def _ensure_required_dependencies(
    *,
    indicator_id: str,
    spec: SignalRuleSpecV1,
    dependency_outputs: Mapping[str, np.ndarray],
    expected_length: int,
) -> None:
    """
    Validate dependency presence and shape for rule families that require extra outputs.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/indicators/domain/definitions/structure.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        indicator_id: Indicator identifier for deterministic error messages.
        spec: Rule specification with dependency requirements.
        dependency_outputs: Provided dependency output series mapping.
        expected_length: Expected timeline length.
    Returns:
        None.
    Assumptions:
        Dependencies are indexed by wrapper indicator ids.
    Raises:
        ValueError: If dependency is absent or length-mismatched.
    Side Effects:
        None.
    """
    for dependency_id in spec.required_dependency_ids:
        series = dependency_outputs.get(dependency_id)
        if series is None:
            raise ValueError(f"{indicator_id}: missing dependency output '{dependency_id}'")
        normalized = _normalize_series_array(
            name=f"{indicator_id}.{dependency_id}",
            values=series,
        )
        if normalized.shape[0] != expected_length:
            raise ValueError(
                f"{indicator_id}: dependency '{dependency_id}' length must equal candles length"
            )


def _require_dependency_series(
    *,
    dependency_outputs: Mapping[str, np.ndarray],
    dependency_id: str,
    expected_length: int,
) -> np.ndarray:
    """
    Fetch and normalize one dependency output series by id.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/indicators/domain/definitions/structure.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        dependency_outputs: Dependency output map.
        dependency_id: Required dependency id.
        expected_length: Expected timeline length.
    Returns:
        np.ndarray: Normalized dependency series.
    Assumptions:
        Dependency arrays are one-dimensional.
    Raises:
        ValueError: If dependency is missing or timeline length mismatches.
    Side Effects:
        None.
    """
    values = dependency_outputs.get(dependency_id)
    if values is None:
        raise ValueError(f"missing dependency output '{dependency_id}'")
    normalized = _normalize_series_array(name=dependency_id, values=values)
    if normalized.shape[0] != expected_length:
        raise ValueError(
            f"dependency '{dependency_id}' length must equal candles timeline length"
        )
    return normalized


def _require_float_param(*, signal_params: Mapping[str, SignalRuleScalar], name: str) -> float:
    """
    Parse one numeric signal parameter into finite float.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        signal_params: Signal parameter mapping.
        name: Parameter name.
    Returns:
        float: Finite parsed numeric value.
    Assumptions:
        Bool values are rejected to avoid implicit numeric casting ambiguity.
    Raises:
        ValueError: If value is missing, non-numeric, or not finite.
    Side Effects:
        None.
    """
    raw = signal_params.get(name)
    if raw is None:
        raise ValueError(f"missing signal param '{name}'")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
        raise ValueError(f"signal param '{name}' must be numeric")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"signal param '{name}' must be finite")
    return value


def _require_positive_abs_int_param(
    *,
    signal_params: Mapping[str, SignalRuleScalar],
    name: str,
) -> int:
    """
    Parse signed integer period param and return positive absolute lookback.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - configs/prod/indicators.yaml
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        signal_params: Signal parameter mapping.
        name: Parameter name.
    Returns:
        int: Positive absolute integer period.
    Assumptions:
        UI/config may provide negative values; absolute value semantics are used.
    Raises:
        ValueError: If value is missing, non-integer, or normalizes to zero.
    Side Effects:
        None.
    """
    raw = signal_params.get(name)
    if raw is None:
        raise ValueError(f"missing signal param '{name}'")
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise ValueError(f"signal param '{name}' must be integer")
    period = abs(int(raw))
    if period <= 0:
        raise ValueError(f"signal param '{name}' absolute value must be > 0")
    return period


def _sort_indicator_grids_deterministically(
    *,
    indicator_grids: Sequence[GridSpec],
) -> tuple[GridSpec, ...]:
    """
    Sort grid list by canonical deterministic key.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        indicator_grids: Unordered grid sequence.
    Returns:
        tuple[GridSpec, ...]: Deterministically sorted grids.
    Assumptions:
        Grid params/source values can be materialized to stable tuples.
    Raises:
        ValueError: If one grid cannot produce deterministic key.
    Side Effects:
        None.
    """
    return tuple(sorted(indicator_grids, key=lambda grid: _grid_deterministic_key(grid=grid)))


def _clone_grid_with_indicator_id(*, grid: GridSpec, indicator_id: str) -> GridSpec:
    """
    Clone one GridSpec with a different indicator id while preserving axes/config payload.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/indicators/domain/entities/indicator_id.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py

    Args:
        grid: Source grid.
        indicator_id: New indicator id.
    Returns:
        GridSpec: Cloned grid with same params/source/layout and replaced id.
    Assumptions:
        Wrapper dependencies for pivots share the same parameterization.
    Raises:
        ValueError: If new indicator id is invalid.
    Side Effects:
        None.
    """
    return GridSpec(
        indicator_id=IndicatorId(indicator_id),
        params=grid.params,
        source=grid.source,
        layout_preference=grid.layout_preference,
    )


def _grid_deterministic_key(*, grid: GridSpec) -> tuple[object, ...]:
    """
    Build deterministic hashable key for one GridSpec.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
    Related:
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        grid: Grid specification.
    Returns:
        tuple[object, ...]: Canonical sortable key.
    Assumptions:
        Parameter materialization order is stable by key sorting.
    Raises:
        ValueError: If one of materialized values is unsupported.
    Side Effects:
        None.
    """
    params_key: list[tuple[str, tuple[SignalRuleScalar, ...]]] = []
    for param_name in sorted(grid.params.keys()):
        values = tuple(grid.params[param_name].materialize())
        params_key.append((param_name, values))

    source_key: tuple[SignalRuleScalar, ...] | None = None
    if grid.source is not None:
        source_key = tuple(grid.source.materialize())

    layout_key = grid.layout_preference.value if grid.layout_preference is not None else ""
    return (
        grid.indicator_id.value,
        tuple(params_key),
        source_key,
        layout_key,
    )


def _normalize_series_array(*, name: str, values: np.ndarray) -> np.ndarray:
    """
    Normalize one 1D numeric array to contiguous float32.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-overview.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/indicators/application/dto/candle_arrays.py
      - src/trading/contexts/indicators/application/dto/indicator_tensor.py

    Args:
        name: Field name used in deterministic validation errors.
        values: Candidate ndarray.
    Returns:
        np.ndarray: Contiguous float32 1D array.
    Assumptions:
        Signal rules are evaluated on numeric float-compatible series.
    Raises:
        ValueError: If input is not ndarray-like or not one-dimensional.
    Side Effects:
        None.
    """
    try:
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1D array")
    except AttributeError as error:
        raise ValueError(f"{name} must be a numpy ndarray") from error
    return np.ascontiguousarray(values, dtype=np.float32)


def _normalize_scalar_mapping(
    *,
    values: Mapping[str, SignalRuleScalar] | None,
    field_name: str,
) -> dict[str, SignalRuleScalar]:
    """
    Normalize optional scalar mapping into deterministic key-sorted dict.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        values: Optional scalar mapping.
        field_name: Deterministic field identifier for validation messages.
    Returns:
        dict[str, SignalRuleScalar]: Normalized mapping.
    Assumptions:
        Scalar values are JSON-compatible and consumed as-is.
    Raises:
        ValueError: If a mapping key is blank after normalization.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, SignalRuleScalar] = {}
    for key in sorted(values.keys()):
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            raise ValueError(f"{field_name} keys must be non-empty")
        normalized[normalized_key] = values[key]
    return normalized


def _normalize_dependency_mapping(
    *,
    values: Mapping[str, np.ndarray] | None,
    field_name: str,
) -> dict[str, np.ndarray]:
    """
    Normalize dependency output mapping into key-sorted float32 arrays.

    Docs:
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/indicators/domain/definitions/structure.py
      - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py

    Args:
        values: Optional dependency mapping by wrapper indicator id.
        field_name: Deterministic field identifier for validation messages.
    Returns:
        dict[str, np.ndarray]: Normalized mapping.
    Assumptions:
        Dependency ids are stable wrapper indicator ids.
    Raises:
        ValueError: If dependency id is blank or array is invalid.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, np.ndarray] = {}
    for key in sorted(values.keys()):
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            raise ValueError(f"{field_name} keys must be non-empty")
        normalized[normalized_key] = _normalize_series_array(
            name=f"{field_name}.{normalized_key}",
            values=values[key],
        )
    return normalized
