from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Mapping

import numpy as np

from trading.contexts.backtest.application.ports import (
    BacktestStagedVariantScorer,
    BacktestVariantScoreDetailsV1,
)
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantScalar,
    ExecutionParamsV1,
    RiskParamsV1,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    IndicatorTensor,
    IndicatorVariantSelection,
)
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec

from .execution_engine_v1 import BacktestExecutionEngineV1
from .grid_builder_v1 import STAGE_A_LITERAL, STAGE_B_LITERAL
from .signals_from_indicators_v1 import (
    build_indicator_signal_inputs_from_tensors_v1,
    evaluate_and_aggregate_signals_encoded_v1,
    indicator_primary_output_series_from_tensor_v1,
    signal_rule_spec_v1,
)
from .staged_runner_v1 import TOTAL_RETURN_METRIC_LITERAL

_DEFAULT_FEE_PCT_BY_MARKET_ID = {
    1: 0.075,
    2: 0.1,
    3: 0.075,
    4: 0.1,
}

_DEFAULT_SIGNALS_CACHE_MAX_ENTRIES = 2048
_DEFAULT_SIGNALS_CACHE_MAX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _SignalCacheValue:
    """
    Internal immutable cached payload for aggregated signal vector reuse.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_close_fill_scorer_v1.py
    """

    final_signal: np.ndarray

    def __post_init__(self) -> None:
        """
        Normalize cached signal payload to compact contiguous `np.int8` representation.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Cache stores only aggregated final-signal vectors.
        Raises:
            ValueError: If provided signal array is not one-dimensional.
        Side Effects:
            Replaces stored array with contiguous compact copy.
        """
        if self.final_signal.ndim != 1:
            raise ValueError("_SignalCacheValue.final_signal must be a 1D array")
        object.__setattr__(
            self,
            "final_signal",
            np.ascontiguousarray(self.final_signal, dtype=np.int8),
        )


class CloseFillBacktestStagedScorerV1(BacktestStagedVariantScorer):
    """
    Concrete staged scorer using close-fill engine v1 and indicator signal aggregation.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    def __init__(
        self,
        *,
        indicator_compute: IndicatorCompute,
        direction_mode: str,
        sizing_mode: str,
        execution_params: Mapping[str, BacktestVariantScalar],
        market_id: int,
        target_slice: slice,
        init_cash_quote_default: float,
        fixed_quote_default: float,
        safe_profit_percent_default: float,
        slippage_pct_default: float,
        fee_pct_default_by_market_id: Mapping[int, float] | None = None,
        max_variants_guard: int = 600_000,
        signals_cache_max_entries: int = _DEFAULT_SIGNALS_CACHE_MAX_ENTRIES,
        signals_cache_max_bytes: int = _DEFAULT_SIGNALS_CACHE_MAX_BYTES,
        execution_engine: BacktestExecutionEngineV1 | None = None,
    ) -> None:
        """
        Initialize staged scorer with runtime defaults and deterministic execution settings.

        Args:
            indicator_compute: Indicators compute port.
            direction_mode: Direction mode literal for all variants in current run.
            sizing_mode: Sizing mode literal for all variants in current run.
            execution_params: Execution override mapping from request/template.
            market_id: Stable market id used for fee defaults lookup.
            target_slice: Trading/reporting window within warmup-inclusive candle arrays.
            init_cash_quote_default: Runtime default initial strategy balance.
            fixed_quote_default: Runtime default fixed notional for `fixed_quote` sizing.
            safe_profit_percent_default: Runtime default lock percent for profit-lock mode.
            slippage_pct_default: Runtime default slippage percent.
            fee_pct_default_by_market_id: Runtime fee defaults per market id.
            max_variants_guard: Guard forwarded to indicator compute requests.
            signals_cache_max_entries: Maximum cached signal vectors retained in memory
                (default: `2048`).
            signals_cache_max_bytes: Maximum total bytes retained by cached signal vectors
                (default: `32 * 1024 * 1024`).
            execution_engine: Optional custom close-fill engine implementation.
        Returns:
            None.
        Assumptions:
            Constructor stores dependencies and performs invariant checks only.
        Raises:
            ValueError: If one scalar/default invariant is invalid.
        Side Effects:
            None.
        """
        if indicator_compute is None:  # type: ignore[truthy-bool]
            raise ValueError("CloseFillBacktestStagedScorerV1 requires indicator_compute")
        if market_id <= 0:
            raise ValueError("CloseFillBacktestStagedScorerV1 market_id must be > 0")
        if init_cash_quote_default <= 0.0:
            raise ValueError("init_cash_quote_default must be > 0")
        if fixed_quote_default <= 0.0:
            raise ValueError("fixed_quote_default must be > 0")
        if safe_profit_percent_default < 0.0 or safe_profit_percent_default > 100.0:
            raise ValueError("safe_profit_percent_default must be in [0, 100]")
        if slippage_pct_default < 0.0:
            raise ValueError("slippage_pct_default must be >= 0")
        if max_variants_guard <= 0:
            raise ValueError("max_variants_guard must be > 0")
        if signals_cache_max_entries <= 0:
            raise ValueError("signals_cache_max_entries must be > 0")
        if signals_cache_max_bytes <= 0:
            raise ValueError("signals_cache_max_bytes must be > 0")
        if target_slice.start is None or target_slice.stop is None:
            raise ValueError("target_slice must define explicit start and stop")
        if target_slice.start < 0:
            raise ValueError("target_slice.start must be >= 0")
        if target_slice.stop < target_slice.start:
            raise ValueError("target_slice.stop must be >= target_slice.start")

        self._indicator_compute = indicator_compute
        self._direction_mode = direction_mode
        self._sizing_mode = sizing_mode
        self._execution_params = MappingProxyType(
            _normalize_scalar_mapping(values=execution_params)
        )
        self._market_id = market_id
        self._target_slice = target_slice
        self._init_cash_quote_default = init_cash_quote_default
        self._fixed_quote_default = fixed_quote_default
        self._safe_profit_percent_default = safe_profit_percent_default
        self._slippage_pct_default = slippage_pct_default
        self._fee_pct_default_by_market_id = _normalize_fee_defaults(
            values=fee_pct_default_by_market_id
        )
        self._max_variants_guard = max_variants_guard
        self._signals_cache_max_entries = signals_cache_max_entries
        self._signals_cache_max_bytes = signals_cache_max_bytes
        self._execution_engine = execution_engine or BacktestExecutionEngineV1()

        self._signals_cache: OrderedDict[str, _SignalCacheValue] = OrderedDict()
        self._signals_cache_total_bytes = 0
        self._signals_cache_lock = Lock()

    def score_variant(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]],
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Score one variant using deterministic close-fill execution over aggregated signals.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Warmup-inclusive candles.
            indicator_selections: Deterministic explicit indicator selections for variant.
            signal_params: Signal parameter mapping per indicator.
            risk_params: Risk payload for Stage B variant expansion.
            indicator_variant_key: Deterministic indicators-only variant key.
            variant_key: Deterministic full backtest variant key.
        Returns:
            Mapping[str, float]: Mapping with `Total Return [%]` ranking metric.
        Assumptions:
            Stage A disables SL/TP regardless of provided `risk_params` values.
        Raises:
            ValueError: If stage literal is unsupported or one scalar payload is invalid.
        Side Effects:
            Uses internal in-memory signal cache keyed by compute+signal identity.
        """
        details = self.score_variant_with_details(
            stage=stage,
            candles=candles,
            indicator_selections=indicator_selections,
            signal_params=signal_params,
            risk_params=risk_params,
            indicator_variant_key=indicator_variant_key,
            variant_key=variant_key,
        )
        return details.metrics

    def score_variant_with_details(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]],
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> BacktestVariantScoreDetailsV1:
        """
        Score one variant and return detailed deterministic execution payload for reporting.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Warmup-inclusive candles.
            indicator_selections: Deterministic explicit indicator selections for variant.
            signal_params: Signal parameter mapping per indicator.
            risk_params: Risk payload for Stage B variant expansion.
            indicator_variant_key: Deterministic indicators-only variant key.
            variant_key: Deterministic full backtest variant key.
        Returns:
            BacktestVariantScoreDetailsV1: Detailed payload with score metrics and execution data.
        Assumptions:
            Stage A still disables SL/TP regardless of provided `risk_params` values.
        Raises:
            ValueError: If stage literal is unsupported or one scalar payload is invalid.
        Side Effects:
            Uses internal in-memory signal cache keyed by compute+signal identity.
        """
        _ = variant_key
        resolved_stage = stage.strip().lower()
        if resolved_stage not in {STAGE_A_LITERAL, STAGE_B_LITERAL}:
            raise ValueError("stage must be stage_a or stage_b")

        normalized_signal_params = _normalize_nested_scalar_mapping(values=signal_params)
        cache_key = _signal_cache_key(
            indicator_variant_key=indicator_variant_key,
            signal_params=normalized_signal_params,
        )
        cached_signal = self._cached_signal(cache_key=cache_key)
        if cached_signal is None:
            final_signal = self._build_final_signal(
                candles=candles,
                indicator_selections=indicator_selections,
                signal_params=normalized_signal_params,
            )
            self._insert_cached_signal(cache_key=cache_key, final_signal=final_signal)
        else:
            final_signal = cached_signal.final_signal

        execution_params = self._resolve_execution_params()
        resolved_risk_params = self._resolve_risk_params(
            stage=resolved_stage,
            risk_params=risk_params,
        )
        outcome = self._execution_engine.run(
            candles=candles,
            target_slice=self._target_slice,
            final_signal=final_signal,
            execution_params=execution_params,
            risk_params=resolved_risk_params,
        )
        metrics = MappingProxyType({TOTAL_RETURN_METRIC_LITERAL: float(outcome.total_return_pct)})
        return BacktestVariantScoreDetailsV1(
            metrics=metrics,
            target_slice=self._target_slice,
            execution_params=execution_params,
            risk_params=resolved_risk_params,
            execution_outcome=outcome,
        )

    def _cached_signal(self, *, cache_key: str) -> _SignalCacheValue | None:
        """
        Return cached final-signal payload by deterministic cache key when available.

        Args:
            cache_key: Deterministic cache key.
        Returns:
            _SignalCacheValue | None: Cached payload or `None`.
        Assumptions:
            Cache key is stable across repeated calls for the same variant identity.
        Raises:
            None.
        Side Effects:
            Reads shared in-memory cache under lock.
        """
        with self._signals_cache_lock:
            cached = self._signals_cache.get(cache_key)
            if cached is None:
                return None
            self._signals_cache.move_to_end(cache_key)
            return cached

    def _insert_cached_signal(self, *, cache_key: str, final_signal: np.ndarray) -> None:
        """
        Insert compact signal vector into bounded in-memory cache with LRU eviction.

        Docs:
          - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
          - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
          - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
          - tests/unit/contexts/backtest/application/services/test_close_fill_scorer_v1.py

        Args:
            cache_key: Deterministic cache key.
            final_signal: Compact aggregated signal vector (`np.int8`).
        Returns:
            None.
        Assumptions:
            Cache stores compact vectors only; value size is derived from `nbytes`.
        Raises:
            None.
        Side Effects:
            Mutates shared cache map and may evict older entries.
        """
        cache_value = _SignalCacheValue(
            final_signal=np.ascontiguousarray(final_signal, dtype=np.int8),
        )
        cache_value_bytes = int(cache_value.final_signal.nbytes)
        with self._signals_cache_lock:
            existing = self._signals_cache.pop(cache_key, None)
            if existing is not None:
                self._signals_cache_total_bytes -= int(existing.final_signal.nbytes)

            # HOT PATH: bounded cache insert keeps memory stable for large variant spaces.
            self._signals_cache[cache_key] = cache_value
            self._signals_cache_total_bytes += cache_value_bytes
            self._signals_cache.move_to_end(cache_key)

            while (
                len(self._signals_cache) > self._signals_cache_max_entries
                or self._signals_cache_total_bytes > self._signals_cache_max_bytes
            ):
                _, evicted_value = self._signals_cache.popitem(last=False)
                self._signals_cache_total_bytes -= int(evicted_value.final_signal.nbytes)

    def _build_final_signal(
        self,
        *,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]],
    ) -> np.ndarray:
        """
        Compute deterministic aggregated final signal for one Stage A base variant.

        Args:
            candles: Warmup-inclusive candles.
            indicator_selections: Explicit indicator selections.
            signal_params: Normalized signal-parameter mapping.
        Returns:
            object: Compact `np.int8` final signal array.
        Assumptions:
            Indicator selections are unique by indicator id.
        Raises:
            ValueError: If dependency tensors cannot be computed deterministically.
        Side Effects:
            Calls indicator compute port for base and dependency indicators.
        """
        ordered_selections = tuple(
            sorted(indicator_selections, key=lambda item: item.indicator_id)
        )
        tensors_by_indicator: dict[str, IndicatorTensor] = {}
        indicator_inputs: dict[str, Mapping[str, BacktestVariantScalar]] = {}
        dependency_outputs: dict[str, Mapping[str, np.ndarray]] = {}

        for selection in ordered_selections:
            tensor = self._compute_tensor_for_selection(
                candles=candles,
                selection=selection,
            )
            indicator_id = selection.indicator_id
            tensors_by_indicator[indicator_id] = tensor
            indicator_inputs[indicator_id] = _normalize_scalar_mapping(values=selection.inputs)

            rule_spec = signal_rule_spec_v1(indicator_id=indicator_id)
            if len(rule_spec.required_dependency_ids) == 0:
                continue

            selection_dependency_outputs: dict[str, np.ndarray] = {}
            for dependency_id in rule_spec.required_dependency_ids:
                dependency_selection = IndicatorVariantSelection(
                    indicator_id=dependency_id,
                    inputs=selection.inputs,
                    params=selection.params,
                )
                dependency_tensor = self._compute_tensor_for_selection(
                    candles=candles,
                    selection=dependency_selection,
                )
                selection_dependency_outputs[dependency_id] = (
                    indicator_primary_output_series_from_tensor_v1(tensor=dependency_tensor)
                )
            dependency_outputs[indicator_id] = MappingProxyType(
                dict(sorted(selection_dependency_outputs.items(), key=lambda item: item[0]))
            )

        signal_inputs = build_indicator_signal_inputs_from_tensors_v1(
            tensors=tensors_by_indicator,
            indicator_inputs=indicator_inputs,
            signal_params=signal_params,
            dependency_outputs=dependency_outputs,
        )
        final_signal = evaluate_and_aggregate_signals_encoded_v1(
            candles=candles,
            indicator_inputs=signal_inputs,
        )
        return final_signal

    def _compute_tensor_for_selection(
        self,
        *,
        candles: CandleArrays,
        selection: IndicatorVariantSelection,
    ) -> IndicatorTensor:
        """
        Compute one indicator tensor for explicit variant selection.

        Args:
            candles: Warmup-inclusive candles.
            selection: Explicit indicator selection.
        Returns:
            object: Indicator tensor object returned by compute port.
        Assumptions:
            Selection maps represent explicit single-value axes.
        Raises:
            ValueError: If explicit axis construction fails.
        Side Effects:
            Calls indicator compute port.
        """
        source_axis = None
        params_axes: dict[str, ExplicitValuesSpec] = {}

        for input_name in sorted(selection.inputs.keys()):
            input_value = selection.inputs[input_name]
            if input_name == "source":
                source_axis = ExplicitValuesSpec(name="source", values=(str(input_value),))
            elif input_name not in selection.params:
                params_axes[input_name] = ExplicitValuesSpec(
                    name=input_name,
                    values=(_normalize_variant_scalar(value=input_value),),
                )

        for param_name in sorted(selection.params.keys()):
            param_value = selection.params[param_name]
            params_axes[param_name] = ExplicitValuesSpec(
                name=param_name,
                values=(_normalize_variant_scalar(value=param_value),),
            )

        grid = GridSpec(
            indicator_id=IndicatorId(selection.indicator_id),
            params=params_axes,
            source=source_axis,
        )
        return self._indicator_compute.compute(
            ComputeRequest(
                candles=candles,
                grid=grid,
                max_variants_guard=self._max_variants_guard,
                dtype="float32",
            )
        )

    def _resolve_execution_params(self) -> ExecutionParamsV1:
        """
        Resolve execution parameter object from runtime defaults and request overrides.

        Args:
            None.
        Returns:
            ExecutionParamsV1: Validated immutable execution settings.
        Assumptions:
            Missing fee override falls back to market-specific runtime defaults.
        Raises:
            ValueError: If one override value is invalid.
        Side Effects:
            None.
        """
        init_cash_quote = _resolve_number(
            values=self._execution_params,
            primary_key="init_cash_quote",
            secondary_key="init_cash",
            default=self._init_cash_quote_default,
        )
        fixed_quote = _resolve_number(
            values=self._execution_params,
            primary_key="fixed_quote",
            secondary_key="",
            default=self._fixed_quote_default,
        )
        safe_profit_percent = _resolve_number(
            values=self._execution_params,
            primary_key="safe_profit_percent",
            secondary_key="",
            default=self._safe_profit_percent_default,
        )
        slippage_pct = _resolve_number(
            values=self._execution_params,
            primary_key="slippage_pct",
            secondary_key="",
            default=self._slippage_pct_default,
        )
        fee_pct = _resolve_number(
            values=self._execution_params,
            primary_key="fee_pct",
            secondary_key="market_fee_pct",
            default=self._fee_pct_default_by_market_id[self._market_id],
        )

        return ExecutionParamsV1(
            direction_mode=self._direction_mode,
            sizing_mode=self._sizing_mode,
            init_cash_quote=init_cash_quote,
            fixed_quote=fixed_quote,
            safe_profit_percent=safe_profit_percent,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
        )

    def _resolve_risk_params(
        self,
        *,
        stage: str,
        risk_params: Mapping[str, BacktestVariantScalar],
    ) -> RiskParamsV1:
        """
        Resolve stage-aware risk settings where Stage A always disables SL/TP.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            risk_params: Raw risk mapping from staged runner.
        Returns:
            RiskParamsV1: Validated immutable risk settings.
        Assumptions:
            Stage A ignores incoming risk values and forces disabled SL/TP axes.
        Raises:
            ValueError: If one risk scalar type is invalid.
        Side Effects:
            None.
        """
        if stage == STAGE_A_LITERAL:
            return RiskParamsV1(
                sl_enabled=False,
                sl_pct=None,
                tp_enabled=False,
                tp_pct=None,
            )

        sl_enabled = _resolve_bool(values=risk_params, key="sl_enabled", default=False)
        tp_enabled = _resolve_bool(values=risk_params, key="tp_enabled", default=False)
        sl_pct = _resolve_optional_number(values=risk_params, key="sl_pct")
        tp_pct = _resolve_optional_number(values=risk_params, key="tp_pct")
        return RiskParamsV1(
            sl_enabled=sl_enabled,
            sl_pct=sl_pct,
            tp_enabled=tp_enabled,
            tp_pct=tp_pct,
        )


def _signal_cache_key(
    *,
    indicator_variant_key: str,
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]],
) -> str:
    """
    Build deterministic cache key for computed aggregated signal vector.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_close_fill_scorer_v1.py

    Args:
        indicator_variant_key: Deterministic indicators-only key.
        signal_params: Normalized signal-parameter mapping.
    Returns:
        str: Deterministic cache key string.
    Assumptions:
        Key serialization uses key-sorted canonical JSON.
    Raises:
        None.
    Side Effects:
        None.
    """
    payload = {
        "indicator_variant_key": indicator_variant_key.strip().lower(),
        "signal_params": _signal_params_json_payload(signal_params=signal_params),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _signal_params_json_payload(
    *,
    signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]],
) -> dict[str, dict[str, BacktestVariantScalar]]:
    """
    Convert normalized nested signal params mapping into JSON-serializable structure.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - tests/unit/contexts/backtest/application/services/test_close_fill_scorer_v1.py

    Args:
        signal_params: Normalized signal-params mapping.
    Returns:
        dict[str, dict[str, BacktestVariantScalar]]: Deterministic plain-dict payload.
    Assumptions:
        Nested parameter mappings are key-addressable by strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    payload: dict[str, dict[str, BacktestVariantScalar]] = {}
    for indicator_id in sorted(signal_params.keys()):
        params = signal_params[indicator_id]
        payload[indicator_id] = {
            param_name: params[param_name] for param_name in sorted(params.keys())
        }
    return payload


def _resolve_number(
    *,
    values: Mapping[str, BacktestVariantScalar],
    primary_key: str,
    secondary_key: str,
    default: float,
) -> float:
    """
    Resolve numeric override by primary/secondary key with deterministic fallback.

    Args:
        values: Scalar mapping with optional override values.
        primary_key: Preferred key name.
        secondary_key: Optional backward-compatible key name.
        default: Fallback value.
    Returns:
        float: Resolved numeric value.
    Assumptions:
        Bool values are rejected despite being `int` subclass.
    Raises:
        ValueError: If override is not numeric scalar.
    Side Effects:
        None.
    """
    candidate: BacktestVariantScalar | None = None
    if primary_key in values:
        candidate = values[primary_key]
    elif secondary_key and secondary_key in values:
        candidate = values[secondary_key]
    if candidate is None:
        return float(default)
    if isinstance(candidate, bool) or not isinstance(candidate, int | float):
        raise ValueError(f"execution field '{primary_key}' must be numeric")
    return float(candidate)


def _resolve_optional_number(
    *,
    values: Mapping[str, BacktestVariantScalar],
    key: str,
) -> float | None:
    """
    Resolve optional numeric scalar from mapping.

    Args:
        values: Scalar mapping payload.
        key: Requested key name.
    Returns:
        float | None: Numeric value or `None` when key is absent/null.
    Assumptions:
        Bool values are rejected despite being `int` subclass.
    Raises:
        ValueError: If provided scalar is not numeric.
    Side Effects:
        None.
    """
    candidate = values.get(key)
    if candidate is None:
        return None
    if isinstance(candidate, bool) or not isinstance(candidate, int | float):
        raise ValueError(f"risk field '{key}' must be numeric when provided")
    return float(candidate)


def _resolve_bool(
    *,
    values: Mapping[str, BacktestVariantScalar],
    key: str,
    default: bool,
) -> bool:
    """
    Resolve boolean scalar from mapping with deterministic fallback.

    Args:
        values: Scalar mapping payload.
        key: Requested key name.
        default: Fallback boolean when key is missing.
    Returns:
        bool: Resolved boolean value.
    Assumptions:
        Backtest risk flags must be explicit booleans.
    Raises:
        ValueError: If provided value is not boolean.
    Side Effects:
        None.
    """
    if key not in values:
        return default
    candidate = values[key]
    if not isinstance(candidate, bool):
        raise ValueError(f"risk field '{key}' must be boolean")
    return candidate


def _normalize_variant_scalar(*, value: object) -> int | float | str:
    """
    Normalize explicit scalar value into supported indicator grid scalar type.

    Args:
        value: Raw scalar value.
    Returns:
        int | float | str: Normalized scalar value.
    Assumptions:
        Indicator grid explicit values must be JSON-compatible scalars.
    Raises:
        ValueError: If scalar type is unsupported.
    Side Effects:
        None.
    """
    if isinstance(value, bool):
        raise ValueError("indicator scalar values must not be bool")
    if isinstance(value, int | float | str):
        return value
    raise ValueError("indicator scalar values must be int, float, or str")


def _normalize_scalar_mapping(
    *,
    values: Mapping[str, BacktestVariantScalar],
) -> dict[str, BacktestVariantScalar]:
    """
    Normalize scalar mapping into deterministic lowercase key-sorted dictionary.

    Args:
        values: Scalar mapping payload.
    Returns:
        dict[str, BacktestVariantScalar]: Normalized deterministic mapping.
    Assumptions:
        Values are JSON-compatible scalars.
    Raises:
        ValueError: If one key is blank after normalization.
    Side Effects:
        None.
    """
    normalized: dict[str, BacktestVariantScalar] = {}
    for raw_key in sorted(values.keys(), key=lambda item: str(item).strip().lower()):
        key = str(raw_key).strip().lower()
        if not key:
            raise ValueError("mapping keys must be non-empty")
        normalized[key] = values[raw_key]
    return normalized


def _normalize_nested_scalar_mapping(
    *,
    values: Mapping[str, Mapping[str, BacktestVariantScalar]],
) -> dict[str, Mapping[str, BacktestVariantScalar]]:
    """
    Normalize nested signal parameter mapping with deterministic key ordering.

    Args:
        values: Mapping `indicator_id -> param_name -> scalar`.
    Returns:
        dict[str, Mapping[str, BacktestVariantScalar]]: Normalized nested mapping.
    Assumptions:
        Nested values are scalar JSON-compatible values.
    Raises:
        ValueError: If one indicator id or parameter name is blank.
    Side Effects:
        None.
    """
    normalized: dict[str, Mapping[str, BacktestVariantScalar]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda item: str(item).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("signal_params indicator_id keys must be non-empty")
        params = values[raw_indicator_id]
        normalized[indicator_id] = MappingProxyType(_normalize_scalar_mapping(values=params))
    return normalized


def _normalize_fee_defaults(
    *,
    values: Mapping[int, float] | None,
) -> Mapping[int, float]:
    """
    Normalize fee defaults mapping and apply v1 fallback values when mapping is absent.

    Args:
        values: Optional mapping `market_id -> fee_pct`.
    Returns:
        Mapping[int, float]: Normalized immutable mapping.
    Assumptions:
        Fee values use human percent units.
    Raises:
        ValueError: If one market id or fee value is invalid.
    Side Effects:
        None.
    """
    source = _DEFAULT_FEE_PCT_BY_MARKET_ID if values is None else values
    normalized: dict[int, float] = {}
    for raw_key in sorted(source.keys()):
        market_id = int(raw_key)
        fee_pct = float(source[raw_key])
        if market_id <= 0:
            raise ValueError("fee default market ids must be > 0")
        if fee_pct < 0.0:
            raise ValueError("fee defaults must be >= 0")
        normalized[market_id] = fee_pct

    if len(normalized) == 0:
        raise ValueError("fee defaults mapping must be non-empty")
    return MappingProxyType(normalized)


__all__ = [
    "CloseFillBacktestStagedScorerV1",
]
