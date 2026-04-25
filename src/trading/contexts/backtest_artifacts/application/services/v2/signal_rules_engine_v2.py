"""Deterministic v2-aligned signal-rules engine for precompute pipeline (R4-01)."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.backtest.application.services.signals_from_indicators_v1 import (
    IndicatorSignalEvaluationInputV1,
    evaluate_indicator_signal_encoded_v1,
    signal_rule_spec_v1,
    supported_indicator_ids_for_signals_v1,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    SIGNAL_CODE_DTYPE_LITERAL_V2,
    SIGNAL_CODE_LONG_V2,
    SIGNAL_CODE_NEUTRAL_V2,
    SIGNAL_CODE_SHORT_V2,
    SIGNAL_CODE_VALUE_SET_V2,
    SIGNALS_V1_PARAMS_PATH_LITERAL_V2,
    SIGNALS_V1_PARAMS_POLICY_LITERAL_V2,
    SignalRuleEvaluationRequestV2,
    SignalRuleEvaluationResultV2,
    SignalRuleScalarV2,
    SignalRuleSpecV2,
    validate_signal_input_source_v2,
    validate_signal_rule_family_v2,
)
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec

_RULE_COMPARE_PRICE_TO_OUTPUT = "compare_price_to_output"


def _build_signal_rule_registry_v2() -> dict[str, SignalRuleSpecV2]:
    """
    Build the explicit indicator-id -> rule-spec registry for the R4-01 v2 engine.

    The registry is derived from the already frozen v1 rule catalog to minimize semantic drift
    while exposing an explicit v2 contract surface for precompute-oriented callers.

    Args:
        None.
    Returns:
        dict[str, SignalRuleSpecV2]: Deterministic sorted registry.
    Assumptions:
        The v1 registry already represents the approved supported indicator catalog.
    Raises:
        ValueError: If one derived v2 rule specification is invalid.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - configs/prod/indicators.yaml
    """
    registry: dict[str, SignalRuleSpecV2] = {}
    for indicator_id in supported_indicator_ids_for_signals_v1():
        v1_spec = signal_rule_spec_v1(indicator_id=indicator_id)
        registry[indicator_id] = SignalRuleSpecV2(
            indicator_id=indicator_id,
            rule_family=validate_signal_rule_family_v2(v1_spec.rule_family),
            required_signal_params=v1_spec.required_signal_params,
            required_dependency_ids=v1_spec.required_dependency_ids,
            uses_inputs_source=v1_spec.rule_family == _RULE_COMPARE_PRICE_TO_OUTPUT,
            threshold_center=v1_spec.threshold_center,
            candle_body_min_param_name=v1_spec.candle_body_min_param_name,
        )
    return dict(sorted(registry.items(), key=lambda item: item[0]))


_SIGNAL_RULE_REGISTRY_V2 = MappingProxyType(_build_signal_rule_registry_v2())


def supported_indicator_ids_for_signal_rules_v2() -> tuple[str, ...]:
    """
    Return the deterministic ordered indicator catalog supported by the v2 rules engine.

    Args:
        None.
    Returns:
        tuple[str, ...]: Stable sorted supported indicator ids.
    Assumptions:
        Registry order is canonicalized once at module import time.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - configs/prod/indicators.yaml
    """
    return tuple(_SIGNAL_RULE_REGISTRY_V2.keys())


def list_signal_rule_registry_v2() -> tuple[tuple[str, str], ...]:
    """
    Return `(indicator_id, rule_family)` pairs for deterministic registry introspection.

    Args:
        None.
    Returns:
        tuple[tuple[str, str], ...]: Stable sorted registry projection.
    Assumptions:
        Rule-family semantics are fixed for R4-01 and deterministic by indicator id.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_rules_engine_v2.py
    """
    return tuple(
        (indicator_id, _SIGNAL_RULE_REGISTRY_V2[indicator_id].rule_family)
        for indicator_id in supported_indicator_ids_for_signal_rules_v2()
    )


def signal_rule_spec_v2(*, indicator_id: str) -> SignalRuleSpecV2:
    """
    Resolve one v2 rule specification by indicator id.

    Args:
        indicator_id: Requested indicator identifier.
    Returns:
        SignalRuleSpecV2: Explicit v2 rule specification.
    Assumptions:
        Identifier lookup is case-insensitive after normalization.
    Raises:
        ValueError: If the indicator id is blank or not supported by the v2 registry.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - configs/prod/indicators.yaml
    """
    normalized_indicator_id = indicator_id.strip().lower()
    if not normalized_indicator_id:
        raise ValueError("indicator_id must be non-empty")
    spec = _SIGNAL_RULE_REGISTRY_V2.get(normalized_indicator_id)
    if spec is None:
        raise ValueError(
            f"Unsupported indicator_id for signal rules v2: {normalized_indicator_id}"
        )
    return spec


@dataclass(frozen=True, slots=True)
class BacktestSignalRulesEngineV2:
    """
    Production signal-rules engine for R4-01 precompute-facing signal evaluation.

    The engine keeps `signals.v1.params` on a strict `default-only` policy, resolves explicit
    `inputs.source` semantics from runtime defaults, and reuses the proven v1 kernels to preserve
    signal semantics while exposing a dedicated v2 API.

    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
    """

    defaults_provider: BacktestGridDefaultsProvider

    def __post_init__(self) -> None:
        """
        Validate startup invariants between runtime defaults and the explicit v2 registry.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Engine construction happens during startup/wiring and should fail fast on drift.
        Raises:
            ValueError: If supported indicator ids, source catalogs, or default-only params drift.
        Side Effects:
            None.
        Docs:
        Related:
          - configs/prod/indicators.yaml
          - src/trading/contexts/backtest/application/services/v2/contracts.py
        """
        _validate_defaults_provider_contract_v2(defaults_provider=self.defaults_provider)

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return the canonical supported indicator catalog for this engine instance.

        Args:
            None.
        Returns:
            tuple[str, ...]: Stable sorted indicator ids.
        Assumptions:
            Catalog is validated against the defaults provider during engine construction.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
          - configs/prod/indicators.yaml
        """
        return supported_indicator_ids_for_signal_rules_v2()

    def rule_spec(self, *, indicator_id: str) -> SignalRuleSpecV2:
        """
        Resolve one explicit v2 rule specification by indicator id.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            SignalRuleSpecV2: Explicit v2 rule specification.
        Assumptions:
            All returned specs were validated at import/startup time.
        Raises:
            ValueError: If the indicator is blank or unsupported.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - configs/prod/indicators.yaml
        """
        return signal_rule_spec_v2(indicator_id=indicator_id)

    def resolved_defaults(
        self,
        *,
        indicator_id: str,
    ) -> tuple[str | None, Mapping[str, SignalRuleScalarV2]]:
        """
        Resolve deterministic default-only inputs required for chunked signal execution.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            tuple[str | None, Mapping[str, SignalRuleScalarV2]]: Default `inputs.source`
                literal and the materialized `signals.v1.params` mapping.
        Assumptions:
            Artifact-precompute chunk workers must avoid re-reading mutable startup state and may
            reuse this resolved contract across every chunk of the same indicator target.
        Raises:
            ValueError: If the indicator is blank, unsupported, or defaults drift from startup
                validation invariants.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
          - configs/prod/indicators.yaml
        """
        spec = signal_rule_spec_v2(indicator_id=indicator_id)
        return (
            _default_inputs_source_from_defaults_v2(
                indicator_id=spec.indicator_id,
                defaults_provider=self.defaults_provider,
                spec=spec,
            ),
            dict(
                _materialize_default_signal_params_v2(
                    indicator_id=spec.indicator_id,
                    defaults_provider=self.defaults_provider,
                    spec=spec,
                )
            ),
        )

    def evaluate(
        self,
        *,
        request: SignalRuleEvaluationRequestV2,
    ) -> SignalRuleEvaluationResultV2:
        """
        Evaluate one indicator into compact `{-1,0,1}` v2 signal codes.

        Args:
            request: Typed v2 evaluation envelope for one indicator variant/series.
        Returns:
            SignalRuleEvaluationResultV2: Compact deterministic evaluation result.
        Assumptions:
            `request.primary_output` and dependency outputs are bar-aligned with candles.
        Raises:
            ValueError: If indicator id is unsupported, `inputs.source` is invalid, or
                `signals.v1.params` violates the `default-only` contract.
        Side Effects:
            None.
        Docs:
        Related:
          - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
          - src/trading/contexts/backtest/application/services/v2/contracts.py
          - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
        """
        spec = signal_rule_spec_v2(indicator_id=request.indicator_id)
        resolved_source = _resolve_inputs_source_v2(
            indicator_id=spec.indicator_id,
            request_source=request.inputs_source,
            defaults_provider=self.defaults_provider,
            spec=spec,
        )
        resolved_signal_params = _resolve_signal_params_default_only_v2(
            indicator_id=spec.indicator_id,
            request_signal_params=request.signal_params,
            defaults_provider=self.defaults_provider,
            spec=spec,
        )
        v1_indicator_input = IndicatorSignalEvaluationInputV1(
            indicator_id=spec.indicator_id,
            primary_output=request.primary_output,
            indicator_inputs=_indicator_inputs_mapping_v2(
                resolved_source=resolved_source,
                spec=spec,
            ),
            signal_params=resolved_signal_params,
            dependency_outputs=request.dependency_outputs,
        )
        signal_codes = evaluate_indicator_signal_encoded_v1(
            candles=request.candles,
            indicator_input=v1_indicator_input,
        )
        normalized_signal_codes = _normalize_signal_codes_v2(
            indicator_id=spec.indicator_id,
            signal_codes=signal_codes,
        )
        return SignalRuleEvaluationResultV2(
            indicator_id=spec.indicator_id,
            rule_family=spec.rule_family,
            inputs_source=resolved_source,
            signal_params=MappingProxyType(dict(resolved_signal_params)),
            signal_codes=normalized_signal_codes,
        )


def _validate_defaults_provider_contract_v2(
    *,
    defaults_provider: BacktestGridDefaultsProvider,
) -> None:
    """
    Validate that runtime defaults exactly cover the explicit v2 rules-engine contract.

    Args:
        defaults_provider: Runtime defaults provider used as source of truth.
    Returns:
        None.
    Assumptions:
        Startup validation should reject catalog drift before precompute/runtime execution starts.
    Raises:
        ValueError: If indicator coverage, source catalogs, or default-only params are invalid.
    Side Effects:
        None.
    Docs:
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
    """
    provider_indicator_ids = tuple(defaults_provider.supported_indicator_ids())
    registry_indicator_ids = supported_indicator_ids_for_signal_rules_v2()
    provider_set = set(provider_indicator_ids)
    registry_set = set(registry_indicator_ids)
    missing_in_registry = tuple(sorted(provider_set - registry_set))
    missing_in_defaults = tuple(sorted(registry_set - provider_set))
    if len(missing_in_registry) > 0 or len(missing_in_defaults) > 0:
        raise ValueError(
            "signal rules v2 indicator catalog drift: "
            f"missing_in_registry={missing_in_registry}, "
            f"missing_in_defaults={missing_in_defaults}"
        )

    for indicator_id in registry_indicator_ids:
        spec = signal_rule_spec_v2(indicator_id=indicator_id)
        _validate_allowed_source_catalog_v2(
            indicator_id=indicator_id,
            defaults_provider=defaults_provider,
        )
        _default_inputs_source_from_defaults_v2(
            indicator_id=indicator_id,
            defaults_provider=defaults_provider,
            spec=spec,
        )
        _materialize_default_signal_params_v2(
            indicator_id=indicator_id,
            defaults_provider=defaults_provider,
            spec=spec,
        )


def _validate_allowed_source_catalog_v2(
    *,
    indicator_id: str,
    defaults_provider: BacktestGridDefaultsProvider,
) -> tuple[str, ...]:
    """
    Validate and normalize allowed `inputs.source` literals for one indicator id.

    Args:
        indicator_id: Indicator identifier.
        defaults_provider: Runtime defaults provider.
    Returns:
        tuple[str, ...]: Stable validated allowed source literals.
    Assumptions:
        Source catalogs are authoritative runtime/defaults metadata and may be empty.
    Raises:
        ValueError: If one configured source literal is unsupported.
    Side Effects:
        None.
    Docs:
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    validated_values: list[str] = []
    for value in defaults_provider.allowed_source_values(indicator_id=indicator_id):
        validated_values.append(validate_signal_input_source_v2(value))
    return tuple(validated_values)


def _default_inputs_source_from_defaults_v2(
    *,
    indicator_id: str,
    defaults_provider: BacktestGridDefaultsProvider,
    spec: SignalRuleSpecV2,
) -> str | None:
    """
    Resolve the deterministic default `inputs.source` literal for one indicator id.

    Args:
        indicator_id: Indicator identifier.
        defaults_provider: Runtime defaults provider.
        spec: Explicit rule specification for the indicator.
    Returns:
        str | None: Default `inputs.source` literal or `None` when source is irrelevant.
    Assumptions:
        When compare-price rules have no configurable source axis, they default to `close`.
    Raises:
        ValueError: If the configured default source literal is unsupported or missing.
    Side Effects:
        None.
    Docs:
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    compute_defaults = defaults_provider.compute_defaults(indicator_id=indicator_id)
    source_values = _materialized_source_values_from_grid_v2(
        indicator_id=indicator_id,
        compute_defaults=compute_defaults,
    )
    if len(source_values) > 0:
        return source_values[0]
    if spec.uses_inputs_source:
        return "close"
    return None


def _materialized_source_values_from_grid_v2(
    *,
    indicator_id: str,
    compute_defaults: GridSpec | None,
) -> tuple[str, ...]:
    """
    Materialize ordered source-axis defaults from one compute grid definition.

    Args:
        indicator_id: Indicator identifier.
        compute_defaults: Optional compute defaults grid.
    Returns:
        tuple[str, ...]: Deterministic materialized source literals.
    Assumptions:
        Source-axis values in compute defaults preserve the authoritative default ordering.
    Raises:
        ValueError: If a materialized source value is blank or unsupported.
    Side Effects:
        None.
    Docs:
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    if compute_defaults is None or compute_defaults.source is None:
        return ()
    resolved: list[str] = []
    for raw_value in compute_defaults.source.materialize():
        if not isinstance(raw_value, str):
            raise ValueError(f"{indicator_id}: inputs.source defaults must be strings")
        resolved.append(validate_signal_input_source_v2(raw_value))
    return tuple(resolved)


def _materialize_default_signal_params_v2(
    *,
    indicator_id: str,
    defaults_provider: BacktestGridDefaultsProvider,
    spec: SignalRuleSpecV2,
) -> Mapping[str, SignalRuleScalarV2]:
    """
    Materialize authoritative scalar defaults for `signals.v1.params` of one indicator id.

    Args:
        indicator_id: Indicator identifier.
        defaults_provider: Runtime defaults provider.
        spec: Explicit v2 rule specification.
    Returns:
        Mapping[str, SignalRuleScalarV2]: Immutable resolved default param mapping.
    Assumptions:
        R4-01 keeps `signals.v1.params` strictly `default-only`, so each param must resolve to one
            scalar value.
    Raises:
        ValueError: If one default param materializes to zero/multiple values or a required param
            is missing.
    Side Effects:
        None.
    Docs:
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    default_specs = defaults_provider.signal_param_defaults(indicator_id=indicator_id)
    resolved_params: dict[str, SignalRuleScalarV2] = {}
    for param_name in sorted(default_specs.keys()):
        resolved_params[param_name] = _single_default_scalar_from_grid_spec_v2(
            indicator_id=indicator_id,
            param_name=param_name,
            spec=default_specs[param_name],
        )
    for required_param_name in spec.required_signal_params:
        if required_param_name not in resolved_params:
            raise ValueError(
                f"{indicator_id}: missing default {SIGNALS_V1_PARAMS_PATH_LITERAL_V2} "
                f"param '{required_param_name}'"
            )
    return MappingProxyType(resolved_params)


def _single_default_scalar_from_grid_spec_v2(
    *,
    indicator_id: str,
    param_name: str,
    spec: GridParamSpec,
) -> SignalRuleScalarV2:
    """
    Materialize one default-only signal param into exactly one scalar value.

    Args:
        indicator_id: Indicator identifier.
        param_name: Signal parameter name.
        spec: Grid/defaults spec from the runtime defaults provider.
    Returns:
        SignalRuleScalarV2: Single authoritative scalar default value.
    Assumptions:
        Default-only R4-01 signal params must not expand into ranges or multi-value grids.
    Raises:
        ValueError: If materialization yields zero or multiple values or a non-scalar value.
    Side Effects:
        None.
    Docs:
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
    """
    materialized = tuple(spec.materialize())
    if len(materialized) != 1:
        raise ValueError(
            f"{indicator_id}: {SIGNALS_V1_PARAMS_PATH_LITERAL_V2} is "
            f"{SIGNALS_V1_PARAMS_POLICY_LITERAL_V2} and requires a single default value for "
            f"'{param_name}'"
        )
    return _normalize_signal_scalar_v2(
        value=materialized[0],
        field_name=f"{indicator_id}.{SIGNALS_V1_PARAMS_PATH_LITERAL_V2}.{param_name}",
    )


def _normalize_signal_scalar_v2(
    *,
    value: object,
    field_name: str,
) -> SignalRuleScalarV2:
    """
    Validate that one signal/default value is a supported scalar.

    Args:
        value: Candidate scalar value.
        field_name: Stable field label used in deterministic error messages.
    Returns:
        SignalRuleScalarV2: Normalized scalar value.
    Assumptions:
        Signal params are scalar config/runtime literals and must not carry list/array payloads.
    Raises:
        ValueError: If the value is not a supported scalar type.
    Side Effects:
        Converts NumPy scalar wrappers into Python scalars.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - configs/prod/indicators.yaml
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"{field_name} must be scalar")


def _resolve_inputs_source_v2(
    *,
    indicator_id: str,
    request_source: str | None,
    defaults_provider: BacktestGridDefaultsProvider,
    spec: SignalRuleSpecV2,
) -> str | None:
    """
    Resolve and validate the effective `inputs.source` literal for one evaluation request.

    Args:
        indicator_id: Indicator identifier.
        request_source: Optional request-level `inputs.source`.
        defaults_provider: Runtime defaults provider.
        spec: Explicit v2 rule specification.
    Returns:
        str | None: Effective source literal or `None` when the indicator has no source axis.
    Assumptions:
        Compare-price rules fall back to `close` when no configurable source axis exists.
    Raises:
        ValueError: If the request source is unsupported or not allowed by defaults.
    Side Effects:
        None.
    Docs:
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    allowed_sources = _validate_allowed_source_catalog_v2(
        indicator_id=indicator_id,
        defaults_provider=defaults_provider,
    )
    default_source = _default_inputs_source_from_defaults_v2(
        indicator_id=indicator_id,
        defaults_provider=defaults_provider,
        spec=spec,
    )
    if request_source is None or not request_source.strip():
        return default_source

    normalized_source = validate_signal_input_source_v2(request_source)
    if len(allowed_sources) > 0:
        if normalized_source not in allowed_sources:
            raise ValueError(
                f"{indicator_id}: inputs.source '{normalized_source}' is not allowed; "
                f"expected one of {allowed_sources}"
            )
        return normalized_source
    if default_source is not None and normalized_source != default_source:
        raise ValueError(
            f"{indicator_id}: inputs.source '{normalized_source}' is not allowed; "
            f"expected '{default_source}'"
        )
    if default_source is None:
        raise ValueError(
            f"{indicator_id}: inputs.source is not supported by this indicator"
        )
    return default_source


def _resolve_signal_params_default_only_v2(
    *,
    indicator_id: str,
    request_signal_params: Mapping[str, SignalRuleScalarV2],
    defaults_provider: BacktestGridDefaultsProvider,
    spec: SignalRuleSpecV2,
) -> Mapping[str, SignalRuleScalarV2]:
    """
    Resolve effective signal params while enforcing `signals.v1.params = default-only`.

    Args:
        indicator_id: Indicator identifier.
        request_signal_params: Optional request-level signal-param mapping.
        defaults_provider: Runtime defaults provider.
        spec: Explicit v2 rule specification.
    Returns:
        Mapping[str, SignalRuleScalarV2]: Immutable resolved signal-param mapping.
    Assumptions:
        Omitted params inherit defaults; provided params must match the exact default scalar.
    Raises:
        ValueError: If a requested param is unknown or differs from the authoritative default.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    resolved_defaults = dict(
        _materialize_default_signal_params_v2(
            indicator_id=indicator_id,
            defaults_provider=defaults_provider,
            spec=spec,
        )
    )
    for raw_param_name in sorted(request_signal_params.keys()):
        param_name = str(raw_param_name).strip().lower()
        if not param_name:
            raise ValueError("signal_params keys must be non-empty")
        requested_value = _normalize_signal_scalar_v2(
            value=request_signal_params[raw_param_name],
            field_name=f"{indicator_id}.{SIGNALS_V1_PARAMS_PATH_LITERAL_V2}.{param_name}",
        )
        default_value = resolved_defaults.get(param_name)
        if default_value is None and param_name not in resolved_defaults:
            raise ValueError(
                f"{indicator_id}: {SIGNALS_V1_PARAMS_PATH_LITERAL_V2} is "
                f"{SIGNALS_V1_PARAMS_POLICY_LITERAL_V2}"
            )
        if requested_value != default_value:
            raise ValueError(
                f"{indicator_id}: {SIGNALS_V1_PARAMS_PATH_LITERAL_V2} is "
                f"{SIGNALS_V1_PARAMS_POLICY_LITERAL_V2}"
            )
        resolved_defaults[param_name] = requested_value
    return MappingProxyType(resolved_defaults)


def _indicator_inputs_mapping_v2(
    *,
    resolved_source: str | None,
    spec: SignalRuleSpecV2,
) -> Mapping[str, SignalRuleScalarV2]:
    """
    Build the v1-compatible indicator-input mapping used by the shared evaluation kernel.

    Args:
        resolved_source: Effective resolved `inputs.source` literal.
        spec: Explicit v2 rule specification.
    Returns:
        Mapping[str, SignalRuleScalarV2]: Deterministic indicator-input mapping.
    Assumptions:
        Only compare-price rules need `inputs.source` for same-bar price/output comparison.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    if not spec.uses_inputs_source or resolved_source is None:
        return {}
    return MappingProxyType({"source": resolved_source})


def _normalize_signal_codes_v2(
    *,
    indicator_id: str,
    signal_codes: np.ndarray,
) -> np.ndarray:
    """
    Normalize output signal codes into canonical compact `int8` representation.

    Args:
        indicator_id: Indicator identifier used in diagnostics.
        signal_codes: Raw compact signal array returned by the shared kernel.
    Returns:
        np.ndarray: C-contiguous `np.int8` signal array with value set `{-1,0,1}`.
    Assumptions:
        Shared kernels already implement the canonical signal semantics.
    Raises:
        ValueError: If shape, dtype conversion, or value set violates the R4-01 contract.
    Side Effects:
        Copies the array only when dtype/layout normalization is needed.
    Docs:
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
      - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
    """
    normalized = np.ascontiguousarray(signal_codes, dtype=np.int8)
    if normalized.ndim != 1:
        raise ValueError(f"{indicator_id}: signal_codes must be 1D")
    invalid_mask = (
        (normalized != SIGNAL_CODE_SHORT_V2)
        & (normalized != SIGNAL_CODE_NEUTRAL_V2)
        & (normalized != SIGNAL_CODE_LONG_V2)
    )
    if np.any(invalid_mask):
        raise ValueError(
            f"{indicator_id}: signal_codes must use {SIGNAL_CODE_DTYPE_LITERAL_V2} "
            f"value set {SIGNAL_CODE_VALUE_SET_V2}"
        )
    return normalized


__all__ = [
    "BacktestSignalRulesEngineV2",
    "list_signal_rule_registry_v2",
    "signal_rule_spec_v2",
    "supported_indicator_ids_for_signal_rules_v2",
]
