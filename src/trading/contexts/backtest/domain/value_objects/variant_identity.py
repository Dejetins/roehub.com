from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

BacktestVariantScalar = int | float | str | bool | None
BacktestSignalsScalarMap = Mapping[str, BacktestVariantScalar]
BacktestSignalsMap = Mapping[str, BacktestSignalsScalarMap]


@dataclass(frozen=True, slots=True)
class BacktestVariantIdentity:
    """
    Stable backtest variant identity exposed to API/UI contracts.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
    """

    variant_index: int
    variant_key: str

    def __post_init__(self) -> None:
        """
        Validate identity invariants for deterministic variant addressing.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `variant_key` must be hex SHA-256 string from canonical payload v1.
        Raises:
            ValueError: If index is negative or key shape is invalid.
        Side Effects:
            Normalizes `variant_key` to lowercase trimmed representation.
        """
        if self.variant_index < 0:
            raise ValueError("BacktestVariantIdentity.variant_index must be >= 0")
        normalized_key = self.variant_key.strip().lower()
        object.__setattr__(self, "variant_key", normalized_key)
        if len(normalized_key) != 64:
            raise ValueError("BacktestVariantIdentity.variant_key must be 64 hex chars")
        for char in normalized_key:
            if char not in "0123456789abcdef":
                raise ValueError("BacktestVariantIdentity.variant_key must be lowercase hex")


def build_backtest_variant_key_v1(
    *,
    indicator_variant_key: str,
    direction_mode: str,
    sizing_mode: str,
    signals: BacktestSignalsMap | None = None,
    risk_params: Mapping[str, BacktestVariantScalar] | None = None,
    execution_params: Mapping[str, BacktestVariantScalar] | None = None,
) -> str:
    """
    Build deterministic backtest `variant_key` v1 from canonical JSON payload.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - tests/unit/contexts/backtest/application/test_backtest_errors.py

    Args:
        indicator_variant_key: Stable indicators variant key built by indicators v1 contract.
        direction_mode: Direction mode literal (`long-only`, `short-only`, `long-short`).
        sizing_mode: Position sizing mode literal.
        signals: Optional per-indicator signal parameter mapping.
        risk_params: Optional risk parameter mapping.
        execution_params: Optional execution parameter mapping.
    Returns:
        str: Hex SHA-256 backtest variant key.
    Assumptions:
        Indicators key semantics stay owned by indicators context; backtest composes on top.
    Raises:
        ValueError: If required literals are blank.
    Side Effects:
        None.
    """
    normalized_indicator_key = indicator_variant_key.strip().lower()
    if not normalized_indicator_key:
        raise ValueError("indicator_variant_key must be non-empty")
    normalized_direction_mode = direction_mode.strip().lower()
    if not normalized_direction_mode:
        raise ValueError("direction_mode must be non-empty")
    normalized_sizing_mode = sizing_mode.strip().lower()
    if not normalized_sizing_mode:
        raise ValueError("sizing_mode must be non-empty")

    payload = {
        "direction_mode": normalized_direction_mode,
        "execution": _normalize_scalar_mapping(values=execution_params),
        "indicator_variant_key": normalized_indicator_key,
        "risk": _normalize_scalar_mapping(values=risk_params),
        "schema_version": 1,
        "signals": _normalize_signals_mapping(values=signals),
        "sizing_mode": normalized_sizing_mode,
    }
    canonical_json = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _normalize_scalar_mapping(
    *,
    values: Mapping[str, BacktestVariantScalar] | None,
) -> dict[str, BacktestVariantScalar]:
    """
    Normalize optional scalar mapping into deterministic key-sorted plain dictionary.

    Args:
        values: Optional scalar mapping payload.
    Returns:
        dict[str, BacktestVariantScalar]: Deterministic sorted dictionary.
    Assumptions:
        Mapping values are JSON-compatible scalars.
    Raises:
        ValueError: If one of keys is blank after normalization.
    Side Effects:
        None.
    """
    if values is None:
        return {}
    if _is_pre_normalized_scalar_mapping(values=values):
        return {key: values[key] for key in values.keys()}

    normalized: dict[str, BacktestVariantScalar] = {}
    for key in sorted(values.keys()):
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("variant scalar mapping keys must be non-empty")
        normalized[normalized_key] = values[key]
    return normalized


def _normalize_signals_mapping(
    *,
    values: BacktestSignalsMap | None,
) -> dict[str, dict[str, BacktestVariantScalar]]:
    """
    Normalize nested `signals` mapping with lowercase sorted indicator/param names.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/indicators/application/dto/variant_key.py

    Args:
        values: Optional mapping `indicator_id -> {signal_param_name: scalar}`.
    Returns:
        dict[str, dict[str, BacktestVariantScalar]]: Deterministic normalized mapping.
    Assumptions:
        Signal values are JSON-compatible scalar values.
    Raises:
        ValueError: If indicator or signal parameter names are blank after normalization.
        TypeError: If one indicator payload is not mapping-like.
    Side Effects:
        None.
    """
    if values is None:
        return {}
    if _is_pre_normalized_signals_mapping(values=values):
        normalized_fast: dict[str, dict[str, BacktestVariantScalar]] = {}
        for indicator_id, params in values.items():
            normalized_fast[indicator_id] = {
                param_name: params[param_name] for param_name in params.keys()
            }
        return normalized_fast

    normalized: dict[str, dict[str, BacktestVariantScalar]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("signals indicator_id keys must be non-empty")
        if indicator_id in normalized:
            raise ValueError(
                "signals contains duplicate indicator_id after normalization: "
                f"{indicator_id}"
            )

        raw_params = values[raw_indicator_id]
        if not isinstance(raw_params, Mapping):
            raise TypeError("signals indicator payload must be a mapping")

        normalized_params: dict[str, BacktestVariantScalar] = {}
        for raw_param_name in sorted(raw_params.keys(), key=lambda key: str(key).strip().lower()):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("signals parameter names must be non-empty")
            if param_name in normalized_params:
                raise ValueError(
                    "signals contains duplicate parameter name after normalization: "
                    f"{indicator_id}.{param_name}"
                )
            normalized_params[param_name] = raw_params[raw_param_name]

        normalized[indicator_id] = normalized_params

    return normalized


def _is_pre_normalized_scalar_mapping(
    *,
    values: Mapping[str, BacktestVariantScalar],
) -> bool:
    """
    Check whether scalar mapping already matches normalized deterministic key order.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py

    Args:
        values: Candidate scalar mapping payload.
    Returns:
        bool: `True` when mapping can be reused without sort/normalize pass.
    Assumptions:
        Fast-path accepts only immutable `MappingProxyType` payloads.
    Raises:
        None.
    Side Effects:
        None.
    """
    if not isinstance(values, MappingProxyType):
        return False
    previous_key = ""
    for raw_key in values.keys():
        if not isinstance(raw_key, str):
            return False
        normalized_key = raw_key.strip()
        if not normalized_key or normalized_key != raw_key:
            return False
        if normalized_key < previous_key:
            return False
        previous_key = normalized_key
    return True


def _is_pre_normalized_signals_mapping(
    *,
    values: BacktestSignalsMap,
) -> bool:
    """
    Check whether nested signals mapping already matches canonical lowercase sorted shape.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py

    Args:
        values: Candidate nested `indicator_id -> param_name -> scalar` mapping.
    Returns:
        bool: `True` when nested mapping can be reused without re-normalization.
    Assumptions:
        Fast-path must remain conservative and reject mutable/non-canonical structures.
    Raises:
        None.
    Side Effects:
        None.
    """
    if not isinstance(values, MappingProxyType):
        return False
    previous_indicator_id = ""
    for indicator_id, params in values.items():
        if not isinstance(indicator_id, str):
            return False
        normalized_indicator_id = indicator_id.strip().lower()
        if not normalized_indicator_id or normalized_indicator_id != indicator_id:
            return False
        if normalized_indicator_id < previous_indicator_id:
            return False
        if not isinstance(params, MappingProxyType):
            return False
        previous_param_name = ""
        for param_name in params.keys():
            if not isinstance(param_name, str):
                return False
            normalized_param_name = param_name.strip().lower()
            if not normalized_param_name or normalized_param_name != param_name:
                return False
            if normalized_param_name < previous_param_name:
                return False
            previous_param_name = normalized_param_name
        previous_indicator_id = normalized_indicator_id
    return True
