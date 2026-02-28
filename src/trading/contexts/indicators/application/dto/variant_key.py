"""
Deterministic variant_key v1 builder bound to instrument_id and timeframe.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

VariantScalar = int | float | str


@dataclass(frozen=True, slots=True)
class IndicatorVariantSelection:
    """
    Explicit indicator configuration used in variant_key payload.
    """

    indicator_id: str
    inputs: Mapping[str, VariantScalar]
    params: Mapping[str, VariantScalar]

    def __post_init__(self) -> None:
        """
        Validate and freeze one explicit indicator selection.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Inputs and params contain scalar JSON-serializable values.
        Raises:
            ValueError: If indicator_id is blank or any key is blank.
        Side Effects:
            Replaces input/param mappings with immutable mapping proxies.
        """
        normalized_id = self.indicator_id.strip().lower()
        if not normalized_id:
            raise ValueError("IndicatorVariantSelection requires non-empty indicator_id")
        object.__setattr__(self, "indicator_id", normalized_id)

        normalized_inputs: dict[str, VariantScalar] = {}
        for key, value in self.inputs.items():
            name = key.strip()
            if not name:
                raise ValueError("IndicatorVariantSelection input names must be non-empty")
            normalized_inputs[name] = value

        normalized_params: dict[str, VariantScalar] = {}
        for key, value in self.params.items():
            name = key.strip()
            if not name:
                raise ValueError("IndicatorVariantSelection parameter names must be non-empty")
            normalized_params[name] = value

        object.__setattr__(self, "inputs", MappingProxyType(normalized_inputs))
        object.__setattr__(self, "params", MappingProxyType(normalized_params))


def build_variant_key_v1(
    *,
    instrument_id: str,
    timeframe: str,
    indicators: tuple[IndicatorVariantSelection, ...],
) -> str:
    """
    Build variant_key v1 from canonical payload and SHA-256.

    Args:
        instrument_id: Canonical instrument identifier string.
        timeframe: Canonical timeframe string.
        indicators: Explicit indicator selections.
    Returns:
        str: Hex SHA-256 variant key.
    Assumptions:
        Inputs are already explicit and do not contain range specs.
    Raises:
        ValueError: If instrument_id/timeframe are blank or indicator ids duplicate.
    Side Effects:
        None.
    """
    payload = _build_payload_v1(
        instrument_id=instrument_id,
        timeframe=timeframe,
        indicators=indicators,
    )
    canonical_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _build_payload_v1(
    *,
    instrument_id: str,
    timeframe: str,
    indicators: tuple[IndicatorVariantSelection, ...],
) -> dict[str, object]:
    """
    Build canonical payload structure for variant_key v1.

    Args:
        instrument_id: Canonical instrument identifier string.
        timeframe: Canonical timeframe string.
        indicators: Explicit indicator selections.
    Returns:
        dict[str, object]: JSON-compatible canonical payload.
    Assumptions:
        Canonical ordering is achieved by sorting indicator ids and pair keys.
    Raises:
        ValueError: If core fields are blank or indicator ids duplicate.
    Side Effects:
        None.
    """
    normalized_instrument_id = instrument_id.strip()
    if not normalized_instrument_id:
        raise ValueError("instrument_id must be non-empty")

    normalized_timeframe = timeframe.strip().lower()
    if not normalized_timeframe:
        raise ValueError("timeframe must be non-empty")

    ordered_indicators = indicators
    if not _is_pre_sorted_indicator_selections(indicators=indicators):
        ordered_indicators = tuple(sorted(indicators, key=lambda item: item.indicator_id))
    previous_indicator_id: str | None = None
    for item in ordered_indicators:
        if previous_indicator_id == item.indicator_id:
            raise ValueError("indicators must not contain duplicate indicator_id")
        previous_indicator_id = item.indicator_id

    serialized = []
    for item in ordered_indicators:
        serialized.append(
            {
                "id": item.indicator_id,
                "inputs": _sorted_pairs(values=item.inputs),
                "params": _sorted_pairs(values=item.params),
            }
        )

    return {
        "indicators": serialized,
        "instrument_id": normalized_instrument_id,
        "schema_version": 1,
        "timeframe": normalized_timeframe,
    }


def _sorted_pairs(values: Mapping[str, VariantScalar]) -> list[list[VariantScalar | str]]:
    """
    Convert mapping to a key-sorted list of `[name, value]` pairs.

    Args:
        values: Mapping with arbitrary insertion order.
    Returns:
        list[list[VariantScalar | str]]: Deterministic key-sorted pair list.
    Assumptions:
        Keys are plain strings.
    Raises:
        None.
    Side Effects:
        None.
    """
    if _is_pre_sorted_mapping(values=values):
        return [[key, values[key]] for key in values.keys()]
    pairs: list[list[VariantScalar | str]] = []
    for key in sorted(values.keys()):
        pairs.append([key, values[key]])
    return pairs


def _is_pre_sorted_indicator_selections(
    *,
    indicators: tuple[IndicatorVariantSelection, ...],
) -> bool:
    """
    Check whether indicator selections are already sorted by `indicator_id` asc.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
    Related:
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py

    Args:
        indicators: Indicator selections tuple.
    Returns:
        bool: `True` when tuple is already strictly sorted and duplicate-free.
    Assumptions:
        `IndicatorVariantSelection.indicator_id` values are normalized to lowercase.
    Raises:
        None.
    Side Effects:
        None.
    """
    previous = ""
    for item in indicators:
        if item.indicator_id <= previous:
            return False
        previous = item.indicator_id
    return True


def _is_pre_sorted_mapping(
    *,
    values: Mapping[str, VariantScalar],
) -> bool:
    """
    Check whether mapping is immutable and already sorted by key asc.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
    Related:
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py

    Args:
        values: Candidate scalar mapping.
    Returns:
        bool: `True` when mapping can be reused without extra sort pass.
    Assumptions:
        Fast-path accepts only `MappingProxyType` payloads from normalized DTOs.
    Raises:
        None.
    Side Effects:
        None.
    """
    if not isinstance(values, MappingProxyType):
        return False
    previous = ""
    for key in values.keys():
        if not isinstance(key, str):
            return False
        normalized = key.strip()
        if not normalized or normalized != key:
            return False
        if normalized < previous:
            return False
        previous = normalized
    return True
