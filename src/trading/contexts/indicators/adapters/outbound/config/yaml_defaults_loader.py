"""
Filesystem YAML loader for indicator UI defaults.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
Related: trading.contexts.indicators.application.dto.registry_view
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from trading.contexts.indicators.application.dto.registry_view import (
    ExplicitDefaultSpec,
    IndicatorDefaults,
    IndicatorDefaultsDocument,
    RangeDefaultSpec,
)


def load_indicator_defaults_yaml(path: str | Path) -> IndicatorDefaultsDocument:
    """
    Load indicator defaults YAML from filesystem.

    Args:
        path: Config path to `indicators.yaml`.
    Returns:
        IndicatorDefaultsDocument: Parsed defaults payload.
    Assumptions:
        YAML top-level object is a mapping with `schema_version` and `defaults`.
    Raises:
        FileNotFoundError: If config path does not exist.
        ValueError: If YAML shape is invalid.
    Side Effects:
        Reads one UTF-8 file from disk.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"indicators defaults config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("indicators defaults config must be a mapping at top-level")

    schema_version = _require_int(raw, key="schema_version", yaml_path="schema_version")
    defaults_map = _require_mapping(raw, key="defaults", yaml_path="defaults")

    parsed_defaults: dict[str, IndicatorDefaults] = {}
    for indicator_id, indicator_payload in defaults_map.items():
        if not isinstance(indicator_id, str) or not indicator_id.strip():
            raise ValueError("defaults keys must be non-empty indicator_id strings")
        indicator_yaml_path = f"defaults.{indicator_id}"
        indicator_mapping = _require_mapping(
            defaults_map,
            key=indicator_id,
            yaml_path=indicator_yaml_path,
        )
        parsed_defaults[indicator_id] = _parse_indicator_defaults(
            indicator_mapping,
            yaml_path=indicator_yaml_path,
        )

    return IndicatorDefaultsDocument(schema_version=schema_version, defaults=parsed_defaults)


def _parse_indicator_defaults(
    payload: Mapping[str, Any],
    *,
    yaml_path: str,
) -> IndicatorDefaults:
    """
    Parse one indicator defaults block.

    Args:
        payload: Raw mapping under `defaults.<indicator_id>`.
        yaml_path: Dot-path prefix for error messages.
    Returns:
        IndicatorDefaults: Parsed defaults section.
    Assumptions:
        `inputs` and `params` are optional mappings.
    Raises:
        ValueError: If section payload shape is invalid.
    Side Effects:
        None.
    """
    inputs_map = _optional_mapping(payload, key="inputs", yaml_path=f"{yaml_path}.inputs")
    params_map = _optional_mapping(payload, key="params", yaml_path=f"{yaml_path}.params")

    parsed_inputs: dict[str, ExplicitDefaultSpec | RangeDefaultSpec] = {}
    for input_name in sorted(inputs_map.keys()):
        input_yaml_path = f"{yaml_path}.inputs.{input_name}"
        spec_mapping = _require_mapping(inputs_map, key=input_name, yaml_path=input_yaml_path)
        parsed_inputs[input_name] = _parse_default_spec(spec_mapping, yaml_path=input_yaml_path)

    parsed_params: dict[str, ExplicitDefaultSpec | RangeDefaultSpec] = {}
    for param_name in sorted(params_map.keys()):
        param_yaml_path = f"{yaml_path}.params.{param_name}"
        spec_mapping = _require_mapping(params_map, key=param_name, yaml_path=param_yaml_path)
        parsed_params[param_name] = _parse_default_spec(spec_mapping, yaml_path=param_yaml_path)

    return IndicatorDefaults(inputs=parsed_inputs, params=parsed_params)


def _parse_default_spec(
    payload: Mapping[str, Any],
    *,
    yaml_path: str,
) -> ExplicitDefaultSpec | RangeDefaultSpec:
    """
    Parse one defaults spec node (`explicit` or `range`).

    Args:
        payload: Raw mapping for one defaults node.
        yaml_path: Dot-path prefix for error messages.
    Returns:
        ExplicitDefaultSpec | RangeDefaultSpec: Parsed spec.
    Assumptions:
        `mode` is either `explicit` or `range`.
    Raises:
        ValueError: If required keys are missing or value types are invalid.
    Side Effects:
        None.
    """
    mode_value = _require_str(payload, key="mode", yaml_path=f"{yaml_path}.mode").lower()

    if mode_value == "explicit":
        values = _require_list(payload, key="values", yaml_path=f"{yaml_path}.values")
        parsed_values = _parse_scalar_values(values, yaml_path=f"{yaml_path}.values")
        return ExplicitDefaultSpec(mode="explicit", values=parsed_values)

    if mode_value == "range":
        start = _require_numeric(payload, key="start", yaml_path=f"{yaml_path}.start")
        stop_incl = _require_numeric(payload, key="stop_incl", yaml_path=f"{yaml_path}.stop_incl")
        step = _require_numeric(payload, key="step", yaml_path=f"{yaml_path}.step")
        return RangeDefaultSpec(mode="range", start=start, stop_incl=stop_incl, step=step)

    raise ValueError(f"{yaml_path}.mode must be 'explicit' or 'range', got {mode_value!r}")


def _require_mapping(
    payload: Mapping[str, Any],
    *,
    key: str,
    yaml_path: str,
) -> Mapping[str, Any]:
    """
    Require mapping value at key.

    Args:
        payload: Parent mapping.
        key: Child key.
        yaml_path: Full YAML path for errors.
    Returns:
        Mapping[str, Any]: Child mapping.
    Assumptions:
        `payload` is already a mapping.
    Raises:
        ValueError: If key missing or value is not a mapping.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing required mapping at {yaml_path}")
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping at {yaml_path}, got {type(value).__name__}")
    return value


def _optional_mapping(
    payload: Mapping[str, Any],
    *,
    key: str,
    yaml_path: str,
) -> Mapping[str, Any]:
    """
    Read optional mapping key or return empty mapping.

    Args:
        payload: Parent mapping.
        key: Child key.
        yaml_path: Full YAML path for errors.
    Returns:
        Mapping[str, Any]: Child mapping or empty mapping if absent.
    Assumptions:
        Absence means no defaults for this section.
    Raises:
        ValueError: If present value is not a mapping.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping at {yaml_path}, got {type(value).__name__}")
    return value


def _require_int(
    payload: Mapping[str, Any],
    *,
    key: str,
    yaml_path: str,
) -> int:
    """
    Read required integer key.

    Args:
        payload: Parent mapping.
        key: Child key.
        yaml_path: Full YAML path for errors.
    Returns:
        int: Parsed integer value.
    Assumptions:
        YAML scalar is already decoded to Python value.
    Raises:
        ValueError: If key missing or value is not an integer.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing required key at {yaml_path}")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"expected int at {yaml_path}, got {type(value).__name__}")
    return value


def _require_str(
    payload: Mapping[str, Any],
    *,
    key: str,
    yaml_path: str,
) -> str:
    """
    Read required non-empty string key.

    Args:
        payload: Parent mapping.
        key: Child key.
        yaml_path: Full YAML path for errors.
    Returns:
        str: Parsed normalized string.
    Assumptions:
        String values may include surrounding spaces.
    Raises:
        ValueError: If key missing, value is not string, or blank after strip.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing required key at {yaml_path}")
    if not isinstance(value, str):
        raise ValueError(f"expected string at {yaml_path}, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"expected non-empty string at {yaml_path}")
    return normalized


def _require_list(
    payload: Mapping[str, Any],
    *,
    key: str,
    yaml_path: str,
) -> list[Any]:
    """
    Read required list key.

    Args:
        payload: Parent mapping.
        key: Child key.
        yaml_path: Full YAML path for errors.
    Returns:
        list[Any]: Raw list value.
    Assumptions:
        Empty lists are invalid for defaults.
    Raises:
        ValueError: If key missing, value is not list, or list is empty.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing required list at {yaml_path}")
    if not isinstance(value, list):
        raise ValueError(f"expected list at {yaml_path}, got {type(value).__name__}")
    if len(value) == 0:
        raise ValueError(f"expected non-empty list at {yaml_path}")
    return value


def _require_numeric(
    payload: Mapping[str, Any],
    *,
    key: str,
    yaml_path: str,
) -> int | float:
    """
    Read required numeric key (int or float, excluding bool).

    Args:
        payload: Parent mapping.
        key: Child key.
        yaml_path: Full YAML path for errors.
    Returns:
        int | float: Numeric scalar.
    Assumptions:
        YAML has already decoded plain numeric scalars.
    Raises:
        ValueError: If key missing or value is not numeric.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        raise ValueError(f"missing required numeric value at {yaml_path}")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected numeric scalar at {yaml_path}, got {type(value).__name__}")
    return value


def _parse_scalar_values(values: list[Any], *, yaml_path: str) -> tuple[int | float | str, ...]:
    """
    Parse list of explicit scalar values.

    Args:
        values: Raw list items.
        yaml_path: Full YAML path for errors.
    Returns:
        tuple[int | float | str, ...]: Immutable explicit values list.
    Assumptions:
        Scalars are int/float/string and bool is rejected.
    Raises:
        ValueError: If an item is not a supported scalar.
    Side Effects:
        None.
    """
    parsed: list[int | float | str] = []
    for index, value in enumerate(values):
        item_path = f"{yaml_path}[{index}]"
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(
                f"expected scalar (int|float|str) at {item_path}, got {type(value).__name__}"
            )
        parsed.append(value)
    return tuple(parsed)
