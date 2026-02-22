from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
    RangeValuesSpec,
)

_ENV_NAME_KEY = "ROEHUB_ENV"
_CONFIG_PATH_KEY = "ROEHUB_INDICATORS_CONFIG"
_ALLOWED_ENVS = ("dev", "prod", "test")


@dataclass(frozen=True, slots=True)
class YamlBacktestGridDefaultsProvider(BacktestGridDefaultsProvider):
    """
    Backtest defaults provider loading compute and signal defaults from indicators YAML.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - apps/api/wiring/modules/backtest.py
    """

    compute_defaults_by_indicator_id: Mapping[str, GridSpec]
    signal_defaults_by_indicator_id: Mapping[str, Mapping[str, GridParamSpec]]

    def __post_init__(self) -> None:
        """
        Freeze loaded defaults mappings into deterministic immutable payloads.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Indicator ids and nested param keys are normalized during parsing.
        Raises:
            ValueError: If one mapping key is blank.
        Side Effects:
            Replaces mutable mappings with read-only mapping proxies.
        """
        normalized_compute: dict[str, GridSpec] = {}
        for raw_indicator_id in sorted(
            self.compute_defaults_by_indicator_id.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            indicator_id = str(raw_indicator_id).strip().lower()
            if not indicator_id:
                raise ValueError("compute defaults indicator_id keys must be non-empty")
            normalized_compute[indicator_id] = self.compute_defaults_by_indicator_id[
                raw_indicator_id
            ]

        normalized_signal: dict[str, Mapping[str, GridParamSpec]] = {}
        for raw_indicator_id in sorted(
            self.signal_defaults_by_indicator_id.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            indicator_id = str(raw_indicator_id).strip().lower()
            if not indicator_id:
                raise ValueError("signal defaults indicator_id keys must be non-empty")
            params = self.signal_defaults_by_indicator_id[raw_indicator_id]
            normalized_params: dict[str, GridParamSpec] = {}
            for raw_param_name in sorted(params.keys(), key=lambda key: str(key).strip().lower()):
                param_name = str(raw_param_name).strip().lower()
                if not param_name:
                    raise ValueError("signal defaults param keys must be non-empty")
                normalized_params[param_name] = params[raw_param_name]
            normalized_signal[indicator_id] = MappingProxyType(normalized_params)

        object.__setattr__(
            self,
            "compute_defaults_by_indicator_id",
            MappingProxyType(normalized_compute),
        )
        object.__setattr__(
            self,
            "signal_defaults_by_indicator_id",
            MappingProxyType(normalized_signal),
        )

    @classmethod
    def from_environ(cls, *, environ: Mapping[str, str]) -> YamlBacktestGridDefaultsProvider:
        """
        Build defaults provider from environment-aware indicators YAML path resolution.

        Args:
            environ: Runtime environment mapping.
        Returns:
            YamlBacktestGridDefaultsProvider: Loaded defaults provider.
        Assumptions:
            Path precedence is `ROEHUB_INDICATORS_CONFIG`
            then `configs/<ROEHUB_ENV>/indicators.yaml`.
        Raises:
            FileNotFoundError: If resolved indicators YAML path does not exist.
            ValueError: If YAML payload shape is invalid.
        Side Effects:
            Reads indicators YAML file from filesystem.
        """
        config_path = _resolve_indicators_config_path(environ=environ)
        return cls.from_yaml(config_path=config_path)

    @classmethod
    def from_yaml(cls, *, config_path: str | Path) -> YamlBacktestGridDefaultsProvider:
        """
        Build defaults provider from one indicators YAML file path.

        Args:
            config_path: Path to indicators defaults YAML file.
        Returns:
            YamlBacktestGridDefaultsProvider: Loaded defaults provider.
        Assumptions:
            Unknown keys in YAML are ignored by this adapter.
        Raises:
            FileNotFoundError: If config path does not exist.
            ValueError: If defaults payload shape cannot be parsed deterministically.
        Side Effects:
            Reads one YAML file from disk.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"indicators defaults config not found: {path}")

        raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw_payload is None:
            raw_payload = {}
        if not isinstance(raw_payload, Mapping):
            raise ValueError("indicators defaults config must be mapping at top-level")

        defaults_payload = _optional_mapping(
            payload=raw_payload,
            key="defaults",
            field_path="defaults",
        )
        compute_defaults: dict[str, GridSpec] = {}
        signal_defaults: dict[str, Mapping[str, GridParamSpec]] = {}
        for raw_indicator_id in sorted(
            defaults_payload.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            indicator_id = str(raw_indicator_id).strip().lower()
            if not indicator_id:
                raise ValueError("defaults indicator_id keys must be non-empty")
            indicator_payload_raw = defaults_payload[raw_indicator_id]
            if not isinstance(indicator_payload_raw, Mapping):
                raise ValueError(f"defaults.{indicator_id} must be mapping")

            compute_grid = _compute_defaults_grid(
                indicator_id=indicator_id,
                indicator_payload=indicator_payload_raw,
            )
            if compute_grid is not None:
                compute_defaults[indicator_id] = compute_grid

            signal_payload = _signal_defaults(
                indicator_id=indicator_id,
                indicator_payload=indicator_payload_raw,
            )
            if len(signal_payload) > 0:
                signal_defaults[indicator_id] = signal_payload

        return cls(
            compute_defaults_by_indicator_id=compute_defaults,
            signal_defaults_by_indicator_id=signal_defaults,
        )

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Resolve compute grid defaults for one indicator id.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            GridSpec | None: Defaults grid for indicator or `None` when absent.
        Assumptions:
            Indicator id lookup is case-insensitive after normalization.
        Raises:
            ValueError: If indicator id is blank.
        Side Effects:
            None.
        """
        normalized_indicator_id = indicator_id.strip().lower()
        if not normalized_indicator_id:
            raise ValueError("compute_defaults requires non-empty indicator_id")
        return self.compute_defaults_by_indicator_id.get(normalized_indicator_id)

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, GridParamSpec]:
        """
        Resolve signal parameter defaults for one indicator id.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            Mapping[str, GridParamSpec]: Deterministic signal-params mapping, empty when absent.
        Assumptions:
            Indicator id lookup is case-insensitive after normalization.
        Raises:
            ValueError: If indicator id is blank.
        Side Effects:
            None.
        """
        normalized_indicator_id = indicator_id.strip().lower()
        if not normalized_indicator_id:
            raise ValueError("signal_param_defaults requires non-empty indicator_id")
        return self.signal_defaults_by_indicator_id.get(normalized_indicator_id, {})


def _resolve_indicators_config_path(*, environ: Mapping[str, str]) -> Path:
    """
    Resolve indicators YAML path from override env or environment-specific default path.

    Args:
        environ: Runtime environment mapping.
    Returns:
        Path: Resolved indicators YAML path.
    Assumptions:
        Missing `ROEHUB_ENV` defaults to `dev`.
    Raises:
        ValueError: If environment name is unsupported.
    Side Effects:
        None.
    """
    override = environ.get(_CONFIG_PATH_KEY, "").strip()
    if override:
        return Path(override)

    raw_env = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if raw_env not in _ALLOWED_ENVS:
        raise ValueError(f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {raw_env!r}")
    return Path("configs") / raw_env / "indicators.yaml"


def _compute_defaults_grid(
    *,
    indicator_id: str,
    indicator_payload: Mapping[str, Any],
) -> GridSpec | None:
    """
    Parse one compute defaults grid from indicators YAML defaults payload.

    Args:
        indicator_id: Indicator identifier.
        indicator_payload: One defaults block from YAML payload.
    Returns:
        GridSpec | None: Parsed defaults grid or `None` when no compute defaults exist.
    Assumptions:
        Inputs and params sections follow indicators defaults explicit/range spec contract.
    Raises:
        ValueError: If nested defaults payload shape is invalid.
    Side Effects:
        None.
    """
    inputs_payload = _optional_mapping(
        payload=indicator_payload,
        key="inputs",
        field_path=f"defaults.{indicator_id}.inputs",
    )
    params_payload = _optional_mapping(
        payload=indicator_payload,
        key="params",
        field_path=f"defaults.{indicator_id}.params",
    )

    source_spec: GridParamSpec | None = None
    if "source" in inputs_payload:
        source_spec = _grid_param_spec_from_node(
            name="source",
            node=inputs_payload["source"],
            field_path=f"defaults.{indicator_id}.inputs.source",
        )

    grid_params: dict[str, GridParamSpec] = {}
    for raw_param_name in sorted(params_payload.keys(), key=lambda key: str(key).strip().lower()):
        param_name = str(raw_param_name).strip().lower()
        if not param_name:
            raise ValueError(f"defaults.{indicator_id}.params keys must be non-empty")
        grid_params[param_name] = _grid_param_spec_from_node(
            name=param_name,
            node=params_payload[raw_param_name],
            field_path=f"defaults.{indicator_id}.params.{param_name}",
        )

    for raw_input_name in sorted(inputs_payload.keys(), key=lambda key: str(key).strip().lower()):
        input_name = str(raw_input_name).strip().lower()
        if not input_name:
            raise ValueError(f"defaults.{indicator_id}.inputs keys must be non-empty")
        if input_name == "source" or input_name in grid_params:
            continue
        grid_params[input_name] = _grid_param_spec_from_node(
            name=input_name,
            node=inputs_payload[raw_input_name],
            field_path=f"defaults.{indicator_id}.inputs.{input_name}",
        )

    if source_spec is None and len(grid_params) == 0:
        return None

    return GridSpec(
        indicator_id=IndicatorId(indicator_id),
        params=grid_params,
        source=source_spec,
    )


def _signal_defaults(
    *,
    indicator_id: str,
    indicator_payload: Mapping[str, Any],
) -> Mapping[str, GridParamSpec]:
    """
    Parse one optional `signals.v1.params` defaults block for indicator id.

    Args:
        indicator_id: Indicator identifier.
        indicator_payload: One defaults block from YAML payload.
    Returns:
        Mapping[str, GridParamSpec]: Deterministic signal defaults mapping.
    Assumptions:
        Signal defaults are stored under `defaults.<id>.signals.v1.params.*`.
    Raises:
        ValueError: If nested payload shape is invalid.
    Side Effects:
        None.
    """
    signals_payload = _optional_mapping(
        payload=indicator_payload,
        key="signals",
        field_path=f"defaults.{indicator_id}.signals",
    )
    v1_payload = _optional_mapping(
        payload=signals_payload,
        key="v1",
        field_path=f"defaults.{indicator_id}.signals.v1",
    )
    params_payload = _optional_mapping(
        payload=v1_payload,
        key="params",
        field_path=f"defaults.{indicator_id}.signals.v1.params",
    )

    signal_params: dict[str, GridParamSpec] = {}
    for raw_param_name in sorted(params_payload.keys(), key=lambda key: str(key).strip().lower()):
        param_name = str(raw_param_name).strip().lower()
        if not param_name:
            raise ValueError(
                f"defaults.{indicator_id}.signals.v1.params keys must be non-empty"
            )
        signal_params[param_name] = _grid_param_spec_from_node(
            name=param_name,
            node=params_payload[raw_param_name],
            field_path=f"defaults.{indicator_id}.signals.v1.params.{param_name}",
        )
    return MappingProxyType(signal_params)


def _grid_param_spec_from_node(
    *,
    name: str,
    node: Any,
    field_path: str,
) -> GridParamSpec:
    """
    Parse one grid axis spec node with explicit/range modes or scalar shorthand.

    Args:
        name: Axis name.
        node: Raw defaults YAML node.
        field_path: Dot-like path for deterministic error messages.
    Returns:
        GridParamSpec: Parsed deterministic grid spec object.
    Assumptions:
        Scalars represent one explicit value for the axis.
    Raises:
        ValueError: If node cannot be parsed into deterministic grid spec.
    Side Effects:
        None.
    """
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError(f"{field_path} axis name must be non-empty")

    if isinstance(node, Mapping):
        raw_mode = node.get("mode")
        if not isinstance(raw_mode, str):
            raise ValueError(f"{field_path}.mode must be string")
        mode = raw_mode.strip().lower()
        if mode == "explicit":
            raw_values = node.get("values")
            if not isinstance(raw_values, list) or len(raw_values) == 0:
                raise ValueError(f"{field_path}.values must be non-empty list")
            values: list[int | float | str] = []
            for raw_item in raw_values:
                if isinstance(raw_item, bool) or not isinstance(raw_item, int | float | str):
                    raise ValueError(f"{field_path}.values items must be scalar")
                values.append(raw_item)
            return ExplicitValuesSpec(name=normalized_name, values=tuple(values))

        if mode == "range":
            start = _numeric_required(node.get("start"), field_path=f"{field_path}.start")
            stop_incl = _numeric_required(
                node.get("stop_incl"),
                field_path=f"{field_path}.stop_incl",
            )
            step = _numeric_required(node.get("step"), field_path=f"{field_path}.step")
            return RangeValuesSpec(
                name=normalized_name,
                start=start,
                stop_inclusive=stop_incl,
                step=step,
            )

        raise ValueError(f"{field_path}.mode must be 'explicit' or 'range'")

    if isinstance(node, bool) or not isinstance(node, int | float | str):
        raise ValueError(f"{field_path} must be scalar or mapping")
    return ExplicitValuesSpec(name=normalized_name, values=(node,))


def _optional_mapping(
    *,
    payload: Mapping[str, Any],
    key: str,
    field_path: str,
) -> Mapping[str, Any]:
    """
    Read optional mapping key from payload, returning empty mapping when key is absent.

    Args:
        payload: Parent mapping payload.
        key: Child key.
        field_path: Dot-like path for deterministic errors.
    Returns:
        Mapping[str, Any]: Child mapping or empty mapping.
    Assumptions:
        Missing section means defaults are not configured for this branch.
    Raises:
        ValueError: If provided value is not mapping.
    Side Effects:
        None.
    """
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_path} must be mapping when provided")
    return value


def _numeric_required(value: Any, *, field_path: str) -> int | float:
    """
    Parse required numeric value while rejecting booleans.

    Args:
        value: Raw payload value.
        field_path: Dot-like path for deterministic errors.
    Returns:
        int | float: Numeric scalar value.
    Assumptions:
        Range-axis numerics follow indicators range defaults semantics.
    Raises:
        ValueError: If value is missing or non-numeric.
    Side Effects:
        None.
    """
    if value is None:
        raise ValueError(f"{field_path} is required")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_path} must be numeric")
    return value


__all__ = ["YamlBacktestGridDefaultsProvider"]
