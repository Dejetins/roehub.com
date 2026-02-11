"""
Indicator registry adapter backed by hard code definitions and YAML defaults.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
Related: trading.contexts.indicators.application.ports.registry.indicator_registry,
  trading.contexts.indicators.adapters.outbound.config.yaml_defaults_validator
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from trading.contexts.indicators.adapters.outbound.config import (
    load_indicator_defaults_yaml,
    validate_indicator_defaults,
)
from trading.contexts.indicators.application.dto.registry_view import (
    DefaultSpec,
    IndicatorDefaults,
    IndicatorDefaultsDocument,
    MergedIndicatorView,
    MergedInputView,
    MergedParamView,
)
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.domain.entities import IndicatorDef, IndicatorId, Layout
from trading.contexts.indicators.domain.errors import UnknownIndicatorError

_EMPTY_DEFAULTS = IndicatorDefaults(inputs={}, params={})


class YamlIndicatorRegistry(IndicatorRegistry):
    """
    Read-only indicator registry with fail-fast YAML defaults validation.
    """

    def __init__(
        self,
        *,
        defs: tuple[IndicatorDef, ...],
        defaults: IndicatorDefaultsDocument,
        config_path: str | Path,
    ) -> None:
        """
        Validate defaults and build deterministic lookup/merged views.

        Args:
            defs: Hard indicator definitions.
            defaults: Parsed defaults document.
            config_path: Defaults file path for diagnostic context.
        Returns:
            None.
        Assumptions:
            Defaults must be validated before serving API requests.
        Raises:
            IndicatorDefaultsValidationError: If defaults violate hard definitions.
            ValueError: If hard defs contain duplicate indicator_id.
        Side Effects:
            None.
        """
        normalized_config_path = str(config_path)
        validate_indicator_defaults(
            config_path=normalized_config_path,
            defs=defs,
            defaults=defaults,
        )

        ordered_defs = tuple(sorted(defs, key=lambda definition: definition.indicator_id.value))
        defs_by_id: dict[str, IndicatorDef] = {}
        for definition in ordered_defs:
            indicator_id = definition.indicator_id.value
            if indicator_id in defs_by_id:
                raise ValueError(f"duplicate indicator_id in hard defs: {indicator_id}")
            defs_by_id[indicator_id] = definition

        merged = _merge_registry_views(
            defs=ordered_defs,
            defaults=defaults,
        )
        merged_by_id = {item.indicator_id: item for item in merged}

        self._config_path = normalized_config_path
        self._defs = ordered_defs
        self._defs_by_id: Mapping[str, IndicatorDef] = MappingProxyType(defs_by_id)
        self._defaults = defaults
        self._merged = merged
        self._merged_by_id: Mapping[str, MergedIndicatorView] = MappingProxyType(merged_by_id)

    @classmethod
    def from_yaml(
        cls,
        *,
        defs: tuple[IndicatorDef, ...],
        config_path: str | Path,
    ) -> YamlIndicatorRegistry:
        """
        Build registry instance from YAML file path.

        Args:
            defs: Hard indicator definitions.
            config_path: Defaults file path.
        Returns:
            YamlIndicatorRegistry: Initialized registry instance.
        Assumptions:
            YAML file exists and is readable.
        Raises:
            FileNotFoundError: If config file is missing.
            ValueError: If YAML cannot be parsed or validated.
        Side Effects:
            Reads one YAML file from disk.
        """
        defaults = load_indicator_defaults_yaml(config_path)
        return cls(defs=defs, defaults=defaults, config_path=config_path)

    def list_defs(self) -> tuple[IndicatorDef, ...]:
        """
        Return all hard definitions in deterministic order.

        Args:
            None.
        Returns:
            tuple[IndicatorDef, ...]: Sorted by `indicator_id`.
        Assumptions:
            Registry state is immutable after construction.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._defs

    def get_def(self, indicator_id: IndicatorId) -> IndicatorDef:
        """
        Resolve one hard indicator definition by id.

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            IndicatorDef: Matching hard definition.
        Assumptions:
            `indicator_id` is normalized by domain value-object invariants.
        Raises:
            UnknownIndicatorError: If indicator id is not registered.
        Side Effects:
            None.
        """
        definition = self._defs_by_id.get(indicator_id.value)
        if definition is None:
            raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")
        return definition

    def list_merged(self) -> tuple[MergedIndicatorView, ...]:
        """
        Return merged registry view (hard defs + defaults) in stable order.

        Args:
            None.
        Returns:
            tuple[MergedIndicatorView, ...]: Sorted by `indicator_id`.
        Assumptions:
            Merged views are precomputed and immutable.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._merged

    def get_merged(self, indicator_id: IndicatorId) -> MergedIndicatorView:
        """
        Resolve one merged registry view by id.

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            MergedIndicatorView: Matching merged indicator representation.
        Assumptions:
            Merged views are indexed by normalized id.
        Raises:
            UnknownIndicatorError: If indicator id is not registered.
        Side Effects:
            None.
        """
        merged = self._merged_by_id.get(indicator_id.value)
        if merged is None:
            raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")
        return merged


def _merge_registry_views(
    *,
    defs: tuple[IndicatorDef, ...],
    defaults: IndicatorDefaultsDocument,
) -> tuple[MergedIndicatorView, ...]:
    """
    Merge hard definitions with validated YAML defaults.

    Args:
        defs: Ordered hard indicator definitions.
        defaults: Parsed and validated defaults document.
    Returns:
        tuple[MergedIndicatorView, ...]: Deterministic merged view for API layer.
    Assumptions:
        Defaults are already validated against hard defs.
    Raises:
        None.
    Side Effects:
        None.
    """
    merged: list[MergedIndicatorView] = []
    for definition in defs:
        indicator_id = definition.indicator_id.value
        indicator_defaults = defaults.defaults.get(indicator_id, _EMPTY_DEFAULTS)

        merged_inputs = _build_merged_inputs(
            definition=definition,
            indicator_defaults=indicator_defaults,
        )
        merged_params = _build_merged_params(
            definition=definition,
            indicator_defaults=indicator_defaults,
        )

        group = _group_of(indicator_id)
        required_inputs = _build_required_inputs(definition=definition)
        merged.append(
            MergedIndicatorView(
                indicator_id=indicator_id,
                group=group,
                title=definition.title,
                required_inputs=required_inputs,
                inputs=merged_inputs,
                params=merged_params,
                output_names=definition.output.names,
                default_layout=Layout.TIME_MAJOR,
            )
        )

    return tuple(merged)


def _build_required_inputs(*, definition: IndicatorDef) -> tuple[str, ...]:
    """
    Compute deterministic list of fixed required input series.

    Args:
        definition: Hard indicator definition.
    Returns:
        tuple[str, ...]: Sorted fixed input series names.
    Assumptions:
        Source-axis indicators externalize source selection and keep fixed list empty.
    Raises:
        None.
    Side Effects:
        None.
    """
    if "source" in definition.axes:
        return ()
    return tuple(sorted({series.value for series in definition.inputs}))


def _build_merged_inputs(
    *,
    definition: IndicatorDef,
    indicator_defaults: IndicatorDefaults,
) -> tuple[MergedInputView, ...]:
    """
    Build merged input-axis defaults for one indicator.

    Args:
        definition: Hard indicator definition.
        indicator_defaults: Defaults loaded from YAML for this indicator.
    Returns:
        tuple[MergedInputView, ...]: Sorted merged input descriptors.
    Assumptions:
        v1 supports only `source` as configurable input axis.
    Raises:
        None.
    Side Effects:
        None.
    """
    inputs: list[MergedInputView] = []
    if "source" in definition.axes:
        allowed_values = tuple(sorted({series.value for series in definition.inputs}))
        default_spec = indicator_defaults.inputs.get("source")
        inputs.append(
            MergedInputView(
                name="source",
                allowed_values=allowed_values,
                default=default_spec,
            )
        )

    return tuple(sorted(inputs, key=lambda item: item.name))


def _build_merged_params(
    *,
    definition: IndicatorDef,
    indicator_defaults: IndicatorDefaults,
) -> tuple[MergedParamView, ...]:
    """
    Build merged parameter defaults for one indicator.

    Args:
        definition: Hard indicator definition.
        indicator_defaults: Defaults loaded from YAML for this indicator.
    Returns:
        tuple[MergedParamView, ...]: Sorted merged parameter descriptors.
    Assumptions:
        Parameter names are unique in hard definitions.
    Raises:
        None.
    Side Effects:
        None.
    """
    params: list[MergedParamView] = []
    for param in sorted(definition.params, key=lambda item: item.name):
        default_spec: DefaultSpec | None = indicator_defaults.params.get(param.name)
        params.append(
            MergedParamView(
                name=param.name,
                kind=param.kind,
                hard_min=param.hard_min,
                hard_max=param.hard_max,
                step=param.step,
                enum_values=param.enum_values,
                default=default_spec,
            )
        )
    return tuple(params)


def _group_of(indicator_id: str) -> str:
    """
    Extract indicator group prefix from indicator_id.

    Args:
        indicator_id: Full indicator identifier.
    Returns:
        str: Group prefix before first dot or `unknown` fallback.
    Assumptions:
        Indicator ids use `group.name` format in v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    head, separator, _tail = indicator_id.partition(".")
    if separator and head:
        return head
    return "unknown"
