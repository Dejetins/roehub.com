"""
Application DTOs for merged indicator registry view and YAML defaults model.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
Related: trading.contexts.indicators.application.ports.registry.indicator_registry,
  trading.contexts.indicators.domain.entities.indicator_def
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from trading.contexts.indicators.domain.entities import Layout, ParamKind

RegistryScalar = int | float | str


@dataclass(frozen=True, slots=True)
class ExplicitDefaultSpec:
    """
    Explicit defaults for one input or parameter axis.
    """

    mode: str
    values: tuple[RegistryScalar, ...]

    def __post_init__(self) -> None:
        """
        Validate explicit defaults payload shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Explicit defaults provide at least one deterministic value.
        Raises:
            ValueError: If mode is invalid or values are empty.
        Side Effects:
            None.
        """
        if self.mode != "explicit":
            raise ValueError(f"ExplicitDefaultSpec mode must be 'explicit', got {self.mode!r}")
        if len(self.values) == 0:
            raise ValueError("ExplicitDefaultSpec requires at least one value")


@dataclass(frozen=True, slots=True)
class RangeDefaultSpec:
    """
    Inclusive range defaults for one numeric axis.
    """

    mode: str
    start: int | float
    stop_incl: int | float
    step: int | float

    def __post_init__(self) -> None:
        """
        Validate range defaults payload shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Range defaults use inclusive stop semantics.
        Raises:
            ValueError: If mode is invalid, step is non-positive, or start > stop_incl.
        Side Effects:
            None.
        """
        if self.mode != "range":
            raise ValueError(f"RangeDefaultSpec mode must be 'range', got {self.mode!r}")
        if self.step <= 0:
            raise ValueError("RangeDefaultSpec requires step > 0")
        if self.start > self.stop_incl:
            raise ValueError("RangeDefaultSpec requires start <= stop_incl")


DefaultSpec = ExplicitDefaultSpec | RangeDefaultSpec


@dataclass(frozen=True, slots=True)
class IndicatorDefaults:
    """
    YAML defaults for one indicator.
    """

    inputs: Mapping[str, DefaultSpec]
    params: Mapping[str, DefaultSpec]

    def __post_init__(self) -> None:
        """
        Freeze defaults mappings to keep deterministic immutable shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Names are provided as plain strings from YAML parser.
        Raises:
            ValueError: If any key is blank.
        Side Effects:
            Replaces mutable mappings with read-only mapping proxies.
        """
        normalized_inputs: dict[str, DefaultSpec] = {}
        for key, value in self.inputs.items():
            name = key.strip()
            if not name:
                raise ValueError("IndicatorDefaults input names must be non-empty")
            normalized_inputs[name] = value

        normalized_params: dict[str, DefaultSpec] = {}
        for key, value in self.params.items():
            name = key.strip()
            if not name:
                raise ValueError("IndicatorDefaults parameter names must be non-empty")
            normalized_params[name] = value

        object.__setattr__(self, "inputs", MappingProxyType(normalized_inputs))
        object.__setattr__(self, "params", MappingProxyType(normalized_params))


@dataclass(frozen=True, slots=True)
class IndicatorDefaultsDocument:
    """
    Parsed defaults YAML document.
    """

    schema_version: int
    defaults: Mapping[str, IndicatorDefaults]

    def __post_init__(self) -> None:
        """
        Validate top-level defaults document and freeze mapping.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Schema version is validated by loader/validator combination.
        Raises:
            ValueError: If schema_version is not positive or keys are blank.
        Side Effects:
            Replaces mutable defaults mapping with read-only mapping proxy.
        """
        if self.schema_version <= 0:
            raise ValueError("IndicatorDefaultsDocument requires schema_version > 0")

        normalized: dict[str, IndicatorDefaults] = {}
        for key, value in self.defaults.items():
            indicator_id = key.strip()
            if not indicator_id:
                raise ValueError("IndicatorDefaultsDocument indicator_id must be non-empty")
            normalized[indicator_id] = value

        object.__setattr__(self, "defaults", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class MergedInputView:
    """
    Merged hard definition and defaults for one parameterized input axis.
    """

    name: str
    allowed_values: tuple[str, ...]
    default: DefaultSpec | None


@dataclass(frozen=True, slots=True)
class MergedParamView:
    """
    Merged hard definition and defaults for one indicator parameter.
    """

    name: str
    kind: ParamKind
    hard_min: int | float | None
    hard_max: int | float | None
    step: int | float | None
    enum_values: tuple[str, ...] | None
    default: DefaultSpec | None


@dataclass(frozen=True, slots=True)
class MergedIndicatorView:
    """
    Full merged view returned by GET /indicators.
    """

    indicator_id: str
    group: str
    title: str
    required_inputs: tuple[str, ...]
    inputs: tuple[MergedInputView, ...]
    params: tuple[MergedParamView, ...]
    output_names: tuple[str, ...]
    default_layout: Layout
