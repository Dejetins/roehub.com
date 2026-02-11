from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from trading.contexts.indicators.domain.entities.indicator_id import IndicatorId
from trading.contexts.indicators.domain.entities.layout import Layout

from .grid_param_spec import GridParamSpec


@dataclass(frozen=True, slots=True)
class GridSpec:
    """
    Grid configuration for one indicator compute/estimate request.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .grid_param_spec, ..entities.indicator_def
    """

    indicator_id: IndicatorId
    params: Mapping[str, GridParamSpec]
    source: GridParamSpec | None = None
    layout_preference: Layout | None = None

    def __post_init__(self) -> None:
        """
        Validate and freeze the mapping of grid parameter specifications.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Parameter keys are unique and each spec materializes a non-empty sequence.
        Raises:
            ValueError: If indicator id is missing, keys are invalid, or specs are empty.
        Side Effects:
            Replaces `params` with an immutable mapping proxy.
        """
        if self.indicator_id is None:  # type: ignore[truthy-bool]
            raise ValueError("GridSpec requires indicator_id")

        normalized: dict[str, GridParamSpec] = {}
        for key, spec in self.params.items():
            normalized_key = key.strip()
            if not normalized_key:
                raise ValueError("GridSpec parameter names must be non-empty")
            if normalized_key in normalized:
                raise ValueError("GridSpec parameter names must be unique")
            if len(spec.materialize()) == 0:
                raise ValueError(
                    f"GridSpec parameter '{normalized_key}' materialized to empty values"
                )
            normalized[normalized_key] = spec

        if self.source is not None and len(self.source.materialize()) == 0:
            raise ValueError("GridSpec source materialized to empty values")

        object.__setattr__(self, "params", MappingProxyType(normalized))
