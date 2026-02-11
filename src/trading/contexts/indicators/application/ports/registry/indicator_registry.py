from __future__ import annotations

from typing import Protocol

from trading.contexts.indicators.domain.entities import IndicatorDef, IndicatorId


class IndicatorRegistry(Protocol):
    """
    Port for listing and resolving indicator definitions.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ....domain.entities.indicator_def, ....domain.errors.unknown_indicator_error
    """

    def list_defs(self) -> tuple[IndicatorDef, ...]:
        """
        Return all indicator definitions available in this registry.

        Args:
            None.
        Returns:
            tuple[IndicatorDef, ...]: Stable tuple of registered indicator definitions.
        Assumptions:
            Returned definitions are immutable and safe to reuse across requests.
        Raises:
            None.
        Side Effects:
            None.
        """
        ...

    def get_def(self, indicator_id: IndicatorId) -> IndicatorDef:
        """
        Resolve one indicator definition by identifier.

        Args:
            indicator_id: Indicator identifier to resolve.
        Returns:
            IndicatorDef: Matching indicator definition.
        Assumptions:
            Registry identity space is unique and deterministic.
        Raises:
            UnknownIndicatorError: If the indicator is not present in the registry.
        Side Effects:
            None.
        """
        ...
