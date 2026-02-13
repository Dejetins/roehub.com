from __future__ import annotations

from typing import Protocol

from trading.contexts.indicators.application.dto.registry_view import MergedIndicatorView
from trading.contexts.indicators.domain.entities import IndicatorDef, IndicatorId


class IndicatorRegistry(Protocol):
    """
    Port for listing and resolving indicator definitions.

    Docs:
      - docs/architecture/indicators/indicators-overview.md
      - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
      - docs/architecture/indicators/README.md
    Related:
      - src/trading/contexts/indicators/domain/entities/indicator_def.py
      - src/trading/contexts/indicators/domain/errors/unknown_indicator_error.py
      - src/trading/contexts/indicators/adapters/outbound/registry/yaml_indicator_registry.py
    """

    def list_defs(self) -> tuple[IndicatorDef, ...]:
        """
        Return all indicator definitions available in this registry.

        Docs:
            - docs/architecture/indicators/indicators-overview.md
            - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md

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

        Docs:
            - docs/architecture/indicators/indicators-overview.md
            - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md

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

    def list_merged(self) -> tuple[MergedIndicatorView, ...]:
        """
        Return merged registry view (hard definitions + defaults).

        Docs:
            - docs/architecture/indicators/indicators-overview.md
            - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md

        Args:
            None.
        Returns:
            tuple[MergedIndicatorView, ...]: Stable merged view for API/UI usage.
        Assumptions:
            Defaults are pre-validated and merged deterministically.
        Raises:
            None.
        Side Effects:
            None.
        """
        ...

    def get_merged(self, indicator_id: IndicatorId) -> MergedIndicatorView:
        """
        Resolve one merged registry view by identifier.

        Docs:
            - docs/architecture/indicators/indicators-overview.md
            - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md

        Args:
            indicator_id: Indicator identifier to resolve.
        Returns:
            MergedIndicatorView: Matching merged definition and defaults.
        Assumptions:
            Indicator id namespace is unique in registry.
        Raises:
            UnknownIndicatorError: If indicator is not present in registry.
        Side Effects:
            None.
        """
        ...
