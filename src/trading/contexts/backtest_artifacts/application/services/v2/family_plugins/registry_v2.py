"""Deterministic registry for proposal-only family accelerators."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, cast

from ..execution_profile_v2 import ExecutionProfileModeLiteralV2
from .contracts_v2 import (
    FamilyAccelerationPluginV2,
    FamilyPluginPlanningContextV2,
    FamilyPluginSelectionKeyV2,
    FamilyPluginWarningV2,
)
from .ma_family_plugin_v2 import MAFamilyAccelerationPluginV2

type FamilyPluginRegistryStatusLiteralV2 = Literal[
    "resolved",
    "disabled",
    "not_applicable",
    "missing_plugin",
]

ALLOWED_FAMILY_PLUGIN_REGISTRY_STATUSES_V2: tuple[
    FamilyPluginRegistryStatusLiteralV2, ...
] = (
    "resolved",
    "disabled",
    "not_applicable",
    "missing_plugin",
)


@dataclass(frozen=True, slots=True)
class FamilyPluginRegistryResolutionV2:
    """
    Deterministic registry resolution result for proposal-only family-plugin selection.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py
    """

    status: FamilyPluginRegistryStatusLiteralV2
    selection_key: FamilyPluginSelectionKeyV2 | None = None
    plugin: FamilyAccelerationPluginV2 | None = None
    warning: FamilyPluginWarningV2 | None = None

    def __post_init__(self) -> None:
        """
        Validate one registry resolution result against explicit proposal-layer statuses.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Resolution keeps family-plugin selection explicit: callers can distinguish disabled,
            not-applicable, missing-plugin, and resolved states without inferring from `None`.
        Raises:
            ValueError: If status-dependent plugin/warning requirements are violated.
        Side Effects:
            None.
        """
        normalized_status = self.status.strip().lower()
        if normalized_status not in ALLOWED_FAMILY_PLUGIN_REGISTRY_STATUSES_V2:
            raise ValueError(
                "FamilyPluginRegistryResolutionV2.status must be one of "
                f"{ALLOWED_FAMILY_PLUGIN_REGISTRY_STATUSES_V2}, got {self.status!r}"
            )
        object.__setattr__(
            self,
            "status",
            cast(FamilyPluginRegistryStatusLiteralV2, normalized_status),
        )
        if self.status == "resolved":
            if self.selection_key is None:  # type: ignore[truthy-bool]
                raise ValueError(
                    "FamilyPluginRegistryResolutionV2.selection_key is required when "
                    "status='resolved'"
                )
            if self.plugin is None:  # type: ignore[truthy-bool]
                raise ValueError(
                    "FamilyPluginRegistryResolutionV2.plugin is required when "
                    "status='resolved'"
                )
            if self.warning is not None:
                raise ValueError(
                    "FamilyPluginRegistryResolutionV2.warning must be None when "
                    "status='resolved'"
                )
            return
        if self.plugin is not None:
            raise ValueError(
                "FamilyPluginRegistryResolutionV2.plugin must be None unless status='resolved'"
            )
        if self.status == "missing_plugin":
            if self.selection_key is None:  # type: ignore[truthy-bool]
                raise ValueError(
                    "FamilyPluginRegistryResolutionV2.selection_key is required when "
                    "status='missing_plugin'"
                )
            if self.warning is None:  # type: ignore[truthy-bool]
                raise ValueError(
                    "FamilyPluginRegistryResolutionV2.warning is required when "
                    "status='missing_plugin'"
                )
            return
        if self.status == "not_applicable":
            if self.selection_key is not None:
                raise ValueError(
                    "FamilyPluginRegistryResolutionV2.selection_key must be None when "
                    "status='not_applicable'"
                )
            if self.warning is None:  # type: ignore[truthy-bool]
                raise ValueError(
                    "FamilyPluginRegistryResolutionV2.warning is required when "
                    "status='not_applicable'"
                )
            return
        if self.warning is not None:
            raise ValueError(
                "FamilyPluginRegistryResolutionV2.warning must be None unless "
                "status='missing_plugin' or status='not_applicable'"
            )


@dataclass(frozen=True, slots=True)
class FamilyPluginRegistryV2:
    """
    Startup-validated deterministic registry for proposal-only family accelerators.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py
    """

    plugins: tuple[FamilyAccelerationPluginV2, ...] = ()
    _plugins_by_selection_key: Mapping[FamilyPluginSelectionKeyV2, FamilyAccelerationPluginV2] = (
        field(init=False, repr=False)
    )

    def __post_init__(self) -> None:
        """
        Normalize registry ordering and fail fast on duplicate family/profile registrations.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Registry selection must be deterministic and fail fast on startup instead of letting
            ambiguous plugin matches leak into runtime proposal planning.
        Raises:
            ValueError: If plugin ids are duplicated or two plugins claim the same
                `(execution_profile_mode, indicator_family_literal)` selection key.
        Side Effects:
            Reorders plugins by canonical `plugin_id` and freezes the lookup mapping.
        """
        sorted_plugins = tuple(
            sorted(self.plugins, key=lambda plugin: plugin.metadata.plugin_id)
        )
        object.__setattr__(self, "plugins", sorted_plugins)
        seen_plugin_ids: set[str] = set()
        plugins_by_selection_key: dict[
            FamilyPluginSelectionKeyV2,
            FamilyAccelerationPluginV2,
        ] = {}
        for plugin in sorted_plugins:
            plugin_id = plugin.metadata.plugin_id
            if plugin_id in seen_plugin_ids:
                raise ValueError(
                    f"FamilyPluginRegistryV2 plugin_id must be unique, got {plugin_id!r}"
                )
            seen_plugin_ids.add(plugin_id)
            for selection_key in _selection_keys_for_plugin_v2(plugin=plugin):
                existing_plugin = plugins_by_selection_key.get(selection_key)
                if existing_plugin is not None:
                    raise ValueError(
                        "FamilyPluginRegistryV2 selection key collision for "
                        f"{selection_key!r}: {existing_plugin.metadata.plugin_id!r} vs "
                        f"{plugin_id!r}"
                    )
                plugins_by_selection_key[selection_key] = plugin
        object.__setattr__(
            self,
            "_plugins_by_selection_key",
            MappingProxyType(plugins_by_selection_key),
        )

    def resolve(
        self,
        *,
        context: FamilyPluginPlanningContextV2,
    ) -> FamilyPluginRegistryResolutionV2:
        """
        Resolve one proposal-only family plugin from the prepared runtime planning context.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

        Args:
            context: Narrow immutable planning context derived from the shared runtime plan.
        Returns:
            FamilyPluginRegistryResolutionV2: Explicit deterministic resolution outcome.
        Assumptions:
            Lookup depends only on the resolved execution profile, the explicit
            `family_plugin_enabled` flag, and the deterministic indicator-family literal derived
            from the prepared runtime plan.
        Raises:
            ValueError: If the context is missing.
        Side Effects:
            None.
        """
        if context is None:  # type: ignore[truthy-bool]
            raise ValueError("FamilyPluginRegistryV2.resolve requires context")
        return self.resolve_selection(
            execution_profile_mode=context.runtime_plan.execution_profile.mode,
            indicator_family_literal=context.indicator_family_literal,
            family_plugin_enabled=(
                context.runtime_plan.execution_profile.feature_flags.family_plugin_enabled
            ),
        )

    def resolve_selection(
        self,
        *,
        execution_profile_mode: ExecutionProfileModeLiteralV2,
        indicator_family_literal: str | None,
        family_plugin_enabled: bool,
    ) -> FamilyPluginRegistryResolutionV2:
        """
        Resolve one proposal-only family-plugin candidate from explicit selector metadata.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

        Args:
            execution_profile_mode: Candidate execution-profile mode under evaluation.
            indicator_family_literal:
                Deterministic indicator-family literal derived from planning-time indicator ids,
                or `None` for mixed-family / unsupported requests.
            family_plugin_enabled:
                Explicit profile-level routing gate carried by the execution-profile catalog.
        Returns:
            FamilyPluginRegistryResolutionV2: Explicit deterministic registry resolution outcome.
        Assumptions:
            Adaptive selection and live runtime execution must share the same registry semantics
            so `hybrid_family` cannot be recommended when plugin routing is unavailable.
        Raises:
            ValueError: If the execution-profile mode literal is unsupported.
        Side Effects:
            None.
        """
        if not family_plugin_enabled:
            return FamilyPluginRegistryResolutionV2(status="disabled")
        if indicator_family_literal is None:
            return FamilyPluginRegistryResolutionV2(
                status="not_applicable",
                warning=FamilyPluginWarningV2(
                    reason="not_applicable",
                    message=(
                        "Prepared runtime plan spans mixed indicator families and does not map "
                        "to one proposal-only family plugin; warning + universal fallback applies."
                    ),
                ),
            )
        selection_key = FamilyPluginSelectionKeyV2(
            execution_profile_mode=execution_profile_mode,
            indicator_family_literal=indicator_family_literal,
        )
        plugin = self._plugins_by_selection_key.get(selection_key)
        if plugin is None:
            return FamilyPluginRegistryResolutionV2(
                status="missing_plugin",
                selection_key=selection_key,
                warning=FamilyPluginWarningV2(
                    reason="missing_plugin",
                    message=(
                        "No proposal-only family plugin is registered for "
                        f"{selection_key.execution_profile_mode!r} / "
                        f"{selection_key.indicator_family_literal!r}; "
                        "warning + universal fallback applies."
                    ),
                ),
            )
        return FamilyPluginRegistryResolutionV2(
            status="resolved",
            selection_key=selection_key,
            plugin=plugin,
        )


def _selection_keys_for_plugin_v2(
    *,
    plugin: FamilyAccelerationPluginV2,
) -> tuple[FamilyPluginSelectionKeyV2, ...]:
    """
    Expand one plugin's applicability metadata into deterministic registry selection keys.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

    Args:
        plugin: One proposal-only plugin instance carrying immutable applicability metadata.
    Returns:
        tuple[FamilyPluginSelectionKeyV2, ...]: Sorted lookup keys claimed by the plugin.
    Assumptions:
        Applicability is a pure cross-product of supported execution-profile modes and supported
        indicator-family literals until future selector work broadens the routing model.
    Raises:
        ValueError: If metadata-derived lookup keys drift from the immutable contract.
    Side Effects:
        None.
    """
    return tuple(
        sorted(
            (
                FamilyPluginSelectionKeyV2(
                    execution_profile_mode=execution_profile_mode,
                    indicator_family_literal=indicator_family_literal,
                )
                for execution_profile_mode in plugin.metadata.applicability.execution_profile_modes
                for indicator_family_literal in (
                    plugin.metadata.applicability.indicator_family_literals
                )
            ),
            key=lambda selection_key: (
                selection_key.execution_profile_mode,
                selection_key.indicator_family_literal,
            ),
        )
    )


def build_default_family_plugin_registry_v2() -> FamilyPluginRegistryV2:
    """
    Build the startup-validated default registry for shipped proposal-only family plugins.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/indicators/indicators-ma.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/
        ma_family_plugin_v2.py
      - src/trading/contexts/backtest/application/services/v2/
        hierarchical_shortlist_builder_v2.py

    Args:
        None.
    Returns:
        FamilyPluginRegistryV2: Registry populated with the shipped first `MA-family` plugin.
    Assumptions:
        Concrete plugins must register through the shared proposal-layer registry instead of
        ad-hoc runtime branching.
    Raises:
        ValueError: Propagated when shipped plugin metadata violates registry uniqueness rules.
    Side Effects:
        None.
    """
    return FamilyPluginRegistryV2(plugins=(MAFamilyAccelerationPluginV2(),))


__all__ = [
    "ALLOWED_FAMILY_PLUGIN_REGISTRY_STATUSES_V2",
    "build_default_family_plugin_registry_v2",
    "FamilyPluginRegistryResolutionV2",
    "FamilyPluginRegistryStatusLiteralV2",
    "FamilyPluginRegistryV2",
]
