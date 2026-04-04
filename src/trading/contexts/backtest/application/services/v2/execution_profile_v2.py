"""Typed execution-profile contracts for artifact-backed backtest runtime v2.

Docs:
  - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type ExecutionProfileModeLiteralV2 = Literal[
    "exact_small",
    "exact_parallel",
    "hybrid_conservative",
    "hybrid_family",
]

ALLOWED_EXECUTION_PROFILE_MODES_V2: tuple[ExecutionProfileModeLiteralV2, ...] = (
    "exact_small",
    "exact_parallel",
    "hybrid_conservative",
    "hybrid_family",
)
DEFAULT_EXECUTION_PROFILE_MODE_V2: ExecutionProfileModeLiteralV2 = "exact_small"
_EXACT_EXECUTION_PROFILE_MODES_V2: tuple[ExecutionProfileModeLiteralV2, ...] = (
    "exact_small",
    "exact_parallel",
)


def validate_execution_profile_mode_v2(
    *,
    value: str,
) -> ExecutionProfileModeLiteralV2:
    """
    Validate one execution-profile mode literal against the frozen v2 contract.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py

    Args:
        value: Raw execution-profile mode literal.
    Returns:
        ExecutionProfileModeLiteralV2: Normalized contract-approved mode literal.
    Assumptions:
        Mode literals are lowercase snake_case strings from the approved roadmap surface.
    Raises:
        ValueError: If the literal is blank or not part of the approved v2 set.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if not normalized_value:
        raise ValueError("ExecutionProfile.mode must be non-empty")
    if normalized_value not in ALLOWED_EXECUTION_PROFILE_MODES_V2:
        raise ValueError(
            "ExecutionProfile.mode must be one of "
            f"{ALLOWED_EXECUTION_PROFILE_MODES_V2}, got {value!r}"
        )
    return normalized_value


@dataclass(frozen=True, slots=True)
class ExecutionProfileShortlistConfigV2:
    """
    Typed shortlist contract for one execution profile.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py
    """

    enabled: bool = False
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        """
        Validate shortlist knobs for deterministic runtime/profile discovery contracts.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `max_candidates` is optional for exact profiles and strict-positive when present.
        Raises:
            ValueError: If one shortlist field violates deterministic bounds.
        Side Effects:
            None.
        """
        if not isinstance(self.enabled, bool):
            raise ValueError("ExecutionProfileShortlistConfigV2.enabled must be bool")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError(
                "ExecutionProfileShortlistConfigV2.max_candidates must be > 0 when provided"
            )


@dataclass(frozen=True, slots=True)
class ExecutionProfileParallelismConfigV2:
    """
    Typed parallelism contract for one execution profile.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py
    """

    stage_a_workers: int = 1
    stage_b_workers: int = 1

    def __post_init__(self) -> None:
        """
        Validate parallelism knobs for deterministic profile discovery and later rollout work.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Worker counts are strict-positive integers; current milestone only publishes them.
        Raises:
            ValueError: If one worker count is non-positive.
        Side Effects:
            None.
        """
        if self.stage_a_workers <= 0:
            raise ValueError("ExecutionProfileParallelismConfigV2.stage_a_workers must be > 0")
        if self.stage_b_workers <= 0:
            raise ValueError("ExecutionProfileParallelismConfigV2.stage_b_workers must be > 0")


@dataclass(frozen=True, slots=True)
class ExecutionProfileFeatureFlagsV2:
    """
    Typed feature-flag surface for one execution profile.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py
    """

    runtime_enabled: bool = True
    heuristic_shortlist_enabled: bool = False
    parallel_stage_b_enabled: bool = False
    family_plugin_enabled: bool = False

    def __post_init__(self) -> None:
        """
        Validate boolean feature flags exposed by execution-profile contracts.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Current milestone only publishes flags and keeps launch/runtime behavior unchanged.
        Raises:
            ValueError: If one field is not boolean.
        Side Effects:
            None.
        """
        for field_name, field_value in (
            ("runtime_enabled", self.runtime_enabled),
            ("heuristic_shortlist_enabled", self.heuristic_shortlist_enabled),
            ("parallel_stage_b_enabled", self.parallel_stage_b_enabled),
            ("family_plugin_enabled", self.family_plugin_enabled),
        ):
            if not isinstance(field_value, bool):
                raise ValueError(
                    f"ExecutionProfileFeatureFlagsV2.{field_name} must be bool"
                )


@dataclass(frozen=True, slots=True)
class ExecutionProfileV2:
    """
    Explicit typed execution profile for artifact-backed runtime planning and contract discovery.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    mode: ExecutionProfileModeLiteralV2
    shortlist_config: ExecutionProfileShortlistConfigV2
    parallelism: ExecutionProfileParallelismConfigV2
    feature_flags: ExecutionProfileFeatureFlagsV2
    planning_budget_ms: int

    def __post_init__(self) -> None:
        """
        Validate the execution-profile contract and normalize its mode literal.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Profile objects are immutable and reused across config/planner/DTO layers.
        Raises:
            ValueError: If one nested contract is missing or the planning budget is invalid.
        Side Effects:
            Normalizes `mode` to the approved lowercase literal.
        """
        object.__setattr__(
            self,
            "mode",
            validate_execution_profile_mode_v2(value=self.mode),
        )
        if self.shortlist_config is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionProfileV2.shortlist_config is required")
        if self.parallelism is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionProfileV2.parallelism is required")
        if self.feature_flags is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionProfileV2.feature_flags is required")
        if self.planning_budget_ms <= 0:
            raise ValueError("ExecutionProfileV2.planning_budget_ms must be > 0")


@dataclass(frozen=True, slots=True)
class ExecutionProfilesCatalogV2:
    """
    Ordered catalog of execution profiles published and consumed across the v2 runtime.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    default_mode: ExecutionProfileModeLiteralV2 = DEFAULT_EXECUTION_PROFILE_MODE_V2
    available_profiles: tuple[ExecutionProfileV2, ...] = ()

    def __post_init__(self) -> None:
        """
        Validate ordered catalog invariants and fail fast on contract drift.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Ordering of `available_profiles` is part of the browser/runtime contract surface.
        Raises:
            ValueError: If the catalog is empty, contains duplicates, misses known literals,
                or points default mode at a disabled/non-exact profile.
        Side Effects:
            Normalizes `default_mode` to the approved lowercase literal.
        """
        object.__setattr__(
            self,
            "default_mode",
            validate_execution_profile_mode_v2(value=self.default_mode),
        )
        if len(self.available_profiles) == 0:
            raise ValueError("ExecutionProfilesCatalogV2.available_profiles must be non-empty")

        seen_modes: set[ExecutionProfileModeLiteralV2] = set()
        for profile in self.available_profiles:
            if profile.mode in seen_modes:
                raise ValueError(f"duplicate ExecutionProfile.mode in catalog: {profile.mode}")
            seen_modes.add(profile.mode)

        missing_modes = [
            mode for mode in ALLOWED_EXECUTION_PROFILE_MODES_V2 if mode not in seen_modes
        ]
        if missing_modes:
            raise ValueError(
                "ExecutionProfilesCatalogV2.available_profiles must include all approved modes, "
                f"missing {tuple(missing_modes)}"
            )

        default_profile = self.profile_for_mode(mode=self.default_mode)
        if default_profile.mode not in _EXACT_EXECUTION_PROFILE_MODES_V2:
            raise ValueError(
                "ExecutionProfilesCatalogV2.default_mode must stay on an exact profile "
                "until hybrid rollout is implemented"
            )
        if not default_profile.feature_flags.runtime_enabled:
            raise ValueError(
                "ExecutionProfilesCatalogV2.default_mode must reference a runtime-enabled profile"
            )

    def default_profile(self) -> ExecutionProfileV2:
        """
        Return the ordered catalog entry configured as the default execution profile.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            None.
        Returns:
            ExecutionProfileV2: Catalog entry referenced by `default_mode`.
        Assumptions:
            Catalog invariants were validated during dataclass construction.
        Raises:
            ValueError: If `default_mode` no longer matches any available profile.
        Side Effects:
            None.
        """
        return self.profile_for_mode(mode=self.default_mode)

    def profile_for_mode(
        self,
        *,
        mode: ExecutionProfileModeLiteralV2,
    ) -> ExecutionProfileV2:
        """
        Resolve one ordered profile from the catalog by its stable mode literal.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            mode: Stable execution-profile mode literal.
        Returns:
            ExecutionProfileV2: Matching catalog entry preserving configured order.
        Assumptions:
            Catalog contains exactly one entry for every approved mode literal.
        Raises:
            ValueError: If the requested mode is not present in the catalog.
        Side Effects:
            None.
        """
        for profile in self.available_profiles:
            if profile.mode == mode:
                return profile
        raise ValueError(f"ExecutionProfilesCatalogV2 does not contain mode {mode!r}")


def default_execution_profiles_catalog_v2() -> ExecutionProfilesCatalogV2:
    """
    Build the default ordered execution-profile catalog for additive A1 rollout.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py

    Args:
        None.
    Returns:
        ExecutionProfilesCatalogV2: Default catalog with all known profile literals in
            deterministic order.
    Assumptions:
        Current milestone keeps `exact_small` as the default runtime-enabled baseline while
        future profiles remain explicitly represented for later EPICs.
    Raises:
        ValueError: If one default profile literal violates catalog invariants.
    Side Effects:
        None.
    """
    return ExecutionProfilesCatalogV2(
        default_mode=DEFAULT_EXECUTION_PROFILE_MODE_V2,
        available_profiles=(
            ExecutionProfileV2(
                mode="exact_small",
                shortlist_config=ExecutionProfileShortlistConfigV2(
                    enabled=False,
                    max_candidates=None,
                ),
                parallelism=ExecutionProfileParallelismConfigV2(
                    stage_a_workers=1,
                    stage_b_workers=1,
                ),
                feature_flags=ExecutionProfileFeatureFlagsV2(
                    runtime_enabled=True,
                    heuristic_shortlist_enabled=False,
                    parallel_stage_b_enabled=False,
                    family_plugin_enabled=False,
                ),
                planning_budget_ms=25,
            ),
            ExecutionProfileV2(
                mode="exact_parallel",
                shortlist_config=ExecutionProfileShortlistConfigV2(
                    enabled=False,
                    max_candidates=None,
                ),
                parallelism=ExecutionProfileParallelismConfigV2(
                    stage_a_workers=1,
                    stage_b_workers=4,
                ),
                feature_flags=ExecutionProfileFeatureFlagsV2(
                    runtime_enabled=False,
                    heuristic_shortlist_enabled=False,
                    parallel_stage_b_enabled=False,
                    family_plugin_enabled=False,
                ),
                planning_budget_ms=50,
            ),
            ExecutionProfileV2(
                mode="hybrid_conservative",
                shortlist_config=ExecutionProfileShortlistConfigV2(
                    enabled=True,
                    max_candidates=5000,
                ),
                parallelism=ExecutionProfileParallelismConfigV2(
                    stage_a_workers=1,
                    stage_b_workers=4,
                ),
                feature_flags=ExecutionProfileFeatureFlagsV2(
                    runtime_enabled=False,
                    heuristic_shortlist_enabled=False,
                    parallel_stage_b_enabled=False,
                    family_plugin_enabled=False,
                ),
                planning_budget_ms=75,
            ),
            ExecutionProfileV2(
                mode="hybrid_family",
                shortlist_config=ExecutionProfileShortlistConfigV2(
                    enabled=True,
                    max_candidates=2000,
                ),
                parallelism=ExecutionProfileParallelismConfigV2(
                    stage_a_workers=1,
                    stage_b_workers=4,
                ),
                feature_flags=ExecutionProfileFeatureFlagsV2(
                    runtime_enabled=False,
                    heuristic_shortlist_enabled=False,
                    parallel_stage_b_enabled=False,
                    family_plugin_enabled=False,
                ),
                planning_budget_ms=100,
            ),
        ),
    )


__all__ = [
    "ALLOWED_EXECUTION_PROFILE_MODES_V2",
    "DEFAULT_EXECUTION_PROFILE_MODE_V2",
    "ExecutionProfileFeatureFlagsV2",
    "ExecutionProfileModeLiteralV2",
    "ExecutionProfileParallelismConfigV2",
    "ExecutionProfileShortlistConfigV2",
    "ExecutionProfileV2",
    "ExecutionProfilesCatalogV2",
    "default_execution_profiles_catalog_v2",
    "validate_execution_profile_mode_v2",
]
