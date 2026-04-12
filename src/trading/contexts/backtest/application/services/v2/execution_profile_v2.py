"""Typed execution-profile contracts for artifact-backed backtest runtime v2.

Docs:
  - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, cast

from trading.contexts.backtest.domain.entities import BacktestJobStageWeights

type ExecutionProfileModeLiteralV2 = Literal[
    "exact_small",
    "exact_parallel",
    "hybrid_conservative",
    "hybrid_family",
]
type ExecutionProfileShortlistDiversityBucketLiteralV2 = Literal[
    "activity_band",
    "direction_band",
    "transition_band",
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
ALLOWED_EXECUTION_PROFILE_SHORTLIST_DIVERSITY_BUCKETS_V2: tuple[
    ExecutionProfileShortlistDiversityBucketLiteralV2, ...
] = (
    "activity_band",
    "direction_band",
    "transition_band",
)


def validate_execution_profile_shortlist_diversity_bucket_v2(
    *,
    value: str,
) -> ExecutionProfileShortlistDiversityBucketLiteralV2:
    """
    Validate one shortlist diversity bucket literal against the frozen D1 contract.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

    Args:
        value: Raw diversity-bucket literal from config or defaults payload.
    Returns:
        ExecutionProfileShortlistDiversityBucketLiteralV2: Canonical approved bucket literal.
    Assumptions:
        Conservative shortlist retention may use only exported scorer bucket axes to keep future
        rollout reviewable and deterministic.
    Raises:
        ValueError: If the literal is blank or outside the approved bucket set.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if not normalized_value:
        raise ValueError(
            "ExecutionProfile shortlist diversity bucket literal must be non-empty"
        )
    if normalized_value not in ALLOWED_EXECUTION_PROFILE_SHORTLIST_DIVERSITY_BUCKETS_V2:
        raise ValueError(
            "ExecutionProfile shortlist diversity bucket must be one of "
            f"{ALLOWED_EXECUTION_PROFILE_SHORTLIST_DIVERSITY_BUCKETS_V2}, got {value!r}"
        )
    return cast(ExecutionProfileShortlistDiversityBucketLiteralV2, normalized_value)


@dataclass(frozen=True, slots=True)
class ExecutionProfileShortlistScoringConfigV2:
    """
    Typed generic-row scoring weights for conservative shortlist foundation work.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py
    """

    activity_ratio_weight: float = 0.40
    direction_balance_weight: float = 0.25
    transition_ratio_weight: float = 0.25
    active_span_ratio_weight: float = 0.10

    def __post_init__(self) -> None:
        """
        Validate normalized shortlist scoring weights for universal row-scoring payloads.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Weights stay additive and reviewable so future rollout can reuse the same generic
            scorer without hidden heuristics.
        Raises:
            ValueError: If one weight is negative, non-finite, or the total weight is zero.
        Side Effects:
            Normalizes all weight fields to builtin `float`.
        """
        total_weight = 0.0
        for field_name in (
            "activity_ratio_weight",
            "direction_balance_weight",
            "transition_ratio_weight",
            "active_span_ratio_weight",
        ):
            raw_value = getattr(self, field_name)
            field_value = float(raw_value)
            if not math.isfinite(field_value):
                raise ValueError(
                    f"ExecutionProfileShortlistScoringConfigV2.{field_name} must be finite"
                )
            if field_value < 0.0:
                raise ValueError(
                    f"ExecutionProfileShortlistScoringConfigV2.{field_name} must be >= 0"
                )
            object.__setattr__(self, field_name, field_value)
            total_weight += field_value
        if total_weight <= 0.0:
            raise ValueError(
                "ExecutionProfileShortlistScoringConfigV2 must define at least one positive "
                "weight"
            )


@dataclass(frozen=True, slots=True)
class ExecutionProfileShortlistRetentionConfigV2:
    """
    Typed diversified-retention knobs for deterministic shortlist survivor selection.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py
    """

    diversity_buckets: tuple[ExecutionProfileShortlistDiversityBucketLiteralV2, ...] = (
        "activity_band",
        "direction_band",
    )
    max_per_bucket: int | None = None

    def __post_init__(self) -> None:
        """
        Validate deterministic diversity-bucket ordering and optional per-bucket survivor caps.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-runtime-kernels-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Bucket identity and iteration order remain explicit so later hybrid rollout can
            reason about `low_activity` and correlation-sensitive slices without hidden grouping.
        Raises:
            ValueError: If bucket literals are blank/duplicated/unsupported or one cap is
                non-positive.
        Side Effects:
            Normalizes bucket literals into an immutable tuple preserving configured order.
        """
        if len(self.diversity_buckets) == 0:
            raise ValueError(
                "ExecutionProfileShortlistRetentionConfigV2.diversity_buckets must be non-empty"
            )
        normalized_buckets: list[ExecutionProfileShortlistDiversityBucketLiteralV2] = []
        seen_buckets: set[ExecutionProfileShortlistDiversityBucketLiteralV2] = set()
        for raw_bucket in self.diversity_buckets:
            typed_bucket = validate_execution_profile_shortlist_diversity_bucket_v2(
                value=raw_bucket
            )
            if typed_bucket in seen_buckets:
                raise ValueError(
                    "ExecutionProfileShortlistRetentionConfigV2.diversity_buckets must not "
                    f"contain duplicates, got {raw_bucket!r}"
                )
            normalized_buckets.append(typed_bucket)
            seen_buckets.add(typed_bucket)
        object.__setattr__(self, "diversity_buckets", tuple(normalized_buckets))
        if self.max_per_bucket is not None and self.max_per_bucket <= 0:
            raise ValueError(
                "ExecutionProfileShortlistRetentionConfigV2.max_per_bucket must be > 0 when "
                "provided"
            )


@dataclass(frozen=True, slots=True)
class ExecutionProfileLaunchBudgetV2:
    """
    Deterministic request-shape budget used for exact profile selection and sync launch routing.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    max_stage_a_variants_total: int
    max_stage_b_variants_total: int
    max_estimated_memory_bytes: int

    def __post_init__(self) -> None:
        """
        Validate strict-positive launch-budget thresholds for deterministic profile routing.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Thresholds are explicit reviewable sync-launch budgets, not adaptive heuristics.
        Raises:
            ValueError: If one threshold is non-positive.
        Side Effects:
            None.
        """
        for field_name, field_value in (
            ("max_stage_a_variants_total", self.max_stage_a_variants_total),
            ("max_stage_b_variants_total", self.max_stage_b_variants_total),
            ("max_estimated_memory_bytes", self.max_estimated_memory_bytes),
        ):
            if field_value <= 0:
                raise ValueError(
                    f"ExecutionProfileLaunchBudgetV2.{field_name} must be > 0"
                )

    def allows(
        self,
        *,
        stage_a_variants_total: int,
        stage_b_variants_total: int,
        estimated_memory_bytes: int,
    ) -> bool:
        """
        Return whether one prepared exact request fits this profile's explicit launch budget.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            stage_a_variants_total: Deterministic prepared Stage A variants count.
            stage_b_variants_total: Deterministic prepared Stage B variants count.
            estimated_memory_bytes: Deterministic estimated runtime memory footprint.
        Returns:
            bool: `True` when the request fits all configured budget thresholds.
        Assumptions:
            Inputs were already validated by planner guard calculations and are deterministic.
        Raises:
            None.
        Side Effects:
            None.
        """
        return (
            stage_a_variants_total <= self.max_stage_a_variants_total
            and stage_b_variants_total <= self.max_stage_b_variants_total
            and estimated_memory_bytes <= self.max_estimated_memory_bytes
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


def execution_profile_uses_hierarchical_shortlist_runtime_v2(
    *,
    profile: "ExecutionProfileV2",
) -> bool:
    """
    Return whether one resolved profile may execute the live hybrid shortlist runtime.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

    Args:
        profile: Resolved execution profile candidate.
    Returns:
        bool: `True` only for the explicit opt-in hybrid shortlist runtime paths with all
            required rollout flags enabled.
    Assumptions:
        `hybrid_conservative` remains the universal shortlist path, while `hybrid_family`
        additionally requires `family_plugin_enabled` before the shared runtime may execute the
        proposal layer.
    Raises:
        None.
    Side Effects:
        None.
    """
    if (
        not profile.shortlist_config.enabled
        or not profile.feature_flags.runtime_enabled
        or not profile.feature_flags.heuristic_shortlist_enabled
    ):
        return False
    if profile.mode == "hybrid_conservative":
        return True
    return profile.mode == "hybrid_family" and profile.feature_flags.family_plugin_enabled


def execution_profile_supports_requested_runtime_v2(
    *,
    profile: "ExecutionProfileV2",
) -> bool:
    """
    Return whether one explicitly requested profile may run in the current live runtime.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py

    Args:
        profile: Resolved execution profile candidate.
    Returns:
        bool: `True` when the profile may be used as an internal requested runtime mode.
    Assumptions:
        Exact profiles require only `runtime_enabled`, while hybrid runtimes remain internal-only
        opt-in paths gated through the shared shortlist runtime checks.
    Raises:
        None.
    Side Effects:
        None.
    """
    if profile.mode in _EXACT_EXECUTION_PROFILE_MODES_V2:
        return profile.feature_flags.runtime_enabled
    return execution_profile_uses_hierarchical_shortlist_runtime_v2(profile=profile)


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
    scoring: ExecutionProfileShortlistScoringConfigV2 = field(
        default_factory=ExecutionProfileShortlistScoringConfigV2
    )
    retention: ExecutionProfileShortlistRetentionConfigV2 = field(
        default_factory=ExecutionProfileShortlistRetentionConfigV2
    )

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
            `max_candidates` is optional for exact profiles and strict-positive when present,
            while generic scoring/retention knobs stay additive until rollout activation.
        Raises:
            ValueError: If one shortlist field violates deterministic bounds.
        Side Effects:
            None.
        """
        if not isinstance(self.enabled, bool):
            raise ValueError("ExecutionProfileShortlistConfigV2.enabled must be bool")
        if self.scoring is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionProfileShortlistConfigV2.scoring is required")
        if self.retention is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionProfileShortlistConfigV2.retention is required")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError(
                "ExecutionProfileShortlistConfigV2.max_candidates must be > 0 when provided"
            )
        if self.enabled and self.max_candidates is None:
            raise ValueError(
                "ExecutionProfileShortlistConfigV2.max_candidates must be provided when "
                "shortlist is enabled"
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
            Worker counts are strict-positive integers; Stage B process fan-out stays available
            only for explicit non-default profiles that opt into it alongside these limits.
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
            Process-based Stage B is a non-default opt-in controlled by this feature-flag
            contract and the configured Stage B worker count together.
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
    launch_budget: ExecutionProfileLaunchBudgetV2
    progress_weights: BacktestJobStageWeights
    family_plugin_budget_ms: int
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
            Profile objects are immutable and reused across config/planner/DTO layers; any
            family-plugin timeout budget must stay explicitly tied to the profile budget surface.
        Raises:
            ValueError: If one nested contract is missing or one planning/plugin budget is
                invalid.
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
        if self.launch_budget is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionProfileV2.launch_budget is required")
        if self.progress_weights is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionProfileV2.progress_weights is required")
        if self.family_plugin_budget_ms <= 0:
            raise ValueError("ExecutionProfileV2.family_plugin_budget_ms must be > 0")
        if self.planning_budget_ms <= 0:
            raise ValueError("ExecutionProfileV2.planning_budget_ms must be > 0")
        if self.family_plugin_budget_ms > self.planning_budget_ms:
            raise ValueError(
                "ExecutionProfileV2.family_plugin_budget_ms must be <= "
                "ExecutionProfileV2.planning_budget_ms"
            )


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

    def runtime_enabled_exact_profiles(self) -> tuple[ExecutionProfileV2, ...]:
        """
        Return ordered runtime-enabled exact profiles available for current rollout decisions.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - apps/api/dto/backtest_runtime_defaults.py

        Args:
            None.
        Returns:
            tuple[ExecutionProfileV2, ...]: Ordered runtime-enabled exact profiles only.
        Assumptions:
            Exact-only rollout may activate multiple exact profiles before hybrid rollout exists.
        Raises:
            ValueError: If no runtime-enabled exact profiles remain in the catalog.
        Side Effects:
            None.
        """
        exact_profiles = tuple(
            profile
            for profile in self.available_profiles
            if profile.mode in _EXACT_EXECUTION_PROFILE_MODES_V2
            and profile.feature_flags.runtime_enabled
        )
        if len(exact_profiles) == 0:
            raise ValueError(
                "ExecutionProfilesCatalogV2 must include at least one runtime-enabled exact "
                "profile"
            )
        return exact_profiles

    def background_exact_profile(self) -> ExecutionProfileV2:
        """
        Return the heaviest runtime-enabled exact profile used for heavy background execution.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/backtest-job-runner-worker-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - apps/api/wiring/modules/backtest.py

        Args:
            None.
        Returns:
            ExecutionProfileV2: The last runtime-enabled exact profile in configured order.
        Assumptions:
            Catalog order reflects progressively heavier exact runtime profiles.
        Raises:
            ValueError: If there are no runtime-enabled exact profiles.
        Side Effects:
            None.
        """
        return self.runtime_enabled_exact_profiles()[-1]


def _default_launch_budget_for_mode_v2(
    *,
    mode: ExecutionProfileModeLiteralV2,
) -> ExecutionProfileLaunchBudgetV2:
    """
    Return default deterministic launch-budget thresholds for one approved profile literal.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - configs/prod/backtest.yaml

    Args:
        mode: Stable execution-profile mode literal.
    Returns:
        ExecutionProfileLaunchBudgetV2: Default launch budget for the requested profile.
    Assumptions:
        Exact profiles use explicit reviewable thresholds; hybrid defaults stay inert until
        runtime rollout activates them.
    Raises:
        ValueError: If the mode literal is unsupported.
    Side Effects:
        None.
    """
    budgets_by_mode: dict[ExecutionProfileModeLiteralV2, ExecutionProfileLaunchBudgetV2] = {
        "exact_small": ExecutionProfileLaunchBudgetV2(
            max_stage_a_variants_total=1500,
            max_stage_b_variants_total=12000,
            max_estimated_memory_bytes=268435456,
        ),
        "exact_parallel": ExecutionProfileLaunchBudgetV2(
            max_stage_a_variants_total=25000,
            max_stage_b_variants_total=180000,
            max_estimated_memory_bytes=1610612736,
        ),
        "hybrid_conservative": ExecutionProfileLaunchBudgetV2(
            max_stage_a_variants_total=50000,
            max_stage_b_variants_total=250000,
            max_estimated_memory_bytes=2147483648,
        ),
        "hybrid_family": ExecutionProfileLaunchBudgetV2(
            max_stage_a_variants_total=75000,
            max_stage_b_variants_total=300000,
            max_estimated_memory_bytes=2684354560,
        ),
    }
    try:
        return budgets_by_mode[mode]
    except KeyError as error:  # pragma: no cover - guarded by validated literal type
        raise ValueError(
            f"Unsupported execution profile mode for launch budget: {mode!r}"
        ) from error


def _default_progress_weights_for_mode_v2(
    *,
    mode: ExecutionProfileModeLiteralV2,
) -> BacktestJobStageWeights:
    """
    Return default deterministic progress weights for one approved execution profile literal.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-runs-history-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
      - configs/prod/backtest.yaml

    Args:
        mode: Stable execution-profile mode literal.
    Returns:
        BacktestJobStageWeights: Deterministic progress weights summing to `100`.
    Assumptions:
        Progress weights live inside the profile contract once profile selection becomes active,
        while internal row prefilter, combo prefilter, and retained-candidate exact work remain
        collapsed into stable public `stage_a` / `stage_b` vocabulary.
    Raises:
        ValueError: If the mode literal is unsupported.
    Side Effects:
        None.
    """
    weights_by_mode: dict[ExecutionProfileModeLiteralV2, BacktestJobStageWeights] = {
        "exact_small": BacktestJobStageWeights(stage_a=40, stage_b=55, finalizing=5),
        "exact_parallel": BacktestJobStageWeights(stage_a=45, stage_b=50, finalizing=5),
        "hybrid_conservative": BacktestJobStageWeights(
            stage_a=55,
            stage_b=40,
            finalizing=5,
        ),
        "hybrid_family": BacktestJobStageWeights(stage_a=60, stage_b=35, finalizing=5),
    }
    try:
        return weights_by_mode[mode]
    except KeyError as error:  # pragma: no cover - guarded by validated literal type
        raise ValueError(
            f"Unsupported execution profile mode for progress weights: {mode!r}"
        ) from error


def _default_family_plugin_budget_ms_for_mode_v2(
    *,
    mode: ExecutionProfileModeLiteralV2,
) -> int:
    """
    Return the default typed family-plugin planning budget for one execution profile.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/backtest-family-accelerators-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
      - apps/api/dto/backtest_runtime_defaults.py

    Args:
        mode: Stable execution-profile mode literal.
    Returns:
        int: Default budget in milliseconds reserved for future proposal-only family plugins.
    Assumptions:
        Plugin timeout stays budget-aware and is explicitly bounded by the execution-profile
        contract instead of an unrelated hardcoded timeout.
    Raises:
        ValueError: If the mode literal is unsupported.
    Side Effects:
        None.
    """
    budgets_by_mode: dict[ExecutionProfileModeLiteralV2, int] = {
        "exact_small": 10,
        "exact_parallel": 20,
        "hybrid_conservative": 30,
        "hybrid_family": 40,
    }
    try:
        return budgets_by_mode[mode]
    except KeyError as error:  # pragma: no cover - guarded by validated literal type
        raise ValueError(
            f"Unsupported execution profile mode for family-plugin budget: {mode!r}"
        ) from error


def default_execution_profiles_catalog_v2() -> ExecutionProfilesCatalogV2:
    """
    Build the default ordered execution-profile catalog for the in-process exact Stage B rollout.

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
        `exact_small` remains the active default exact baseline, while `exact_parallel` keeps the
        benchmark-sized Stage B path in-process by default and leaves process-pool fan-out as an
        explicit non-default opt-in for later rollout policy work.
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
                    scoring=ExecutionProfileShortlistScoringConfigV2(),
                    retention=ExecutionProfileShortlistRetentionConfigV2(),
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
                launch_budget=_default_launch_budget_for_mode_v2(mode="exact_small"),
                progress_weights=_default_progress_weights_for_mode_v2(mode="exact_small"),
                family_plugin_budget_ms=_default_family_plugin_budget_ms_for_mode_v2(
                    mode="exact_small"
                ),
                planning_budget_ms=25,
            ),
            ExecutionProfileV2(
                mode="exact_parallel",
                shortlist_config=ExecutionProfileShortlistConfigV2(
                    enabled=False,
                    max_candidates=None,
                    scoring=ExecutionProfileShortlistScoringConfigV2(),
                    retention=ExecutionProfileShortlistRetentionConfigV2(),
                ),
                parallelism=ExecutionProfileParallelismConfigV2(
                    stage_a_workers=4,
                    stage_b_workers=1,
                ),
                feature_flags=ExecutionProfileFeatureFlagsV2(
                    runtime_enabled=True,
                    heuristic_shortlist_enabled=False,
                    parallel_stage_b_enabled=False,
                    family_plugin_enabled=False,
                ),
                launch_budget=_default_launch_budget_for_mode_v2(mode="exact_parallel"),
                progress_weights=_default_progress_weights_for_mode_v2(mode="exact_parallel"),
                family_plugin_budget_ms=_default_family_plugin_budget_ms_for_mode_v2(
                    mode="exact_parallel"
                ),
                planning_budget_ms=50,
            ),
            ExecutionProfileV2(
                mode="hybrid_conservative",
                shortlist_config=ExecutionProfileShortlistConfigV2(
                    enabled=True,
                    max_candidates=5000,
                    scoring=ExecutionProfileShortlistScoringConfigV2(
                        activity_ratio_weight=0.40,
                        direction_balance_weight=0.25,
                        transition_ratio_weight=0.25,
                        active_span_ratio_weight=0.10,
                    ),
                    retention=ExecutionProfileShortlistRetentionConfigV2(
                        diversity_buckets=("activity_band", "direction_band"),
                        max_per_bucket=750,
                    ),
                ),
                parallelism=ExecutionProfileParallelismConfigV2(
                    stage_a_workers=4,
                    stage_b_workers=3,
                ),
                feature_flags=ExecutionProfileFeatureFlagsV2(
                    runtime_enabled=False,
                    heuristic_shortlist_enabled=False,
                    parallel_stage_b_enabled=False,
                    family_plugin_enabled=False,
                ),
                launch_budget=_default_launch_budget_for_mode_v2(
                    mode="hybrid_conservative"
                ),
                progress_weights=_default_progress_weights_for_mode_v2(
                    mode="hybrid_conservative"
                ),
                family_plugin_budget_ms=_default_family_plugin_budget_ms_for_mode_v2(
                    mode="hybrid_conservative"
                ),
                planning_budget_ms=75,
            ),
            ExecutionProfileV2(
                mode="hybrid_family",
                shortlist_config=ExecutionProfileShortlistConfigV2(
                    enabled=True,
                    max_candidates=2000,
                    scoring=ExecutionProfileShortlistScoringConfigV2(
                        activity_ratio_weight=0.35,
                        direction_balance_weight=0.20,
                        transition_ratio_weight=0.30,
                        active_span_ratio_weight=0.15,
                    ),
                    retention=ExecutionProfileShortlistRetentionConfigV2(
                        diversity_buckets=("activity_band", "transition_band"),
                        max_per_bucket=300,
                    ),
                ),
                parallelism=ExecutionProfileParallelismConfigV2(
                    stage_a_workers=3,
                    stage_b_workers=2,
                ),
                feature_flags=ExecutionProfileFeatureFlagsV2(
                    runtime_enabled=False,
                    heuristic_shortlist_enabled=False,
                    parallel_stage_b_enabled=False,
                    family_plugin_enabled=False,
                ),
                launch_budget=_default_launch_budget_for_mode_v2(mode="hybrid_family"),
                progress_weights=_default_progress_weights_for_mode_v2(
                    mode="hybrid_family"
                ),
                family_plugin_budget_ms=_default_family_plugin_budget_ms_for_mode_v2(
                    mode="hybrid_family"
                ),
                planning_budget_ms=100,
            ),
        ),
    )


__all__ = [
    "ALLOWED_EXECUTION_PROFILE_SHORTLIST_DIVERSITY_BUCKETS_V2",
    "ALLOWED_EXECUTION_PROFILE_MODES_V2",
    "DEFAULT_EXECUTION_PROFILE_MODE_V2",
    "ExecutionProfileFeatureFlagsV2",
    "ExecutionProfileLaunchBudgetV2",
    "ExecutionProfileModeLiteralV2",
    "ExecutionProfileParallelismConfigV2",
    "ExecutionProfileShortlistDiversityBucketLiteralV2",
    "ExecutionProfileShortlistRetentionConfigV2",
    "ExecutionProfileShortlistScoringConfigV2",
    "ExecutionProfileShortlistConfigV2",
    "ExecutionProfileV2",
    "ExecutionProfilesCatalogV2",
    "default_execution_profiles_catalog_v2",
    "execution_profile_supports_requested_runtime_v2",
    "execution_profile_uses_hierarchical_shortlist_runtime_v2",
    "validate_execution_profile_shortlist_diversity_bucket_v2",
    "validate_execution_profile_mode_v2",
]
