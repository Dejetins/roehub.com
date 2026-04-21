"""Proposal-only family-accelerator contracts for future `hybrid_family` rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

from ..execution_profile_v2 import (
    ExecutionProfileModeLiteralV2,
    validate_execution_profile_mode_v2,
)

if TYPE_CHECKING:
    from ..artifact_runtime_plan_v2 import BacktestArtifactRuntimePlanV2

type FamilyPluginProposalCapabilityLiteralV2 = Literal[
    "row_shortlist",
    "pair_shortlist",
    "proxy_score",
]
type FamilyPluginWarningReasonLiteralV2 = Literal[
    "missing_plugin",
    "not_applicable",
    "timeout",
    "error",
    "open_breaker",
]

ALLOWED_FAMILY_PLUGIN_PROPOSAL_CAPABILITIES_V2: tuple[
    FamilyPluginProposalCapabilityLiteralV2, ...
] = (
    "row_shortlist",
    "pair_shortlist",
    "proxy_score",
)
ALLOWED_FAMILY_PLUGIN_WARNING_REASONS_V2: tuple[FamilyPluginWarningReasonLiteralV2, ...] = (
    "missing_plugin",
    "not_applicable",
    "timeout",
    "error",
    "open_breaker",
)
FAMILY_PLUGIN_WARNING_FALLBACK_ACTION_V2 = "warning + universal fallback"


def normalize_family_plugin_identifier_v2(*, value: str, field_name: str) -> str:
    """
    Normalize one internal family-plugin identifier literal to stable lower-case form.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py

    Args:
        value: Candidate identifier literal from config, registry metadata, or tests.
        field_name: Human-readable field path used in validation errors.
    Returns:
        str: Canonical lower-case identifier.
    Assumptions:
        Family and plugin ids remain internal deterministic literals and are never sourced from
        the public `POST /backtests` payload.
    Raises:
        ValueError: If the identifier is blank after normalization.
    Side Effects:
        None.
    """
    normalized_value = value.strip().lower()
    if not normalized_value:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized_value


def validate_family_plugin_proposal_capability_v2(
    *,
    value: str,
) -> FamilyPluginProposalCapabilityLiteralV2:
    """
    Validate one proposal-capability literal for proposal-only family-plugin contracts.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py

    Args:
        value: Candidate proposal-capability literal.
    Returns:
        FamilyPluginProposalCapabilityLiteralV2: Canonical approved capability literal.
    Assumptions:
        Plugins may propose only `row shortlist`, `pair shortlist`, and/or `proxy score`
        suggestions while final exact Stage B scoring remains canonical.
    Raises:
        ValueError: If the literal is blank or outside the approved proposal-only set.
    Side Effects:
        None.
    """
    normalized_value = normalize_family_plugin_identifier_v2(
        value=value,
        field_name="FamilyPlugin proposal capability",
    )
    if normalized_value not in ALLOWED_FAMILY_PLUGIN_PROPOSAL_CAPABILITIES_V2:
        raise ValueError(
            "FamilyPlugin proposal capability must be one of "
            f"{ALLOWED_FAMILY_PLUGIN_PROPOSAL_CAPABILITIES_V2}, got {value!r}"
        )
    return cast(FamilyPluginProposalCapabilityLiteralV2, normalized_value)


def validate_family_plugin_warning_reason_v2(
    *,
    value: str,
) -> FamilyPluginWarningReasonLiteralV2:
    """
    Validate one warning-reason literal for family-plugin failure-handling contracts.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_family_plugin_circuit_breaker_v2.py

    Args:
        value: Candidate warning-reason literal.
    Returns:
        FamilyPluginWarningReasonLiteralV2: Canonical approved warning-reason literal.
    Assumptions:
        Family-plugin failure handling is explicit and limited to missing-plugin,
        not-applicable, timeout, error, and open-breaker paths.
    Raises:
        ValueError: If the literal is blank or outside the approved warning set.
    Side Effects:
        None.
    """
    normalized_value = normalize_family_plugin_identifier_v2(
        value=value,
        field_name="FamilyPlugin warning reason",
    )
    if normalized_value not in ALLOWED_FAMILY_PLUGIN_WARNING_REASONS_V2:
        raise ValueError(
            "FamilyPlugin warning reason must be one of "
            f"{ALLOWED_FAMILY_PLUGIN_WARNING_REASONS_V2}, got {value!r}"
        )
    return cast(FamilyPluginWarningReasonLiteralV2, normalized_value)


def resolve_family_plugin_indicator_family_v2(
    *,
    indicator_ids: tuple[str, ...],
) -> str | None:
    """
    Resolve one deterministic indicator-family literal from normalized indicator ids.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

    Args:
        indicator_ids: Deterministic normalized indicator-id tuple from the prepared runtime plan.
    Returns:
        str | None: Shared family prefix before the first `.` when all ids belong to one family,
            else `None`.
    Assumptions:
        Family selection stays internal and explicit: mixed indicator families must fall back to
        the universal proposal path until a future selector defines something more specific.
    Raises:
        ValueError: If one provided indicator id is blank after normalization.
    Side Effects:
        None.
    """
    if len(indicator_ids) == 0:
        return None
    normalized_indicator_ids = tuple(
        normalize_family_plugin_identifier_v2(
            value=indicator_id,
            field_name="FamilyPluginPlanningContextV2.indicator_ids[]",
        )
        for indicator_id in indicator_ids
    )
    resolved_families = {
        normalized_indicator_id.split(".", 1)[0]
        for normalized_indicator_id in normalized_indicator_ids
    }
    if len(resolved_families) != 1:
        return None
    return next(iter(resolved_families))


def build_family_plugin_planning_context_v2(
    *,
    runtime_plan: BacktestArtifactRuntimePlanV2,
    requested_execution_profile_mode: ExecutionProfileModeLiteralV2 | None = None,
) -> "FamilyPluginPlanningContextV2":
    """
    Build the narrow immutable planning context reused by future family-plugin proposal work.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

    Args:
        runtime_plan: Prepared artifact-backed runtime plan owned by the shared planner.
        requested_execution_profile_mode:
            Optional explicit requested profile mode copied from internal request metadata.
    Returns:
        FamilyPluginPlanningContextV2: Narrow proposal-only context referencing the prepared plan.
    Assumptions:
        The context must not duplicate full orchestration state: it reuses the shared runtime
        plan, exposes normalized indicator ids, and carries the typed family-plugin budget.
    Raises:
        ValueError: If the runtime plan is missing or its plugin budget/profile metadata drift.
    Side Effects:
        None.
    """
    if runtime_plan is None:  # type: ignore[truthy-bool]
        raise ValueError("build_family_plugin_planning_context_v2 requires runtime_plan")
    indicator_ids = tuple(
        sorted({plan.indicator_id.strip().lower() for plan in runtime_plan.indicator_plans})
    )
    return FamilyPluginPlanningContextV2(
        runtime_plan=runtime_plan,
        requested_execution_profile_mode=requested_execution_profile_mode,
        indicator_ids=indicator_ids,
        indicator_family_literal=resolve_family_plugin_indicator_family_v2(
            indicator_ids=indicator_ids
        ),
        plugin_budget_ms=runtime_plan.execution_profile.family_plugin_budget_ms,
    )


@dataclass(frozen=True, slots=True)
class FamilyPluginSelectionKeyV2:
    """
    Deterministic registry lookup key for one proposal-only family-plugin candidate.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py
    """

    execution_profile_mode: ExecutionProfileModeLiteralV2
    indicator_family_literal: str

    def __post_init__(self) -> None:
        """
        Validate and normalize the immutable family-plugin registry selection key.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Registry lookup may depend only on stable execution-profile and indicator-family
            metadata already available from runtime planning.
        Raises:
            ValueError: If one lookup literal is blank or unsupported.
        Side Effects:
            Normalizes lookup literals to canonical lower-case form.
        """
        object.__setattr__(
            self,
            "execution_profile_mode",
            validate_execution_profile_mode_v2(value=self.execution_profile_mode),
        )
        object.__setattr__(
            self,
            "indicator_family_literal",
            normalize_family_plugin_identifier_v2(
                value=self.indicator_family_literal,
                field_name="FamilyPluginSelectionKeyV2.indicator_family_literal",
            ),
        )


@dataclass(frozen=True, slots=True)
class FamilyPluginPlanningContextV2:
    """
    Narrow immutable planning context passed to proposal-only family accelerators.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
    """

    runtime_plan: BacktestArtifactRuntimePlanV2
    requested_execution_profile_mode: ExecutionProfileModeLiteralV2 | None
    indicator_ids: tuple[str, ...]
    indicator_family_literal: str | None
    plugin_budget_ms: int

    def __post_init__(self) -> None:
        """
        Validate the narrow family-plugin context derived from prepared runtime planning.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The context reuses the prepared runtime plan, carries only explicit request metadata,
            and keeps the family-plugin budget tied to the resolved execution profile.
        Raises:
            ValueError: If the runtime plan is missing, the budget is invalid, or normalized
                indicator metadata drifts from the resolved family literal.
        Side Effects:
            Normalizes indicator ids and optional request/profile literals to canonical form.
        """
        if self.runtime_plan is None:  # type: ignore[truthy-bool]
            raise ValueError("FamilyPluginPlanningContextV2.runtime_plan is required")
        if self.requested_execution_profile_mode is not None:
            object.__setattr__(
                self,
                "requested_execution_profile_mode",
                validate_execution_profile_mode_v2(
                    value=self.requested_execution_profile_mode
                ),
            )
        normalized_indicator_ids = tuple(
            sorted(
                {
                    normalize_family_plugin_identifier_v2(
                        value=indicator_id,
                        field_name="FamilyPluginPlanningContextV2.indicator_ids[]",
                    )
                    for indicator_id in self.indicator_ids
                }
            )
        )
        object.__setattr__(self, "indicator_ids", normalized_indicator_ids)
        if self.indicator_family_literal is not None:
            object.__setattr__(
                self,
                "indicator_family_literal",
                normalize_family_plugin_identifier_v2(
                    value=self.indicator_family_literal,
                    field_name="FamilyPluginPlanningContextV2.indicator_family_literal",
                ),
            )
        resolved_indicator_family = resolve_family_plugin_indicator_family_v2(
            indicator_ids=normalized_indicator_ids
        )
        if resolved_indicator_family != self.indicator_family_literal:
            raise ValueError(
                "FamilyPluginPlanningContextV2.indicator_family_literal must match the "
                "deterministic family resolution rule derived from indicator_ids"
            )
        if self.plugin_budget_ms <= 0:
            raise ValueError("FamilyPluginPlanningContextV2.plugin_budget_ms must be > 0")
        if self.plugin_budget_ms > self.runtime_plan.execution_profile.planning_budget_ms:
            raise ValueError(
                "FamilyPluginPlanningContextV2.plugin_budget_ms must be <= the resolved "
                "ExecutionProfileV2.planning_budget_ms"
            )
        if self.plugin_budget_ms != self.runtime_plan.execution_profile.family_plugin_budget_ms:
            raise ValueError(
                "FamilyPluginPlanningContextV2.plugin_budget_ms must reuse the resolved "
                "ExecutionProfileV2.family_plugin_budget_ms"
            )


@dataclass(frozen=True, slots=True)
class FamilyPluginApplicabilityV2:
    """
    Immutable applicability metadata for deterministic family-plugin registry selection.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py
    """

    execution_profile_modes: tuple[ExecutionProfileModeLiteralV2, ...] = ("hybrid_family",)
    indicator_family_literals: tuple[str, ...] = ()
    feature_flag_required: bool = True

    def __post_init__(self) -> None:
        """
        Validate applicability metadata used by startup-time registry normalization.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Applicability must stay explicit so registry lookup depends only on stable family and
            execution-profile metadata from the prepared runtime plan.
        Raises:
            ValueError: If the applicability surface is empty, duplicated, or contains unsupported
                literals.
        Side Effects:
            Normalizes supported modes and family literals to immutable sorted tuples.
        """
        if not isinstance(self.feature_flag_required, bool):
            raise ValueError("FamilyPluginApplicabilityV2.feature_flag_required must be bool")
        normalized_execution_profile_modes = tuple(
            sorted(
                {
                    validate_execution_profile_mode_v2(value=mode)
                    for mode in self.execution_profile_modes
                }
            )
        )
        if len(normalized_execution_profile_modes) == 0:
            raise ValueError(
                "FamilyPluginApplicabilityV2.execution_profile_modes must be non-empty"
            )
        object.__setattr__(
            self,
            "execution_profile_modes",
            normalized_execution_profile_modes,
        )
        normalized_indicator_family_literals = tuple(
            sorted(
                {
                    normalize_family_plugin_identifier_v2(
                        value=indicator_family_literal,
                        field_name=(
                            "FamilyPluginApplicabilityV2.indicator_family_literals[]"
                        ),
                    )
                    for indicator_family_literal in self.indicator_family_literals
                }
            )
        )
        if len(normalized_indicator_family_literals) == 0:
            raise ValueError(
                "FamilyPluginApplicabilityV2.indicator_family_literals must be non-empty"
            )
        object.__setattr__(
            self,
            "indicator_family_literals",
            normalized_indicator_family_literals,
        )


@dataclass(frozen=True, slots=True)
class FamilyPluginMetadataV2:
    """
    Immutable identity and applicability metadata for one proposal-only family plugin.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py
    """

    plugin_id: str
    display_name: str
    applicability: FamilyPluginApplicabilityV2
    proposal_capabilities: tuple[FamilyPluginProposalCapabilityLiteralV2, ...]

    def __post_init__(self) -> None:
        """
        Validate plugin identity metadata published to the deterministic registry.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Plugin identity and capability metadata remain startup-validated and do not redefine
            final exact scoring semantics.
        Raises:
            ValueError: If the plugin id/display name is blank, applicability is missing, or the
                capability set is empty/duplicated.
        Side Effects:
            Normalizes plugin id and proposal capabilities to canonical immutable tuples.
        """
        object.__setattr__(
            self,
            "plugin_id",
            normalize_family_plugin_identifier_v2(
                value=self.plugin_id,
                field_name="FamilyPluginMetadataV2.plugin_id",
            ),
        )
        normalized_display_name = self.display_name.strip()
        if not normalized_display_name:
            raise ValueError("FamilyPluginMetadataV2.display_name must be non-empty")
        object.__setattr__(self, "display_name", normalized_display_name)
        if self.applicability is None:  # type: ignore[truthy-bool]
            raise ValueError("FamilyPluginMetadataV2.applicability is required")
        normalized_capabilities = tuple(
            sorted(
                {
                    validate_family_plugin_proposal_capability_v2(value=capability)
                    for capability in self.proposal_capabilities
                }
            )
        )
        if len(normalized_capabilities) == 0:
            raise ValueError(
                "FamilyPluginMetadataV2.proposal_capabilities must be non-empty"
            )
        object.__setattr__(self, "proposal_capabilities", normalized_capabilities)


@dataclass(frozen=True, slots=True)
class FamilyPluginPairCandidateV2:
    """
    Proposal-only Stage B pair shortlist candidate addressed by Stage A row and risk indexes.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py
    """

    stage_a_index: int
    risk_index: int

    def __post_init__(self) -> None:
        """
        Validate one deterministic pair-shortlist candidate reference.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Pair shortlist references existing shared Stage A / Stage B coordinates instead of
            creating a family-specific runtime engine.
        Raises:
            ValueError: If one index is negative.
        Side Effects:
            None.
        """
        if self.stage_a_index < 0:
            raise ValueError("FamilyPluginPairCandidateV2.stage_a_index must be >= 0")
        if self.risk_index < 0:
            raise ValueError("FamilyPluginPairCandidateV2.risk_index must be >= 0")


@dataclass(frozen=True, slots=True)
class FamilyPluginProxyScoreV2:
    """
    Proposal-only proxy-score suggestion for one row or one explicit Stage B pair candidate.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py
    """

    stage_a_index: int
    proxy_score: float
    risk_index: int | None = None

    def __post_init__(self) -> None:
        """
        Validate one deterministic `proxy score` suggestion without changing exact semantics.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Proxy scores are proposal-only hints for future shortlist ordering; exact Stage B
            scoring remains canonical on retained survivors.
        Raises:
            ValueError: If one index is negative or the proxy score is non-finite.
        Side Effects:
            Normalizes `proxy_score` to builtin `float`.
        """
        if self.stage_a_index < 0:
            raise ValueError("FamilyPluginProxyScoreV2.stage_a_index must be >= 0")
        if self.risk_index is not None and self.risk_index < 0:
            raise ValueError("FamilyPluginProxyScoreV2.risk_index must be >= 0 when provided")
        normalized_proxy_score = float(self.proxy_score)
        if not math.isfinite(normalized_proxy_score):
            raise ValueError("FamilyPluginProxyScoreV2.proxy_score must be finite")
        object.__setattr__(self, "proxy_score", normalized_proxy_score)


@dataclass(frozen=True, slots=True)
class FamilyPluginProposalResultV2:
    """
    Proposal-only family-plugin output for `row shortlist`, `pair shortlist`, and `proxy score`.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py
    """

    plugin_id: str
    row_shortlist: tuple[int, ...] = ()
    pair_shortlist: tuple[FamilyPluginPairCandidateV2, ...] = ()
    proxy_scores: tuple[FamilyPluginProxyScoreV2, ...] = ()

    def __post_init__(self) -> None:
        """
        Validate and normalize proposal-only shortlist outputs into deterministic order.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Proposal ordering must be deterministic and reviewable; consumers may still rank by
            `proxy_score`, but the normalized transport contract sorts by explicit candidate keys.
        Raises:
            ValueError: If one shortlist entry is negative or if proxy-score targets are
                duplicated.
        Side Effects:
            Normalizes `plugin_id`, deduplicates/sorts shortlist coordinates, and canonicalizes
            proxy-score ordering.
        """
        object.__setattr__(
            self,
            "plugin_id",
            normalize_family_plugin_identifier_v2(
                value=self.plugin_id,
                field_name="FamilyPluginProposalResultV2.plugin_id",
            ),
        )
        normalized_row_shortlist = tuple(
            sorted(
                {
                    _validate_family_plugin_row_index_v2(stage_a_index=stage_a_index)
                    for stage_a_index in self.row_shortlist
                }
            )
        )
        object.__setattr__(self, "row_shortlist", normalized_row_shortlist)
        normalized_pair_shortlist = tuple(
            sorted(
                {
                    FamilyPluginPairCandidateV2(
                        stage_a_index=pair_candidate.stage_a_index,
                        risk_index=pair_candidate.risk_index,
                    )
                    for pair_candidate in self.pair_shortlist
                },
                key=lambda pair_candidate: (
                    pair_candidate.stage_a_index,
                    pair_candidate.risk_index,
                ),
            )
        )
        object.__setattr__(self, "pair_shortlist", normalized_pair_shortlist)
        normalized_proxy_scores = [
            FamilyPluginProxyScoreV2(
                stage_a_index=proxy_score.stage_a_index,
                risk_index=proxy_score.risk_index,
                proxy_score=proxy_score.proxy_score,
            )
            for proxy_score in self.proxy_scores
        ]
        seen_proxy_targets: set[tuple[int, int | None]] = set()
        for proxy_score in normalized_proxy_scores:
            target_key = (proxy_score.stage_a_index, proxy_score.risk_index)
            if target_key in seen_proxy_targets:
                raise ValueError(
                    "FamilyPluginProposalResultV2.proxy_scores must not contain duplicate "
                    f"candidate targets, got {target_key!r}"
                )
            seen_proxy_targets.add(target_key)
        object.__setattr__(
            self,
            "proxy_scores",
            tuple(
                sorted(
                    normalized_proxy_scores,
                    key=lambda proxy_score: (
                        proxy_score.stage_a_index,
                        -1 if proxy_score.risk_index is None else proxy_score.risk_index,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class FamilyPluginWarningV2:
    """
    Reusable warning payload for explicit family-plugin fallback semantics.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py
    """

    reason: FamilyPluginWarningReasonLiteralV2
    message: str
    plugin_id: str | None = None
    fallback_action: str = FAMILY_PLUGIN_WARNING_FALLBACK_ACTION_V2

    def __post_init__(self) -> None:
        """
        Validate one explicit family-plugin warning and fallback action payload.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/
            circuit_breaker_v2.py

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warning payloads are the reusable diagnostic surface for timeout, error,
            open-breaker, missing-plugin, and not-applicable fallback paths.
        Raises:
            ValueError: If the reason or fallback action drift from the frozen contract, or if
                `message` is blank.
        Side Effects:
            Normalizes warning reason and optional plugin id to canonical lower-case form.
        """
        object.__setattr__(
            self,
            "reason",
            validate_family_plugin_warning_reason_v2(value=self.reason),
        )
        normalized_message = self.message.strip()
        if not normalized_message:
            raise ValueError("FamilyPluginWarningV2.message must be non-empty")
        object.__setattr__(self, "message", normalized_message)
        if self.plugin_id is not None:
            object.__setattr__(
                self,
                "plugin_id",
                normalize_family_plugin_identifier_v2(
                    value=self.plugin_id,
                    field_name="FamilyPluginWarningV2.plugin_id",
                ),
            )
        if self.reason in {"timeout", "error", "open_breaker"} and self.plugin_id is None:
            raise ValueError(
                "FamilyPluginWarningV2.plugin_id is required for "
                "timeout/error/open_breaker warnings"
            )
        normalized_fallback_action = self.fallback_action.strip().lower()
        if normalized_fallback_action != FAMILY_PLUGIN_WARNING_FALLBACK_ACTION_V2:
            raise ValueError(
                "FamilyPluginWarningV2.fallback_action must equal "
                f"{FAMILY_PLUGIN_WARNING_FALLBACK_ACTION_V2!r}"
            )
        object.__setattr__(self, "fallback_action", normalized_fallback_action)


class FamilyAccelerationPluginV2(Protocol):
    """
    Proposal-only plugin protocol for future family accelerators behind `hybrid_family`.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_registry_v2.py
    """

    metadata: FamilyPluginMetadataV2

    def propose(
        self,
        *,
        context: FamilyPluginPlanningContextV2,
    ) -> FamilyPluginProposalResultV2:
        """
        Produce proposal-only shortlist/proxy suggestions for one prepared runtime plan.

        Docs:
          - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
          - docs/architecture/backtest/README.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
          - src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py
          - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py

        Args:
            context: Narrow immutable planning context referencing the prepared runtime plan.
        Returns:
            FamilyPluginProposalResultV2: Proposal-only row/pair shortlist and/or proxy scores.
        Assumptions:
            Implementations must never replace the shared exact scorer; they may only provide
            proposal-layer hints that later runtime code can validate or ignore.
        Raises:
            Exception: Implementations may raise, after which the caller applies explicit warning
                plus universal fallback semantics.
        Side Effects:
            None.
        """
        ...


def _validate_family_plugin_row_index_v2(*, stage_a_index: int) -> int:
    """
    Validate one proposal-only Stage A row shortlist index.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py
      - tests/unit/contexts/backtest/application/services/v2/test_family_plugin_contracts_v2.py

    Args:
        stage_a_index: Candidate Stage A row index proposed by one family plugin.
    Returns:
        int: The same index when valid.
    Assumptions:
        Row shortlist entries refer to shared Stage A coordinates already owned by the runtime
        planner.
    Raises:
        ValueError: If the index is negative.
    Side Effects:
        None.
    """
    if stage_a_index < 0:
        raise ValueError("FamilyPlugin row shortlist indexes must be >= 0")
    return stage_a_index


__all__ = [
    "ALLOWED_FAMILY_PLUGIN_PROPOSAL_CAPABILITIES_V2",
    "ALLOWED_FAMILY_PLUGIN_WARNING_REASONS_V2",
    "FAMILY_PLUGIN_WARNING_FALLBACK_ACTION_V2",
    "FamilyAccelerationPluginV2",
    "FamilyPluginApplicabilityV2",
    "FamilyPluginMetadataV2",
    "FamilyPluginPairCandidateV2",
    "FamilyPluginPlanningContextV2",
    "FamilyPluginProposalCapabilityLiteralV2",
    "FamilyPluginProposalResultV2",
    "FamilyPluginProxyScoreV2",
    "FamilyPluginSelectionKeyV2",
    "FamilyPluginWarningReasonLiteralV2",
    "FamilyPluginWarningV2",
    "build_family_plugin_planning_context_v2",
    "normalize_family_plugin_identifier_v2",
    "resolve_family_plugin_indicator_family_v2",
    "validate_family_plugin_proposal_capability_v2",
    "validate_family_plugin_warning_reason_v2",
]
