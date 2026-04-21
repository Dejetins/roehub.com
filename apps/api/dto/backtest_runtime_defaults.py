"""
Pydantic models and deterministic mapper for Backtest runtime defaults API endpoint.

Docs:
  - configs/prod/backtest.yaml
  - docs/architecture/backtest/README.md
  - docs/architecture/backtest/README.md
  - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from trading.contexts.backtest.adapters.outbound import BacktestRuntimeConfig
from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider


class BacktestRuntimeExecutionDefaultsResponse(BaseModel):
    """
    API response model for non-secret execution defaults used by `/backtests` web UI.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/dist/backtest_ui.js
    """

    model_config = ConfigDict(extra="forbid")

    init_cash_quote_default: float
    fixed_quote_default: float
    safe_profit_percent_default: float
    slippage_pct_default: float
    fee_pct_default_by_market_id: dict[str, float]


class BacktestRuntimeJobsDefaultsResponse(BaseModel):
    """
    API response model for jobs defaults required by browser-side validation hints.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/dist/backtest_ui.js
    """

    model_config = ConfigDict(extra="forbid")

    top_k_persisted_default: int


class BacktestRuntimeRankingDefaultsResponse(BaseModel):
    """
    API response model for ranking defaults used by browser-side ranking controls.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/dist/backtest_ui.js
    """

    model_config = ConfigDict(extra="forbid")

    primary_metric_default: str


class BacktestRuntimeRequestTimeframesContractResponse(BaseModel):
    """
    API response model for frozen R0 request-timeframe contract literals.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    allowed: list[str]
    forbidden: list[str]


class BacktestRuntimeSummaryContractResponse(BaseModel):
    """
    API response model for frozen R0 summary/ranking contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    top_n_default: int
    top_n_max: int
    ranking_metrics: list[str]
    sortable_columns: list[str]


class BacktestRuntimeSignalsContractResponse(BaseModel):
    """
    API response model for frozen R0 signal-params contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    params_path: str
    params_policy: str


class BacktestRuntimeAdaptiveSelectorCandidateResponse(BaseModel):
    """
    API response model for one candidate-specific adaptive-selector rollout policy.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    rollout_mode: str
    min_grid_cardinality: int
    min_stage_a_variants_total: int
    min_stage_b_variants_total: int
    min_estimated_memory_bytes: int
    minimum_exceeded_signals: int


class BacktestRuntimeAdaptiveSelectorResponse(BaseModel):
    """
    API response model for read-only adaptive-selector rollout status.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    mode: str
    hybrid_conservative: BacktestRuntimeAdaptiveSelectorCandidateResponse
    hybrid_family: BacktestRuntimeAdaptiveSelectorCandidateResponse


class BacktestRuntimeExecutionContractResponse(BaseModel):
    """
    API response model for frozen R0 execution semantics contract.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    risk_model: str
    default_execution_profile: str
    available_execution_profiles: list["BacktestRuntimeExecutionProfileResponse"]
    adaptive_selector: BacktestRuntimeAdaptiveSelectorResponse


class BacktestRuntimeExecutionProfileShortlistResponse(BaseModel):
    """
    API response model for typed execution-profile shortlist knobs.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    max_candidates: int | None = None
    scoring: "BacktestRuntimeExecutionProfileShortlistScoringResponse"
    retention: "BacktestRuntimeExecutionProfileShortlistRetentionResponse"


class BacktestRuntimeExecutionProfileShortlistScoringResponse(BaseModel):
    """
    API response model for generic-row shortlist scoring weights.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
    """

    model_config = ConfigDict(extra="forbid")

    activity_ratio_weight: float
    direction_balance_weight: float
    transition_ratio_weight: float
    active_span_ratio_weight: float


class BacktestRuntimeExecutionProfileShortlistRetentionResponse(BaseModel):
    """
    API response model for deterministic diversified-retention shortlist knobs.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
    """

    model_config = ConfigDict(extra="forbid")

    diversity_buckets: list[str]
    max_per_bucket: int | None = None


class BacktestRuntimeExecutionProfileParallelismResponse(BaseModel):
    """
    API response model for typed execution-profile parallelism knobs.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    stage_a_workers: int
    stage_b_workers: int


class BacktestRuntimeExecutionProfileFeatureFlagsResponse(BaseModel):
    """
    API response model for typed execution-profile feature flags.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    runtime_enabled: bool
    heuristic_shortlist_enabled: bool
    parallel_stage_b_enabled: bool
    family_plugin_enabled: bool


class BacktestRuntimeExecutionProfileLaunchBudgetResponse(BaseModel):
    """
    API response model for deterministic execution-profile launch-budget hints.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    max_stage_a_variants_total: int
    max_stage_b_variants_total: int
    max_estimated_memory_bytes: int


class BacktestRuntimeExecutionProfileProgressWeightsResponse(BaseModel):
    """
    API response model for deterministic progress/ETA stage weights per execution profile.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    """

    model_config = ConfigDict(extra="forbid")

    stage_a: int
    stage_b: int
    finalizing: int


class BacktestRuntimeExecutionProfileResponse(BaseModel):
    """
    API response model for one typed execution profile in runtime-defaults discovery payload.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    mode: str
    shortlist_config: BacktestRuntimeExecutionProfileShortlistResponse
    parallelism: BacktestRuntimeExecutionProfileParallelismResponse
    feature_flags: BacktestRuntimeExecutionProfileFeatureFlagsResponse
    launch_budget: BacktestRuntimeExecutionProfileLaunchBudgetResponse
    progress_weights: BacktestRuntimeExecutionProfileProgressWeightsResponse
    family_plugin_budget_ms: int
    planning_budget_ms: int


class BacktestRuntimeLaunchContractResponse(BaseModel):
    """
    API response model for frozen R0 launch/execution-mode contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    execution_mode: str
    auto_preflight_enabled: bool
    auto_fallback_to_background_enabled: bool
    supported_indicator_ids: list[str]
    source_values_by_indicator_id: dict[str, list[str]]


class BacktestRuntimeContractsResponse(BaseModel):
    """
    API response model for additive frozen R0 contract surface.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    model_config = ConfigDict(extra="forbid")

    request_timeframes: BacktestRuntimeRequestTimeframesContractResponse
    summary: BacktestRuntimeSummaryContractResponse
    signals: BacktestRuntimeSignalsContractResponse
    execution: BacktestRuntimeExecutionContractResponse
    launch: BacktestRuntimeLaunchContractResponse


class BacktestRuntimeDefaultsResponse(BaseModel):
    """
    API response model for deterministic runtime defaults contract.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/web/templates/backtests.html
    """

    model_config = ConfigDict(extra="forbid")

    top_k_default: int
    preselect_default: int
    ranking: BacktestRuntimeRankingDefaultsResponse
    execution: BacktestRuntimeExecutionDefaultsResponse
    jobs: BacktestRuntimeJobsDefaultsResponse
    contracts: BacktestRuntimeContractsResponse


BacktestRuntimeExecutionProfileShortlistResponse.model_rebuild()
BacktestRuntimeExecutionContractResponse.model_rebuild()
BacktestRuntimeContractsResponse.model_rebuild()
BacktestRuntimeDefaultsResponse.model_rebuild()


def build_backtest_runtime_defaults_response(
    *,
    config: BacktestRuntimeConfig,
    defaults_provider: BacktestGridDefaultsProvider | None = None,
) -> BacktestRuntimeDefaultsResponse:
    """
    Convert loaded runtime config into deterministic non-secret browser defaults payload.

    Docs:
      - configs/prod/backtest.yaml
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - apps/api/dto/backtest_runtime_defaults.py
      - apps/api/routes/backtests.py
      - apps/api/wiring/modules/backtest.py

    Args:
        config: Parsed startup-validated runtime config.
        defaults_provider: Optional startup-validated indicators defaults provider/catalog.
    Returns:
        BacktestRuntimeDefaultsResponse: Deterministic response DTO for
            `/backtests/runtime-defaults`.
    Assumptions:
        Runtime config already passed fail-fast startup validation.
    Raises:
        None.
    Side Effects:
        None.
    """
    fee_defaults = {
        str(market_id): config.execution.fee_pct_default_by_market_id[market_id]
        for market_id in sorted(config.execution.fee_pct_default_by_market_id.keys())
    }
    supported_indicator_ids = (
        list(defaults_provider.supported_indicator_ids()) if defaults_provider is not None else []
    )
    source_values_by_indicator_id = (
        {
            indicator_id: list(defaults_provider.allowed_source_values(indicator_id=indicator_id))
            for indicator_id in supported_indicator_ids
        }
        if defaults_provider is not None
        else {}
    )
    available_execution_profiles = [
        BacktestRuntimeExecutionProfileResponse(
            mode=profile.mode,
            shortlist_config=BacktestRuntimeExecutionProfileShortlistResponse(
                enabled=profile.shortlist_config.enabled,
                max_candidates=profile.shortlist_config.max_candidates,
                scoring=BacktestRuntimeExecutionProfileShortlistScoringResponse(
                    activity_ratio_weight=(
                        profile.shortlist_config.scoring.activity_ratio_weight
                    ),
                    direction_balance_weight=(
                        profile.shortlist_config.scoring.direction_balance_weight
                    ),
                    transition_ratio_weight=(
                        profile.shortlist_config.scoring.transition_ratio_weight
                    ),
                    active_span_ratio_weight=(
                        profile.shortlist_config.scoring.active_span_ratio_weight
                    ),
                ),
                retention=BacktestRuntimeExecutionProfileShortlistRetentionResponse(
                    diversity_buckets=list(
                        profile.shortlist_config.retention.diversity_buckets
                    ),
                    max_per_bucket=profile.shortlist_config.retention.max_per_bucket,
                ),
            ),
            parallelism=BacktestRuntimeExecutionProfileParallelismResponse(
                stage_a_workers=profile.parallelism.stage_a_workers,
                stage_b_workers=profile.parallelism.stage_b_workers,
            ),
            feature_flags=BacktestRuntimeExecutionProfileFeatureFlagsResponse(
                runtime_enabled=profile.feature_flags.runtime_enabled,
                heuristic_shortlist_enabled=(
                    profile.feature_flags.heuristic_shortlist_enabled
                ),
                parallel_stage_b_enabled=profile.feature_flags.parallel_stage_b_enabled,
                family_plugin_enabled=profile.feature_flags.family_plugin_enabled,
            ),
            launch_budget=BacktestRuntimeExecutionProfileLaunchBudgetResponse(
                max_stage_a_variants_total=(
                    profile.launch_budget.max_stage_a_variants_total
                ),
                max_stage_b_variants_total=(
                    profile.launch_budget.max_stage_b_variants_total
                ),
                max_estimated_memory_bytes=(
                    profile.launch_budget.max_estimated_memory_bytes
                ),
            ),
            progress_weights=BacktestRuntimeExecutionProfileProgressWeightsResponse(
                stage_a=profile.progress_weights.stage_a,
                stage_b=profile.progress_weights.stage_b,
                finalizing=profile.progress_weights.finalizing,
            ),
            family_plugin_budget_ms=profile.family_plugin_budget_ms,
            planning_budget_ms=profile.planning_budget_ms,
        )
        for profile in config.execution_profiles.available_profiles
    ]
    return BacktestRuntimeDefaultsResponse(
        top_k_default=config.top_k_default,
        preselect_default=config.preselect_default,
        ranking=BacktestRuntimeRankingDefaultsResponse(
            primary_metric_default=config.ranking.primary_metric_default,
        ),
        execution=BacktestRuntimeExecutionDefaultsResponse(
            init_cash_quote_default=config.execution.init_cash_quote_default,
            fixed_quote_default=config.execution.fixed_quote_default,
            safe_profit_percent_default=config.execution.safe_profit_percent_default,
            slippage_pct_default=config.execution.slippage_pct_default,
            fee_pct_default_by_market_id=fee_defaults,
        ),
        jobs=BacktestRuntimeJobsDefaultsResponse(
            top_k_persisted_default=config.jobs.top_k_persisted_default,
        ),
        contracts=BacktestRuntimeContractsResponse(
            request_timeframes=BacktestRuntimeRequestTimeframesContractResponse(
                allowed=list(config.contracts.allowed_request_timeframes),
                forbidden=list(config.contracts.forbidden_request_timeframes),
            ),
            summary=BacktestRuntimeSummaryContractResponse(
                top_n_default=config.contracts.top_n_default,
                top_n_max=config.contracts.top_n_max,
                ranking_metrics=list(config.contracts.ranking_metrics),
                sortable_columns=list(config.contracts.sortable_summary_columns),
            ),
            signals=BacktestRuntimeSignalsContractResponse(
                params_path=config.contracts.signals_v1_params_path,
                params_policy=config.contracts.signals_v1_params_policy,
            ),
            execution=BacktestRuntimeExecutionContractResponse(
                risk_model=config.contracts.risk_model,
                default_execution_profile=config.execution_profiles.default_mode,
                available_execution_profiles=available_execution_profiles,
                adaptive_selector=BacktestRuntimeAdaptiveSelectorResponse(
                    mode=config.adaptive_selector_policy.mode,
                    hybrid_conservative=BacktestRuntimeAdaptiveSelectorCandidateResponse(
                        rollout_mode=(
                            config.adaptive_selector_policy.hybrid_conservative.rollout_mode
                        ),
                        min_grid_cardinality=(
                            config.adaptive_selector_policy.hybrid_conservative.min_grid_cardinality
                        ),
                        min_stage_a_variants_total=(
                            config.adaptive_selector_policy.hybrid_conservative
                            .min_stage_a_variants_total
                        ),
                        min_stage_b_variants_total=(
                            config.adaptive_selector_policy.hybrid_conservative
                            .min_stage_b_variants_total
                        ),
                        min_estimated_memory_bytes=(
                            config.adaptive_selector_policy.hybrid_conservative
                            .min_estimated_memory_bytes
                        ),
                        minimum_exceeded_signals=(
                            config.adaptive_selector_policy.hybrid_conservative
                            .minimum_exceeded_signals
                        ),
                    ),
                    hybrid_family=BacktestRuntimeAdaptiveSelectorCandidateResponse(
                        rollout_mode=config.adaptive_selector_policy.hybrid_family.rollout_mode,
                        min_grid_cardinality=(
                            config.adaptive_selector_policy.hybrid_family.min_grid_cardinality
                        ),
                        min_stage_a_variants_total=(
                            config.adaptive_selector_policy.hybrid_family
                            .min_stage_a_variants_total
                        ),
                        min_stage_b_variants_total=(
                            config.adaptive_selector_policy.hybrid_family
                            .min_stage_b_variants_total
                        ),
                        min_estimated_memory_bytes=(
                            config.adaptive_selector_policy.hybrid_family
                            .min_estimated_memory_bytes
                        ),
                        minimum_exceeded_signals=(
                            config.adaptive_selector_policy.hybrid_family
                            .minimum_exceeded_signals
                        ),
                    ),
                ),
            ),
            launch=BacktestRuntimeLaunchContractResponse(
                execution_mode=config.contracts.execution_mode,
                auto_preflight_enabled=config.contracts.auto_preflight_enabled,
                auto_fallback_to_background_enabled=(
                    config.contracts.auto_fallback_to_background_enabled
                ),
                supported_indicator_ids=supported_indicator_ids,
                source_values_by_indicator_id=source_values_by_indicator_id,
            ),
        ),
    )


__all__ = [
    "BacktestRuntimeAdaptiveSelectorCandidateResponse",
    "BacktestRuntimeAdaptiveSelectorResponse",
    "BacktestRuntimeContractsResponse",
    "BacktestRuntimeDefaultsResponse",
    "BacktestRuntimeExecutionDefaultsResponse",
    "BacktestRuntimeExecutionContractResponse",
    "BacktestRuntimeExecutionProfileFeatureFlagsResponse",
    "BacktestRuntimeExecutionProfileLaunchBudgetResponse",
    "BacktestRuntimeExecutionProfileProgressWeightsResponse",
    "BacktestRuntimeExecutionProfileParallelismResponse",
    "BacktestRuntimeExecutionProfileResponse",
    "BacktestRuntimeExecutionProfileShortlistRetentionResponse",
    "BacktestRuntimeExecutionProfileShortlistScoringResponse",
    "BacktestRuntimeExecutionProfileShortlistResponse",
    "BacktestRuntimeJobsDefaultsResponse",
    "BacktestRuntimeLaunchContractResponse",
    "BacktestRuntimeRankingDefaultsResponse",
    "BacktestRuntimeRequestTimeframesContractResponse",
    "BacktestRuntimeSignalsContractResponse",
    "BacktestRuntimeSummaryContractResponse",
    "build_backtest_runtime_defaults_response",
]
