from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.domain.entities import (
    BacktestArtifactSlotLiteral,
    BacktestJobExecutionMode,
    BacktestJobStageANoRiskExactRow,
    BacktestJobState,
    TradeV1,
)
from trading.contexts.backtest.domain.entities.backtest_job_results import (
    BacktestJobParityRuntimeState,
    BacktestJobStageAShortlist,
)
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec
from trading.shared_kernel.primitives import InstrumentId, Timeframe, TimeRange

BacktestRequestScalar = int | float | str | bool | None
BacktestSignalGridMap = Mapping[str, Mapping[str, GridParamSpec]]
BacktestSignalScalarMap = Mapping[str, Mapping[str, BacktestRequestScalar]]
_ALLOWED_DIRECTION_MODES = {"long-only", "short-only", "long-short"}
_ALLOWED_SIZING_MODES = {
    "all_in",
    "fixed_quote",
    "strategy_compound",
    "strategy_compound_profit_lock",
}
BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1 = "total_return_pct"
BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1: str | None = None
BACKTEST_RANKING_METRIC_LITERALS_V1: tuple[str, ...] = (
    "total_return_pct",
    "max_drawdown_pct",
    "return_over_max_drawdown",
    "profit_factor",
    "sharpe_trades",
    "win_rate_pct",
)
_BACKTEST_ALLOWED_RANKING_METRICS_V1 = frozenset(BACKTEST_RANKING_METRIC_LITERALS_V1)


def normalize_backtest_ranking_metric_literal(*, metric: str, field_path: str) -> str:
    """
    Normalize and validate one ranking metric literal against v1 allowed contract.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

    Args:
        metric: Raw ranking metric literal.
        field_path: Field path used in deterministic validation error messages.
    Returns:
        str: Lowercase normalized metric literal.
    Assumptions:
        Ranking identifiers are lowercase snake_case literals from fixed v1 list.
    Raises:
        ValueError: If value is not a supported metric literal.
    Side Effects:
        None.
    """
    normalized_metric = metric.strip().lower()
    if not normalized_metric:
        raise ValueError(f"{field_path} must be non-empty")
    if normalized_metric not in _BACKTEST_ALLOWED_RANKING_METRICS_V1:
        allowed_values = ", ".join(sorted(_BACKTEST_ALLOWED_RANKING_METRICS_V1))
        raise ValueError(f"{field_path} must be one of: {allowed_values}")
    return normalized_metric


@dataclass(frozen=True, slots=True)
class BacktestRankingConfig:
    """
    Ranking override payload for request/runtime contracts with deterministic normalization.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
    """

    primary_metric: str = BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1
    secondary_metric: str | None = BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1

    def __post_init__(self) -> None:
        """
        Validate ranking metric literals and enforce deterministic tie-break contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Secondary metric is optional and cannot duplicate primary metric.
        Raises:
            ValueError: If one metric literal is invalid or duplicates primary metric.
        Side Effects:
            Normalizes ranking literals to lowercase snake_case.
        """
        normalized_primary_metric = normalize_backtest_ranking_metric_literal(
            metric=self.primary_metric,
            field_path="ranking.primary_metric",
        )
        object.__setattr__(self, "primary_metric", normalized_primary_metric)

        if self.secondary_metric is None:
            return

        normalized_secondary_metric = normalize_backtest_ranking_metric_literal(
            metric=self.secondary_metric,
            field_path="ranking.secondary_metric",
        )
        if normalized_secondary_metric == normalized_primary_metric:
            raise ValueError(
                "ranking.secondary_metric must be different from ranking.primary_metric"
            )
        object.__setattr__(self, "secondary_metric", normalized_secondary_metric)


@dataclass(frozen=True, slots=True)
class BacktestRiskGridSpec:
    """
    Stage B risk axes specification with explicit enable flags and percent semantics.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
    """

    sl_enabled: bool = False
    tp_enabled: bool = False
    sl: GridParamSpec | None = None
    tp: GridParamSpec | None = None

    def __post_init__(self) -> None:
        """
        Validate risk-grid semantic invariants for Stage B expansion.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            SL/TP values are percentages where `3.0 == 3%`.
        Raises:
            ValueError: If enabled axis does not provide a materializable specification.
        Side Effects:
            None.
        """
        if self.sl_enabled and self.sl is None:
            raise ValueError("BacktestRiskGridSpec.sl must be provided when sl_enabled is true")
        if self.tp_enabled and self.tp is None:
            raise ValueError("BacktestRiskGridSpec.tp must be provided when tp_enabled is true")
        if self.sl is not None and len(self.sl.materialize()) == 0:
            raise ValueError("BacktestRiskGridSpec.sl materialized to empty values")
        if self.tp is not None and len(self.tp.materialize()) == 0:
            raise ValueError("BacktestRiskGridSpec.tp materialized to empty values")


@dataclass(frozen=True, slots=True)
class RunBacktestTemplate:
    """
    Ad-hoc backtest template payload (instrument/timeframe/indicator-grid contract).

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
    """

    instrument_id: InstrumentId
    timeframe: Timeframe
    indicator_grids: tuple[GridSpec, ...]
    indicator_selections: tuple[IndicatorVariantSelection, ...] = ()
    signal_grids: BacktestSignalGridMap | None = None
    risk_grid: BacktestRiskGridSpec | None = None
    direction_mode: str = "long-short"
    sizing_mode: str = "all_in"
    risk_params: Mapping[str, BacktestRequestScalar] | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None

    def __post_init__(self) -> None:
        """
        Validate and normalize ad-hoc template invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            EPIC-01 keeps grid/run settings minimal and defers staged execution details.
        Raises:
            ValueError: If required fields are missing or mode literals are unsupported.
        Side Effects:
            Normalizes mode literals and freezes mapping payloads into immutable mapping proxies.
        """
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestTemplate.instrument_id is required")
        if self.timeframe is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestTemplate.timeframe is required")
        if len(self.indicator_grids) == 0:
            raise ValueError("RunBacktestTemplate.indicator_grids must be non-empty")

        normalized_direction_mode = self.direction_mode.strip().lower()
        object.__setattr__(self, "direction_mode", normalized_direction_mode)
        if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
            raise ValueError(
                "RunBacktestTemplate.direction_mode must be one of: "
                f"{sorted(_ALLOWED_DIRECTION_MODES)}"
            )

        normalized_sizing_mode = self.sizing_mode.strip().lower()
        object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
        if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
            raise ValueError(
                "RunBacktestTemplate.sizing_mode must be one of: "
                f"{sorted(_ALLOWED_SIZING_MODES)}"
            )

        object.__setattr__(
            self,
            "risk_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.risk_params)),
        )
        object.__setattr__(
            self,
            "signal_grids",
            MappingProxyType(_normalize_signal_grid_mapping(values=self.signal_grids)),
        )
        resolved_risk_grid = self.risk_grid
        if resolved_risk_grid is None:
            resolved_risk_grid = BacktestRiskGridSpec()
        object.__setattr__(self, "risk_grid", resolved_risk_grid)
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
        )


@dataclass(frozen=True, slots=True)
class RunBacktestSavedOverrides:
    """
    Optional saved-mode override payload applied over loaded strategy snapshot template.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - apps/api/dto/backtests.py
    """

    direction_mode: str | None = None
    sizing_mode: str | None = None
    signal_grids: BacktestSignalGridMap | None = None
    risk_grid: BacktestRiskGridSpec | None = None
    risk_params: Mapping[str, BacktestRequestScalar] | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None

    def __post_init__(self) -> None:
        """
        Validate optional saved-mode overrides and normalize nested payload mappings.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Missing fields keep loaded saved-strategy values unchanged.
        Raises:
            ValueError: If provided mode literal is unsupported.
        Side Effects:
            Freezes mapping fields into deterministic immutable mapping proxies.
        """
        if self.direction_mode is not None:
            normalized_direction_mode = self.direction_mode.strip().lower()
            object.__setattr__(self, "direction_mode", normalized_direction_mode)
            if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
                raise ValueError(
                    "RunBacktestSavedOverrides.direction_mode must be one of: "
                    f"{sorted(_ALLOWED_DIRECTION_MODES)}"
                )

        if self.sizing_mode is not None:
            normalized_sizing_mode = self.sizing_mode.strip().lower()
            object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
            if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
                raise ValueError(
                    "RunBacktestSavedOverrides.sizing_mode must be one of: "
                    f"{sorted(_ALLOWED_SIZING_MODES)}"
                )

        object.__setattr__(
            self,
            "signal_grids",
            MappingProxyType(_normalize_signal_grid_mapping(values=self.signal_grids)),
        )
        object.__setattr__(
            self,
            "risk_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.risk_params)),
        )
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
        )


@dataclass(frozen=True, slots=True)
class RunBacktestRequest:
    """
    Backtest use-case request supporting both `saved` and `template` modes.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/current_user.py
      - apps/api/routes
    """

    time_range: TimeRange
    strategy_id: UUID | None = None
    template: RunBacktestTemplate | None = None
    overrides: RunBacktestSavedOverrides | None = None
    warmup_bars: int | None = None
    top_k: int | None = None
    preselect: int | None = None
    ranking: BacktestRankingConfig | None = None

    def __post_init__(self) -> None:
        """
        Validate request-mode exclusivity and scalar override invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Exactly one mode is selected: `strategy_id` (saved) xor `template` (ad-hoc).
        Raises:
            ValueError: If mode selection or override numbers violate v1 contract.
        Side Effects:
            None.
        """
        if self.time_range is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestRequest.time_range is required")

        has_saved_mode = self.strategy_id is not None
        has_template_mode = self.template is not None
        if has_saved_mode == has_template_mode:
            raise ValueError(
                "RunBacktestRequest requires exactly one mode: strategy_id xor template"
            )
        if self.overrides is not None and not has_saved_mode:
            raise ValueError(
                "RunBacktestRequest.overrides is allowed only in saved mode"
            )

        _validate_positive_optional_int(name="warmup_bars", value=self.warmup_bars)
        _validate_positive_optional_int(name="top_k", value=self.top_k)
        _validate_positive_optional_int(name="preselect", value=self.preselect)

    @property
    def mode(self) -> str:
        """
        Return normalized request mode literal.

        Args:
            None.
        Returns:
            str: `saved` when `strategy_id` is used, otherwise `template`.
        Assumptions:
            Mode exclusivity has been validated during object initialization.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.strategy_id is not None:
            return "saved"
        return "template"


@dataclass(frozen=True, slots=True)
class BacktestMetricRowV1:
    """
    One deterministic reporting row rendered in `|Metric|Value|` table contract.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/table_formatter_v1.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    metric: str
    value: str

    def __post_init__(self) -> None:
        """
        Validate one reporting metric row payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Value string is already deterministically formatted by reporting formatter.
        Raises:
            ValueError: If metric name or formatted value is empty.
        Side Effects:
            Normalizes metric/value literals by stripping surrounding whitespace.
        """
        normalized_metric = self.metric.strip()
        object.__setattr__(self, "metric", normalized_metric)
        if not normalized_metric:
            raise ValueError("BacktestMetricRowV1.metric must be non-empty")

        normalized_value = self.value.strip()
        object.__setattr__(self, "value", normalized_value)
        if not normalized_value:
            raise ValueError("BacktestMetricRowV1.value must be non-empty")


@dataclass(frozen=True, slots=True)
class BacktestReportV1:
    """
    Deterministic reporting payload with metric rows, markdown table, and optional trades.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    rows: tuple[BacktestMetricRowV1, ...]
    table_md: str | None = None
    trades: tuple[TradeV1, ...] | None = None

    def __post_init__(self) -> None:
        """
        Validate deterministic report payload shape and markdown-table contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Rows are provided in fixed contract order by reporting formatter.
        Raises:
            ValueError: If rows are empty, table header is invalid, or trades are not sorted.
        Side Effects:
            None.
        """
        if len(self.rows) == 0:
            raise ValueError("BacktestReportV1.rows must be non-empty")

        if self.table_md is not None and not self.table_md.startswith("|Metric|Value|"):
            raise ValueError("BacktestReportV1.table_md must start with '|Metric|Value|'")

        if self.trades is not None:
            previous_trade_id = 0
            for trade in self.trades:
                if trade.trade_id < previous_trade_id:
                    raise ValueError("BacktestReportV1.trades must be ordered by trade_id asc")
                previous_trade_id = trade.trade_id


@dataclass(frozen=True, slots=True)
class BacktestVariantPayloadV1:
    """
    Explicit deterministic variant payload required for saveable API response contract.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
    """

    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_params: BacktestSignalScalarMap | None = None
    risk_params: Mapping[str, BacktestRequestScalar] | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None
    direction_mode: str = "long-short"
    sizing_mode: str = "all_in"

    def __post_init__(self) -> None:
        """
        Validate payload fields and freeze nested mappings into deterministic forms.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Payload is assembled from Stage-B deterministic variant task data.
        Raises:
            ValueError: If one mode literal is unsupported.
        Side Effects:
            Normalizes mode literals and replaces mapping payloads with immutable proxies.
        """
        if len(self.indicator_selections) == 0:
            raise ValueError("BacktestVariantPayloadV1.indicator_selections must be non-empty")
        if not _is_pre_sorted_indicator_selections(indicator_selections=self.indicator_selections):
            object.__setattr__(
                self,
                "indicator_selections",
                tuple(
                    sorted(
                        self.indicator_selections,
                        key=lambda item: item.indicator_id,
                    )
                ),
            )

        normalized_direction_mode = self.direction_mode.strip().lower()
        object.__setattr__(self, "direction_mode", normalized_direction_mode)
        if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
            raise ValueError(
                "BacktestVariantPayloadV1.direction_mode must be one of: "
                f"{sorted(_ALLOWED_DIRECTION_MODES)}"
            )

        normalized_sizing_mode = self.sizing_mode.strip().lower()
        object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
        if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
            raise ValueError(
                "BacktestVariantPayloadV1.sizing_mode must be one of: "
                f"{sorted(_ALLOWED_SIZING_MODES)}"
            )

        object.__setattr__(
            self,
            "signal_params",
            MappingProxyType(_normalize_nested_scalar_mapping(values=self.signal_params)),
        )
        object.__setattr__(
            self,
            "risk_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.risk_params)),
        )
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
        )


@dataclass(frozen=True, slots=True)
class BacktestVariantPreview:
    """
    One deterministic variant preview identity returned by skeleton use-case.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/routes
    """

    variant_index: int
    variant_key: str
    indicator_variant_key: str
    total_return_pct: float = 0.0
    payload: BacktestVariantPayloadV1 | None = None
    report: BacktestReportV1 | None = None
    summary_metrics_json: Mapping[str, float] = field(default_factory=dict)
    best_tp_pct: float | None = None
    best_sl_pct: float | None = None

    def __post_init__(self) -> None:
        """
        Validate deterministic variant identity payload shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Keys are lowercase hex SHA-256 strings produced by canonical builders.
        Raises:
            ValueError: If index/key invariants are violated.
        Side Effects:
            Normalizes string keys to lowercase stripped format.
        """
        if self.variant_index < 0:
            raise ValueError("BacktestVariantPreview.variant_index must be >= 0")

        normalized_variant_key = self.variant_key.strip().lower()
        object.__setattr__(self, "variant_key", normalized_variant_key)
        if len(normalized_variant_key) != 64:
            raise ValueError("BacktestVariantPreview.variant_key must be 64 hex chars")

        normalized_indicator_key = self.indicator_variant_key.strip().lower()
        object.__setattr__(self, "indicator_variant_key", normalized_indicator_key)
        if len(normalized_indicator_key) != 64:
            raise ValueError("BacktestVariantPreview.indicator_variant_key must be 64 hex chars")

        if (
            isinstance(self.total_return_pct, bool)
            or not isinstance(self.total_return_pct, int | float)
        ):
            raise ValueError("BacktestVariantPreview.total_return_pct must be numeric")
        object.__setattr__(self, "total_return_pct", float(self.total_return_pct))

        if self.payload is None:  # pragma: no cover - guarded by staged runner payload assembly
            raise ValueError("BacktestVariantPreview.payload is required")
        resolved_payload = self.payload

        object.__setattr__(
            self,
            "summary_metrics_json",
            MappingProxyType(
                _normalize_summary_metrics_mapping(
                    values=self.summary_metrics_json,
                    total_return_pct=float(self.total_return_pct),
                )
            ),
        )
        object.__setattr__(
            self,
            "best_tp_pct",
            _resolve_best_risk_pct(
                explicit_value=self.best_tp_pct,
                payload=resolved_payload,
                flag_key="tp_enabled",
                value_key="tp_pct",
            ),
        )
        object.__setattr__(
            self,
            "best_sl_pct",
            _resolve_best_risk_pct(
                explicit_value=self.best_sl_pct,
                payload=resolved_payload,
                flag_key="sl_enabled",
                value_key="sl_pct",
            ),
        )


@dataclass(frozen=True, slots=True)
class RunBacktestSyncPersistenceArtifact:
    """
    Internal-only sync persistence payload carrying live Stage A artifacts across `/backtests`.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
    """

    stage_a_indexes: tuple[int, ...]
    stage_a_variants_total: int
    risk_total: int
    preselect_used: int
    no_risk_exact_rows: tuple[BacktestJobStageANoRiskExactRow, ...] | None = None
    parity_runtime_state: BacktestJobParityRuntimeState | None = None

    def to_stage_a_shortlist(
        self,
        *,
        job_id: UUID,
        updated_at: Any,
    ) -> BacktestJobStageAShortlist:
        """
        Materialize the shared Stage A shortlist entity for atomic sync persistence.

        Args:
            job_id: Persisted terminal sync run identifier.
            updated_at: Terminal sync persistence timestamp in UTC.
        Returns:
            BacktestJobStageAShortlist: Shared worker-compatible shortlist snapshot entity.
        Assumptions:
            The live artifact is internal-only and must remain excluded from public API transport.
        Raises:
            ValueError: Propagated from `BacktestJobStageAShortlist` invariant validation.
        Side Effects:
            None.
        """
        return BacktestJobStageAShortlist(
            job_id=job_id,
            stage_a_indexes=self.stage_a_indexes,
            stage_a_variants_total=self.stage_a_variants_total,
            risk_total=self.risk_total,
            preselect_used=self.preselect_used,
            updated_at=updated_at,
            no_risk_exact_rows=self.no_risk_exact_rows,
            parity_runtime_state=self.parity_runtime_state,
        )


@dataclass(frozen=True, slots=True)
class RunBacktestResponse:
    """
    Backtest use-case response skeleton for BKT-EPIC-01.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/entities/backtest_placeholders.py
      - apps/api/routes
    """

    mode: str
    instrument_id: InstrumentId
    timeframe: Timeframe
    strategy_id: UUID | None
    top_k: int
    preselect: int
    variants: tuple[BacktestVariantPreview, ...]
    total_indicator_compute_calls: int
    direction_mode: str | None = None
    sizing_mode: str | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None
    run_id: UUID | None = None
    state: BacktestJobState | None = None
    execution_mode: BacktestJobExecutionMode | None = None
    execution_profile_mode: str | None = None
    engine_version: str | None = None
    artifact_slot: BacktestArtifactSlotLiteral | None = None
    artifact_slot_generation: int | None = None
    artifact_asof_date: str | None = None
    artifact_manifest_hash: str | None = None
    spec_hash: str | None = None
    spec_payload_json: Mapping[str, Any] | None = None
    engine_params_hash: str | None = None
    sync_persistence_artifact: RunBacktestSyncPersistenceArtifact | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate response-level deterministic ordering and scalar invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variants are emitted in deterministic order and indexes are unique.
        Raises:
            ValueError: If mode is unknown, scalar bounds are invalid, or variant ordering breaks.
        Side Effects:
            Normalizes mode literal to lowercase stripped representation.
        """
        normalized_mode = self.mode.strip().lower()
        object.__setattr__(self, "mode", normalized_mode)
        if normalized_mode not in {"saved", "template"}:
            raise ValueError("RunBacktestResponse.mode must be saved or template")

        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestResponse.instrument_id is required")
        if self.timeframe is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestResponse.timeframe is required")

        if self.top_k <= 0:
            raise ValueError("RunBacktestResponse.top_k must be > 0")
        if self.preselect <= 0:
            raise ValueError("RunBacktestResponse.preselect must be > 0")
        if self.total_indicator_compute_calls < 0:
            raise ValueError("RunBacktestResponse.total_indicator_compute_calls must be >= 0")

        if self.direction_mode is not None:
            normalized_direction_mode = self.direction_mode.strip().lower()
            if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
                raise ValueError(
                    "RunBacktestResponse.direction_mode must be one of: "
                    f"{sorted(_ALLOWED_DIRECTION_MODES)}"
                )
            object.__setattr__(self, "direction_mode", normalized_direction_mode)
        if self.sizing_mode is not None:
            normalized_sizing_mode = self.sizing_mode.strip().lower()
            if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
                raise ValueError(
                    "RunBacktestResponse.sizing_mode must be one of: "
                    f"{sorted(_ALLOWED_SIZING_MODES)}"
                )
            object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
        if self.execution_params is not None:
            object.__setattr__(
                self,
                "execution_params",
                MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
            )
        if self.execution_profile_mode is not None:
            from trading.contexts.backtest.application.services.v2.execution_profile_v2 import (
                validate_execution_profile_mode_v2,
            )

            object.__setattr__(
                self,
                "execution_profile_mode",
                validate_execution_profile_mode_v2(value=self.execution_profile_mode),
            )

        variant_indexes = tuple(item.variant_index for item in self.variants)
        if len(set(variant_indexes)) != len(variant_indexes):
            raise ValueError("RunBacktestResponse variants must contain unique variant_index")

        previous_variant: BacktestVariantPreview | None = None
        for current in self.variants:
            if previous_variant is None:
                previous_variant = current
                continue
            if current.total_return_pct > previous_variant.total_return_pct:
                raise ValueError(
                    "RunBacktestResponse variants must be sorted by total_return_pct desc"
                )
            if (
                current.total_return_pct == previous_variant.total_return_pct
                and current.variant_key < previous_variant.variant_key
            ):
                raise ValueError(
                    "RunBacktestResponse variants with equal total_return_pct must be sorted "
                    "by variant_key asc"
                )
            previous_variant = current

        has_artifact_metadata = any(
            item is not None
            for item in (
                self.artifact_slot,
                self.artifact_slot_generation,
                self.artifact_asof_date,
                self.artifact_manifest_hash,
            )
        )
        if has_artifact_metadata:
            if (
                self.artifact_slot is None
                or self.artifact_slot_generation is None
                or self.artifact_asof_date is None
                or self.artifact_manifest_hash is None
            ):
                raise ValueError(
                    "RunBacktestResponse artifact metadata must be fully populated"
                )
            if self.artifact_slot_generation <= 0:
                raise ValueError(
                    "RunBacktestResponse.artifact_slot_generation must be > 0"
                )

        has_persisted_run_metadata = any(
            item is not None
            for item in (
                self.run_id,
                self.state,
                self.execution_mode,
                self.engine_version,
            )
        )
        if has_persisted_run_metadata:
            if (
                self.run_id is None
                or self.state is None
                or self.execution_mode is None
                or self.engine_version is None
                or not has_artifact_metadata
            ):
                raise ValueError(
                    "RunBacktestResponse persisted run metadata must be fully populated"
                )
            normalized_engine_version = self.engine_version.strip()
            if not normalized_engine_version:
                raise ValueError("RunBacktestResponse.engine_version must be non-empty")
            object.__setattr__(self, "engine_version", normalized_engine_version)

        if self.spec_hash is not None and len(self.spec_hash.strip()) != 64:
            raise ValueError("RunBacktestResponse.spec_hash must be 64 hex chars when provided")
        if self.engine_params_hash is not None and len(self.engine_params_hash.strip()) != 64:
            raise ValueError(
                "RunBacktestResponse.engine_params_hash must be 64 hex chars when provided"
            )
        if self.spec_payload_json is not None:
            object.__setattr__(
                self,
                "spec_payload_json",
                MappingProxyType(_normalize_json_payload_mapping(values=self.spec_payload_json)),
            )
        if self.sync_persistence_artifact is not None and not isinstance(
            self.sync_persistence_artifact,
            RunBacktestSyncPersistenceArtifact,
        ):
            raise ValueError(
                "RunBacktestResponse.sync_persistence_artifact must be "
                "RunBacktestSyncPersistenceArtifact"
            )


def _validate_positive_optional_int(*, name: str, value: int | None) -> None:
    """
    Validate optional positive integer scalar used for request override fields.

    Args:
        name: Field name used in deterministic error message.
        value: Optional integer value.
    Returns:
        None.
    Assumptions:
        `None` means fallback to runtime config default.
    Raises:
        ValueError: If provided value is non-positive.
    Side Effects:
        None.
    """
    if value is not None and value <= 0:
        raise ValueError(f"RunBacktestRequest.{name} must be > 0 when provided")


def _normalize_scalar_mapping(
    *,
    values: Mapping[str, BacktestRequestScalar] | None,
) -> dict[str, BacktestRequestScalar]:
    """
    Normalize optional scalar mapping into deterministic key-sorted plain dict.

    Args:
        values: Optional scalar mapping.
    Returns:
        dict[str, BacktestRequestScalar]: Deterministic mapping.
    Assumptions:
        Values are JSON-compatible scalars.
    Raises:
        ValueError: If one of keys is blank.
    Side Effects:
        None.
    """
    if values is None:
        return {}
    if _is_pre_normalized_scalar_mapping(values=values):
        return {key: values[key] for key in values.keys()}

    normalized: dict[str, BacktestRequestScalar] = {}
    for key in sorted(values.keys()):
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("RunBacktestTemplate mapping keys must be non-empty")
        normalized[normalized_key] = values[key]
    return normalized


def _normalize_signal_grid_mapping(
    *,
    values: BacktestSignalGridMap | None,
) -> dict[str, Mapping[str, GridParamSpec]]:
    """
    Normalize nested signal-grid mapping with deterministic lowercase key ordering.

    Args:
        values: Optional `indicator_id -> signal_param_name -> GridParamSpec` mapping.
    Returns:
        dict[str, Mapping[str, GridParamSpec]]: Deterministic normalized nested mapping.
    Assumptions:
        Every `GridParamSpec` materializes non-empty value sequence.
    Raises:
        ValueError: If one indicator or signal parameter key is blank.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, Mapping[str, GridParamSpec]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("RunBacktestTemplate.signal_grids indicator_id keys must be non-empty")
        signal_axes = values[raw_indicator_id]
        signal_axis_map: dict[str, GridParamSpec] = {}
        for raw_param_name in sorted(signal_axes.keys(), key=lambda key: str(key).strip().lower()):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError(
                    "RunBacktestTemplate.signal_grids param keys must be non-empty"
                )
            if len(signal_axes[raw_param_name].materialize()) == 0:
                raise ValueError(
                    "RunBacktestTemplate.signal_grids parameter materialized to empty values"
                )
            signal_axis_map[param_name] = signal_axes[raw_param_name]
        normalized[indicator_id] = MappingProxyType(signal_axis_map)
    return normalized


def _normalize_nested_scalar_mapping(
    *,
    values: BacktestSignalScalarMap | None,
) -> dict[str, Mapping[str, BacktestRequestScalar]]:
    """
    Normalize nested scalar mapping with deterministic lowercase key ordering.

    Args:
        values: Optional `indicator_id -> parameter -> scalar` payload mapping.
    Returns:
        dict[str, Mapping[str, BacktestRequestScalar]]: Deterministic normalized nested mapping.
    Assumptions:
        Scalar payload values are JSON-compatible by API/use-case contracts.
    Raises:
        ValueError: If one indicator id or nested parameter key is blank.
    Side Effects:
        None.
    """
    if values is None:
        return {}
    if _is_pre_normalized_nested_scalar_mapping(values=values):
        normalized_fast: dict[str, Mapping[str, BacktestRequestScalar]] = {}
        for indicator_id, params in values.items():
            normalized_fast[indicator_id] = MappingProxyType(
                {name: params[name] for name in params.keys()}
            )
        return normalized_fast

    normalized: dict[str, Mapping[str, BacktestRequestScalar]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("nested scalar mapping indicator_id keys must be non-empty")
        normalized[indicator_id] = MappingProxyType(
            _normalize_scalar_mapping(values=values[raw_indicator_id])
        )
    return normalized


def _normalize_summary_metrics_mapping(
    *,
    values: Mapping[str, float] | None,
    total_return_pct: float,
) -> dict[str, float]:
    """
    Normalize summary metrics into deterministic key-sorted float mapping for persisted previews.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py
    Args:
        values: Optional raw summary metrics payload.
        total_return_pct: Canonical total-return metric for the preview.
    Returns:
        dict[str, float]: Deterministic float mapping with `total_return_pct` always populated.
    Assumptions:
        Summary metrics remain JSON-compatible scalar floats used for summary-only persistence.
    Raises:
        ValueError: If one metric key is blank or one metric value is non-numeric.
    Side Effects:
        None.
    """
    normalized: dict[str, float] = {}
    source = values or {}
    for raw_key in sorted(source.keys()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("BacktestVariantPreview.summary_metrics_json keys must be non-empty")
        metric_value = source[raw_key]
        if isinstance(metric_value, bool) or not isinstance(metric_value, int | float):
            raise ValueError(
                "BacktestVariantPreview.summary_metrics_json values must be numeric"
            )
        normalized[key] = float(metric_value)
    normalized["total_return_pct"] = float(total_return_pct)
    return normalized


def _resolve_best_risk_pct(
    *,
    explicit_value: float | None,
    payload: BacktestVariantPayloadV1,
    flag_key: str,
    value_key: str,
) -> float | None:
    """
    Resolve persisted best-risk percentage from explicit field or variant payload risk params.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/domain/entities/backtest_job_results.py
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
    Args:
        explicit_value: Optional explicit best-risk scalar provided by the caller.
        payload: Variant payload carrying risk params.
        flag_key: Boolean risk-enabled flag key.
        value_key: Percentage risk value key.
    Returns:
        float | None: Resolved non-negative risk percentage or `None`.
    Assumptions:
        Risk parameters use human percent units and nullable best-risk fields stay additive.
    Raises:
        ValueError: If explicit or payload-derived value is non-numeric or negative.
    Side Effects:
        None.
    """
    candidate = explicit_value
    if candidate is None:
        risk_params = payload.risk_params or {}
        flag_value = risk_params.get(flag_key)
        risk_value = risk_params.get(value_key)
        if flag_value is True and isinstance(risk_value, int | float) and not isinstance(
            risk_value,
            bool,
        ):
            candidate = float(risk_value)
    if candidate is None:
        return None
    if isinstance(candidate, bool) or not isinstance(candidate, int | float):
        raise ValueError(f"BacktestVariantPreview.{value_key} must be numeric when provided")
    normalized = float(candidate)
    if normalized < 0.0:
        raise ValueError(f"BacktestVariantPreview.{value_key} must be >= 0")
    return normalized


def _normalize_json_payload_mapping(
    *,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Normalize JSON payload mapping into deterministic immutable-friendly structure.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        values: Raw JSON-like mapping payload.
    Returns:
        dict[str, Any]: Deterministic key-sorted JSON-compatible mapping.
    Assumptions:
        Payload is used only for reproducibility metadata carried alongside persisted sync runs.
    Raises:
        ValueError: If a key is blank.
    Side Effects:
        None.
    """
    normalized: dict[str, Any] = {}
    for raw_key in sorted(values.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("RunBacktestResponse.spec_payload_json keys must be non-empty")
        normalized[key] = _normalize_json_payload_value(value=values[raw_key])
    return normalized


def _normalize_json_payload_value(*, value: Any) -> Any:
    """
    Normalize arbitrary JSON-like node into deterministic mapping/list/scalar structure.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    Args:
        value: Raw JSON-like node.
    Returns:
        Any: Deterministically normalized node.
    Assumptions:
        Non-mapping/list scalar values are already JSON-compatible or stringifiable.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        return _normalize_json_payload_mapping(values=value)
    if isinstance(value, list | tuple):
        return [_normalize_json_payload_value(value=item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    return value


def _is_pre_sorted_indicator_selections(
    *,
    indicator_selections: tuple[IndicatorVariantSelection, ...],
) -> bool:
    """
    Check whether indicator selections are already sorted by `indicator_id` asc.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py

    Args:
        indicator_selections: Candidate indicator selections tuple.
    Returns:
        bool: `True` when tuple is strictly sorted and duplicate-free.
    Assumptions:
        `IndicatorVariantSelection.indicator_id` is already normalized to lowercase.
    Raises:
        None.
    Side Effects:
        None.
    """
    previous_indicator_id = ""
    for selection in indicator_selections:
        if selection.indicator_id <= previous_indicator_id:
            return False
        previous_indicator_id = selection.indicator_id
    return True


def _is_pre_normalized_scalar_mapping(
    *,
    values: Mapping[str, BacktestRequestScalar],
) -> bool:
    """
    Check whether scalar mapping already matches canonical stripped key order.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py

    Args:
        values: Candidate scalar mapping.
    Returns:
        bool: `True` when mapping can skip extra normalize/sort pass.
    Assumptions:
        Fast-path accepts only immutable `MappingProxyType` payloads.
    Raises:
        None.
    Side Effects:
        None.
    """
    if not isinstance(values, MappingProxyType):
        return False
    previous_key = ""
    for raw_key in values.keys():
        if not isinstance(raw_key, str):
            return False
        normalized_key = raw_key.strip()
        if not normalized_key or normalized_key != raw_key:
            return False
        if normalized_key < previous_key:
            return False
        previous_key = normalized_key
    return True


def _is_pre_normalized_nested_scalar_mapping(
    *,
    values: BacktestSignalScalarMap,
) -> bool:
    """
    Check whether nested scalar mapping already matches lowercase sorted mapping-proxy shape.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py

    Args:
        values: Candidate nested mapping.
    Returns:
        bool: `True` when nested payload can skip re-normalization.
    Assumptions:
        Fast-path must stay conservative and reject mutable/unsorted payloads.
    Raises:
        None.
    Side Effects:
        None.
    """
    if not isinstance(values, MappingProxyType):
        return False
    previous_indicator_id = ""
    for indicator_id, params in values.items():
        if not isinstance(indicator_id, str):
            return False
        normalized_indicator_id = indicator_id.strip().lower()
        if not normalized_indicator_id or normalized_indicator_id != indicator_id:
            return False
        if normalized_indicator_id < previous_indicator_id:
            return False
        if not _is_pre_normalized_scalar_mapping(values=params):
            return False
        previous_indicator_id = normalized_indicator_id
    return True
