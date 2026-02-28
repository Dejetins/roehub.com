"""
Pydantic API models and deterministic converters for backtests sync/report endpoints.

Docs:
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Annotated, Any, Literal, Mapping, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trading.contexts.backtest.application.dto import (
    BacktestMetricRowV1,
    BacktestRankingConfig,
    BacktestReportV1,
    BacktestRequestScalar,
    BacktestRiskGridSpec,
    BacktestVariantPayloadV1,
    BacktestVariantPreview,
    RunBacktestRequest,
    RunBacktestResponse,
    RunBacktestSavedOverrides,
    RunBacktestTemplate,
    normalize_backtest_ranking_metric_literal,
)
from trading.contexts.backtest.application.ports import BacktestStrategySnapshot
from trading.contexts.backtest.domain.entities import TradeV1
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
    RangeValuesSpec,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

BacktestScalar = int | float | str | bool | None
BacktestAxisScalar = int | float | str


class BacktestExplicitAxisSpecRequest(BaseModel):
    """
    Explicit grid axis request DTO (`mode=explicit`).

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/domain/specifications/grid_param_spec.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["explicit"] = "explicit"
    values: list[BacktestAxisScalar]


class BacktestRangeAxisSpecRequest(BaseModel):
    """
    Inclusive range grid axis request DTO (`mode=range`).

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/domain/specifications/grid_param_spec.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["range"] = "range"
    start: int | float
    stop_incl: int | float
    step: int | float


BacktestAxisSpecRequest = Annotated[
    BacktestExplicitAxisSpecRequest | BacktestRangeAxisSpecRequest,
    Field(discriminator="mode"),
]


class BacktestTimeRangeRequest(BaseModel):
    """
    API payload for half-open UTC time range `[start, end)`.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/shared_kernel/primitives/time_range.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime


class BacktestInstrumentIdRequest(BaseModel):
    """
    API payload for market/symbol instrument identity tuple.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/shared_kernel/primitives/instrument_id.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    market_id: int
    symbol: str


class BacktestExecutionRequest(BaseModel):
    """
    API payload for execution runtime overrides in human percent units.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/value_objects/execution_params_v1.py
    """

    model_config = ConfigDict(extra="forbid")

    init_cash_quote: int | float | None = None
    fee_pct: int | float | None = None
    slippage_pct: int | float | None = None
    fixed_quote: int | float | None = None
    safe_profit_percent: int | float | None = None


class BacktestRiskGridRequest(BaseModel):
    """
    API payload for Stage-B risk grid with explicit enable flags.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
    """

    model_config = ConfigDict(extra="forbid")

    sl_enabled: bool = False
    tp_enabled: bool = False
    sl: BacktestAxisSpecRequest | None = None
    tp: BacktestAxisSpecRequest | None = None
    sl_pct: int | float | None = None
    tp_pct: int | float | None = None


class BacktestIndicatorGridRequest(BaseModel):
    """
    API payload for one ad-hoc indicator grid block.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    indicator_id: str
    params: dict[str, BacktestAxisSpecRequest] = Field(default_factory=dict)
    source: BacktestAxisSpecRequest | None = None


class BacktestTemplateRequest(BaseModel):
    """
    API payload for ad-hoc `template` mode in `POST /backtests`.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    instrument_id: BacktestInstrumentIdRequest
    timeframe: str
    indicator_grids: list[BacktestIndicatorGridRequest]
    direction_mode: str | None = None
    sizing_mode: str | None = None
    execution: BacktestExecutionRequest | None = None
    risk_grid: BacktestRiskGridRequest | None = None
    signal_grids: dict[str, dict[str, BacktestAxisSpecRequest]] = Field(default_factory=dict)


class BacktestSavedOverridesRequest(BaseModel):
    """
    API payload for optional saved-mode overrides.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    direction_mode: str | None = None
    sizing_mode: str | None = None
    execution: BacktestExecutionRequest | None = None
    risk_grid: BacktestRiskGridRequest | None = None
    signal_grids: dict[str, dict[str, BacktestAxisSpecRequest]] = Field(default_factory=dict)


class BacktestRankingRequest(BaseModel):
    """
    API payload for optional ranking override block in sync/jobs request envelope.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    primary_metric: str
    secondary_metric: str | None = None

    @field_validator("primary_metric", "secondary_metric")
    @classmethod
    def _normalize_metric_literal(cls, metric: str | None) -> str | None:
        """
        Normalize ranking metric literal into canonical lowercase snake_case identifier.

        Args:
            metric: Raw metric literal from API request.
        Returns:
            str | None: Canonical metric literal or `None`.
        Assumptions:
            Allowed literals are fixed by ranking contract and validated centrally.
        Raises:
            ValueError: If metric literal is unsupported.
        Side Effects:
            None.
        """
        if metric is None:
            return None
        return normalize_backtest_ranking_metric_literal(
            metric=metric,
            field_path="ranking.metric",
        )

    @model_validator(mode="after")
    def _validate_secondary_metric(self) -> BacktestRankingRequest:
        """
        Validate secondary metric invariant for deterministic multi-key ranking behavior.

        Args:
            None.
        Returns:
            BacktestRankingRequest: Validated request model instance.
        Assumptions:
            Secondary metric is optional and cannot duplicate primary metric.
        Raises:
            ValueError: If `secondary_metric` duplicates `primary_metric`.
        Side Effects:
            None.
        """
        if self.secondary_metric is None:
            return self
        if self.secondary_metric == self.primary_metric:
            raise ValueError("secondary_metric must be different from primary_metric")
        return self


class BacktestsPostRequest(BaseModel):
    """
    API request envelope for `POST /backtests` saved/ad-hoc modes.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/api/api-errors-and-422-payload-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    time_range: BacktestTimeRangeRequest
    strategy_id: UUID | None = None
    template: BacktestTemplateRequest | None = None
    overrides: BacktestSavedOverridesRequest | None = None
    warmup_bars: int | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, gt=0)
    preselect: int | None = Field(default=None, gt=0)
    top_trades_n: int | None = Field(default=None, gt=0)
    ranking: BacktestRankingRequest | None = None


class BacktestIndicatorSelectionRequest(BaseModel):
    """
    API request payload for one explicit indicator selection in variant-report endpoint.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    indicator_id: str
    inputs: dict[str, BacktestAxisScalar]
    params: dict[str, BacktestAxisScalar]

    @field_validator("inputs", "params", mode="before")
    @classmethod
    def _reject_boolean_scalars(
        cls,
        value: Any,
    ) -> Any:
        """
        Reject boolean scalars before coercion to preserve strict variant selection contract.

        Docs:
          - docs/architecture/backtest/backtest-api-post-backtests-v1.md
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
        Related:
          - apps/api/dto/backtests.py
          - src/trading/contexts/indicators/application/dto/variant_key.py
          - apps/api/routes/backtests.py

        Args:
            value: Raw mapping payload from request body.
        Returns:
            Any: Unchanged mapping payload when no boolean scalars are present.
        Assumptions:
            Variant-selection values must remain `int|float|str` for key semantics stability.
        Raises:
            ValueError: If mapping contains boolean scalar value.
        Side Effects:
            None.
        """
        if not isinstance(value, Mapping):
            return value
        for raw_key in value.keys():
            scalar = value[raw_key]
            if isinstance(scalar, bool):
                key = str(raw_key).strip()
                raise ValueError(f"{key} must be int, float, or string")
        return value


class BacktestVariantPayloadRequest(BaseModel):
    """
    API request payload for explicit selected variant in lazy report-load endpoint.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    indicator_selections: list[BacktestIndicatorSelectionRequest]
    signal_params: dict[str, dict[str, BacktestScalar]]
    risk_params: dict[str, BacktestScalar]
    execution_params: dict[str, BacktestScalar]
    direction_mode: str
    sizing_mode: str


class BacktestsVariantReportPostRequest(BaseModel):
    """
    API request envelope for on-demand `POST /api/backtests/variant-report`.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    time_range: BacktestTimeRangeRequest
    strategy_id: UUID | None = None
    template: BacktestTemplateRequest | None = None
    overrides: BacktestSavedOverridesRequest | None = None
    warmup_bars: int | None = Field(default=None, gt=0)
    variant: BacktestVariantPayloadRequest
    include_trades: bool = False


class BacktestInstrumentIdResponse(BaseModel):
    """
    API response payload for instrument identity tuple.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/shared_kernel/primitives/instrument_id.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    market_id: int
    symbol: str


class BacktestMetricRowResponse(BaseModel):
    """
    API response payload for one reporting metrics-table row.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: str


class BacktestTradeResponse(BaseModel):
    """
    API response payload for one deterministic closed trade.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    trade_id: int
    direction: str
    entry_bar_index: int
    exit_bar_index: int
    entry_fill_price: float
    exit_fill_price: float
    qty_base: float
    entry_quote_amount: float
    exit_quote_amount: float
    entry_fee_quote: float
    exit_fee_quote: float
    gross_pnl_quote: float
    net_pnl_quote: float
    locked_profit_quote: float
    exit_reason: str


class BacktestReportResponse(BaseModel):
    """
    API response payload for deterministic backtest report block.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    rows: list[BacktestMetricRowResponse]
    table_md: str | None = None
    trades: list[BacktestTradeResponse] | None = None


class BacktestIndicatorSelectionResponse(BaseModel):
    """
    API response payload for one explicit indicator selection.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    indicator_id: str
    inputs: dict[str, BacktestAxisScalar]
    params: dict[str, BacktestAxisScalar]


class BacktestVariantPayloadResponse(BaseModel):
    """
    API response payload for one saveable explicit variant configuration.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    indicator_selections: list[BacktestIndicatorSelectionResponse]
    signal_params: dict[str, dict[str, BacktestScalar]]
    risk_params: dict[str, BacktestScalar]
    execution_params: dict[str, BacktestScalar]
    direction_mode: str
    sizing_mode: str


class BacktestVariantResponse(BaseModel):
    """
    API response payload for one ranked top-K variant.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py
    """

    model_config = ConfigDict(extra="forbid")

    variant_index: int
    variant_key: str
    indicator_variant_key: str
    total_return_pct: float
    report: BacktestReportResponse | None = None
    payload: BacktestVariantPayloadResponse


class BacktestsPostResponse(BaseModel):
    """
    API response payload for `POST /backtests` endpoint.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    mode: str
    strategy_id: UUID | None
    instrument_id: BacktestInstrumentIdResponse
    timeframe: str
    warmup_bars: int
    top_k: int
    preselect: int
    top_trades_n: int
    spec_hash: str | None = None
    grid_request_hash: str | None = None
    engine_params_hash: str
    variants: list[BacktestVariantResponse]


def build_backtest_run_request(*, request: BacktestsPostRequest) -> RunBacktestRequest:
    """
    Convert API request envelope into application `RunBacktestRequest` deterministically.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py

    Args:
        request: Parsed API request payload.
    Returns:
        RunBacktestRequest: Deterministic application-layer request DTO.
    Assumptions:
        Request mode is selected by one-of contract: `strategy_id xor template`.
    Raises:
        BacktestValidationError: If mode contract or override contract is violated.
        ValueError: If primitive conversion into shared-kernel value objects fails.
    Side Effects:
        None.
    """
    has_saved_mode = request.strategy_id is not None
    has_template_mode = request.template is not None
    if has_saved_mode == has_template_mode:
        raise BacktestValidationError(
            "POST /backtests requires exactly one mode: strategy_id xor template",
            errors=(
                {
                    "path": "body.strategy_id",
                    "code": "mode_conflict",
                    "message": "Provide exactly one of strategy_id or template",
                },
                {
                    "path": "body.template",
                    "code": "mode_conflict",
                    "message": "Provide exactly one of strategy_id or template",
                },
            ),
        )

    if request.overrides is not None and not has_saved_mode:
        raise BacktestValidationError(
            "POST /backtests overrides are allowed only in saved mode",
            errors=(
                {
                    "path": "body.overrides",
                    "code": "mode_conflict",
                    "message": "overrides are allowed only with strategy_id",
                },
            ),
        )

    return RunBacktestRequest(
        time_range=_build_time_range(request=request),
        strategy_id=request.strategy_id,
        template=_build_template(request=request.template),
        overrides=_build_saved_overrides(request=request.overrides),
        warmup_bars=request.warmup_bars,
        top_k=request.top_k,
        preselect=request.preselect,
        top_trades_n=request.top_trades_n,
        ranking=_build_ranking_config(request=request.ranking),
    )


def build_backtest_variant_report_run_request(
    *,
    request: BacktestsVariantReportPostRequest,
) -> RunBacktestRequest:
    """
    Convert variant-report run context into application `RunBacktestRequest`.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py

    Args:
        request: Parsed variant-report API request payload.
    Returns:
        RunBacktestRequest: Application request DTO for timeline/ownership resolution.
    Assumptions:
        Mode contract stays `strategy_id xor template` and reuses sync endpoint validation.
    Raises:
        BacktestValidationError: If mode contract or overrides contract is violated.
        ValueError: If primitive conversions fail.
    Side Effects:
        None.
    """
    return build_backtest_run_request(
        request=BacktestsPostRequest(
            time_range=request.time_range,
            strategy_id=request.strategy_id,
            template=request.template,
            overrides=request.overrides,
            warmup_bars=request.warmup_bars,
        )
    )


def build_backtest_variant_report_payload(
    *,
    request: BacktestVariantPayloadRequest,
) -> BacktestVariantPayloadV1:
    """
    Convert explicit variant-report payload into application variant payload DTO.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        request: Parsed explicit variant payload from API request.
    Returns:
        BacktestVariantPayloadV1: Deterministic variant payload for report build use-case.
    Assumptions:
        Nested mappings are normalized and key-sorted for deterministic variant identity.
    Raises:
        ValueError: If one scalar field violates payload invariants.
    Side Effects:
        None.
    """
    sorted_indicator_selections = sorted(
        request.indicator_selections,
        key=lambda item: item.indicator_id.strip().lower(),
    )
    return BacktestVariantPayloadV1(
        indicator_selections=tuple(
            IndicatorVariantSelection(
                indicator_id=item.indicator_id,
                inputs=_normalize_indicator_selection_mapping(
                    values=item.inputs,
                    field_path=f"variant.indicator_selections[{index}].inputs",
                ),
                params=_normalize_indicator_selection_mapping(
                    values=item.params,
                    field_path=f"variant.indicator_selections[{index}].params",
                ),
            )
            for index, item in enumerate(sorted_indicator_selections)
        ),
        signal_params=_normalize_variant_signal_scalar_mapping(
            values=request.signal_params,
        ),
        risk_params=_normalize_variant_scalar_mapping(
            values=request.risk_params,
            field_path="variant.risk_params",
        ),
        execution_params=_normalize_variant_scalar_mapping(
            values=request.execution_params,
            field_path="variant.execution_params",
        ),
        direction_mode=request.direction_mode,
        sizing_mode=request.sizing_mode,
    )


def build_backtest_variant_report_response(
    *,
    report: BacktestReportV1,
) -> BacktestReportResponse:
    """
    Convert application variant report DTO into strict API response payload.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py

    Args:
        report: Application-layer report payload.
    Returns:
        BacktestReportResponse: Strict API report response.
    Assumptions:
        Variant-report endpoint always returns one non-null report.
    Raises:
        BacktestValidationError: If report payload is unexpectedly missing.
    Side Effects:
        None.
    """
    response = _build_report_response(report=report)
    if response is None:  # pragma: no cover - guarded by type contract
        raise BacktestValidationError("Variant report payload is required")
    return response


def build_backtests_post_response(
    *,
    request: BacktestsPostRequest,
    response: RunBacktestResponse,
    strategy_snapshot: BacktestStrategySnapshot | None,
    include_reports: bool,
) -> BacktestsPostResponse:
    """
    Convert application response DTO into strict API response payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py

    Args:
        request: Original parsed API request payload.
        response: Application-layer deterministic response DTO.
        strategy_snapshot: Optional saved strategy snapshot used for `spec_hash`.
        include_reports: Whether variant report bodies should be included in sync response.
    Returns:
        BacktestsPostResponse: Strict API response model.
    Assumptions:
        `response.variants` may be sorted again defensively by ranking contract.
    Raises:
        BacktestValidationError: If required hash source payload is missing.
    Side Effects:
        None.
    """
    variants = sorted(
        response.variants,
        key=lambda item: (-item.total_return_pct, item.variant_key),
    )

    spec_hash: str | None = None
    grid_request_hash: str | None = None
    if response.mode == "saved":
        if strategy_snapshot is None:
            raise BacktestValidationError(
                "Unable to build spec_hash: saved strategy snapshot is missing",
            )
        spec_hash = build_sha256_from_payload(payload=dict(strategy_snapshot.spec_payload or {}))
    else:
        grid_request_hash = build_grid_request_hash(request=request)

    engine_params_hash = build_engine_params_hash(
        request=request,
        response=response,
        variants=variants,
    )

    return BacktestsPostResponse(
        schema_version=1,
        mode=response.mode,
        strategy_id=response.strategy_id,
        instrument_id=BacktestInstrumentIdResponse(
            market_id=response.instrument_id.market_id.value,
            symbol=str(response.instrument_id.symbol),
        ),
        timeframe=response.timeframe.code,
        warmup_bars=response.warmup_bars,
        top_k=response.top_k,
        preselect=response.preselect,
        top_trades_n=response.top_trades_n,
        spec_hash=spec_hash,
        grid_request_hash=grid_request_hash,
        engine_params_hash=engine_params_hash,
        variants=[
            _build_variant_response(variant=item, include_report=include_reports)
            for item in variants
        ],
    )


def build_grid_request_hash(*, request: BacktestsPostRequest) -> str:
    """
    Build deterministic ad-hoc grid request hash from canonical JSON payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py

    Args:
        request: Parsed API request payload.
    Returns:
        str: Hex SHA-256 hash for canonical ad-hoc request payload.
    Assumptions:
        Function is called only for template mode responses.
    Raises:
        BacktestValidationError: If template payload is missing in template mode.
    Side Effects:
        None.
    """
    if request.template is None:
        raise BacktestValidationError("grid_request_hash requires template payload")

    payload = {
        "mode": "template",
        "time_range": request.time_range.model_dump(mode="json"),
        "template": request.template.model_dump(mode="json", exclude_none=True),
        "warmup_bars": request.warmup_bars,
        "top_k": request.top_k,
        "preselect": request.preselect,
        "top_trades_n": request.top_trades_n,
    }
    if request.ranking is not None:
        payload["ranking"] = request.ranking.model_dump(mode="json", exclude_none=True)
    return build_sha256_from_payload(payload=payload)


def build_engine_params_hash(
    *,
    request: BacktestsPostRequest,
    response: RunBacktestResponse,
    variants: Sequence[BacktestVariantPreview],
) -> str:
    """
    Build deterministic engine params hash from effective run-time settings payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - apps/api/dto/backtests.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/domain/value_objects/execution_params_v1.py

    Args:
        request: Parsed API request payload.
        response: Application deterministic response DTO.
        variants: Deterministically ordered variants list.
    Returns:
        str: Hex SHA-256 hash for canonical effective engine params payload.
    Assumptions:
        First ranked variant reflects effective execution settings for this run.
    Raises:
        BacktestValidationError: If engine payload cannot be derived.
    Side Effects:
        None.
    """
    if len(variants) > 0 and variants[0].payload is not None:
        payload_source = variants[0].payload
        payload = {
            "direction_mode": payload_source.direction_mode,
            "sizing_mode": payload_source.sizing_mode,
            "execution": _to_sorted_scalar_mapping(payload_source.execution_params),
        }
        return build_sha256_from_payload(payload=payload)

    if request.template is not None:
        payload = {
            "direction_mode": request.template.direction_mode,
            "sizing_mode": request.template.sizing_mode,
            "execution": request.template.execution.model_dump(exclude_none=True)
            if request.template.execution is not None
            else {},
        }
        return build_sha256_from_payload(payload=payload)

    raise BacktestValidationError(
        "Unable to build engine_params_hash for empty variants saved-mode response",
    )


def build_sha256_from_payload(*, payload: Mapping[str, Any]) -> str:
    """
    Build deterministic SHA-256 hash from canonical JSON representation.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - apps/api/routes/backtests.py

    Args:
        payload: JSON-compatible payload mapping.
    Returns:
        str: Hex SHA-256 hash.
    Assumptions:
        Canonical JSON uses sorted keys and stable separators.
    Raises:
        TypeError: If payload contains unsupported non-JSON-compatible values.
    Side Effects:
        None.
    """
    canonical_json = json.dumps(
        _normalize_json_value(value=payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _build_time_range(*, request: BacktestsPostRequest) -> TimeRange:
    """
    Convert API time-range payload into shared-kernel `TimeRange` value object.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/shared_kernel/primitives/time_range.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py

    Args:
        request: Parsed API request payload.
    Returns:
        TimeRange: Half-open UTC range.
    Assumptions:
        Datetime values are timezone-aware and validated by `UtcTimestamp`.
    Raises:
        ValueError: If timestamps are invalid or range invariant is broken.
    Side Effects:
        None.
    """
    return TimeRange(
        start=UtcTimestamp(request.time_range.start),
        end=UtcTimestamp(request.time_range.end),
    )


def _build_template(*, request: BacktestTemplateRequest | None) -> RunBacktestTemplate | None:
    """
    Convert optional ad-hoc template payload into application `RunBacktestTemplate`.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        request: Optional template payload.
    Returns:
        RunBacktestTemplate | None: Converted template for ad-hoc mode.
    Assumptions:
        Missing template means request is expected to be saved mode.
    Raises:
        ValueError: If conversion of primitives/grid specs fails.
    Side Effects:
        None.
    """
    if request is None:
        return None

    risk_grid = _build_risk_grid_spec(request=request.risk_grid)
    risk_params = _build_risk_params(request=request.risk_grid)
    execution_params = _build_execution_params(request=request.execution)

    direction_mode = request.direction_mode if request.direction_mode is not None else "long-short"
    sizing_mode = request.sizing_mode if request.sizing_mode is not None else "all_in"

    return RunBacktestTemplate(
        instrument_id=InstrumentId(
            market_id=MarketId(request.instrument_id.market_id),
            symbol=Symbol(request.instrument_id.symbol),
        ),
        timeframe=Timeframe(request.timeframe),
        indicator_grids=_build_indicator_grids(request=request),
        signal_grids=_build_signal_grids(signal_grids=request.signal_grids),
        risk_grid=risk_grid,
        direction_mode=direction_mode,
        sizing_mode=sizing_mode,
        risk_params=risk_params,
        execution_params=execution_params,
    )


def _build_saved_overrides(
    *,
    request: BacktestSavedOverridesRequest | None,
) -> RunBacktestSavedOverrides | None:
    """
    Convert optional saved-mode overrides payload into application DTO.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        request: Optional saved-mode overrides payload.
    Returns:
        RunBacktestSavedOverrides | None: Converted overrides DTO.
    Assumptions:
        Missing payload means no overrides over saved strategy snapshot.
    Raises:
        ValueError: If override payload cannot be converted deterministically.
    Side Effects:
        None.
    """
    if request is None:
        return None

    return RunBacktestSavedOverrides(
        direction_mode=request.direction_mode,
        sizing_mode=request.sizing_mode,
        signal_grids=_build_signal_grids(signal_grids=request.signal_grids),
        risk_grid=_build_risk_grid_spec(request=request.risk_grid),
        risk_params=_build_risk_params(request=request.risk_grid),
        execution_params=_build_execution_params(request=request.execution),
    )


def _build_ranking_config(
    *,
    request: BacktestRankingRequest | None,
) -> BacktestRankingConfig | None:
    """
    Convert optional API ranking block into deterministic application ranking config DTO.

    Docs:
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        request: Optional ranking override payload.
    Returns:
        BacktestRankingConfig | None: Application ranking DTO or `None` when omitted.
    Assumptions:
        Missing block keeps runtime-config defaults and preserves backward compatibility.
    Raises:
        ValueError: If ranking metric literals violate DTO invariants.
    Side Effects:
        None.
    """
    if request is None:
        return None
    return BacktestRankingConfig(
        primary_metric=request.primary_metric,
        secondary_metric=request.secondary_metric,
    )


def _build_indicator_grids(*, request: BacktestTemplateRequest) -> tuple[GridSpec, ...]:
    """
    Convert API indicator grid payload list into deterministic `GridSpec` tuple.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py

    Args:
        request: Parsed template-mode payload.
    Returns:
        tuple[GridSpec, ...]: Deterministic indicator grids tuple.
    Assumptions:
        Input list order is preserved to keep duplicate-id overwrite semantics stable.
    Raises:
        ValueError: If one indicator block is malformed.
    Side Effects:
        None.
    """
    grids: list[GridSpec] = []
    for item in request.indicator_grids:
        params: dict[str, GridParamSpec] = {}
        for param_name in sorted(item.params.keys(), key=lambda key: key.strip().lower()):
            params[param_name.strip().lower()] = _build_grid_param_spec(
                name=param_name,
                request=item.params[param_name],
            )

        source: GridParamSpec | None = None
        if item.source is not None:
            source = _build_grid_param_spec(name="source", request=item.source)

        grids.append(
            GridSpec(
                indicator_id=IndicatorId(item.indicator_id.strip().lower()),
                params=params,
                source=source,
            )
        )

    return tuple(grids)


def _build_signal_grids(
    *,
    signal_grids: Mapping[str, Mapping[str, BacktestAxisSpecRequest]] | None,
) -> Mapping[str, Mapping[str, GridParamSpec]]:
    """
    Convert nested API signal grids payload into deterministic nested grid specs mapping.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py

    Args:
        signal_grids: Optional nested `indicator_id -> signal_param -> axis` payload.
    Returns:
        Mapping[str, Mapping[str, GridParamSpec]]: Deterministic nested mapping.
    Assumptions:
        Missing signal payload means empty mapping.
    Raises:
        ValueError: If one indicator id/parameter key is blank.
    Side Effects:
        None.
    """
    if signal_grids is None:
        return {}

    normalized: dict[str, Mapping[str, GridParamSpec]] = {}
    for raw_indicator_id in sorted(signal_grids.keys(), key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("signal_grids indicator_id keys must be non-empty")

        params = signal_grids[raw_indicator_id]
        normalized_params: dict[str, GridParamSpec] = {}
        for raw_param_name in sorted(params.keys(), key=lambda key: str(key).strip().lower()):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("signal_grids parameter keys must be non-empty")
            normalized_params[param_name] = _build_grid_param_spec(
                name=param_name,
                request=params[raw_param_name],
            )

        normalized[indicator_id] = normalized_params
    return normalized


def _build_risk_grid_spec(
    *,
    request: BacktestRiskGridRequest | None,
) -> BacktestRiskGridSpec | None:
    """
    Convert API risk grid payload into `BacktestRiskGridSpec` value object.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py

    Args:
        request: Optional risk grid payload.
    Returns:
        BacktestRiskGridSpec | None: Converted risk grid object.
    Assumptions:
        Scalar fallback (`sl_pct`/`tp_pct`) can be promoted into explicit single-value axes.
    Raises:
        ValueError: If axis payload shape is invalid.
    Side Effects:
        None.
    """
    if request is None:
        return None

    sl = _build_grid_param_spec(name="sl", request=request.sl) if request.sl is not None else None
    tp = _build_grid_param_spec(name="tp", request=request.tp) if request.tp is not None else None

    if request.sl_enabled and sl is None and request.sl_pct is not None:
        sl = ExplicitValuesSpec(
            name="sl",
            values=(_normalize_numeric(value=request.sl_pct, field_path="risk_grid.sl_pct"),),
        )
    if request.tp_enabled and tp is None and request.tp_pct is not None:
        tp = ExplicitValuesSpec(
            name="tp",
            values=(_normalize_numeric(value=request.tp_pct, field_path="risk_grid.tp_pct"),),
        )

    return BacktestRiskGridSpec(
        sl_enabled=request.sl_enabled,
        tp_enabled=request.tp_enabled,
        sl=sl,
        tp=tp,
    )


def _build_risk_params(
    *,
    request: BacktestRiskGridRequest | None,
) -> Mapping[str, BacktestRequestScalar]:
    """
    Convert API risk payload into scalar mapping used by application fallback semantics.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py

    Args:
        request: Optional risk grid payload.
    Returns:
        Mapping[str, BacktestRequestScalar]: Deterministic scalar risk mapping.
    Assumptions:
        Flags are included explicitly to keep saved/ad-hoc overrides deterministic.
    Raises:
        ValueError: If numeric scalar values are invalid.
    Side Effects:
        None.
    """
    if request is None:
        return {}

    values: dict[str, BacktestRequestScalar] = {
        "sl_enabled": bool(request.sl_enabled),
        "tp_enabled": bool(request.tp_enabled),
    }
    if request.sl_pct is not None:
        values["sl_pct"] = _normalize_numeric(value=request.sl_pct, field_path="risk_grid.sl_pct")
    if request.tp_pct is not None:
        values["tp_pct"] = _normalize_numeric(value=request.tp_pct, field_path="risk_grid.tp_pct")
    return values


def _build_execution_params(
    *,
    request: BacktestExecutionRequest | None,
) -> Mapping[str, BacktestRequestScalar]:
    """
    Convert API execution payload into deterministic scalar mapping.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/domain/value_objects/execution_params_v1.py

    Args:
        request: Optional execution payload.
    Returns:
        Mapping[str, BacktestRequestScalar]: Deterministic execution scalars mapping.
    Assumptions:
        Percent units remain human-readable (`0.1` means `0.1%`).
    Raises:
        ValueError: If one numeric field is invalid.
    Side Effects:
        None.
    """
    if request is None:
        return {}

    values: dict[str, BacktestRequestScalar] = {}
    if request.init_cash_quote is not None:
        values["init_cash_quote"] = _normalize_numeric(
            value=request.init_cash_quote,
            field_path="execution.init_cash_quote",
        )
    if request.fee_pct is not None:
        values["fee_pct"] = _normalize_numeric(
            value=request.fee_pct,
            field_path="execution.fee_pct",
        )
    if request.slippage_pct is not None:
        values["slippage_pct"] = _normalize_numeric(
            value=request.slippage_pct,
            field_path="execution.slippage_pct",
        )
    if request.fixed_quote is not None:
        values["fixed_quote"] = _normalize_numeric(
            value=request.fixed_quote,
            field_path="execution.fixed_quote",
        )
    if request.safe_profit_percent is not None:
        values["safe_profit_percent"] = _normalize_numeric(
            value=request.safe_profit_percent,
            field_path="execution.safe_profit_percent",
        )
    return values


def _build_grid_param_spec(
    *,
    name: str,
    request: BacktestAxisSpecRequest | None,
) -> GridParamSpec:
    """
    Convert API axis payload into one deterministic `GridParamSpec` implementation.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/domain/specifications/grid_param_spec.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py

    Args:
        name: Axis name.
        request: Parsed API axis payload.
    Returns:
        GridParamSpec: Explicit or range axis spec object.
    Assumptions:
        Caller enforces non-null request payload.
    Raises:
        ValueError: If axis mode is unknown or values are invalid.
    Side Effects:
        None.
    """
    if request is None:
        raise ValueError(f"axis '{name}' request payload is required")

    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("axis name must be non-empty")

    mode = request.mode.strip().lower()
    if mode == "explicit":
        if not isinstance(request, BacktestExplicitAxisSpecRequest):
            raise ValueError(f"axis '{normalized_name}' mode payload mismatch")
        values: list[BacktestAxisScalar] = []
        for value in request.values:
            if isinstance(value, bool):
                raise ValueError(f"axis '{normalized_name}' explicit values must not be boolean")
            values.append(value)
        if len(values) == 0:
            raise ValueError(f"axis '{normalized_name}' explicit values must be non-empty")
        return ExplicitValuesSpec(name=normalized_name, values=tuple(values))

    if mode == "range":
        if not isinstance(request, BacktestRangeAxisSpecRequest):
            raise ValueError(f"axis '{normalized_name}' mode payload mismatch")
        return RangeValuesSpec(
            name=normalized_name,
            start=_normalize_axis_numeric(
                value=request.start,
                field_path=f"{normalized_name}.start",
            ),
            stop_inclusive=_normalize_axis_numeric(
                value=request.stop_incl,
                field_path=f"{normalized_name}.stop_incl",
            ),
            step=_normalize_axis_numeric(
                value=request.step,
                field_path=f"{normalized_name}.step",
            ),
        )

    raise ValueError(f"axis '{normalized_name}' mode must be explicit or range")


def _build_variant_response(
    *,
    variant: BacktestVariantPreview,
    include_report: bool,
) -> BacktestVariantResponse:
    """
    Convert one application variant preview into strict API variant response payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        variant: Application deterministic variant preview.
        include_report: Whether report body is included in sync response payload.
    Returns:
        BacktestVariantResponse: Strict API variant payload.
    Assumptions:
        Variant payload is always present by application DTO invariant.
    Raises:
        BacktestValidationError: If variant payload is unexpectedly missing.
    Side Effects:
        None.
    """
    if variant.payload is None:
        raise BacktestValidationError("Backtest variant payload is required")

    return BacktestVariantResponse(
        variant_index=variant.variant_index,
        variant_key=variant.variant_key,
        indicator_variant_key=variant.indicator_variant_key,
        total_return_pct=variant.total_return_pct,
        payload=_build_variant_payload_response(payload=variant.payload),
        report=_build_report_response(report=variant.report) if include_report else None,
    )


def _build_variant_payload_response(
    *,
    payload: BacktestVariantPayloadV1,
) -> BacktestVariantPayloadResponse:
    """
    Convert one application variant payload into strict API payload block.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        payload: Application variant payload.
    Returns:
        BacktestVariantPayloadResponse: Strict API payload for one variant.
    Assumptions:
        Nested mapping keys must be sorted for deterministic JSON output.
    Raises:
        None.
    Side Effects:
        None.
    """
    indicator_selections = sorted(
        payload.indicator_selections,
        key=lambda item: item.indicator_id,
    )
    return BacktestVariantPayloadResponse(
        indicator_selections=[
            _build_indicator_selection_response(selection=item) for item in indicator_selections
        ],
        signal_params=_to_sorted_nested_scalar_mapping(values=payload.signal_params),
        risk_params=_to_sorted_scalar_mapping(payload.risk_params),
        execution_params=_to_sorted_scalar_mapping(payload.execution_params),
        direction_mode=payload.direction_mode,
        sizing_mode=payload.sizing_mode,
    )


def _build_indicator_selection_response(
    *,
    selection: IndicatorVariantSelection,
) -> BacktestIndicatorSelectionResponse:
    """
    Convert application indicator selection into strict API payload model.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - apps/api/routes/backtests.py

    Args:
        selection: Application indicator selection payload.
    Returns:
        BacktestIndicatorSelectionResponse: Strict API indicator selection payload.
    Assumptions:
        Inputs/params values are scalar and JSON-compatible.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestIndicatorSelectionResponse(
        indicator_id=selection.indicator_id,
        inputs={name: selection.inputs[name] for name in sorted(selection.inputs.keys())},
        params={name: selection.params[name] for name in sorted(selection.params.keys())},
    )


def _build_report_response(*, report: BacktestReportV1 | None) -> BacktestReportResponse | None:
    """
    Convert optional application report payload into strict API report model.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        report: Optional application report payload.
    Returns:
        BacktestReportResponse | None: Strict report payload or `None`.
    Assumptions:
        Trades are already sorted by `trade_id` in application report DTO.
    Raises:
        None.
    Side Effects:
        None.
    """
    if report is None:
        return None

    return BacktestReportResponse(
        rows=[_build_metric_row_response(row=row) for row in report.rows],
        table_md=report.table_md,
        trades=_build_trade_responses(trades=report.trades),
    )


def _build_metric_row_response(*, row: BacktestMetricRowV1) -> BacktestMetricRowResponse:
    """
    Convert one application metric row into API metric row payload.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        row: Application metrics row.
    Returns:
        BacktestMetricRowResponse: API row payload.
    Assumptions:
        Row values are already deterministically formatted in application layer.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestMetricRowResponse(metric=row.metric, value=row.value)


def _build_trade_responses(
    *,
    trades: tuple[TradeV1, ...] | None,
) -> list[BacktestTradeResponse] | None:
    """
    Convert optional application trades tuple into strict API trades list.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - apps/api/routes/backtests.py

    Args:
        trades: Optional application trades tuple.
    Returns:
        list[BacktestTradeResponse] | None: API trades list or `None`.
    Assumptions:
        Trade ordering is deterministic and preserved.
    Raises:
        None.
    Side Effects:
        None.
    """
    if trades is None:
        return None

    return [_build_trade_response(trade=item) for item in trades]


def _build_trade_response(*, trade: TradeV1) -> BacktestTradeResponse:
    """
    Convert one application trade entity into strict API trade payload.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - apps/api/routes/backtests.py

    Args:
        trade: Application trade entity.
    Returns:
        BacktestTradeResponse: API trade payload.
    Assumptions:
        Trade scalar values are JSON-compatible primitives.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestTradeResponse(
        trade_id=trade.trade_id,
        direction=trade.direction,
        entry_bar_index=trade.entry_bar_index,
        exit_bar_index=trade.exit_bar_index,
        entry_fill_price=trade.entry_fill_price,
        exit_fill_price=trade.exit_fill_price,
        qty_base=trade.qty_base,
        entry_quote_amount=trade.entry_quote_amount,
        exit_quote_amount=trade.exit_quote_amount,
        entry_fee_quote=trade.entry_fee_quote,
        exit_fee_quote=trade.exit_fee_quote,
        gross_pnl_quote=trade.gross_pnl_quote,
        net_pnl_quote=trade.net_pnl_quote,
        locked_profit_quote=trade.locked_profit_quote,
        exit_reason=trade.exit_reason,
    )


def _normalize_indicator_selection_mapping(
    *,
    values: Mapping[str, BacktestAxisScalar],
    field_path: str,
) -> dict[str, BacktestAxisScalar]:
    """
    Normalize explicit indicator selection scalar mapping for variant-report payload.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/application/dto/variant_key.py
      - apps/api/routes/backtests.py

    Args:
        values: Raw mapping with scalar input/parameter values.
        field_path: Dot-path prefix used in deterministic validation errors.
    Returns:
        dict[str, BacktestAxisScalar]: Deterministic key-sorted scalar mapping.
    Assumptions:
        Values are explicit scalars and must not contain booleans or nulls.
    Raises:
        ValueError: If key is blank or scalar value type is unsupported.
    Side Effects:
        None.
    """
    normalized: dict[str, BacktestAxisScalar] = {}
    for raw_key in sorted(values.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{field_path} keys must be non-empty")
        raw_value = values[raw_key]
        if raw_value is None or isinstance(raw_value, bool) or not isinstance(
            raw_value, (int, float, str)
        ):
            raise ValueError(
                f"{field_path}.{key} must be int, float, or string"
            )
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            raise ValueError(
                f"{field_path}.{key} must be finite number"
            )
        normalized[key] = raw_value
    return normalized


def _normalize_variant_scalar_mapping(
    *,
    values: Mapping[str, BacktestScalar],
    field_path: str,
) -> dict[str, BacktestScalar]:
    """
    Normalize scalar mapping payload for variant-report risk/execution blocks.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        values: Raw scalar mapping payload.
        field_path: Dot-path prefix used in deterministic validation errors.
    Returns:
        dict[str, BacktestScalar]: Deterministic key-sorted scalar mapping.
    Assumptions:
        Scalar values are JSON-compatible and finite for numeric types.
    Raises:
        ValueError: If key is blank or numeric scalar is non-finite.
    Side Effects:
        None.
    """
    normalized: dict[str, BacktestScalar] = {}
    for raw_key in sorted(values.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{field_path} keys must be non-empty")
        raw_value = values[raw_key]
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            raise ValueError(f"{field_path}.{key} must be finite number")
        normalized[key] = raw_value
    return normalized


def _normalize_variant_signal_scalar_mapping(
    *,
    values: Mapping[str, Mapping[str, BacktestScalar]],
) -> dict[str, dict[str, BacktestScalar]]:
    """
    Normalize nested signal scalar mapping for deterministic variant-report payloads.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        values: Raw nested `indicator_id -> signal_param -> scalar` payload mapping.
    Returns:
        dict[str, dict[str, BacktestScalar]]: Deterministic nested mapping.
    Assumptions:
        Indicator/parameter keys are normalized to lowercase identifiers.
    Raises:
        ValueError: If indicator/parameter keys are blank or numeric value is non-finite.
    Side Effects:
        None.
    """
    normalized: dict[str, dict[str, BacktestScalar]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("variant.signal_params indicator_id keys must be non-empty")
        signal_params = values[raw_indicator_id]
        normalized_params: dict[str, BacktestScalar] = {}
        for raw_param_name in sorted(
            signal_params.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("variant.signal_params param keys must be non-empty")
            raw_value = signal_params[raw_param_name]
            if isinstance(raw_value, float) and not math.isfinite(raw_value):
                raise ValueError(
                    f"variant.signal_params.{indicator_id}.{param_name} must be finite number"
                )
            normalized_params[param_name] = raw_value
        normalized[indicator_id] = normalized_params
    return normalized


def _normalize_numeric(*, value: int | float, field_path: str) -> float:
    """
    Convert numeric payload scalar to float while rejecting booleans.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        value: Raw numeric payload value.
        field_path: Dot-path used in deterministic error messages.
    Returns:
        float: Normalized numeric scalar.
    Assumptions:
        Numeric payload values use human units from API contract.
    Raises:
        ValueError: If value is boolean.
    Side Effects:
        None.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_path} must be numeric")
    return float(value)


def _normalize_axis_numeric(*, value: int | float, field_path: str) -> int | float:
    """Normalize numeric axis scalar while preserving ints when possible.

    Why:
    - `GridBuilder` validates integer axes strictly (`axis 'window' expects integer values`).
    - The backtests API accepts `int | float` for range axis fields.
    - We must not eagerly cast int axis values to float, otherwise integer params materialize as
      floats (e.g. `5.0`) and fail validation.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/domain/specifications/grid_param_spec.py
      - src/trading/contexts/indicators/application/services/grid_builder.py

    Args:
        value: Raw numeric scalar.
        field_path: Dot-path used in deterministic error messages.
    Returns:
        int | float: Preserved int when possible, otherwise float.
    Assumptions:
        Boolean values are rejected at the API boundary.
    Raises:
        ValueError: If value is boolean or not finite.
    Side Effects:
        None.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_path} must be numeric")
    if isinstance(value, int):
        return value
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_path} must be numeric")
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _to_sorted_nested_scalar_mapping(
    *,
    values: Mapping[str, Mapping[str, BacktestScalar]] | None,
) -> dict[str, dict[str, BacktestScalar]]:
    """
    Convert nested scalar mapping into deterministic sorted plain dictionary.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        values: Optional nested scalar mapping.
    Returns:
        dict[str, dict[str, BacktestScalar]]: Deterministic sorted plain mapping.
    Assumptions:
        Keys are normalized and non-empty by application DTO constructors.
    Raises:
        None.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    nested: dict[str, dict[str, BacktestScalar]] = {}
    for indicator_id in sorted(values.keys()):
        nested[indicator_id] = {
            param_name: values[indicator_id][param_name]
            for param_name in sorted(values[indicator_id].keys())
        }
    return nested


def _to_sorted_scalar_mapping(
    values: Mapping[str, BacktestRequestScalar] | None,
) -> dict[str, BacktestRequestScalar]:
    """
    Convert scalar mapping into deterministic sorted plain dictionary.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/routes/backtests.py

    Args:
        values: Optional scalar mapping.
    Returns:
        dict[str, BacktestRequestScalar]: Deterministic sorted plain mapping.
    Assumptions:
        Keys are normalized and non-empty by upstream DTO constructors.
    Raises:
        None.
    Side Effects:
        None.
    """
    if values is None:
        return {}
    return {key: values[key] for key in sorted(values.keys())}


def _normalize_json_value(*, value: Any) -> Any:
    """
    Convert arbitrary payload node into deterministic JSON-serializable value.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - apps/api/routes/backtests.py

    Args:
        value: Arbitrary payload node.
    Returns:
        Any: JSON-serializable normalized value.
    Assumptions:
        Mapping keys are converted to strings and sorted recursively.
    Raises:
        TypeError: Propagated by `json.dumps` when unsupported nodes remain.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda key: str(key)):
            normalized[str(raw_key)] = _normalize_json_value(value=value[raw_key])
        return normalized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(value=item) for item in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


__all__ = [
    "BacktestAxisSpecRequest",
    "BacktestExplicitAxisSpecRequest",
    "BacktestIndicatorGridRequest",
    "BacktestIndicatorSelectionRequest",
    "BacktestIndicatorSelectionResponse",
    "BacktestInstrumentIdRequest",
    "BacktestInstrumentIdResponse",
    "BacktestMetricRowResponse",
    "BacktestRankingRequest",
    "BacktestRangeAxisSpecRequest",
    "BacktestReportResponse",
    "BacktestRiskGridRequest",
    "BacktestSavedOverridesRequest",
    "BacktestTemplateRequest",
    "BacktestTimeRangeRequest",
    "BacktestTradeResponse",
    "BacktestVariantPayloadResponse",
    "BacktestVariantPayloadRequest",
    "BacktestVariantResponse",
    "BacktestsPostRequest",
    "BacktestsPostResponse",
    "BacktestsVariantReportPostRequest",
    "build_backtest_run_request",
    "build_backtest_variant_report_payload",
    "build_backtest_variant_report_response",
    "build_backtest_variant_report_run_request",
    "build_backtests_post_response",
    "build_engine_params_hash",
    "build_grid_request_hash",
    "build_sha256_from_payload",
]
