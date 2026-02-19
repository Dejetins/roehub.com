from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BacktestVariantPreview,
    RunBacktestRequest,
    RunBacktestResponse,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestStrategyReader,
    BacktestStrategySnapshot,
    CurrentUser,
)
from trading.contexts.backtest.application.services import BacktestCandleTimelineBuilder
from trading.contexts.backtest.application.use_cases.errors import map_backtest_exception
from trading.contexts.backtest.domain.errors import (
    BacktestForbiddenError,
    BacktestNotFoundError,
    BacktestValidationError,
)
from trading.contexts.backtest.domain.value_objects import (
    BacktestVariantIdentity,
    build_backtest_variant_key_v1,
)
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    build_variant_key_v1,
)
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import InstrumentId


@dataclass(frozen=True, slots=True)
class _ResolvedRunContext:
    """
    Internal resolved request context used by run use-case orchestration.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
    """

    mode: str
    strategy_id: UUID | None
    template: RunBacktestTemplate
    warmup_bars: int
    top_k: int
    preselect: int


class RunBacktestUseCase:
    """
    RunBacktestUseCase — BKT-EPIC-01 orchestration skeleton for saved/ad-hoc modes.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
    """

    def __init__(
        self,
        *,
        candle_feed: CandleFeed,
        indicator_compute: IndicatorCompute,
        strategy_reader: BacktestStrategyReader,
        candle_timeline_builder: BacktestCandleTimelineBuilder | None = None,
        warmup_bars_default: int = 200,
        top_k_default: int = 300,
        preselect_default: int = 20000,
    ) -> None:
        """
        Initialize backtest use-case dependencies and runtime defaults.

        Docs:
          - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
          - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
          - src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

        Args:
            candle_feed: Indicators candle-feed port producing dense timeline arrays.
            indicator_compute: Indicators compute port.
            strategy_reader: Backtest ACL strategy reader without owner filtering.
            candle_timeline_builder: Optional custom timeline builder (BKT-EPIC-02).
            warmup_bars_default: Runtime default warmup bars.
            top_k_default: Runtime default top-k response limit.
            preselect_default: Runtime default preselect shortlist limit.
        Returns:
            None.
        Assumptions:
            Runtime defaults come from fail-fast `configs/<env>/backtest.yaml` loader.
        Raises:
            ValueError: If dependencies are missing or defaults are non-positive.
        Side Effects:
            None.
        """
        if candle_feed is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestUseCase requires candle_feed")
        if indicator_compute is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestUseCase requires indicator_compute")
        if strategy_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestUseCase requires strategy_reader")
        if warmup_bars_default <= 0:
            raise ValueError("RunBacktestUseCase.warmup_bars_default must be > 0")
        if top_k_default <= 0:
            raise ValueError("RunBacktestUseCase.top_k_default must be > 0")
        if preselect_default <= 0:
            raise ValueError("RunBacktestUseCase.preselect_default must be > 0")

        resolved_timeline_builder = candle_timeline_builder
        if resolved_timeline_builder is None:
            resolved_timeline_builder = BacktestCandleTimelineBuilder(candle_feed=candle_feed)

        self._candle_timeline_builder = resolved_timeline_builder
        self._indicator_compute = indicator_compute
        self._strategy_reader = strategy_reader
        self._warmup_bars_default = warmup_bars_default
        self._top_k_default = top_k_default
        self._preselect_default = preselect_default

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
    ) -> RunBacktestResponse:
        """
        Execute backtest skeleton flow and return deterministic variant preview response.

        Docs:
          - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
          - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
          - src/trading/contexts/backtest/application/dto/run_backtest.py
          - src/trading/contexts/indicators/application/dto/compute_request.py

        Args:
            request: Saved/ad-hoc backtest request.
            current_user: Authenticated user for ownership checks in saved mode.
        Returns:
            RunBacktestResponse: Deterministic skeleton response with one variant identity.
        Assumptions:
            EPIC-01 intentionally runs minimal compute orchestration and does not execute trades.
        Raises:
            RoehubError: Canonical mapped error for validation/forbidden/not-found/conflict/
                unexpected.
        Side Effects:
            Reads canonical-based candles via `CandleFeed`, invokes indicators compute, and
                loads saved strategy snapshot when needed.
        """
        try:
            if request is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError("RunBacktestUseCase.execute requires request")
            if current_user is None:  # type: ignore[truthy-bool]
                raise BacktestValidationError("RunBacktestUseCase.execute requires current_user")

            resolved = self._resolve_run_context(request=request, current_user=current_user)
            timeline = self._candle_timeline_builder.build(
                market_id=resolved.template.instrument_id.market_id,
                symbol=resolved.template.instrument_id.symbol,
                timeframe=resolved.template.timeframe,
                requested_time_range=request.time_range,
                warmup_bars=resolved.warmup_bars,
            )
            compute_calls = self._run_indicator_compute(
                candles=timeline.candles,
                template=resolved.template,
                preselect=resolved.preselect,
            )
            variant_preview = self._build_variant_preview(template=resolved.template)

            return RunBacktestResponse(
                mode=resolved.mode,
                instrument_id=resolved.template.instrument_id,
                timeframe=resolved.template.timeframe,
                strategy_id=resolved.strategy_id,
                warmup_bars=resolved.warmup_bars,
                top_k=resolved.top_k,
                preselect=resolved.preselect,
                variants=(variant_preview,),
                total_indicator_compute_calls=compute_calls,
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_backtest_exception(error=error) from error

    def _resolve_run_context(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
    ) -> _ResolvedRunContext:
        """
        Resolve final run-mode/template/default values before external port calls.

        Args:
            request: Input backtest request.
            current_user: Authenticated user for saved-mode ownership checks.
        Returns:
            _ResolvedRunContext: Fully resolved run context.
        Assumptions:
            Request mode exclusivity is already validated by DTO invariants.
        Raises:
            BacktestNotFoundError: If saved strategy is missing/deleted.
            BacktestForbiddenError: If saved strategy does not belong to current user.
            BacktestValidationError: If resolved template cannot be built.
        Side Effects:
            Reads saved strategy snapshot through ACL port in saved mode.
        """
        warmup_bars = self._resolve_with_default(
            value=request.warmup_bars,
            default=self._warmup_bars_default,
        )
        top_k = self._resolve_with_default(
            value=request.top_k,
            default=self._top_k_default,
        )
        preselect = self._resolve_with_default(
            value=request.preselect,
            default=self._preselect_default,
        )

        if request.strategy_id is not None:
            snapshot = self._strategy_reader.load_any(strategy_id=request.strategy_id)
            template = self._template_from_snapshot(
                strategy_id=request.strategy_id,
                snapshot=snapshot,
                current_user=current_user,
            )
            return _ResolvedRunContext(
                mode="saved",
                strategy_id=request.strategy_id,
                template=template,
                warmup_bars=warmup_bars,
                top_k=top_k,
                preselect=preselect,
            )

        if request.template is None:  # pragma: no cover - guarded by request DTO invariant
            raise BacktestValidationError(
                "RunBacktestRequest.template is required for template mode"
            )

        return _ResolvedRunContext(
            mode="template",
            strategy_id=None,
            template=request.template,
            warmup_bars=warmup_bars,
            top_k=top_k,
            preselect=preselect,
        )

    def _template_from_snapshot(
        self,
        *,
        strategy_id: UUID,
        snapshot: BacktestStrategySnapshot | None,
        current_user: CurrentUser,
    ) -> RunBacktestTemplate:
        """
        Convert saved strategy snapshot into template after ownership/deletion checks.

        Args:
            strategy_id: Requested saved strategy identifier.
            snapshot: Loaded snapshot or `None`.
            current_user: Authenticated principal.
        Returns:
            RunBacktestTemplate: Template equivalent used by compute flow.
        Assumptions:
            Missing and deleted snapshots are hidden behind one `not_found` contract.
        Raises:
            BacktestNotFoundError: If snapshot is missing or soft-deleted.
            BacktestForbiddenError: If snapshot owner differs from current user.
        Side Effects:
            None.
        """
        if snapshot is None or snapshot.is_deleted:
            raise BacktestNotFoundError(strategy_id=strategy_id)
        if snapshot.user_id != current_user.user_id:
            raise BacktestForbiddenError(strategy_id=strategy_id)

        return RunBacktestTemplate(
            instrument_id=snapshot.instrument_id,
            timeframe=snapshot.timeframe,
            indicator_grids=snapshot.indicator_grids,
            indicator_selections=snapshot.indicator_selections,
        )

    def _run_indicator_compute(
        self,
        *,
        candles: CandleArrays,
        template: RunBacktestTemplate,
        preselect: int,
    ) -> int:
        """
        Invoke indicators compute for each template indicator block in deterministic order.

        Args:
            candles: Dense candle arrays from candle-feed port.
            template: Resolved run template.
            preselect: Current preselect limit used as compute variants guard in skeleton.
        Returns:
            int: Number of compute invocations performed.
        Assumptions:
            EPIC-01 validates orchestration wiring, not full staged variant execution semantics.
        Raises:
            GridValidationError: Propagated from indicators compute for invalid grid payloads.
            MissingInputSeriesError: Propagated when feed cannot satisfy series requirements.
        Side Effects:
            Calls indicators compute port one time per template indicator grid.
        """
        compute_calls = 0
        for indicator_grid in template.indicator_grids:
            self._indicator_compute.compute(
                ComputeRequest(
                    candles=candles,
                    grid=indicator_grid,
                    max_variants_guard=preselect,
                )
            )
            compute_calls += 1
        return compute_calls

    def _build_variant_preview(self, *, template: RunBacktestTemplate) -> BacktestVariantPreview:
        """
        Build deterministic variant preview with `variant_index` and composed `variant_key` v1.

        Args:
            template: Resolved template payload.
        Returns:
            BacktestVariantPreview: Deterministic one-variant skeleton identity.
        Assumptions:
            Indicator key semantics are delegated to indicators `build_variant_key_v1`.
        Raises:
            ValueError: If key canonicalization inputs are invalid.
        Side Effects:
            None.
        """
        indicator_variant_key = build_variant_key_v1(
            instrument_id=self._instrument_id_literal(instrument_id=template.instrument_id),
            timeframe=template.timeframe.code,
            indicators=template.indicator_selections,
        )
        variant_key = build_backtest_variant_key_v1(
            indicator_variant_key=indicator_variant_key,
            direction_mode=template.direction_mode,
            sizing_mode=template.sizing_mode,
            risk_params=template.risk_params,
            execution_params=template.execution_params,
        )
        identity = BacktestVariantIdentity(
            variant_index=0,
            variant_key=variant_key,
        )
        return BacktestVariantPreview(
            variant_index=identity.variant_index,
            variant_key=identity.variant_key,
            indicator_variant_key=indicator_variant_key,
        )

    def _resolve_with_default(self, *, value: int | None, default: int) -> int:
        """
        Resolve optional positive integer override against runtime default.

        Args:
            value: Optional override from request DTO.
            default: Runtime default loaded from config.
        Returns:
            int: Effective positive integer value.
        Assumptions:
            Defaults are validated in use-case constructor.
        Raises:
            BacktestValidationError: If provided override is non-positive.
        Side Effects:
            None.
        """
        if value is None:
            return default
        if value <= 0:
            raise BacktestValidationError("Backtest request override values must be > 0")
        return value

    def _instrument_id_literal(self, *, instrument_id: InstrumentId) -> str:
        """
        Build deterministic instrument literal for indicators variant-key builder.

        Args:
            instrument_id: Shared-kernel instrument identity.
        Returns:
            str: Canonical `<market_id>:<symbol>` string.
        Assumptions:
            `market_id` and `symbol` value objects are already validated.
        Raises:
            None.
        Side Effects:
            None.
        """
        return f"{instrument_id.market_id.value}:{instrument_id.symbol.value}"
