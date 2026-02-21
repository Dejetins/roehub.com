from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    RunBacktestRequest,
    RunBacktestResponse,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestGridDefaultsProvider,
    BacktestStagedVariantScorer,
    BacktestStrategyReader,
    BacktestStrategySnapshot,
    CurrentUser,
)
from trading.contexts.backtest.application.services import (
    TOTAL_RETURN_METRIC_LITERAL,
    BacktestCandleTimelineBuilder,
    BacktestStagedRunnerV1,
)
from trading.contexts.backtest.application.use_cases.errors import map_backtest_exception
from trading.contexts.backtest.domain.errors import (
    BacktestForbiddenError,
    BacktestNotFoundError,
    BacktestValidationError,
)
from trading.contexts.backtest.domain.value_objects import BacktestVariantScalar
from trading.contexts.indicators.application.dto import CandleArrays, IndicatorVariantSelection
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.contexts.indicators.application.services.grid_builder import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
)
from trading.platform.errors import RoehubError


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


class _ConstantBacktestStagedScorer:
    """
    Deterministic fallback scorer returning constant `Total Return [%]` value.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
    """

    def score_variant(
        self,
        *,
        stage: str,
        candles: CandleArrays,
        indicator_selections: tuple[IndicatorVariantSelection, ...],
        signal_params: Mapping[str, Mapping[str, BacktestVariantScalar]],
        risk_params: Mapping[str, BacktestVariantScalar],
        indicator_variant_key: str,
        variant_key: str,
    ) -> Mapping[str, float]:
        """
        Return deterministic constant ranking metric for fallback staged execution.

        Args:
            stage: Stage literal (`stage_a` or `stage_b`).
            candles: Dense candles payload.
            indicator_selections: Explicit indicator selections for the variant.
            signal_params: Signal parameters for the variant.
            risk_params: Risk payload for the variant.
            indicator_variant_key: Deterministic indicator key.
            variant_key: Deterministic backtest variant key.
        Returns:
            Mapping[str, float]: Constant metric payload with `Total Return [%]` key.
        Assumptions:
            Fallback scorer is used only when no scorer port is injected by composition root.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (
            stage,
            candles,
            indicator_selections,
            signal_params,
            risk_params,
            indicator_variant_key,
            variant_key,
        )
        return {TOTAL_RETURN_METRIC_LITERAL: 0.0}


class RunBacktestUseCase:
    """
    RunBacktestUseCase — staged sync backtest orchestration for saved/ad-hoc modes.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/ports/staged_runner.py
    """

    def __init__(
        self,
        *,
        candle_feed: CandleFeed,
        indicator_compute: IndicatorCompute,
        strategy_reader: BacktestStrategyReader,
        candle_timeline_builder: BacktestCandleTimelineBuilder | None = None,
        staged_runner: BacktestStagedRunnerV1 | None = None,
        staged_scorer: BacktestStagedVariantScorer | None = None,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        warmup_bars_default: int = 200,
        top_k_default: int = 300,
        preselect_default: int = 20000,
        max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    ) -> None:
        """
        Initialize staged backtest use-case dependencies and runtime defaults.

        Docs:
          - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py

        Args:
            candle_feed: Indicators candle-feed port producing dense timeline arrays.
            indicator_compute:
                Indicators compute port used for staged grid estimate/materialization.
            strategy_reader: Backtest ACL strategy reader without owner filtering.
            candle_timeline_builder: Optional custom timeline builder (BKT-EPIC-02).
            staged_runner: Optional custom staged runner implementation.
            staged_scorer: Optional Stage A/Stage B scorer port implementation.
            defaults_provider: Optional defaults provider for compute/signal grid fallback.
            warmup_bars_default: Runtime default warmup bars.
            top_k_default: Runtime default top-k response limit.
            preselect_default: Runtime default preselect shortlist limit.
            max_variants_per_compute: Stage variants guard limit.
            max_compute_bytes_total: Stage memory guard limit.
        Returns:
            None.
        Assumptions:
            Runtime defaults come from fail-fast `configs/<env>/backtest.yaml` loader.
        Raises:
            ValueError: If dependencies are missing or scalar defaults/guards are non-positive.
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
        if max_variants_per_compute <= 0:
            raise ValueError("RunBacktestUseCase.max_variants_per_compute must be > 0")
        if max_compute_bytes_total <= 0:
            raise ValueError("RunBacktestUseCase.max_compute_bytes_total must be > 0")

        resolved_timeline_builder = candle_timeline_builder
        if resolved_timeline_builder is None:
            resolved_timeline_builder = BacktestCandleTimelineBuilder(candle_feed=candle_feed)

        self._candle_timeline_builder = resolved_timeline_builder
        self._indicator_compute = indicator_compute
        self._strategy_reader = strategy_reader
        self._staged_runner = staged_runner or BacktestStagedRunnerV1()
        self._staged_scorer = staged_scorer or _ConstantBacktestStagedScorer()
        self._defaults_provider = defaults_provider
        self._warmup_bars_default = warmup_bars_default
        self._top_k_default = top_k_default
        self._preselect_default = preselect_default
        self._max_variants_per_compute = max_variants_per_compute
        self._max_compute_bytes_total = max_compute_bytes_total

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
    ) -> RunBacktestResponse:
        """
        Execute staged sync flow and return deterministic top-k variant preview response.

        Docs:
          - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
          - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/staged_runner_v1.py
          - src/trading/contexts/backtest/application/dto/run_backtest.py
          - src/trading/contexts/backtest/application/use_cases/errors.py

        Args:
            request: Saved/ad-hoc backtest request.
            current_user: Authenticated user for ownership checks in saved mode.
        Returns:
            RunBacktestResponse: Deterministic staged response with ranked top-k variants.
        Assumptions:
            Trade execution/metrics engine is delegated to scorer port implementation.
        Raises:
            RoehubError: Canonical mapped error for validation/forbidden/not-found/conflict/
                unexpected.
        Side Effects:
            Reads candles via `CandleFeed`, resolves staged variants, and calls scorer port.
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
            staged = self._staged_runner.run(
                template=resolved.template,
                candles=timeline.candles,
                preselect=resolved.preselect,
                top_k=resolved.top_k,
                indicator_compute=self._indicator_compute,
                scorer=self._staged_scorer,
                defaults_provider=self._defaults_provider,
                max_variants_per_compute=self._max_variants_per_compute,
                max_compute_bytes_total=self._max_compute_bytes_total,
            )

            return RunBacktestResponse(
                mode=resolved.mode,
                instrument_id=resolved.template.instrument_id,
                timeframe=resolved.template.timeframe,
                strategy_id=resolved.strategy_id,
                warmup_bars=resolved.warmup_bars,
                top_k=resolved.top_k,
                preselect=resolved.preselect,
                variants=staged.variants,
                total_indicator_compute_calls=staged.indicator_estimate_calls,
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
            RunBacktestTemplate: Template equivalent used by staged flow.
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
            signal_grids=snapshot.signal_grids,
            risk_grid=snapshot.risk_grid,
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
