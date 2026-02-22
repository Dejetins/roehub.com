from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
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
    BacktestCandleTimelineBuilder,
    BacktestStagedRunnerV1,
    CloseFillBacktestStagedScorerV1,
)
from trading.contexts.backtest.application.use_cases.errors import map_backtest_exception
from trading.contexts.backtest.domain.errors import (
    BacktestForbiddenError,
    BacktestNotFoundError,
    BacktestValidationError,
)
from trading.contexts.indicators.application.ports.compute import IndicatorCompute
from trading.contexts.indicators.application.ports.feeds import CandleFeed
from trading.contexts.indicators.application.services.grid_builder import (
    MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
    MAX_VARIANTS_PER_COMPUTE_DEFAULT,
)
from trading.platform.errors import RoehubError

_DEFAULT_FEE_PCT_BY_MARKET_ID = {
    1: 0.075,
    2: 0.1,
    3: 0.075,
    4: 0.1,
}


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
        top_trades_n_default: int = 3,
        init_cash_quote_default: float = 10000.0,
        fixed_quote_default: float = 100.0,
        safe_profit_percent_default: float = 30.0,
        slippage_pct_default: float = 0.01,
        fee_pct_default_by_market_id: Mapping[int, float] | None = None,
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
            top_trades_n_default: Runtime default number of variants with full trades payload.
            init_cash_quote_default: Runtime default initial strategy quote balance.
            fixed_quote_default: Runtime default fixed quote notional for `fixed_quote`.
            safe_profit_percent_default: Runtime default profit-lock percent.
            slippage_pct_default: Runtime default slippage percent.
            fee_pct_default_by_market_id: Runtime default fee mapping by market id.
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
        if top_trades_n_default <= 0:
            raise ValueError("RunBacktestUseCase.top_trades_n_default must be > 0")
        if init_cash_quote_default <= 0.0:
            raise ValueError("RunBacktestUseCase.init_cash_quote_default must be > 0")
        if fixed_quote_default <= 0.0:
            raise ValueError("RunBacktestUseCase.fixed_quote_default must be > 0")
        if safe_profit_percent_default < 0.0 or safe_profit_percent_default > 100.0:
            raise ValueError(
                "RunBacktestUseCase.safe_profit_percent_default must be in [0, 100]"
            )
        if slippage_pct_default < 0.0:
            raise ValueError("RunBacktestUseCase.slippage_pct_default must be >= 0")
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
        self._staged_scorer = staged_scorer
        self._defaults_provider = defaults_provider
        self._warmup_bars_default = warmup_bars_default
        self._top_k_default = top_k_default
        self._preselect_default = preselect_default
        self._top_trades_n_default = top_trades_n_default
        self._init_cash_quote_default = init_cash_quote_default
        self._fixed_quote_default = fixed_quote_default
        self._safe_profit_percent_default = safe_profit_percent_default
        self._slippage_pct_default = slippage_pct_default
        self._fee_pct_default_by_market_id = _normalize_fee_defaults(
            values=fee_pct_default_by_market_id
        )
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
            resolved_scorer = self._resolve_staged_scorer(
                template=resolved.template,
                target_slice=timeline.target_slice,
            )
            staged = self._staged_runner.run(
                template=resolved.template,
                candles=timeline.candles,
                preselect=resolved.preselect,
                top_k=resolved.top_k,
                indicator_compute=self._indicator_compute,
                scorer=resolved_scorer,
                defaults_provider=self._defaults_provider,
                max_variants_per_compute=self._max_variants_per_compute,
                max_compute_bytes_total=self._max_compute_bytes_total,
                requested_time_range=request.time_range,
                top_trades_n=self._top_trades_n_default,
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

    def _resolve_staged_scorer(
        self,
        *,
        template: RunBacktestTemplate,
        target_slice: slice,
    ) -> BacktestStagedVariantScorer:
        """
        Resolve scorer for current execution, building default close-fill scorer when absent.

        Args:
            template: Resolved run template containing direction/sizing/execution settings.
            target_slice: Trading/reporting target slice inside warmup-inclusive timeline.
        Returns:
            BacktestStagedVariantScorer: Scorer used by staged runner.
        Assumptions:
            Injected scorer takes precedence over default close-fill scorer composition.
        Raises:
            ValueError: Propagated from default scorer constructor on invalid settings.
        Side Effects:
            None.
        """
        if self._staged_scorer is not None:
            return self._staged_scorer

        return CloseFillBacktestStagedScorerV1(
            indicator_compute=self._indicator_compute,
            direction_mode=template.direction_mode,
            sizing_mode=template.sizing_mode,
            execution_params=template.execution_params or {},
            market_id=template.instrument_id.market_id.value,
            target_slice=target_slice,
            init_cash_quote_default=self._init_cash_quote_default,
            fixed_quote_default=self._fixed_quote_default,
            safe_profit_percent_default=self._safe_profit_percent_default,
            slippage_pct_default=self._slippage_pct_default,
            fee_pct_default_by_market_id=self._fee_pct_default_by_market_id,
            max_variants_guard=self._max_variants_per_compute,
        )


def _normalize_fee_defaults(
    *,
    values: Mapping[int, float] | None,
) -> Mapping[int, float]:
    """
    Normalize and validate runtime fee-default mapping by market id.

    Args:
        values: Optional mapping `market_id -> fee_pct`.
    Returns:
        Mapping[int, float]: Immutable normalized mapping.
    Assumptions:
        Fee values are human percent units and must be non-negative.
    Raises:
        ValueError: If one market id/fee value is invalid or mapping is empty.
    Side Effects:
        None.
    """
    source = _DEFAULT_FEE_PCT_BY_MARKET_ID if values is None else values
    normalized: dict[int, float] = {}
    for raw_market_id in sorted(source.keys()):
        market_id = int(raw_market_id)
        fee_pct = float(source[raw_market_id])
        if market_id <= 0:
            raise ValueError("fee_pct_default_by_market_id keys must be > 0")
        if fee_pct < 0.0:
            raise ValueError("fee_pct_default_by_market_id values must be >= 0")
        normalized[market_id] = fee_pct

    if len(normalized) == 0:
        raise ValueError("fee_pct_default_by_market_id must be non-empty")
    return MappingProxyType(normalized)
