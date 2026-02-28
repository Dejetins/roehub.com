from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
    BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1,
    BacktestRankingConfig,
    RunBacktestRequest,
    RunBacktestResponse,
    RunBacktestSavedOverrides,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestGridDefaultsProvider,
    BacktestStagedVariantMetricScorer,
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
from trading.contexts.backtest.application.services.numba_runtime_v1 import (
    apply_backtest_numba_threads,
)
from trading.contexts.backtest.application.services.run_control_v1 import BacktestRunControlV1
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
from trading.contexts.indicators.domain.specifications import GridParamSpec
from trading.platform.errors import RoehubError

_DEFAULT_FEE_PCT_BY_MARKET_ID = {
    1: 0.075,
    2: 0.1,
    3: 0.075,
    4: 0.1,
}
_DEFAULT_MAX_NUMBA_THREADS = max(1, os.cpu_count() or 1)
MetricScorerV1 = BacktestStagedVariantMetricScorer | BacktestStagedVariantScorer


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
    top_trades_n: int
    ranking: BacktestRankingConfig


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
        staged_scorer: MetricScorerV1 | None = None,
        defaults_provider: BacktestGridDefaultsProvider | None = None,
        warmup_bars_default: int = 200,
        top_k_default: int = 300,
        preselect_default: int = 20000,
        top_trades_n_default: int = 3,
        ranking_primary_metric_default: str = BACKTEST_RANKING_PRIMARY_METRIC_DEFAULT_V1,
        ranking_secondary_metric_default: str | None = (
            BACKTEST_RANKING_SECONDARY_METRIC_DEFAULT_V1
        ),
        configurable_ranking_enabled: bool = True,
        init_cash_quote_default: float = 10000.0,
        fixed_quote_default: float = 100.0,
        safe_profit_percent_default: float = 30.0,
        slippage_pct_default: float = 0.01,
        fee_pct_default_by_market_id: Mapping[int, float] | None = None,
        max_variants_per_compute: int = MAX_VARIANTS_PER_COMPUTE_DEFAULT,
        max_compute_bytes_total: int = MAX_COMPUTE_BYTES_TOTAL_DEFAULT,
        max_numba_threads: int = _DEFAULT_MAX_NUMBA_THREADS,
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
            ranking_primary_metric_default:
                Runtime default for ranking primary metric literal.
            ranking_secondary_metric_default:
                Runtime default for ranking secondary metric literal.
            configurable_ranking_enabled:
                Feature-flag guard for configurable ranking behavior rollout.
            init_cash_quote_default: Runtime default initial strategy quote balance.
            fixed_quote_default: Runtime default fixed quote notional for `fixed_quote`.
            safe_profit_percent_default: Runtime default profit-lock percent.
            slippage_pct_default: Runtime default slippage percent.
            fee_pct_default_by_market_id: Runtime default fee mapping by market id.
            max_variants_per_compute: Stage variants guard limit.
            max_compute_bytes_total: Stage memory guard limit.
            max_numba_threads:
                Runtime CPU knob for backtest runs mapped to maximum Numba threads.
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
            raise ValueError("RunBacktestUseCase.safe_profit_percent_default must be in [0, 100]")
        if slippage_pct_default < 0.0:
            raise ValueError("RunBacktestUseCase.slippage_pct_default must be >= 0")
        if max_variants_per_compute <= 0:
            raise ValueError("RunBacktestUseCase.max_variants_per_compute must be > 0")
        if max_compute_bytes_total <= 0:
            raise ValueError("RunBacktestUseCase.max_compute_bytes_total must be > 0")
        if max_numba_threads <= 0:
            raise ValueError("RunBacktestUseCase.max_numba_threads must be > 0")
        if not isinstance(configurable_ranking_enabled, bool):
            raise ValueError("RunBacktestUseCase.configurable_ranking_enabled must be bool")

        ranking_defaults = BacktestRankingConfig(
            primary_metric=ranking_primary_metric_default,
            secondary_metric=ranking_secondary_metric_default,
        )

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
        self._ranking_defaults = ranking_defaults
        self._configurable_ranking_enabled = configurable_ranking_enabled
        self._init_cash_quote_default = init_cash_quote_default
        self._fixed_quote_default = fixed_quote_default
        self._safe_profit_percent_default = safe_profit_percent_default
        self._slippage_pct_default = slippage_pct_default
        self._fee_pct_default_by_market_id = _normalize_fee_defaults(
            values=fee_pct_default_by_market_id
        )
        self._max_variants_per_compute = max_variants_per_compute
        self._max_compute_bytes_total = max_compute_bytes_total
        self._max_numba_threads = max_numba_threads

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        run_control: BacktestRunControlV1 | None = None,
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
            run_control: Optional cooperative cancellation/deadline control object.
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

            apply_backtest_numba_threads(max_numba_threads=self._max_numba_threads)
            if run_control is not None:
                run_control.raise_if_cancelled(stage="stage_a")
            resolved = self._resolve_run_context(request=request, current_user=current_user)
            timeline = self._candle_timeline_builder.build(
                market_id=resolved.template.instrument_id.market_id,
                symbol=resolved.template.instrument_id.symbol,
                timeframe=resolved.template.timeframe,
                requested_time_range=request.time_range,
                warmup_bars=resolved.warmup_bars,
            )
            if run_control is not None:
                run_control.raise_if_cancelled(stage="stage_a")
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
                ranking=resolved.ranking,
                defaults_provider=self._defaults_provider,
                max_variants_per_compute=self._max_variants_per_compute,
                max_compute_bytes_total=self._max_compute_bytes_total,
                requested_time_range=request.time_range,
                top_trades_n=resolved.top_trades_n,
                run_control=run_control,
            )

            return RunBacktestResponse(
                mode=resolved.mode,
                instrument_id=resolved.template.instrument_id,
                timeframe=resolved.template.timeframe,
                strategy_id=resolved.strategy_id,
                warmup_bars=resolved.warmup_bars,
                top_k=resolved.top_k,
                preselect=resolved.preselect,
                top_trades_n=resolved.top_trades_n,
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
        top_trades_n = self._resolve_with_default(
            value=request.top_trades_n,
            default=self._top_trades_n_default,
        )
        ranking = self._resolve_ranking_config(request=request)
        if request.top_trades_n is not None and top_trades_n > top_k:
            raise BacktestValidationError("Backtest request top_trades_n must be <= top_k")
        if top_trades_n > top_k:
            top_trades_n = top_k

        if request.strategy_id is not None:
            snapshot = self._strategy_reader.load_any(strategy_id=request.strategy_id)
            base_template = self._template_from_snapshot(
                strategy_id=request.strategy_id,
                snapshot=snapshot,
                current_user=current_user,
            )
            template = self._apply_saved_overrides(
                base_template=base_template,
                overrides=request.overrides,
            )
            return _ResolvedRunContext(
                mode="saved",
                strategy_id=request.strategy_id,
                template=template,
                warmup_bars=warmup_bars,
                top_k=top_k,
                preselect=preselect,
                top_trades_n=top_trades_n,
                ranking=ranking,
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
            top_trades_n=top_trades_n,
            ranking=ranking,
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
            direction_mode=snapshot.direction_mode,
            sizing_mode=snapshot.sizing_mode,
            risk_params=snapshot.risk_params,
            execution_params=snapshot.execution_params,
        )

    def _apply_saved_overrides(
        self,
        *,
        base_template: RunBacktestTemplate,
        overrides: RunBacktestSavedOverrides | None,
    ) -> RunBacktestTemplate:
        """
        Merge optional saved-mode overrides over loaded snapshot template deterministically.

        Args:
            base_template: Template resolved from saved strategy snapshot.
            overrides: Optional saved-mode overrides from request payload.
        Returns:
            RunBacktestTemplate: Effective template used for staged run execution.
        Assumptions:
            Ownership/deletion checks already passed before this merge step.
        Raises:
            ValueError: Propagated from template/override value-object validation.
        Side Effects:
            None.
        """
        if overrides is None:
            return base_template

        direction_mode = (
            overrides.direction_mode
            if overrides.direction_mode is not None
            else base_template.direction_mode
        )
        sizing_mode = (
            overrides.sizing_mode
            if overrides.sizing_mode is not None
            else base_template.sizing_mode
        )
        signal_grids = _merge_signal_grids(
            base=base_template.signal_grids or {},
            updates=overrides.signal_grids or {},
        )
        risk_params = _merge_scalar_mappings(
            base=base_template.risk_params or {},
            updates=overrides.risk_params or {},
        )
        execution_params = _merge_scalar_mappings(
            base=base_template.execution_params or {},
            updates=overrides.execution_params or {},
        )
        risk_grid = (
            overrides.risk_grid if overrides.risk_grid is not None else base_template.risk_grid
        )

        return RunBacktestTemplate(
            instrument_id=base_template.instrument_id,
            timeframe=base_template.timeframe,
            indicator_grids=base_template.indicator_grids,
            indicator_selections=base_template.indicator_selections,
            signal_grids=signal_grids,
            risk_grid=risk_grid,
            direction_mode=direction_mode,
            sizing_mode=sizing_mode,
            risk_params=risk_params,
            execution_params=execution_params,
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

    def _resolve_ranking_config(self, *, request: RunBacktestRequest) -> BacktestRankingConfig:
        """
        Resolve effective ranking config from request override, runtime defaults, and feature flag.

        Args:
            request: Backtest request payload.
        Returns:
            BacktestRankingConfig: Effective deterministic ranking config.
        Assumptions:
            DTO validation already normalized metric literals and duplicate checks.
        Raises:
            ValueError: If runtime ranking defaults are invalid.
        Side Effects:
            None.
        """
        if not self._configurable_ranking_enabled:
            return BacktestRankingConfig()
        if request.ranking is not None:
            return request.ranking
        return self._ranking_defaults

    def _resolve_staged_scorer(
        self,
        *,
        template: RunBacktestTemplate,
        target_slice: slice,
    ) -> MetricScorerV1:
        """
        Resolve scorer for current execution, building default close-fill scorer when absent.

        Args:
            template: Resolved run template containing direction/sizing/execution settings.
            target_slice: Trading/reporting target slice inside warmup-inclusive timeline.
        Returns:
            MetricScorerV1: Scorer used by staged runner.
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
            max_compute_bytes_total=self._max_compute_bytes_total,
        )


def _merge_scalar_mappings(
    *,
    base: Mapping[str, int | float | str | bool | None],
    updates: Mapping[str, int | float | str | bool | None],
) -> Mapping[str, int | float | str | bool | None]:
    """
    Merge scalar mappings deterministically with update precedence and sorted keys.

    Args:
        base: Base scalar payload mapping.
        updates: Override scalar payload mapping.
    Returns:
        Mapping[str, int | float | str | bool | None]: Immutable merged scalar mapping.
    Assumptions:
        Input mappings use non-empty string keys and JSON-compatible scalar values.
    Raises:
        ValueError: If one key is blank after normalization.
    Side Effects:
        None.
    """
    merged: dict[str, int | float | str | bool | None] = {}
    for raw_key in sorted(base.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("saved-mode scalar override key must be non-empty")
        merged[key] = base[raw_key]
    for raw_key in sorted(updates.keys(), key=lambda key: str(key).strip()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("saved-mode scalar override key must be non-empty")
        merged[key] = updates[raw_key]
    return MappingProxyType(merged)


def _merge_signal_grids(
    *,
    base: Mapping[str, Mapping[str, GridParamSpec]],
    updates: Mapping[str, Mapping[str, GridParamSpec]],
) -> Mapping[str, Mapping[str, GridParamSpec]]:
    """
    Merge nested signal-grid mappings deterministically by indicator id and param key.

    Args:
        base: Base signal-grid mapping loaded from saved strategy snapshot.
        updates: Saved-mode signal-grid overrides from request payload.
    Returns:
        Mapping[str, Mapping[str, object]]: Immutable merged nested mapping.
    Assumptions:
        Values are GridParamSpec-compatible objects validated by template DTO.
    Raises:
        ValueError: If one indicator id or param key is blank.
    Side Effects:
        None.
    """
    merged: dict[str, Mapping[str, GridParamSpec]] = {}
    indicator_ids = set(base.keys()) | set(updates.keys())
    for raw_indicator_id in sorted(indicator_ids, key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("saved-mode signal override indicator_id must be non-empty")
        merged_params: dict[str, GridParamSpec] = {}
        base_params = base.get(raw_indicator_id, {})
        updates_params = updates.get(raw_indicator_id, {})
        for raw_param_name in sorted(base_params.keys(), key=lambda key: str(key).strip().lower()):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("saved-mode signal override param key must be non-empty")
            merged_params[param_name] = base_params[raw_param_name]
        for raw_param_name in sorted(
            updates_params.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError("saved-mode signal override param key must be non-empty")
            merged_params[param_name] = updates_params[raw_param_name]
        merged[indicator_id] = MappingProxyType(merged_params)
    return MappingProxyType(merged)


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
