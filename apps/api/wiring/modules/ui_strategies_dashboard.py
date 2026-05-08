from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Literal, Mapping
from uuid import UUID

from fastapi import APIRouter

from apps.api.dto.ui_strategies_dashboard import (
    RefreshStatus,
    SourceStatus,
    StrategyBestWorstDaysResponse,
    StrategyBreakdownPanelResponse,
    StrategyBreakdownRowResponse,
    StrategyChartResponse,
    StrategyDashboardActionsResponse,
    StrategyDashboardFooterStatusResponse,
    StrategyDashboardRefreshControlResponse,
    StrategyDashboardResponse,
    StrategyDashboardSelectedStrategyResponse,
    StrategyDashboardSelectorFiltersResponse,
    StrategyDashboardSelectorResponse,
    StrategyDashboardSelectorRowResponse,
    StrategyDashboardSelectorTotalsResponse,
    StrategyDashboardSourceResponse,
    StrategyHourlyResultResponse,
    StrategyHourlyResultsResponse,
    StrategyMetricGridResponse,
    StrategyMetricResponse,
    StrategyMonthlyStatsResponse,
    StrategySeriesPanelResponse,
    StrategySymbolResultResponse,
    StrategySymbolResultsResponse,
    StrategyTradesResponse,
)
from apps.api.routes.ui_strategies_dashboard import (
    build_ui_strategies_dashboard_router as build_ui_strategies_dashboard_api_router,
)
from apps.api.wiring.modules.strategy import (
    _build_repositories,
    _resolve_strategy_runtime_settings,
    is_strategy_api_enabled,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.application.ports.repositories import (
    StrategyRepository,
    StrategyRunRepository,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategyRun
from trading.shared_kernel.primitives import UserId

_DEFAULT_REFRESH_INTERVAL_SECONDS = 15
_MINIMUM_MANUAL_REFRESH_SECONDS = 10
_STRATEGY_SELECTOR_LIMIT = 25
_CHART_MAX_POINTS = 600
_SERIES_MAX_POINTS = 600
_TRADES_LIMIT = 50

_STAT_SOURCE = "strategy_stat_projections"
_CANDLE_SOURCE = "market_candles"
_TRADES_SOURCE = "execution_fills"
_EVENTS_SOURCE = "strategy_events"


@dataclass(frozen=True, slots=True)
class _RefreshDecision:
    status: RefreshStatus
    next_allowed_refresh_at: datetime | None
    retry_after_seconds: int | None


class StrategyDashboardManualRefreshLimiter:
    def __init__(self, *, interval_seconds: int = _MINIMUM_MANUAL_REFRESH_SECONDS) -> None:
        if interval_seconds < 1:
            raise ValueError(
                "StrategyDashboardManualRefreshLimiter requires positive interval_seconds"
            )
        self._interval = timedelta(seconds=interval_seconds)
        self._next_allowed_by_user: dict[str, datetime] = {}

    def resolve(
        self,
        *,
        user_id: str,
        requested_at: datetime,
        refresh: Literal["initial", "auto", "manual"],
    ) -> _RefreshDecision:
        if refresh != "manual":
            return _RefreshDecision(
                status="fresh",
                next_allowed_refresh_at=self._next_allowed_by_user.get(user_id),
                retry_after_seconds=None,
            )

        next_allowed = self._next_allowed_by_user.get(user_id)
        if next_allowed is not None and requested_at < next_allowed:
            return _RefreshDecision(
                status="rate_limited",
                next_allowed_refresh_at=next_allowed,
                retry_after_seconds=max(1, ceil((next_allowed - requested_at).total_seconds())),
            )

        next_allowed = requested_at + self._interval
        self._next_allowed_by_user[user_id] = next_allowed
        return _RefreshDecision(
            status="fresh",
            next_allowed_refresh_at=next_allowed,
            retry_after_seconds=None,
        )


class StrategyDashboardQueryService:
    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository | None,
        run_repository: StrategyRunRepository | None,
        refresh_limiter: StrategyDashboardManualRefreshLimiter | None = None,
    ) -> None:
        self._strategy_repository = strategy_repository
        self._run_repository = run_repository
        self._refresh_limiter = refresh_limiter or StrategyDashboardManualRefreshLimiter()

    def get_dashboard(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: str | None,
        state: Literal["active", "all"],
        cursor: str | None,
        refresh: Literal["initial", "auto", "manual"],
    ) -> StrategyDashboardResponse:
        generated_at = datetime.now(UTC)
        refresh_decision = self._refresh_limiter.resolve(
            user_id=str(principal.user_id),
            requested_at=generated_at,
            refresh=refresh,
        )
        strategies, runs_by_strategy_id, dynamic_sources = self._load_strategy_state(
            user_id=principal.user_id,
            generated_at=generated_at,
        )
        selected_strategy = _select_strategy(strategies=strategies, strategy_id=strategy_id)
        selected_run = (
            runs_by_strategy_id.get(selected_strategy.strategy_id)
            if selected_strategy is not None
            else None
        )
        sources = [
            *dynamic_sources,
            _source(
                name=_CANDLE_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail="bounded strategy candle/trade chart projection is not migrated yet",
            ),
            _source(
                name=_STAT_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail="monthly, drawdown, equity, hourly and symbol statistics need projections",
            ),
            _source(
                name=_TRADES_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail="bounded strategy fill/trade detail read-model is not migrated yet",
            ),
            _source(
                name=_EVENTS_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail="strategy events are not exposed as a dashboard panel read-model yet",
            ),
            _source(
                name="exchange_account",
                status="unavailable",
                generated_at=generated_at,
                detail="exchange account snapshots are not called from the browser",
            ),
        ]
        effective_refresh_status = _resolve_refresh_status(
            refresh_decision=refresh_decision,
            sources=sources,
        )
        selected_symbol = _selected_symbol(strategy=selected_strategy)

        return StrategyDashboardResponse(
            generated_at=generated_at,
            refresh_status=effective_refresh_status,
            next_allowed_refresh_at=refresh_decision.next_allowed_refresh_at,
            retry_after_seconds=refresh_decision.retry_after_seconds,
            sources=sources,
            selected_strategy=_build_selected_strategy(
                strategy=selected_strategy,
                run=selected_run,
                generated_at=generated_at,
                requested_strategy_id=strategy_id,
            ),
            strategy_selector=_build_strategy_selector(
                strategies=strategies,
                runs_by_strategy_id=runs_by_strategy_id,
                selected_strategy=selected_strategy,
                state=state,
                cursor=cursor,
            ),
            chart=_build_unavailable_chart(symbol=selected_symbol),
            metric_grid=_build_metric_grid(),
            monthly_stats=_build_monthly_stats(),
            long_short=_build_long_short(),
            risk_execution=_build_risk_execution(),
            drawdown=_build_unavailable_series(title="drawdown"),
            equity_curve=_build_unavailable_series(title="equity_curve"),
            best_worst_days=_build_best_worst_days(),
            hourly_results=_build_hourly_results(),
            trades=_build_trades(),
            symbol_results=_build_symbol_results(symbol=selected_symbol),
            footer_status=StrategyDashboardFooterStatusResponse(
                connection_status="degraded" if _has_degraded_sources(sources) else "ok",
                data_status="degraded" if _has_degraded_sources(sources) else "actual",
                api_label="Roehub API",
                latency_ms=None,
                capital_usdt=None,
                server_time=generated_at,
            ),
            refresh_control=StrategyDashboardRefreshControlResponse(
                manual_refresh_available=True,
                autorefresh_enabled=True,
                interval_seconds=_DEFAULT_REFRESH_INTERVAL_SECONDS,
                preset_key="15s",
                generated_at=generated_at,
                next_allowed_refresh_at=refresh_decision.next_allowed_refresh_at,
                retry_after_seconds=refresh_decision.retry_after_seconds,
                last_refresh_reason=refresh,
                refresh_status=effective_refresh_status,
            ),
        )

    def _load_strategy_state(
        self,
        *,
        user_id: UserId,
        generated_at: datetime,
    ) -> tuple[
        tuple[Strategy, ...],
        dict[UUID, StrategyRun],
        list[StrategyDashboardSourceResponse],
    ]:
        if self._strategy_repository is None:
            return (
                (),
                {},
                [
                    _source(
                        name="strategy_strategies",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy API is disabled for this runtime",
                    ),
                    _source(
                        name="strategy_runs",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy run repository is not configured",
                    ),
                ],
            )

        try:
            strategies = self._strategy_repository.list_for_user(
                user_id=user_id,
                include_deleted=False,
            )
            strategy_source = _source(
                name="strategy_strategies",
                status="available",
                generated_at=generated_at,
                detail=f"{len(strategies)} owner strategies loaded",
            )
        except Exception as error:  # noqa: BLE001
            return (
                (),
                {},
                [
                    _source(
                        name="strategy_strategies",
                        status="degraded",
                        generated_at=generated_at,
                        detail=str(error),
                    ),
                    _source(
                        name="strategy_runs",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy list failed before run lookup",
                    ),
                ],
            )

        if self._run_repository is None:
            return (
                tuple(strategies),
                {},
                [
                    strategy_source,
                    _source(
                        name="strategy_runs",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy run repository is not configured",
                    ),
                ],
            )

        runs_by_strategy_id: dict[UUID, StrategyRun] = {}
        run_error: Exception | None = None
        for strategy in tuple(strategies)[:_STRATEGY_SELECTOR_LIMIT]:
            try:
                active_run = self._run_repository.find_active_for_strategy(
                    user_id=user_id,
                    strategy_id=strategy.strategy_id,
                )
            except Exception as error:  # noqa: BLE001
                run_error = error
                break
            if active_run is not None:
                runs_by_strategy_id[strategy.strategy_id] = active_run

        return (
            tuple(strategies),
            runs_by_strategy_id,
            [
                strategy_source,
                _source(
                    name="strategy_runs",
                    status="degraded" if run_error is not None else "available",
                    generated_at=generated_at,
                    detail=(
                        str(run_error)
                        if run_error is not None
                        else f"{len(runs_by_strategy_id)} active owner runs loaded"
                    ),
                ),
            ],
        )


def build_ui_strategies_dashboard_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    service = build_strategy_dashboard_service(environ=environ)
    return build_ui_strategies_dashboard_api_router(
        dashboard_service=service,
        current_user_dependency=current_user_dependency,
    )


def build_strategy_dashboard_service(
    *,
    environ: Mapping[str, str],
) -> StrategyDashboardQueryService:
    if not is_strategy_api_enabled(environ=environ):
        return StrategyDashboardQueryService(
            strategy_repository=None,
            run_repository=None,
        )
    settings = _resolve_strategy_runtime_settings(environ=environ)
    strategy_repository, run_repository, _event_repository = _build_repositories(settings=settings)
    return StrategyDashboardQueryService(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
    )


def _source(
    *,
    name: str,
    status: SourceStatus,
    generated_at: datetime,
    detail: str,
) -> StrategyDashboardSourceResponse:
    return StrategyDashboardSourceResponse(
        name=name,
        status=status,
        generated_at=generated_at,
        age_seconds=0,
        detail=detail,
    )


def _resolve_refresh_status(
    *,
    refresh_decision: _RefreshDecision,
    sources: list[StrategyDashboardSourceResponse],
) -> RefreshStatus:
    if refresh_decision.status == "rate_limited":
        return "rate_limited"
    if _has_degraded_sources(sources):
        return "degraded"
    return "fresh"


def _has_degraded_sources(sources: list[StrategyDashboardSourceResponse]) -> bool:
    return any(source.status in {"degraded", "unavailable"} for source in sources)


def _select_strategy(
    *,
    strategies: tuple[Strategy, ...],
    strategy_id: str | None,
) -> Strategy | None:
    if strategy_id:
        return next(
            (strategy for strategy in strategies if str(strategy.strategy_id) == strategy_id),
            None,
        )
    return strategies[0] if strategies else None


def _build_selected_strategy(
    *,
    strategy: Strategy | None,
    run: StrategyRun | None,
    generated_at: datetime,
    requested_strategy_id: str | None,
) -> StrategyDashboardSelectedStrategyResponse:
    if strategy is None:
        return StrategyDashboardSelectedStrategyResponse(
            source="strategy_strategies",
            state="empty",
            strategy_id=requested_strategy_id,
            name=None,
            version=None,
            exchange=None,
            market_type=None,
            symbols=[],
            timeframe=None,
            direction=None,
            capital_usdt=None,
            commission_percent=None,
            slippage_percent=None,
            created_at=None,
            updated_at=None,
            status="unknown",
            run_state=None,
            latest_update=None,
            actions=StrategyDashboardActionsResponse(
                can_create=True,
                can_clone=False,
                can_delete=False,
                can_run=False,
                can_stop=False,
            ),
            degradation_reason=(
                "selected_strategy_not_found"
                if requested_strategy_id
                else "strategy_read_model_empty"
            ),
        )

    exchange, market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    is_live = run is not None and run.is_active()
    latest_update = run.updated_at if run is not None else strategy.created_at
    return StrategyDashboardSelectedStrategyResponse(
        source="strategy_strategies",
        state="ready",
        strategy_id=str(strategy.strategy_id),
        name=strategy.name,
        version=f"v{strategy.spec.schema_version}",
        exchange=exchange,
        market_type=market_type or strategy.spec.market_type,
        symbols=[symbol],
        timeframe=strategy.spec.timeframe.code,
        direction="long / short",
        capital_usdt=None,
        commission_percent=None,
        slippage_percent=None,
        created_at=strategy.created_at,
        updated_at=latest_update,
        status="live" if is_live else "stopped",
        run_state=run.state if run is not None else None,
        latest_update=latest_update,
        actions=StrategyDashboardActionsResponse(
            can_create=True,
            can_clone=True,
            can_delete=True,
            can_run=not is_live,
            can_stop=is_live,
        ),
        degradation_reason="stat_projection_unavailable",
    )


def _parse_instrument_key(instrument_key: str) -> tuple[str | None, str | None, str]:
    parts = instrument_key.split(":")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2].upper()
    return None, None, instrument_key.upper()


def _selected_symbol(*, strategy: Strategy | None) -> str | None:
    if strategy is None:
        return None
    _exchange, _market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    return symbol


def _build_strategy_selector(
    *,
    strategies: tuple[Strategy, ...],
    runs_by_strategy_id: dict[UUID, StrategyRun],
    selected_strategy: Strategy | None,
    state: Literal["active", "all"],
    cursor: str | None,
) -> StrategyDashboardSelectorResponse:
    rows = [
        _build_selector_row(strategy=strategy, run=runs_by_strategy_id.get(strategy.strategy_id))
        for strategy in strategies[:_STRATEGY_SELECTOR_LIMIT]
    ]
    active_rows = [row for row in rows if row.status == "live"]
    visible_rows = active_rows if state == "active" else rows
    symbols = {symbol for row in rows for symbol in row.symbols}
    return StrategyDashboardSelectorResponse(
        source="strategy_strategies",
        state="ready" if visible_rows else "empty",
        filters=StrategyDashboardSelectorFiltersResponse(
            state=state,
            cursor=cursor,
            limit=_STRATEGY_SELECTOR_LIMIT,
            query="",
            sort="updated",
        ),
        totals=StrategyDashboardSelectorTotalsResponse(
            strategies=len(strategies),
            active=len(active_rows),
            stopped=sum(1 for row in rows if row.status == "stopped"),
            degraded=sum(1 for row in rows if row.status == "degraded"),
            symbols=len(symbols),
        ),
        items=visible_rows,
        selected_strategy_id=str(selected_strategy.strategy_id) if selected_strategy else None,
        next_cursor=None if len(strategies) <= _STRATEGY_SELECTOR_LIMIT else "next",
        degradation_reason=None if visible_rows else "strategy_read_model_empty",
    )


def _build_selector_row(
    *,
    strategy: Strategy,
    run: StrategyRun | None,
) -> StrategyDashboardSelectorRowResponse:
    exchange, market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    is_live = run is not None and run.is_active()
    return StrategyDashboardSelectorRowResponse(
        strategy_id=str(strategy.strategy_id),
        name=strategy.name,
        version=f"v{strategy.spec.schema_version}",
        exchange=exchange,
        market_type=market_type or strategy.spec.market_type,
        symbols=[symbol],
        timeframe=strategy.spec.timeframe.code,
        status="live" if is_live else "stopped",
        run_state=run.state if run is not None else None,
        latest_activity=run.updated_at if run is not None else strategy.created_at,
    )


def _build_unavailable_chart(*, symbol: str | None) -> StrategyChartResponse:
    return StrategyChartResponse(
        source=_CANDLE_SOURCE,
        state="unavailable",
        symbol=symbol,
        range="1y",
        max_points=_CHART_MAX_POINTS,
        candles=[],
        markers=[],
        degradation_reason="strategy_candles_projection_unavailable",
    )


def _build_metric_grid() -> StrategyMetricGridResponse:
    metrics = [
        "total_return",
        "best_sharpe",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "trades",
        "avg_hold",
        "exposure",
        "avg_trade",
    ]
    return StrategyMetricGridResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        items=[_unavailable_metric(key=metric, source=_STAT_SOURCE) for metric in metrics],
        degradation_reason="strategy_statistics_projection_unavailable",
    )


def _unavailable_metric(*, key: str, source: str) -> StrategyMetricResponse:
    return StrategyMetricResponse(
        key=key,
        label=key,
        value=None,
        formatted="Unavailable",
        direction="neutral",
        status="unavailable",
        source=source,
    )


def _build_monthly_stats() -> StrategyMonthlyStatsResponse:
    return StrategyMonthlyStatsResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        columns=[
            "year",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
            "total",
        ],
        rows=[],
        summary=[
            _unavailable_metric(key="best_month", source=_STAT_SOURCE),
            _unavailable_metric(key="worst_month", source=_STAT_SOURCE),
            _unavailable_metric(key="profitable_months", source=_STAT_SOURCE),
        ],
        degradation_reason="monthly_statistics_projection_unavailable",
    )


def _build_long_short() -> StrategyBreakdownPanelResponse:
    return StrategyBreakdownPanelResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        rows=[
            _unavailable_breakdown("win_rate"),
            _unavailable_breakdown("trades"),
            _unavailable_breakdown("return"),
            _unavailable_breakdown("profit_factor"),
            _unavailable_breakdown("avg_trade"),
        ],
        degradation_reason="long_short_statistics_projection_unavailable",
    )


def _build_risk_execution() -> StrategyBreakdownPanelResponse:
    return StrategyBreakdownPanelResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        rows=[
            _unavailable_breakdown("avg_holding_time"),
            _unavailable_breakdown("avg_bars_in_trade"),
            _unavailable_breakdown("commissions"),
            _unavailable_breakdown("execution_paid"),
            _unavailable_breakdown("max_consecutive_losses"),
            _unavailable_breakdown("worst_trade"),
        ],
        degradation_reason="risk_execution_projection_unavailable",
    )


def _unavailable_breakdown(key: str) -> StrategyBreakdownRowResponse:
    return StrategyBreakdownRowResponse(
        key=key,
        label=key,
        long_value=None,
        short_value=None,
        total_value="Unavailable",
        direction="neutral",
    )


def _build_unavailable_series(*, title: str) -> StrategySeriesPanelResponse:
    return StrategySeriesPanelResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        title=title,
        max_points=_SERIES_MAX_POINTS,
        points=[],
        degradation_reason=f"{title}_projection_unavailable",
    )


def _build_best_worst_days() -> StrategyBestWorstDaysResponse:
    return StrategyBestWorstDaysResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        best_days=[],
        worst_days=[],
        degradation_reason="daily_statistics_projection_unavailable",
    )


def _build_hourly_results() -> StrategyHourlyResultsResponse:
    empty_total = StrategyHourlyResultResponse(
        hour_bucket="total",
        win_rate_percent=None,
        pnl_percent=None,
        direction="neutral",
    )
    return StrategyHourlyResultsResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        items=[],
        total=empty_total,
        degradation_reason="hourly_statistics_projection_unavailable",
    )


def _build_trades() -> StrategyTradesResponse:
    return StrategyTradesResponse(
        source=_TRADES_SOURCE,
        state="unavailable",
        limit=_TRADES_LIMIT,
        items=[],
        next_cursor=None,
        degradation_reason="strategy_trades_projection_unavailable",
    )


def _build_symbol_results(*, symbol: str | None) -> StrategySymbolResultsResponse:
    total = StrategySymbolResultResponse(
        symbol=symbol or "total",
        trades=None,
        win_rate_percent=None,
        pnl_percent=None,
        pnl_usdt=None,
        direction="neutral",
    )
    return StrategySymbolResultsResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        items=[],
        total=total,
        degradation_reason="symbol_statistics_projection_unavailable",
    )
