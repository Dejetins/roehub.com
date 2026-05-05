from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Literal, Mapping
from uuid import UUID

from fastapi import APIRouter

from apps.api.dto.ui_dashboard import (
    DashboardAlertsResponse,
    DashboardEquityPnlSeriesResponse,
    DashboardExecutionsResponse,
    DashboardFooterStatusResponse,
    DashboardHealthCheckResponse,
    DashboardHealthRiskResponse,
    DashboardMetricResponse,
    DashboardPositionsResponse,
    DashboardRefreshControlResponse,
    DashboardSelectedStrategySnapshotResponse,
    DashboardSourceResponse,
    DashboardStrategyActionsResponse,
    DashboardStrategyListFiltersResponse,
    DashboardStrategyListResponse,
    DashboardStrategyListRowResponse,
    DashboardStrategyListTotalsResponse,
    DashboardSummaryResponse,
    DashboardSymbolAllocationResponse,
    FinancialDirection,
    RefreshStatus,
    SourceStatus,
)
from apps.api.routes.ui_dashboard import build_ui_dashboard_router as build_ui_dashboard_api_router
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
_STRATEGY_LIST_LIMIT = 20
_OPEN_POSITIONS_LIMIT = 20
_RECENT_EXECUTIONS_LIMIT = 20
_EQUITY_MAX_POINTS = 600


@dataclass(frozen=True, slots=True)
class _RefreshDecision:
    status: RefreshStatus
    next_allowed_refresh_at: datetime | None
    retry_after_seconds: int | None


class DashboardManualRefreshLimiter:
    def __init__(self, *, interval_seconds: int = _MINIMUM_MANUAL_REFRESH_SECONDS) -> None:
        if interval_seconds < 1:
            raise ValueError("DashboardManualRefreshLimiter requires positive interval_seconds")
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


class DashboardSummaryQueryService:
    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository | None,
        run_repository: StrategyRunRepository | None,
        refresh_limiter: DashboardManualRefreshLimiter | None = None,
    ) -> None:
        self._strategy_repository = strategy_repository
        self._run_repository = run_repository
        self._refresh_limiter = refresh_limiter or DashboardManualRefreshLimiter()

    def get_summary(
        self,
        *,
        principal: CurrentUserPrincipal,
        refresh: Literal["initial", "auto", "manual"],
    ) -> DashboardSummaryResponse:
        generated_at = datetime.now(UTC)
        user_id = principal.user_id
        refresh_decision = self._refresh_limiter.resolve(
            user_id=str(user_id),
            requested_at=generated_at,
            refresh=refresh,
        )
        strategies, runs_by_strategy_id, dynamic_sources = self._load_strategy_state(
            user_id=user_id,
            generated_at=generated_at,
        )
        inventory_sources = [
            *dynamic_sources,
            _source(
                name="strategy_events",
                status="unavailable",
                generated_at=generated_at,
                detail="strategy event alerts are not yet exposed as a dashboard read-model",
            ),
            _source(
                name="portfolio_snapshots",
                status="unavailable",
                generated_at=generated_at,
                detail="typed portfolio snapshots are not migrated yet",
            ),
            _source(
                name="position_snapshots",
                status="unavailable",
                generated_at=generated_at,
                detail="typed open-position snapshots are not migrated yet",
            ),
            _source(
                name="execution_fills",
                status="unavailable",
                generated_at=generated_at,
                detail="typed execution fill snapshots are not migrated yet",
            ),
            _source(
                name="equity_points",
                status="unavailable",
                generated_at=generated_at,
                detail="typed equity/PnL point snapshots are not migrated yet",
            ),
            _source(
                name="symbol_allocations",
                status="unavailable",
                generated_at=generated_at,
                detail="typed symbol allocation snapshots are not migrated yet",
            ),
            _source(
                name="exchange_account",
                status="unavailable",
                generated_at=generated_at,
                detail="exchange account snapshots are not wired for dashboard refresh",
            ),
            _source(
                name="backtest_jobs",
                status="unavailable",
                generated_at=generated_at,
                detail="recent backtest jobs are intentionally omitted from Stage 4 summary",
            ),
        ]
        effective_refresh_status = _resolve_refresh_status(
            refresh_decision=refresh_decision,
            sources=inventory_sources,
        )
        selected_strategy = strategies[0] if strategies else None
        selected_run = (
            runs_by_strategy_id.get(selected_strategy.strategy_id)
            if selected_strategy is not None
            else None
        )

        return DashboardSummaryResponse(
            generated_at=generated_at,
            refresh_status=effective_refresh_status,
            next_allowed_refresh_at=refresh_decision.next_allowed_refresh_at,
            retry_after_seconds=refresh_decision.retry_after_seconds,
            sources=inventory_sources,
            selected_strategy_snapshot=_build_selected_strategy_snapshot(
                strategy=selected_strategy,
                run=selected_run,
                generated_at=generated_at,
            ),
            equity_pnl_series=_unavailable_equity_series(),
            metric_grid=_build_metric_grid(run=selected_run, generated_at=generated_at),
            open_positions=DashboardPositionsResponse(
                source="position_snapshots",
                state="unavailable",
                limit=_OPEN_POSITIONS_LIMIT,
                items=[],
                degradation_reason="position_snapshots_unavailable",
            ),
            recent_executions=DashboardExecutionsResponse(
                source="execution_fills",
                state="unavailable",
                limit=_RECENT_EXECUTIONS_LIMIT,
                items=[],
                next_cursor=None,
                degradation_reason="execution_fills_unavailable",
            ),
            health_risk=_build_health_risk(sources=inventory_sources),
            alerts=DashboardAlertsResponse(
                source="strategy_events",
                state="unavailable",
                items=[],
                next_cursor=None,
                degradation_reason="strategy_events_dashboard_read_model_unavailable",
            ),
            symbol_allocation=DashboardSymbolAllocationResponse(
                source="symbol_allocations",
                state="unavailable",
                items=[],
                degradation_reason="symbol_allocations_unavailable",
            ),
            strategy_list=_build_strategy_list(
                strategies=strategies,
                runs_by_strategy_id=runs_by_strategy_id,
            ),
            footer_status=DashboardFooterStatusResponse(
                system_status="degraded" if _has_degraded_sources(inventory_sources) else "ok",
                account_tier=str(principal.paid_level),
                mode="LIVE",
                api_label="Roehub API",
                latency_ms=None,
                server_time=generated_at,
            ),
            refresh_control=DashboardRefreshControlResponse(
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
    ) -> tuple[tuple[Strategy, ...], dict[UUID, StrategyRun], list[DashboardSourceResponse]]:
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
        for strategy in tuple(strategies)[:_STRATEGY_LIST_LIMIT]:
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

        run_source = _source(
            name="strategy_runs",
            status="degraded" if run_error is not None else "available",
            generated_at=generated_at,
            detail=(
                str(run_error)
                if run_error is not None
                else f"{len(runs_by_strategy_id)} active owner runs loaded"
            ),
        )
        return tuple(strategies), runs_by_strategy_id, [strategy_source, run_source]


def build_ui_dashboard_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    service = build_dashboard_summary_service(environ=environ)
    return build_ui_dashboard_api_router(
        summary_service=service,
        current_user_dependency=current_user_dependency,
    )


def build_dashboard_summary_service(*, environ: Mapping[str, str]) -> DashboardSummaryQueryService:
    if not is_strategy_api_enabled(environ=environ):
        return DashboardSummaryQueryService(
            strategy_repository=None,
            run_repository=None,
        )
    settings = _resolve_strategy_runtime_settings(environ=environ)
    strategy_repository, run_repository, _event_repository = _build_repositories(settings=settings)
    return DashboardSummaryQueryService(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
    )


def _source(
    *,
    name: str,
    status: SourceStatus,
    generated_at: datetime,
    detail: str,
) -> DashboardSourceResponse:
    return DashboardSourceResponse(
        name=name,
        status=status,
        generated_at=generated_at,
        age_seconds=0,
        detail=detail,
    )


def _resolve_refresh_status(
    *,
    refresh_decision: _RefreshDecision,
    sources: list[DashboardSourceResponse],
) -> RefreshStatus:
    if refresh_decision.status == "rate_limited":
        return "rate_limited"
    if _has_degraded_sources(sources):
        return "degraded"
    return "fresh"


def _has_degraded_sources(sources: list[DashboardSourceResponse]) -> bool:
    return any(source.status in {"degraded", "unavailable"} for source in sources)


def _build_selected_strategy_snapshot(
    *,
    strategy: Strategy | None,
    run: StrategyRun | None,
    generated_at: datetime,
) -> DashboardSelectedStrategySnapshotResponse:
    if strategy is None:
        return DashboardSelectedStrategySnapshotResponse(
            source="strategy_strategies",
            state="empty",
            strategy_id=None,
            name=None,
            version=None,
            exchange=None,
            symbols=[],
            direction=None,
            mode=None,
            timeframe=None,
            capital=None,
            leverage=None,
            status="unknown",
            latest_update=None,
            uptime_seconds=None,
            actions=DashboardStrategyActionsResponse(
                can_start=False,
                can_stop=False,
                can_restart=False,
                can_open_settings=False,
            ),
            degradation_reason="strategy_read_model_empty",
        )

    exchange, _market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    is_live = run is not None and run.is_active()
    latest_update = run.updated_at if run is not None else strategy.created_at
    uptime_seconds = (
        max(0, int((generated_at - run.started_at).total_seconds()))
        if run is not None and run.is_active()
        else None
    )
    return DashboardSelectedStrategySnapshotResponse(
        source="strategy_strategies",
        state="ready",
        strategy_id=str(strategy.strategy_id),
        name=strategy.name,
        version=f"v{strategy.spec.schema_version}",
        exchange=exchange,
        symbols=[symbol],
        direction=None,
        mode=run.state if run is not None else "stopped",
        timeframe=strategy.spec.timeframe.code,
        capital=None,
        leverage=None,
        status="live" if is_live else "stopped",
        latest_update=latest_update,
        uptime_seconds=uptime_seconds,
        actions=DashboardStrategyActionsResponse(
            can_start=not is_live,
            can_stop=is_live,
            can_restart=is_live,
            can_open_settings=True,
        ),
        degradation_reason="portfolio_snapshot_unavailable",
    )


def _parse_instrument_key(instrument_key: str) -> tuple[str | None, str | None, str]:
    parts = instrument_key.split(":")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2].upper()
    return None, None, instrument_key.upper()


def _unavailable_equity_series() -> DashboardEquityPnlSeriesResponse:
    return DashboardEquityPnlSeriesResponse(
        source="equity_points",
        state="unavailable",
        range="1d",
        max_points=_EQUITY_MAX_POINTS,
        points=[],
        degradation_reason="equity_points_unavailable",
    )


def _build_metric_grid(
    *,
    run: StrategyRun | None,
    generated_at: datetime,
) -> list[DashboardMetricResponse]:
    uptime_seconds = (
        max(0, int((generated_at - run.started_at).total_seconds()))
        if run is not None and run.is_active()
        else None
    )
    return [
        _metric("total_pnl", None, "unavailable", "portfolio_snapshots"),
        _metric("unrealized_pnl", None, "unavailable", "portfolio_snapshots"),
        _metric("realized_pnl", None, "unavailable", "portfolio_snapshots"),
        _metric("roi", None, "unavailable", "portfolio_snapshots"),
        _metric("win_rate", None, "unavailable", "portfolio_snapshots"),
        _metric("open_positions", None, "unavailable", "position_snapshots"),
        _metric("equity", None, "unavailable", "portfolio_snapshots"),
        _metric("max_drawdown", None, "unavailable", "portfolio_snapshots"),
        _metric("exposure", None, "unavailable", "portfolio_snapshots"),
        _metric("trades_today", None, "unavailable", "execution_fills"),
        _metric(
            "uptime",
            uptime_seconds,
            "available" if uptime_seconds is not None else "unavailable",
            "strategy_runs",
            formatted=_format_duration(uptime_seconds) if uptime_seconds is not None else None,
        ),
    ]


def _metric(
    key: str,
    value: float | int | None,
    status: SourceStatus,
    source: str,
    *,
    formatted: str | None = None,
) -> DashboardMetricResponse:
    return DashboardMetricResponse(
        key=key,
        label=key,
        value=value,
        formatted=formatted or "Unavailable",
        direction=_financial_direction(value=value),
        status=status,
        source=source,
    )


def _financial_direction(*, value: float | int | None) -> FinancialDirection:
    if value is None:
        return "neutral"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _format_duration(seconds: int) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}"
    return f"{hours:02d}:{minutes:02d}"


def _build_health_risk(
    *,
    sources: list[DashboardSourceResponse],
) -> DashboardHealthRiskResponse:
    checks = [
        DashboardHealthCheckResponse(
            key=source.name,
            label=source.name,
            state=_source_status_to_health(source.status),
            value=source.status.upper(),
            ratio=_source_status_to_ratio(source.status),
            source=source.name,
        )
        for source in sources[:6]
    ]
    return DashboardHealthRiskResponse(
        source="sources",
        state="warn" if _has_degraded_sources(sources) else "ok",
        checks=checks,
        degradation_reason=(
            "one_or_more_dashboard_sources_unavailable" if _has_degraded_sources(sources) else None
        ),
    )


def _source_status_to_health(status: SourceStatus) -> Literal["ok", "warn", "error", "unknown"]:
    if status == "available":
        return "ok"
    if status == "degraded":
        return "warn"
    return "unknown"


def _source_status_to_ratio(status: SourceStatus) -> float:
    if status == "available":
        return 1.0
    if status == "degraded":
        return 0.45
    return 0.0


def _build_strategy_list(
    *,
    strategies: tuple[Strategy, ...],
    runs_by_strategy_id: dict[UUID, StrategyRun],
) -> DashboardStrategyListResponse:
    rows = [
        _build_strategy_row(strategy=strategy, run=runs_by_strategy_id.get(strategy.strategy_id))
        for strategy in strategies[:_STRATEGY_LIST_LIMIT]
    ]
    running_count = sum(1 for row in rows if row.status == "live")
    stopped_count = sum(1 for row in rows if row.status == "stopped")
    symbols = {
        symbol
        for row in rows
        for symbol in row.symbols
    }
    return DashboardStrategyListResponse(
        source="strategy_strategies",
        state="ready" if rows else "empty",
        filters=DashboardStrategyListFiltersResponse(
            state="running",
            exchange="all",
            mode="all",
            query="",
            sort="activity",
        ),
        totals=DashboardStrategyListTotalsResponse(
            running=running_count,
            stopped=stopped_count,
            degraded=0,
            symbols=len(symbols),
            strategies=len(strategies),
            open_positions=None,
        ),
        items=rows,
        next_cursor=None if len(strategies) <= _STRATEGY_LIST_LIMIT else "next",
        degradation_reason=None if rows else "strategy_read_model_empty",
    )


def _build_strategy_row(
    *,
    strategy: Strategy,
    run: StrategyRun | None,
) -> DashboardStrategyListRowResponse:
    exchange, _market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    is_live = run is not None and run.is_active()
    return DashboardStrategyListRowResponse(
        strategy_id=str(strategy.strategy_id),
        name=strategy.name,
        version=f"v{strategy.spec.schema_version}",
        exchange=exchange,
        symbols=[symbol],
        latest_activity=run.updated_at if run is not None else strategy.created_at,
        pnl=None,
        pnl_percent=None,
        mode=run.state if run is not None else "stopped",
        open_positions=None,
        status="live" if is_live else "stopped",
        mini_sparkline=[],
        sparkline_state="unavailable",
    )
