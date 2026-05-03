from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Mapping
from uuid import UUID

from fastapi import APIRouter

from apps.api.dto import (
    StrategyEquityResponse,
    StrategyFillsResponse,
    StrategyMonitoringAlertResponse,
    StrategyMonitoringLimitsResponse,
    StrategyMonitoringSourceResponse,
    StrategyMonitorItemResponse,
    StrategyMonitorResponse,
    StrategyPositionsResponse,
    StrategySnapshotMetricResponse,
    StrategySnapshotResponse,
    StrategySnapshotRunResponse,
    StrategySnapshotSpecResponse,
)
from apps.api.routes import build_strategy_streams_router, build_ui_strategies_monitoring_router
from apps.api.wiring.modules.strategy import (
    _build_repositories,
    _resolve_strategy_runtime_settings,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.adapters.outbound import (
    RedisStrategyRealtimeOutputReader,
    RedisStrategyRealtimeOutputReaderConfig,
    SystemStrategyClock,
    load_strategy_runtime_config,
    resolve_strategy_config_path,
)
from trading.contexts.strategy.application import (
    CurrentUser as StrategyCurrentUser,
)
from trading.contexts.strategy.application import (
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    StrategyRealtimeOutputReader,
    StrategyRepository,
    StrategyRunRepository,
    UnavailableStrategyRealtimeOutputReader,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategyRun
from trading.shared_kernel.primitives import UserId

_POLL_INTERVAL_SECONDS = 10
_STRATEGY_LIMIT = 50
_ALERT_LIMIT = 10
_POSITIONS_LIMIT = 50
_FILLS_LIMIT = 50
_EQUITY_POINT_LIMIT = 600


@dataclass(frozen=True, slots=True)
class UiStrategyMonitoringApiModule:
    router: APIRouter
    stream_reader: StrategyRealtimeOutputReader
    query_service: StrategyMonitoringQueryService


class StrategyMonitoringQueryService:
    def __init__(
        self,
        *,
        list_use_case: ListMyStrategiesUseCase,
        get_use_case: GetMyStrategyUseCase,
        run_repository: StrategyRunRepository,
        clock: SystemStrategyClock,
    ) -> None:
        if list_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyMonitoringQueryService requires list_use_case")
        if get_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyMonitoringQueryService requires get_use_case")
        if run_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyMonitoringQueryService requires run_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyMonitoringQueryService requires clock")
        self._list_use_case = list_use_case
        self._get_use_case = get_use_case
        self._run_repository = run_repository
        self._clock = clock

    def get_monitor(
        self,
        *,
        principal: CurrentUserPrincipal,
        state: Literal["active", "all"],
        cursor: str | None,
    ) -> StrategyMonitorResponse:
        generated_at = _format_required_datetime(self._clock.now())
        try:
            strategies = self._list_use_case.execute(current_user=_strategy_user(principal))
            active_runs = _active_runs_by_strategy(
                run_repository=self._run_repository,
                user_id=principal.user_id,
                strategy_ids={strategy.strategy_id for strategy in strategies},
            )
            filtered = [
                strategy
                for strategy in strategies
                if state == "all" or strategy.strategy_id in active_runs
            ]
            offset = _parse_cursor(cursor=cursor)
            page = filtered[offset : offset + _STRATEGY_LIMIT]
            next_cursor = (
                str(offset + _STRATEGY_LIMIT)
                if offset + _STRATEGY_LIMIT < len(filtered)
                else None
            )
            items = [
                _monitor_item(
                    strategy=strategy,
                    active_run=active_runs.get(strategy.strategy_id),
                    now=self._clock.now(),
                )
                for strategy in page
            ]
            return StrategyMonitorResponse(
                source=_source(
                    status="available",
                    code="strategies.monitor.available",
                    message="Strategy monitoring read-model is available",
                    updated_at=generated_at,
                ),
                generated_at=generated_at,
                poll_interval_seconds=_POLL_INTERVAL_SECONDS,
                items=items,
                selected_strategy_id=items[0].strategy_id if items else None,
                next_cursor=next_cursor,
                limits=_limits(),
                links={
                    "self": "/api/ui/strategies/monitor",
                    "stream": "/api/stream/strategies",
                    "strategies": "/api/strategies",
                },
            )
        except Exception:
            return StrategyMonitorResponse(
                source=_source(
                    status="degraded",
                    code="strategies.monitor.degraded",
                    message="Strategy monitoring read-model failed",
                    updated_at=generated_at,
                ),
                generated_at=generated_at,
                poll_interval_seconds=_POLL_INTERVAL_SECONDS,
                items=[],
                selected_strategy_id=None,
                next_cursor=None,
                limits=_limits(),
                links={
                    "self": "/api/ui/strategies/monitor",
                    "stream": "/api/stream/strategies",
                    "strategies": "/api/strategies",
                },
            )

    def get_snapshot(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
    ) -> StrategySnapshotResponse:
        strategy = self._owned_strategy(principal=principal, strategy_id=strategy_id)
        active_run = self._run_repository.find_active_for_strategy(
            user_id=principal.user_id,
            strategy_id=strategy_id,
        )
        latest_run = active_run or _latest_run(
            self._run_repository.list_for_strategy(
                user_id=principal.user_id,
                strategy_id=strategy_id,
            )
        )
        generated_at = _format_required_datetime(self._clock.now())
        return StrategySnapshotResponse(
            source=_source(
                status="available",
                code="strategies.snapshot.available",
                message="Strategy snapshot is available",
                updated_at=generated_at,
            ),
            generated_at=generated_at,
            strategy_id=str(strategy.strategy_id),
            name=strategy.name,
            spec=StrategySnapshotSpecResponse(
                instrument_key=strategy.spec.instrument_key,
                market_type=strategy.spec.market_type,
                timeframe=strategy.spec.timeframe.code,
                signal_template=strategy.spec.signal_template,
            ),
            run=_snapshot_run(run=latest_run),
            metrics=_snapshot_metrics(run=latest_run, now=self._clock.now()),
            alerts=_snapshot_alerts(run=latest_run),
            links={
                "self": f"/api/ui/strategies/{strategy.strategy_id}/snapshot",
                "run": f"/api/strategies/{strategy.strategy_id}/run",
                "stop": f"/api/strategies/{strategy.strategy_id}/stop",
                "stream": f"/api/stream/strategies?strategy_id={strategy.strategy_id}",
                "positions": f"/api/ui/strategies/{strategy.strategy_id}/positions",
                "fills": f"/api/ui/strategies/{strategy.strategy_id}/fills",
                "equity": f"/api/ui/strategies/{strategy.strategy_id}/equity",
            },
        )

    def get_positions(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
        limit: int,
    ) -> StrategyPositionsResponse:
        self.ensure_strategy_owner(principal=principal, strategy_id=strategy_id)
        return StrategyPositionsResponse(
            source=_source(
                status="unavailable",
                code="strategies.positions.unavailable",
                message="Position storage is not available for Strategy monitoring v1",
            ),
            strategy_id=str(strategy_id),
            limit=min(limit, _POSITIONS_LIMIT),
            items=[],
        )

    def get_fills(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
        limit: int,
        cursor: str | None,
    ) -> StrategyFillsResponse:
        _ = cursor
        self.ensure_strategy_owner(principal=principal, strategy_id=strategy_id)
        return StrategyFillsResponse(
            source=_source(
                status="unavailable",
                code="strategies.fills.unavailable",
                message="Fill storage is not available for Strategy monitoring v1",
            ),
            strategy_id=str(strategy_id),
            limit=min(limit, _FILLS_LIMIT),
            items=[],
            next_cursor=None,
        )

    def get_equity(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
        range_name: str,
        points: int,
    ) -> StrategyEquityResponse:
        self.ensure_strategy_owner(principal=principal, strategy_id=strategy_id)
        return StrategyEquityResponse(
            source=_source(
                status="unavailable",
                code="strategies.equity.unavailable",
                message="Equity series is not available for Strategy monitoring v1",
            ),
            strategy_id=str(strategy_id),
            range=range_name,
            points=min(points, _EQUITY_POINT_LIMIT),
            items=[],
        )

    def ensure_strategy_owner(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
    ) -> None:
        self._owned_strategy(principal=principal, strategy_id=strategy_id)

    def _owned_strategy(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
    ) -> Strategy:
        return self._get_use_case.execute(
            strategy_id=strategy_id,
            current_user=_strategy_user(principal),
        )


def build_ui_strategy_monitoring_api_module(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
    strategy_repository: StrategyRepository | None = None,
    run_repository: StrategyRunRepository | None = None,
) -> UiStrategyMonitoringApiModule:
    if strategy_repository is None or run_repository is None:
        settings = _resolve_strategy_runtime_settings(environ=environ)
        strategy_repository, run_repository, _event_repository = _build_repositories(
            settings=settings
        )
    query_service = StrategyMonitoringQueryService(
        list_use_case=ListMyStrategiesUseCase(repository=strategy_repository),
        get_use_case=GetMyStrategyUseCase(repository=strategy_repository),
        run_repository=run_repository,
        clock=SystemStrategyClock(),
    )
    stream_reader = _build_stream_reader(environ=environ)
    router = APIRouter()
    router.include_router(
        build_ui_strategies_monitoring_router(
            current_user_dependency=current_user_dependency,
            monitoring_query=query_service,
        )
    )
    router.include_router(
        build_strategy_streams_router(
            current_user_dependency=current_user_dependency,
            stream_reader=stream_reader,
            owner_scope=query_service,
        )
    )
    return UiStrategyMonitoringApiModule(
        router=router,
        stream_reader=stream_reader,
        query_service=query_service,
    )


def _build_stream_reader(*, environ: Mapping[str, str]) -> StrategyRealtimeOutputReader:
    runtime_config = load_strategy_runtime_config(
        resolve_strategy_config_path(environ=environ),
        environ=environ,
    )
    redis_config = runtime_config.realtime_output.redis_streams
    if not redis_config.enabled:
        return UnavailableStrategyRealtimeOutputReader(
            reason="Strategy realtime output Redis Streams are disabled"
        )
    return RedisStrategyRealtimeOutputReader(
        config=RedisStrategyRealtimeOutputReaderConfig(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password_env=redis_config.password_env,
            socket_timeout_s=redis_config.socket_timeout_s,
            connect_timeout_s=redis_config.connect_timeout_s,
            metrics_stream_prefix=redis_config.metrics_stream_prefix,
            events_stream_prefix=redis_config.events_stream_prefix,
        ),
        environ=environ,
    )


def _strategy_user(principal: CurrentUserPrincipal) -> StrategyCurrentUser:
    return StrategyCurrentUser(user_id=principal.user_id)


def _active_runs_by_strategy(
    *,
    run_repository: StrategyRunRepository,
    user_id: UserId,
    strategy_ids: set[UUID],
) -> dict[UUID, StrategyRun]:
    return {
        run.strategy_id: run
        for run in run_repository.list_active_runs()
        if run.user_id == user_id and run.strategy_id in strategy_ids
    }


def _monitor_item(
    *,
    strategy: Strategy,
    active_run: StrategyRun | None,
    now: datetime,
) -> StrategyMonitorItemResponse:
    updated_at = active_run.updated_at if active_run is not None else strategy.created_at
    return StrategyMonitorItemResponse(
        strategy_id=str(strategy.strategy_id),
        name=strategy.name,
        state=active_run.state if active_run is not None else "idle",
        run_id=str(active_run.run_id) if active_run is not None else None,
        instrument_key=strategy.spec.instrument_key,
        timeframe=strategy.spec.timeframe.code,
        checkpoint_ts_open=_format_datetime(active_run.checkpoint_ts_open)
        if active_run is not None
        else None,
        lag_seconds=_lag_seconds(run=active_run, now=now),
        updated_at=_format_required_datetime(updated_at),
    )


def _snapshot_run(*, run: StrategyRun | None) -> StrategySnapshotRunResponse:
    if run is None:
        return StrategySnapshotRunResponse(
            run_id=None,
            state="idle",
            started_at=None,
            stopped_at=None,
            checkpoint_ts_open=None,
            updated_at=None,
            last_error=None,
        )
    return StrategySnapshotRunResponse(
        run_id=str(run.run_id),
        state=run.state,
        started_at=_format_required_datetime(run.started_at),
        stopped_at=_format_datetime(run.stopped_at),
        checkpoint_ts_open=_format_datetime(run.checkpoint_ts_open),
        updated_at=_format_required_datetime(run.updated_at),
        last_error=run.last_error,
    )


def _snapshot_metrics(
    *,
    run: StrategyRun | None,
    now: datetime,
) -> list[StrategySnapshotMetricResponse]:
    if run is None:
        return [
            StrategySnapshotMetricResponse(key="run_state", value="idle"),
            StrategySnapshotMetricResponse(key="lag_seconds", value="--"),
        ]
    lag = _lag_seconds(run=run, now=now)
    return [
        StrategySnapshotMetricResponse(
            key="run_state",
            value=run.state,
            updated_at=_format_required_datetime(run.updated_at),
        ),
        StrategySnapshotMetricResponse(
            key="checkpoint_ts_open",
            value=_format_datetime(run.checkpoint_ts_open) or "--",
            updated_at=_format_required_datetime(run.updated_at),
        ),
        StrategySnapshotMetricResponse(
            key="lag_seconds",
            value=str(lag) if lag is not None else "--",
            tone="negative" if lag is not None and lag > 300 else "neutral",
            updated_at=_format_required_datetime(run.updated_at),
        ),
    ]


def _snapshot_alerts(*, run: StrategyRun | None) -> list[StrategyMonitoringAlertResponse]:
    if run is None or run.last_error is None:
        return []
    return [
        StrategyMonitoringAlertResponse(
            alert_id=f"{run.run_id}:last_error",
            severity="critical",
            title=run.last_error,
            created_at=_format_required_datetime(run.updated_at),
        )
    ][:_ALERT_LIMIT]


def _latest_run(runs: tuple[StrategyRun, ...]) -> StrategyRun | None:
    if not runs:
        return None
    return max(runs, key=lambda run: (run.updated_at, str(run.run_id)))


def _lag_seconds(*, run: StrategyRun | None, now: datetime) -> int | None:
    if run is None or run.checkpoint_ts_open is None:
        return None
    return max(0, int((now - run.checkpoint_ts_open).total_seconds()))


def _parse_cursor(*, cursor: str | None) -> int:
    if cursor is None or not cursor.strip():
        return 0
    try:
        return max(0, int(cursor))
    except ValueError:
        return 0


def _source(
    *,
    status: Literal["available", "degraded", "unavailable"],
    code: str,
    message: str,
    updated_at: str | None = None,
) -> StrategyMonitoringSourceResponse:
    return StrategyMonitoringSourceResponse(
        status=status,
        code=code,
        message=message,
        updated_at=updated_at,
    )


def _limits() -> StrategyMonitoringLimitsResponse:
    return StrategyMonitoringLimitsResponse(
        strategies=_STRATEGY_LIMIT,
        alerts=_ALERT_LIMIT,
        positions=_POSITIONS_LIMIT,
        fills=_FILLS_LIMIT,
        equity_points=_EQUITY_POINT_LIMIT,
    )


def _format_required_datetime(value: datetime) -> str:
    return _format_datetime(value) or ""


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.astimezone(UTC)
    return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "StrategyMonitoringQueryService",
    "UiStrategyMonitoringApiModule",
    "build_ui_strategy_monitoring_api_module",
]
