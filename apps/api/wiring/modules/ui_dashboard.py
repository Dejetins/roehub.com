from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Protocol

from fastapi import APIRouter

from apps.api.dto import (
    DashboardAccountResponse,
    DashboardAlertsResponse,
    DashboardBacktestJobResponse,
    DashboardBacktestsResponse,
    DashboardSourceResponse,
    DashboardSourceStatus,
    DashboardStrategiesResponse,
    DashboardStrategyItemResponse,
    DashboardSummaryResponse,
)
from apps.api.routes import build_ui_dashboard_router
from apps.api.routes.ui_dashboard import CurrentUserDependency
from apps.api.wiring.modules.strategy import is_strategy_api_enabled
from trading.contexts.backtest.adapters.outbound import (
    PostgresBacktestJobRepository,
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest.application.dto import build_backtest_job_read_model
from trading.contexts.backtest.application.ports import BacktestJobListQuery, BacktestJobRepository
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.adapters.outbound import (
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
    PsycopgStrategyPostgresGateway,
)
from trading.contexts.strategy.application import (
    CurrentUser as StrategyCurrentUser,
)
from trading.contexts.strategy.application import (
    ListMyStrategiesUseCase,
    StrategyRunRepository,
)

_DEFAULT_POLL_INTERVAL_SECONDS = 12
_PANEL_ITEM_LIMIT = 5


@dataclass(frozen=True, slots=True)
class UiDashboardApiModule:
    router: APIRouter
    summary_query: DashboardSummaryQueryService


class StrategyDashboardProvider(Protocol):
    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardStrategiesResponse:
        ...


class BacktestsDashboardProvider(Protocol):
    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardBacktestsResponse:
        ...


class AlertsDashboardProvider(Protocol):
    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardAlertsResponse:
        ...


class DashboardSummaryQueryService:
    def __init__(
        self,
        *,
        strategy_provider: StrategyDashboardProvider,
        backtests_provider: BacktestsDashboardProvider,
        alerts_provider: AlertsDashboardProvider,
        poll_interval_seconds: int = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        if not 10 <= poll_interval_seconds <= 15:
            raise ValueError("Dashboard poll interval must stay within 10-15 seconds")
        self._strategy_provider = strategy_provider
        self._backtests_provider = backtests_provider
        self._alerts_provider = alerts_provider
        self._poll_interval_seconds = poll_interval_seconds

    def get_summary(self, *, principal: CurrentUserPrincipal) -> DashboardSummaryResponse:
        generated_at = _format_required_datetime(datetime.now(UTC))
        account = DashboardAccountResponse(
            source=_source(
                status="available",
                code="account.available",
                message="Authenticated account principal is available",
                updated_at=generated_at,
            ),
            user_id=str(principal.user_id),
            paid_level=str(principal.paid_level),
        )
        strategies = _safe_strategies(provider=self._strategy_provider, principal=principal)
        backtests = _safe_backtests(provider=self._backtests_provider, principal=principal)
        alerts = _safe_alerts(provider=self._alerts_provider, principal=principal)
        sources = {
            "account": account.source,
            "strategies": strategies.source,
            "backtests": backtests.source,
            "alerts": alerts.source,
        }
        return DashboardSummaryResponse(
            generated_at=generated_at,
            poll_interval_seconds=self._poll_interval_seconds,
            sources=sources,
            account=account,
            strategies=strategies,
            backtests=backtests,
            alerts=alerts,
            links={
                "self": "/api/ui/dashboard/summary",
                "settings": "/settings",
                "strategies": "/strategies",
                "backtests": "/backtests",
            },
        )


class UnavailableStrategiesDashboardProvider:
    def __init__(self, *, code: str, message: str) -> None:
        self._code = code
        self._message = message

    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardStrategiesResponse:
        _ = principal
        return DashboardStrategiesResponse(
            source=_source(status="unavailable", code=self._code, message=self._message),
            total_count=None,
            active_count=None,
            items=[],
        )


class StrategyUseCaseDashboardProvider:
    def __init__(
        self,
        *,
        list_use_case: ListMyStrategiesUseCase,
        run_repository: StrategyRunRepository,
        item_limit: int = _PANEL_ITEM_LIMIT,
    ) -> None:
        if list_use_case is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyUseCaseDashboardProvider requires list_use_case")
        if run_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyUseCaseDashboardProvider requires run_repository")
        self._list_use_case = list_use_case
        self._run_repository = run_repository
        self._item_limit = item_limit

    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardStrategiesResponse:
        strategies = self._list_use_case.execute(
            current_user=StrategyCurrentUser(user_id=principal.user_id)
        )
        strategy_ids = {strategy.strategy_id for strategy in strategies}
        active_runs = {
            run.strategy_id: run
            for run in self._run_repository.list_active_runs()
            if run.user_id == principal.user_id and run.strategy_id in strategy_ids
        }
        items = [
            DashboardStrategyItemResponse(
                strategy_id=str(strategy.strategy_id),
                name=strategy.name,
                state=active_runs[strategy.strategy_id].state
                if strategy.strategy_id in active_runs
                else "idle",
                instrument_key=strategy.spec.instrument_key,
                timeframe=strategy.spec.timeframe.code,
                updated_at=_format_datetime(
                    active_runs[strategy.strategy_id].updated_at
                    if strategy.strategy_id in active_runs
                    else strategy.created_at
                ),
            )
            for strategy in strategies[: self._item_limit]
        ]
        return DashboardStrategiesResponse(
            source=_source(
                status="available",
                code="strategies.available",
                message="Strategy owner read-model is available",
            ),
            total_count=len(strategies),
            active_count=len(active_runs),
            items=items,
        )


class UnavailableBacktestsDashboardProvider:
    def __init__(self, *, code: str, message: str) -> None:
        self._code = code
        self._message = message

    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardBacktestsResponse:
        _ = principal
        return DashboardBacktestsResponse(
            source=_source(status="unavailable", code=self._code, message=self._message),
            active_count=None,
            items=[],
            next_cursor=None,
        )


class BacktestJobsDashboardProvider:
    def __init__(
        self,
        *,
        job_repository: BacktestJobRepository,
        item_limit: int = _PANEL_ITEM_LIMIT,
    ) -> None:
        if job_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestJobsDashboardProvider requires job_repository")
        self._job_repository = job_repository
        self._item_limit = item_limit

    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardBacktestsResponse:
        page = self._job_repository.list_for_user(
            query=BacktestJobListQuery(user_id=principal.user_id, limit=self._item_limit)
        )
        read_models = [build_backtest_job_read_model(job=job) for job in page.items]
        items = [
            DashboardBacktestJobResponse(
                job_id=model.job_id,
                state=model.state,
                pipeline_stage=model.progress.pipeline_stage,
                progress_percent=model.progress.percent,
                symbol=_extract_symbol(request=dict(model.request)),
                timeframe=_optional_string(dict(model.request).get("timeframe")),
                risk_mode=_optional_string(dict(model.request).get("risk_mode")),
                primary_metric=_optional_string(dict(model.ranking).get("primary_metric")),
                updated_at=_format_required_datetime(model.updated_at),
                links={
                    "page": f"/backtests/{model.job_id}",
                    "api": f"/api/backtests/jobs/{model.job_id}",
                },
            )
            for model in read_models
        ]
        return DashboardBacktestsResponse(
            source=_source(
                status="available",
                code="backtests.available",
                message="Backtest jobs read-model is available",
            ),
            active_count=sum(1 for model in read_models if model.state in {"queued", "running"}),
            items=items,
            next_cursor=None,
        )


class StaticAlertsDashboardProvider:
    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardAlertsResponse:
        _ = principal
        return DashboardAlertsResponse(
            source=_source(
                status="unavailable",
                code="alerts.unavailable",
                message="Alerts read-model is not accepted for Stage 4",
            ),
            items=[],
            next_cursor=None,
        )


def build_ui_dashboard_module(
    *,
    environ: Mapping[str, str],
    current_user_dependency: CurrentUserDependency,
) -> UiDashboardApiModule:
    query = DashboardSummaryQueryService(
        strategy_provider=_build_strategy_provider(environ=environ),
        backtests_provider=_build_backtests_provider(environ=environ),
        alerts_provider=StaticAlertsDashboardProvider(),
    )
    return UiDashboardApiModule(
        router=build_ui_dashboard_router(
            current_user_dependency=current_user_dependency,
            summary_query=query,
        ),
        summary_query=query,
    )


def _build_strategy_provider(*, environ: Mapping[str, str]) -> StrategyDashboardProvider:
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn:
        return UnavailableStrategiesDashboardProvider(
            code="strategies.unconfigured",
            message="Strategy storage is not configured for dashboard reads",
        )
    try:
        if not is_strategy_api_enabled(environ=environ):
            return UnavailableStrategiesDashboardProvider(
                code="strategies.disabled",
                message="Strategy API is disabled by runtime config",
            )
    except Exception as error:  # noqa: BLE001
        _ = error
        return UnavailableStrategiesDashboardProvider(
            code="strategies.config_unavailable",
            message="Strategy runtime config is unavailable",
        )
    gateway = PsycopgStrategyPostgresGateway(dsn=postgres_dsn)
    return StrategyUseCaseDashboardProvider(
        list_use_case=ListMyStrategiesUseCase(
            repository=PostgresStrategyRepository(gateway=gateway)
        ),
        run_repository=PostgresStrategyRunRepository(gateway=gateway),
    )


def _build_backtests_provider(*, environ: Mapping[str, str]) -> BacktestsDashboardProvider:
    postgres_dsn = environ.get("STRATEGY_PG_DSN", "").strip()
    if not postgres_dsn:
        return UnavailableBacktestsDashboardProvider(
            code="backtests.unconfigured",
            message="Backtest jobs storage is not configured for dashboard reads",
        )
    return BacktestJobsDashboardProvider(
        job_repository=PostgresBacktestJobRepository(
            gateway=PsycopgBacktestPostgresGateway(dsn=postgres_dsn)
        )
    )


def _safe_strategies(
    *,
    provider: StrategyDashboardProvider,
    principal: CurrentUserPrincipal,
) -> DashboardStrategiesResponse:
    try:
        return provider.get_panel(principal=principal)
    except Exception as error:  # noqa: BLE001
        _ = error
        return DashboardStrategiesResponse(
            source=_source(
                status="degraded",
                code="strategies.provider_failed",
                message="Strategy dashboard source failed",
            ),
            total_count=None,
            active_count=None,
            items=[],
        )


def _safe_backtests(
    *,
    provider: BacktestsDashboardProvider,
    principal: CurrentUserPrincipal,
) -> DashboardBacktestsResponse:
    try:
        return provider.get_panel(principal=principal)
    except Exception as error:  # noqa: BLE001
        _ = error
        return DashboardBacktestsResponse(
            source=_source(
                status="degraded",
                code="backtests.provider_failed",
                message="Backtest dashboard source failed",
            ),
            active_count=None,
            items=[],
            next_cursor=None,
        )


def _safe_alerts(
    *,
    provider: AlertsDashboardProvider,
    principal: CurrentUserPrincipal,
) -> DashboardAlertsResponse:
    try:
        return provider.get_panel(principal=principal)
    except Exception as error:  # noqa: BLE001
        _ = error
        return DashboardAlertsResponse(
            source=_source(
                status="degraded",
                code="alerts.provider_failed",
                message="Alerts dashboard source failed",
            ),
            items=[],
            next_cursor=None,
        )


def _source(
    *,
    status: DashboardSourceStatus,
    code: str,
    message: str,
    updated_at: str | None = None,
) -> DashboardSourceResponse:
    return DashboardSourceResponse(
        status=status,
        code=code,
        message=message,
        updated_at=updated_at,
    )


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _format_required_datetime(value)


def _format_required_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _extract_symbol(*, request: Mapping[str, object]) -> str | None:
    coordinates = request.get("coordinates")
    if isinstance(coordinates, Mapping):
        return _optional_string(coordinates.get("symbol"))
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


__all__ = [
    "BacktestJobsDashboardProvider",
    "DashboardSummaryQueryService",
    "StaticAlertsDashboardProvider",
    "StrategyUseCaseDashboardProvider",
    "UiDashboardApiModule",
    "UnavailableBacktestsDashboardProvider",
    "UnavailableStrategiesDashboardProvider",
    "build_ui_dashboard_module",
]
