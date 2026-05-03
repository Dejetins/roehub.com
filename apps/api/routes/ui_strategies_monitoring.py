from __future__ import annotations

from typing import Callable, Literal, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.dto import (
    StrategyEquityResponse,
    StrategyFillsResponse,
    StrategyMonitorResponse,
    StrategyPositionsResponse,
    StrategySnapshotResponse,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


class StrategyMonitoringQuery(Protocol):
    def get_monitor(
        self,
        *,
        principal: CurrentUserPrincipal,
        state: Literal["active", "all"],
        cursor: str | None,
    ) -> StrategyMonitorResponse:
        ...

    def get_snapshot(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
    ) -> StrategySnapshotResponse:
        ...

    def get_positions(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
        limit: int,
    ) -> StrategyPositionsResponse:
        ...

    def get_fills(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
        limit: int,
        cursor: str | None,
    ) -> StrategyFillsResponse:
        ...

    def get_equity(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
        range_name: str,
        points: int,
    ) -> StrategyEquityResponse:
        ...


def build_ui_strategies_monitoring_router(
    *,
    current_user_dependency: CurrentUserDependency,
    monitoring_query: StrategyMonitoringQuery,
) -> APIRouter:
    """
    Build owner-scoped Strategy monitoring read-model routes.

    Local contract:
    - browser paths: `/api/ui/strategies/monitor`,
      `/api/ui/strategies/{strategy_id}/snapshot`,
      `/api/ui/strategies/{strategy_id}/positions`,
      `/api/ui/strategies/{strategy_id}/fills`,
      `/api/ui/strategies/{strategy_id}/equity`
    - backend paths: same paths without the `/api` prefix
    - owner scope: identity current-user principal gates every Strategy repository query
    - request DTO: query params only; limits are capped server-side
    - response DTO: compact bounded monitoring DTOs, no fake positions/fills/equity
    - status codes: 200, 401, 403/404 from owner Strategy use-case, 422 validation
    - error payload: canonical RoehubError envelope via global handlers
    - pagination: cursor is offset-based for monitor/fills v1; fills are empty until sourced
    - cache identity: none; user-scoped live read-model, intended `no-store`
    - compatibility: compatible-change, additive `/ui/strategies/*` surface
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_strategies_monitoring_router requires current_user_dependency")
    if monitoring_query is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_strategies_monitoring_router requires monitoring_query")

    router = APIRouter(tags=["ui-strategies-monitoring"])

    def require_monitoring_user(request: Request) -> CurrentUserPrincipal:
        try:
            return current_user_dependency(request)
        except HTTPException as error:
            if error.status_code == 401:
                raise RoehubError(
                    code="auth.required",
                    message="Authentication is required",
                    details={},
                ) from error
            raise

    @router.get("/ui/strategies/monitor", response_model=StrategyMonitorResponse)
    def get_strategy_monitor(
        state: Literal["active", "all"] = "all",
        cursor: str | None = None,
        principal: CurrentUserPrincipal = Depends(require_monitoring_user),
    ) -> StrategyMonitorResponse:
        return monitoring_query.get_monitor(principal=principal, state=state, cursor=cursor)

    @router.get("/ui/strategies/{strategy_id}/snapshot", response_model=StrategySnapshotResponse)
    def get_strategy_snapshot(
        strategy_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_monitoring_user),
    ) -> StrategySnapshotResponse:
        return monitoring_query.get_snapshot(principal=principal, strategy_id=strategy_id)

    @router.get("/ui/strategies/{strategy_id}/positions", response_model=StrategyPositionsResponse)
    def get_strategy_positions(
        strategy_id: UUID,
        limit: int = Query(default=50, ge=1, le=50),
        principal: CurrentUserPrincipal = Depends(require_monitoring_user),
    ) -> StrategyPositionsResponse:
        return monitoring_query.get_positions(
            principal=principal,
            strategy_id=strategy_id,
            limit=limit,
        )

    @router.get("/ui/strategies/{strategy_id}/fills", response_model=StrategyFillsResponse)
    def get_strategy_fills(
        strategy_id: UUID,
        cursor: str | None = None,
        limit: int = Query(default=50, ge=1, le=50),
        principal: CurrentUserPrincipal = Depends(require_monitoring_user),
    ) -> StrategyFillsResponse:
        return monitoring_query.get_fills(
            principal=principal,
            strategy_id=strategy_id,
            limit=limit,
            cursor=cursor,
        )

    @router.get("/ui/strategies/{strategy_id}/equity", response_model=StrategyEquityResponse)
    def get_strategy_equity(
        strategy_id: UUID,
        range: str = "1d",  # noqa: A002 - public query parameter name is fixed.
        points: int = Query(default=600, ge=1, le=600),
        principal: CurrentUserPrincipal = Depends(require_monitoring_user),
    ) -> StrategyEquityResponse:
        return monitoring_query.get_equity(
            principal=principal,
            strategy_id=strategy_id,
            range_name=range,
            points=points,
        )

    return router


__all__ = ["StrategyMonitoringQuery", "build_ui_strategies_monitoring_router"]
