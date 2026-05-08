from __future__ import annotations

from typing import Callable, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.dto.ui_strategies_dashboard import StrategyDashboardResponse
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


class StrategyDashboardService(Protocol):
    def get_dashboard(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: str | None,
        state: Literal["active", "all"],
        cursor: str | None,
        refresh: Literal["initial", "auto", "manual"],
    ) -> StrategyDashboardResponse:
        ...


def build_ui_strategies_dashboard_router(
    *,
    dashboard_service: StrategyDashboardService,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    if dashboard_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_strategies_dashboard_router requires dashboard_service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_strategies_dashboard_router requires current_user_dependency")

    router = APIRouter(tags=["ui-strategies-dashboard"])

    def require_strategy_dashboard_user(request: Request) -> CurrentUserPrincipal:
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

    @router.get("/ui/strategies/dashboard", response_model=StrategyDashboardResponse)
    def get_strategy_dashboard(
        strategy_id: str | None = Query(default=None),
        state: Literal["active", "all"] = Query(default="all"),
        cursor: str | None = Query(default=None),
        refresh: Literal["initial", "auto", "manual"] = Query(default="initial"),
        principal: CurrentUserPrincipal = Depends(require_strategy_dashboard_user),
    ) -> StrategyDashboardResponse:
        normalized_strategy_id = strategy_id.strip() if strategy_id else None
        normalized_cursor = cursor.strip() if cursor else None
        return dashboard_service.get_dashboard(
            principal=principal,
            strategy_id=normalized_strategy_id or None,
            state=state,
            cursor=normalized_cursor or None,
            refresh=refresh,
        )

    return router
