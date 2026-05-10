from __future__ import annotations

from typing import Callable, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.dto.ui_backtests import BacktestWorkstationResponse
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


class BacktestWorkstationService(Protocol):
    def get_workstation(
        self,
        *,
        principal: CurrentUserPrincipal,
        cursor: str | None,
        state: str | None,
        query: str,
        refresh: Literal["initial", "auto", "manual"],
    ) -> BacktestWorkstationResponse:
        ...


def build_ui_backtests_router(
    *,
    workstation_service: BacktestWorkstationService,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    if workstation_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_backtests_router requires workstation_service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_backtests_router requires current_user_dependency")

    router = APIRouter(tags=["ui-backtests"])

    def require_backtest_workstation_user(request: Request) -> CurrentUserPrincipal:
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

    @router.get("/ui/backtests/workstation", response_model=BacktestWorkstationResponse)
    def get_backtest_workstation(
        cursor: str | None = Query(default=None),
        state: str | None = Query(default=None),
        query: str = Query(default=""),
        refresh: Literal["initial", "auto", "manual"] = Query(default="initial"),
        principal: CurrentUserPrincipal = Depends(require_backtest_workstation_user),
    ) -> BacktestWorkstationResponse:
        normalized_cursor = cursor.strip() if cursor else None
        normalized_state = state.strip() if state else None
        normalized_query = query.strip()
        return workstation_service.get_workstation(
            principal=principal,
            cursor=normalized_cursor or None,
            state=normalized_state or None,
            query=normalized_query,
            refresh=refresh,
        )

    return router


__all__ = ["build_ui_backtests_router"]
