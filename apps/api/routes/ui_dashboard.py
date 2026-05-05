from __future__ import annotations

from typing import Callable, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from apps.api.dto.ui_dashboard import DashboardSummaryResponse
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


class DashboardSummaryService(Protocol):
    def get_summary(
        self,
        *,
        principal: CurrentUserPrincipal,
        refresh: Literal["initial", "auto", "manual"],
    ) -> DashboardSummaryResponse:
        ...


def build_ui_dashboard_router(
    *,
    summary_service: DashboardSummaryService,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    if summary_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_dashboard_router requires summary_service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_dashboard_router requires current_user_dependency")

    router = APIRouter(tags=["ui-dashboard"])

    def require_dashboard_user(request: Request) -> CurrentUserPrincipal:
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

    @router.get("/ui/dashboard/summary", response_model=DashboardSummaryResponse)
    def get_dashboard_summary(
        refresh: Literal["initial", "auto", "manual"] = Query(default="initial"),
        principal: CurrentUserPrincipal = Depends(require_dashboard_user),
    ) -> DashboardSummaryResponse:
        return summary_service.get_summary(principal=principal, refresh=refresh)

    return router
