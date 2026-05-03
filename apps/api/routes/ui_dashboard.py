from __future__ import annotations

from typing import Callable, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request

from apps.api.dto import DashboardSummaryResponse
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


class DashboardSummaryQuery(Protocol):
    def get_summary(self, *, principal: CurrentUserPrincipal) -> DashboardSummaryResponse:
        """
        Return owner-scoped compact dashboard summary for the authenticated principal.
        """
        ...


def build_ui_dashboard_router(
    *,
    current_user_dependency: CurrentUserDependency,
    summary_query: DashboardSummaryQuery,
) -> APIRouter:
    """
    Build protected UI dashboard read-model router.

    Local contract:
    - browser path: `GET /api/ui/dashboard/summary`
    - backend path: `GET /ui/dashboard/summary`
    - owner scope: identity current-user principal gates all source queries
    - request DTO: none
    - response DTO: `DashboardSummaryResponse` with per-panel source state
    - status: 200, 401 via `auth.required`, 422 for framework validation, 500 for bugs
    - pagination: none in summary; panels are bounded to small recent slices
    - cache identity: none; response is user-scoped live read-model with `no-store`
    - compatibility: compatible-change, additive `/ui/dashboard/*` surface
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_dashboard_router requires current_user_dependency")
    if summary_query is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_dashboard_router requires summary_query")

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
        principal: CurrentUserPrincipal = Depends(require_dashboard_user),
    ) -> DashboardSummaryResponse:
        return summary_query.get_summary(principal=principal)

    return router


__all__ = ["DashboardSummaryQuery", "build_ui_dashboard_router"]
