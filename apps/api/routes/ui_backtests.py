from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request

from apps.api.dto import UiBacktestCountersResponse, build_ui_backtest_counters_response
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_ui_backtests_router(
    *,
    current_user_dependency: CurrentUserDependency,
    jobs_use_case: BacktestJobsUseCase | None,
) -> APIRouter:
    """
    Build additive UI-only backtests endpoints.
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_backtests_router requires current_user_dependency")

    router = APIRouter(tags=["ui-backtests"])

    def require_backtest_user(request: Request) -> CurrentUserPrincipal:
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

    def require_jobs_use_case() -> BacktestJobsUseCase:
        if jobs_use_case is None:
            raise RoehubError(
                code="backtest.queue_saturated",
                message="Backtest jobs service is not configured",
                details={"reason": "job_repository_unavailable"},
            )
        return jobs_use_case

    @router.get("/ui/backtests/counters", response_model=UiBacktestCountersResponse)
    def get_ui_backtest_counters(
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestJobsUseCase = Depends(require_jobs_use_case),
    ) -> UiBacktestCountersResponse:
        result = use_case.counters(user_id=principal.user_id)
        return build_ui_backtest_counters_response(result=result)

    return router


__all__ = ["build_ui_backtests_router"]
