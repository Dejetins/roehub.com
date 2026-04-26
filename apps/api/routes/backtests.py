from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Body, Depends, Request

from apps.api.dto import (
    BacktestPreflightResponse,
    BacktestRuntimeDefaultsResponse,
    build_backtest_preflight_response,
    build_backtest_runtime_defaults_response,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightRejected,
    BacktestPreflightService,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_backtests_router(
    *,
    runtime_defaults_service: BacktestRuntimeDefaultsService,
    preflight_service: BacktestPreflightService,
    current_user_dependency: CurrentUserDependency,
) -> APIRouter:
    """
    Build Iteration 1 public backtests API shell.
    """
    if runtime_defaults_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires runtime_defaults_service")
    if preflight_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires preflight_service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtests_router requires current_user_dependency")

    router = APIRouter(tags=["backtests"])

    @router.get("/backtests/runtime-defaults", response_model=BacktestRuntimeDefaultsResponse)
    def get_backtest_runtime_defaults(
        _principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestRuntimeDefaultsResponse:
        defaults = runtime_defaults_service.execute()
        return build_backtest_runtime_defaults_response(defaults=defaults)

    @router.post("/backtests/preflight", response_model=BacktestPreflightResponse)
    def post_backtest_preflight(
        payload: Any = Body(...),
        _principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> BacktestPreflightResponse:
        try:
            result = preflight_service.execute(payload)
        except BacktestPreflightRejected as error:
            raise RoehubError(
                code=error.error_code,
                message=error.message,
                details=error.details(),
            ) from error
        return build_backtest_preflight_response(result=result)

    return router


__all__ = ["build_backtests_router"]
