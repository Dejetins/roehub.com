from __future__ import annotations

from fastapi import APIRouter

from apps.api.routes.ui_backtests import build_ui_backtests_router as build_ui_backtests_api_router
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency


def build_ui_backtests_router(
    *,
    current_user_dependency: RequireCurrentUserDependency,
    jobs_use_case: BacktestJobsUseCase | None,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_backtests_router requires current_user_dependency")
    return build_ui_backtests_api_router(
        current_user_dependency=current_user_dependency,
        jobs_use_case=jobs_use_case,
    )


__all__ = ["build_ui_backtests_router"]
