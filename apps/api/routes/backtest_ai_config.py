from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Request

from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_backtest_ai_config_router(
    *,
    current_user_dependency: CurrentUserDependency,
    jobs_use_case: object | None = None,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_ai_config_router requires current_user_dependency")
    _ = jobs_use_case
    return APIRouter(tags=["backtest-ai-config-retired"])


__all__ = ["build_backtest_ai_config_router"]
