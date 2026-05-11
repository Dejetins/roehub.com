from __future__ import annotations

from typing import Mapping

from fastapi import APIRouter

from apps.api.routes import (
    build_backtest_ai_config_router as build_backtest_ai_config_api_router,
)
from apps.api.wiring.modules.backtest import build_backtest_ai_configurator_use_cases
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigFakeWorkerUseCase,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency


def build_backtest_ai_config_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_ai_config_router requires current_user_dependency")
    use_cases = build_backtest_ai_configurator_use_cases(environ=environ)
    return build_backtest_ai_config_api_router(
        current_user_dependency=current_user_dependency,
        jobs_use_case=None if use_cases is None else use_cases.jobs,
    )


def build_backtest_ai_config_fake_worker(
    *,
    environ: Mapping[str, str],
) -> BacktestAiConfigFakeWorkerUseCase | None:
    use_cases = build_backtest_ai_configurator_use_cases(environ=environ)
    if use_cases is None:
        return None
    return BacktestAiConfigFakeWorkerUseCase(
        job_repository=use_cases.jobs.repository,
        lease_repository=use_cases.lease_repository,
        lease_seconds=use_cases.runtime_config.queue.lease_seconds,
        max_attempts=use_cases.runtime_config.queue.repair_attempts + 1,
    )


__all__ = [
    "build_backtest_ai_config_fake_worker",
    "build_backtest_ai_config_router",
]
