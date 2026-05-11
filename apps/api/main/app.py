"""
FastAPI application factory for Roehub API.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping

from fastapi import FastAPI

from apps.api.common import register_api_error_handlers
from apps.api.monitoring import install_metrics_middleware
from apps.api.routes import build_indicators_router, build_operations_router
from apps.api.wiring.modules import (
    bind_indicators_runtime_dependencies,
    build_backtest_ai_config_router,
    build_backtests_router,
    build_identity_api_module,
    build_indicators_compute,
    build_indicators_registry,
    build_market_data_reference_router,
    build_strategy_router,
    build_ui_account_router,
    build_ui_backtests_router,
    build_ui_dashboard_router,
    build_ui_strategies_dashboard_router,
    is_strategy_api_enabled,
)
from trading.platform.config import load_indicators_compute_numba_config

log = logging.getLogger(__name__)


def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    """
    Build FastAPI app with indicators, identity, and strategy modules wired at startup.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md,
      docs/architecture/identity/identity-telegram-login-user-model-v1.md,
      docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md,
      docs/architecture/market_data/market-data-reference-api-v1.md,
      docs/architecture/api/api-errors-and-422-payload-v1.md,
      docs/runbooks/mac-studio-monitoring-plan.md
    Related: apps.api.routes.indicators,
      apps.api.routes.operations,
      apps.api.routes.identity,
      apps.api.wiring.modules.identity,
      apps.api.wiring.modules.indicators,
      trading.contexts.indicators.application.ports.compute.indicator_compute,
      apps.api.monitoring

    Args:
        environ: Optional environment mapping override.
    Returns:
        FastAPI: Application instance with registered routers.
    Assumptions:
        Modules wiring performs fail-fast validation before first request.
    Raises:
        FileNotFoundError: If indicators config path is missing.
        ValueError: If config parsing/validation fails for indicators, identity, or strategy.
    Side Effects:
        Reads indicators YAML, performs Numba warmup, and validates
        identity/strategy runtime settings.
    """
    effective_environ = os.environ if environ is None else environ
    registry = build_indicators_registry(environ=effective_environ)
    compute_config = load_indicators_compute_numba_config(environ=effective_environ)
    compute = build_indicators_compute(
        environ=effective_environ,
        config=compute_config,
    )

    app = FastAPI(
        title="Roehub API",
        version="1.0.0",
    )
    install_metrics_middleware(app=app)
    register_api_error_handlers(app=app)
    app.include_router(build_operations_router())
    identity_module = build_identity_api_module(environ=effective_environ)
    app.include_router(identity_module.router)
    if is_strategy_api_enabled(environ=effective_environ):
        app.include_router(
            build_strategy_router(
                environ=effective_environ,
                current_user_dependency=identity_module.current_user_dependency,
            )
        )
    else:
        log.info("strategy API router disabled by strategy runtime config")
    app.include_router(
        build_market_data_reference_router(
            environ=effective_environ,
            current_user_dependency=identity_module.current_user_dependency,
        )
    )
    app.include_router(
        build_backtests_router(
            environ=effective_environ,
            current_user_dependency=identity_module.current_user_dependency,
        )
    )
    app.include_router(
        build_backtest_ai_config_router(
            environ=effective_environ,
            current_user_dependency=identity_module.current_user_dependency,
        )
    )
    app.include_router(
        build_ui_account_router(
            environ=effective_environ,
            current_user_dependency=identity_module.current_user_dependency,
        )
    )
    app.include_router(
        build_ui_dashboard_router(
            environ=effective_environ,
            current_user_dependency=identity_module.current_user_dependency,
        )
    )
    app.include_router(
        build_ui_backtests_router(
            environ=effective_environ,
            current_user_dependency=identity_module.current_user_dependency,
        )
    )
    app.include_router(
        build_ui_strategies_dashboard_router(
            environ=effective_environ,
            current_user_dependency=identity_module.current_user_dependency,
        )
    )
    bind_indicators_runtime_dependencies(
        app_state=app.state,
        compute=compute,
        candle_feed=None,
    )
    app.include_router(
        build_indicators_router(
            registry=registry,
            compute=compute,
            max_variants_per_compute=compute_config.max_variants_per_compute,
            max_compute_bytes_total=compute_config.max_compute_bytes_total,
        )
    )
    return app


app = create_app()
