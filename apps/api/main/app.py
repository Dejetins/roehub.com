"""
FastAPI application factory for Roehub API.
"""

from __future__ import annotations

import os
from typing import Mapping

from fastapi import FastAPI

from apps.api.routes import build_indicators_router
from apps.api.wiring.modules import (
    bind_indicators_runtime_dependencies,
    build_identity_router,
    build_indicators_compute,
    build_indicators_registry,
)
from trading.contexts.identity.adapters.inbound.api.deps import (
    register_two_factor_required_exception_handler,
)
from trading.platform.config import load_indicators_compute_numba_config


def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    """
    Build FastAPI app with indicators and identity modules wired at startup.

    Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md,
      docs/architecture/identity/identity-telegram-login-user-model-v1.md,
      docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related: apps.api.routes.indicators,
      apps.api.routes.identity,
      apps.api.wiring.modules.identity,
      apps.api.wiring.modules.indicators,
      trading.contexts.indicators.application.ports.compute.indicator_compute

    Args:
        environ: Optional environment mapping override.
    Returns:
        FastAPI: Application instance with registered routers.
    Assumptions:
        Modules wiring performs fail-fast validation before first request.
    Raises:
        FileNotFoundError: If indicators config path is missing.
        ValueError: If config parsing/validation fails for indicators or identity.
    Side Effects:
        Reads indicators YAML, performs Numba warmup, and validates identity runtime settings.
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
    register_two_factor_required_exception_handler(app=app)
    app.include_router(build_identity_router(environ=effective_environ))
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
