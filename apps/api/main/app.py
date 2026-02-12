"""
FastAPI application factory for Roehub API.
"""

from __future__ import annotations

import os
from typing import Mapping

from fastapi import FastAPI

from apps.api.routes import build_indicators_router
from apps.api.wiring.modules import build_indicators_compute, build_indicators_registry
from trading.platform.config import load_indicators_compute_numba_config


def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    """
    Build FastAPI app with indicators registry wired at startup.

    Args:
        environ: Optional environment mapping override.
    Returns:
        FastAPI: Application instance with registered routers.
    Assumptions:
        Registry wiring performs fail-fast validation before first request.
    Raises:
        FileNotFoundError: If indicators config path is missing.
        ValueError: If config parsing/validation fails.
    Side Effects:
        Reads indicators YAML and performs Numba warmup at application creation time.
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
    app.state.indicators_compute = compute
    app.include_router(
        build_indicators_router(
            registry=registry,
            max_variants_per_compute=compute_config.max_variants_per_compute,
            max_compute_bytes_total=compute_config.max_compute_bytes_total,
        )
    )
    return app


app = create_app()
