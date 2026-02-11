"""
Indicators API routes.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
"""

from __future__ import annotations

from fastapi import APIRouter

from apps.api.dto import IndicatorsResponse, build_indicators_response
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry


def build_indicators_router(*, registry: IndicatorRegistry) -> APIRouter:
    """
    Build API router exposing merged indicator registry endpoint.

    Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md

    Args:
        registry: Indicator registry application port implementation.
    Returns:
        APIRouter: Router with `GET /indicators` endpoint.
    Assumptions:
        Registry is fully initialized and validated during app wiring.
    Raises:
        None.
    Side Effects:
        None.
    """
    router = APIRouter(tags=["indicators"])

    @router.get("/indicators", response_model=IndicatorsResponse)
    def get_indicators() -> IndicatorsResponse:
        """
        Return merged registry view (hard defs + defaults).

        Args:
            None.
        Returns:
            IndicatorsResponse: Deterministic merged registry payload.
        Assumptions:
            Registry order is deterministic and defaults already validated.
        Raises:
            None.
        Side Effects:
            None.
        """
        views = registry.list_merged()
        return build_indicators_response(views=views)

    return router
