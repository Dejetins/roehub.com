"""
Operational API routes for health and Prometheus metrics.

Docs:
  - docs/runbooks/mac-studio-monitoring-plan.md
  - docs/runbooks/mac-studio-backend-operations.md
Related:
  - apps/api/monitoring.py
  - apps/api/main/app.py
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from apps.api.monitoring import build_metrics_response


def build_operations_router() -> APIRouter:
    """
    Build unauthenticated operations router for liveness and metrics endpoints.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
      - docs/runbooks/mac-studio-backend-operations.md
    Related:
      - apps/api/monitoring.py
      - apps/api/main/app.py

    Args:
        None.
    Returns:
        APIRouter: Router exposing `/health` and `/metrics`.
    Assumptions:
        Endpoints must remain additive and stable for Prometheus and Blackbox checks.
    Raises:
        None.
    Side Effects:
        None.
    """
    router = APIRouter(tags=["ops"])

    @router.get("/health")
    def get_health() -> dict[str, str]:
        """
        Return stable API health contract for monitoring probes.

        Args:
            None.
        Returns:
            dict[str, str]: Deterministic success payload.
        Assumptions:
            FastAPI process startup already validated required runtime dependencies.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {"status": "ok"}

    @router.get("/metrics", include_in_schema=False)
    def get_metrics() -> Response:
        """
        Return Prometheus exposition payload for API runtime metrics.

        Args:
            None.
        Returns:
            Response: Prometheus text exposition.
        Assumptions:
            Metrics registry is populated by `install_metrics_middleware`.
        Raises:
            None.
        Side Effects:
            Serializes current Prometheus registry state.
        """
        return build_metrics_response()

    return router


__all__ = ["build_operations_router"]
