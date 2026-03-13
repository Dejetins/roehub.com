from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.monitoring import install_metrics_middleware
from apps.api.routes import build_operations_router


def _build_client() -> TestClient:
    """
    Build isolated FastAPI client with operations routes and metrics middleware.

    Args:
        None.
    Returns:
        TestClient: In-memory HTTP client for `/health` and `/metrics`.
    Assumptions:
        Operations router has no external runtime dependencies.
    Raises:
        None.
    Side Effects:
        Registers Prometheus middleware on a throwaway FastAPI app.
    """
    app = FastAPI()
    install_metrics_middleware(app=app)

    @app.get("/ping")
    def _ping() -> dict[str, str]:
        """
        Return deterministic payload for metrics middleware smoke tests.

        Args:
            None.
        Returns:
            dict[str, str]: Static success payload.
        Assumptions:
            Route exists only in test app.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {"ok": "1"}

    app.include_router(build_operations_router())
    return TestClient(app)


def test_health_endpoint_returns_stable_success_payload() -> None:
    """
    Verify `/health` returns HTTP 200 with stable minimal payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Monitoring stack depends on a deterministic success contract.
    Raises:
        AssertionError: If status code or body differs from contract.
    Side Effects:
        None.
    """
    client = _build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_exposes_prometheus_payload() -> None:
    """
    Verify `/metrics` exposes Prometheus counters and in-flight gauge names.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Prometheus middleware registered before router inclusion.
    Raises:
        AssertionError: If exposition content type or metric names are missing.
    Side Effects:
        Performs one `/health` request to seed metrics counters.
    """
    client = _build_client()

    client.get("/ping")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_in_progress" in response.text
    assert "http_requests_total" in response.text
