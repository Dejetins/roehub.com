"""
Prometheus instrumentation helpers for Roehub API.

Docs:
  - docs/runbooks/mac-studio-monitoring-plan.md
  - docs/runbooks/mac-studio-backend-operations.md
Related:
  - apps/api/routes/operations.py
  - apps/api/main/app.py
"""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_EXCLUDED_PATHS = frozenset({"/health", "/metrics"})
_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests handled by Roehub API.",
    ("method", "path", "status_code"),
)
_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Duration of Roehub API HTTP requests in seconds.",
    ("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of in-flight Roehub API HTTP requests.",
)
_STRATEGY_VARIANT_LAUNCH_TOTAL = Counter(
    "strategy_variant_launch_total",
    "Strategy create-from-backtest-variant attempts by result.",
    ("result", "reason"),
)
_LIVE_STRATEGY_PROFILE_READINESS_TOTAL = Counter(
    "live_strategy_profile_readiness_total",
    "Live strategy profile readiness evaluations by result.",
    ("status", "reason"),
)
_STRATEGY_VARIANT_COMPATIBILITY_TOTAL = Counter(
    "strategy_variant_compatibility_total",
    "Strategy variant compatibility checks by result.",
    ("state", "reason"),
)
_MARKET_DATA_READINESS_TOTAL = Counter(
    "market_data_readiness_total",
    "Market-data readiness checks by result.",
    ("state", "reason"),
)


def install_metrics_middleware(*, app: FastAPI) -> None:
    """
    Attach Prometheus HTTP instrumentation middleware to FastAPI app.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
      - docs/runbooks/mac-studio-backend-operations.md
    Related:
      - apps/api/routes/operations.py
      - apps/api/main/app.py

    Args:
        app: FastAPI application instance to instrument.
    Returns:
        None.
    Assumptions:
        Application runs as a single-process Uvicorn service in current production topology.
    Raises:
        None.
    Side Effects:
        Mutates the app middleware stack and registers Prometheus observations per request.
    """

    @app.middleware("http")
    async def _prometheus_http_middleware(request: Request, call_next) -> Response:
        """
        Record request counters, latency histogram, and in-flight gauge.

        Args:
            request: Incoming FastAPI request.
            call_next: Next middleware/router callable.
        Returns:
            Response: HTTP response returned by downstream app.
        Assumptions:
            Route metadata is available in `request.scope["route"]` after dispatch.
        Raises:
            Exception: Propagates downstream application exceptions after metrics update.
        Side Effects:
            Updates global Prometheus metrics in the default registry.
        """
        _REQUESTS_IN_PROGRESS.inc()
        start = perf_counter()
        method = request.method
        try:
            response = await call_next(request)
        except Exception:
            duration = perf_counter() - start
            path_label = _resolve_path_label(request=request)
            if path_label not in _EXCLUDED_PATHS:
                _REQUESTS_TOTAL.labels(
                    method=method,
                    path=path_label,
                    status_code="500",
                ).inc()
                _REQUEST_DURATION_SECONDS.labels(method=method, path=path_label).observe(duration)
            raise
        finally:
            _REQUESTS_IN_PROGRESS.dec()

        duration = perf_counter() - start
        path_label = _resolve_path_label(request=request)
        if path_label in _EXCLUDED_PATHS:
            return response

        _REQUESTS_TOTAL.labels(
            method=method,
            path=path_label,
            status_code=str(response.status_code),
        ).inc()
        _REQUEST_DURATION_SECONDS.labels(method=method, path=path_label).observe(duration)
        return response


def build_metrics_response() -> Response:
    """
    Render Prometheus exposition payload for `/metrics`.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
    Related:
      - apps/api/routes/operations.py
      - apps/api/main/app.py

    Args:
        None.
    Returns:
        Response: Plain-text Prometheus exposition response.
    Assumptions:
        Default Prometheus registry contains API, process, and Python runtime metrics.
    Raises:
        None.
    Side Effects:
        Serializes the global Prometheus registry.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_strategy_variant_launch(*, result: str, reason: str = "none") -> None:
    """
    Record one create-from-backtest-variant attempt without user or strategy labels.
    """
    _STRATEGY_VARIANT_LAUNCH_TOTAL.labels(result=result, reason=reason).inc()


def record_live_strategy_profile_readiness(*, status: str, reason: str) -> None:
    """
    Record one profile readiness result without user, strategy, or connection labels.
    """
    bounded_reason = reason if reason else "unknown"
    _LIVE_STRATEGY_PROFILE_READINESS_TOTAL.labels(
        status=status,
        reason=bounded_reason[:80],
    ).inc()


def record_strategy_variant_compatibility(*, state: str, reason: str) -> None:
    _STRATEGY_VARIANT_COMPATIBILITY_TOTAL.labels(
        state=state,
        reason=(reason or "unknown")[:80],
    ).inc()


def record_market_data_readiness(*, state: str, reason: str) -> None:
    _MARKET_DATA_READINESS_TOTAL.labels(
        state=state,
        reason=(reason or "unknown")[:80],
    ).inc()


def _resolve_path_label(*, request: Request) -> str:
    """
    Resolve deterministic path label for one HTTP request.

    Docs:
      - docs/runbooks/mac-studio-monitoring-plan.md
    Related:
      - apps/api/monitoring.py

    Args:
        request: FastAPI request after routing.
    Returns:
        str: Route template path when available, otherwise raw URL path.
    Assumptions:
        FastAPI stores matched route object under `scope["route"]`.
    Raises:
        None.
    Side Effects:
        None.
    """
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return request.url.path


__all__ = [
    "build_metrics_response",
    "install_metrics_middleware",
    "record_live_strategy_profile_readiness",
    "record_market_data_readiness",
    "record_strategy_variant_launch",
    "record_strategy_variant_compatibility",
]
