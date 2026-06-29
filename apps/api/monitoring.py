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
_EXCHANGE_ACCOUNT_STATE_SYNC_TOTAL = Counter(
    "exchange_account_state_sync_total",
    "Exchange account projection sync/readiness checks by result.",
    ("status", "reason"),
)
_EXCHANGE_CONFIG_GUARD_TOTAL = Counter(
    "exchange_config_guard_total",
    "Verify-only exchange config guard checks by result.",
    ("status", "reason"),
)
_EXCHANGE_ACCOUNT_PROJECTION_STALENESS_SECONDS = Gauge(
    "exchange_account_projection_staleness_seconds",
    "Latest observed exchange account projection age in seconds.",
)
_STRATEGY_POSITION_OWNERSHIP_TOTAL = Counter(
    "strategy_position_ownership_total",
    "Strategy position ownership reserve/release/conflict outcomes.",
    ("result", "reason"),
)
_STRATEGY_CAPITAL_RESERVATION_TOTAL = Counter(
    "strategy_capital_reservation_total",
    "Strategy capital reservation outcomes.",
    ("result", "reason"),
)
_STRATEGY_PAPER_ACCOUNTING_TOTAL = Counter(
    "strategy_paper_accounting_total",
    "Strategy paper order/fill/accounting outcomes.",
    ("result", "reason"),
)
_EXECUTION_SOURCE_EVENT_TOTAL = Counter(
    "execution_source_event_total",
    "Execution source events recorded by producer type.",
    ("source_type", "result"),
)
_EXECUTION_INTENT_TOTAL = Counter(
    "execution_intent_total",
    "Execution intent ingress outcomes.",
    ("source_type", "result", "reason"),
)
_EXECUTION_ORDER_MODEL_REJECTED_TOTAL = Counter(
    "execution_order_model_rejected_total",
    "Unsupported execution order model rejections.",
    ("source_type", "reason"),
)
_EXECUTION_RISK_GATE_TOTAL = Counter(
    "execution_risk_gate_total",
    "Execution risk gate decisions by source type and bounded reason.",
    ("source_type", "result", "reason"),
)
_EXECUTION_RISK_GATE_LATENCY_SECONDS = Histogram(
    "execution_risk_gate_latency_seconds",
    "Execution risk gate evaluation duration in seconds.",
    ("source_type", "result"),
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
_EXECUTION_DISPATCH_TOTAL = Counter(
    "execution_dispatch_total",
    "Execution intent Redis dispatch outcomes.",
    ("result", "reason"),
)
_EXECUTION_DISPATCH_RETRY_TOTAL = Counter(
    "execution_dispatch_retry_total",
    "Execution intent Redis dispatch retry outcomes.",
    ("reason",),
)
_EXECUTION_DISPATCH_DLQ_TOTAL = Counter(
    "execution_dispatch_dlq_total",
    "Execution intent Redis dispatch quarantine/DLQ outcomes.",
    ("reason",),
)
_EXECUTION_DISPATCH_BACKPRESSURE_TOTAL = Counter(
    "execution_dispatch_backpressure_total",
    "Execution intent Redis dispatch backpressure outcomes.",
    ("reason",),
)
_EXECUTION_DISPATCH_REDIS_ERRORS_TOTAL = Counter(
    "execution_dispatch_redis_errors_total",
    "Execution intent Redis dispatch transport errors.",
    ("reason",),
)
_EXECUTION_NOTIFICATION_OUTBOX_TOTAL = Counter(
    "execution_notification_outbox_total",
    "Execution notification outbox events by type and producer source.",
    ("event_type", "source_type", "severity"),
)
_ADMIN_NOTIFICATIONS_TOTAL = Counter(
    "admin_notifications_total",
    "Admin notification events and deliveries by bounded category, severity, and status.",
    ("category", "severity", "status"),
)
_NOTIFICATIONS_DELIVERY_UNKNOWN_TOTAL = Counter(
    "notifications_delivery_unknown_total",
    "Notification deliveries that entered unknown provider state.",
    ("provider", "channel", "category"),
)
_NOTIFICATIONS_PENDING_OLDEST_AGE_SECONDS = Gauge(
    "notifications_pending_oldest_age_seconds",
    "Oldest pending notification delivery age by bounded provider, channel, and severity.",
    ("provider", "channel", "severity"),
)
_NOTIFICATIONS_DELIVERIES_RETRY_TOTAL = Counter(
    "notifications_deliveries_retry_total",
    "Notification delivery retry outcomes by bounded provider, channel, and reason.",
    ("provider", "channel", "reason"),
)
_NOTIFICATIONS_WORKER_UP = Gauge(
    "notifications_worker_up",
    "Notification worker health gauge where 1 is up and 0 is down.",
    ("worker",),
)
_NOTIFICATIONS_REPORT_SCHEDULE_MISSED_TOTAL = Counter(
    "notifications_report_schedule_missed_total",
    "Missed scheduled notification reports by report type and timezone.",
    ("report_type", "timezone"),
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


def record_exchange_account_state_sync(
    *, status: str, reason: str, age_seconds: int | None = None
) -> None:
    _EXCHANGE_ACCOUNT_STATE_SYNC_TOTAL.labels(
        status=status,
        reason=(reason or "unknown")[:80],
    ).inc()
    if age_seconds is not None:
        _EXCHANGE_ACCOUNT_PROJECTION_STALENESS_SECONDS.set(max(0, age_seconds))


def record_exchange_config_guard(*, status: str, reason: str) -> None:
    _EXCHANGE_CONFIG_GUARD_TOTAL.labels(
        status=status,
        reason=(reason or "unknown")[:80],
    ).inc()


def record_strategy_position_ownership(*, result: str, reason: str) -> None:
    _STRATEGY_POSITION_OWNERSHIP_TOTAL.labels(
        result=result,
        reason=(reason or "unknown")[:80],
    ).inc()


def record_strategy_capital_reservation(*, result: str, reason: str) -> None:
    _STRATEGY_CAPITAL_RESERVATION_TOTAL.labels(
        result=result,
        reason=(reason or "unknown")[:80],
    ).inc()


def record_strategy_paper_accounting(*, result: str, reason: str) -> None:
    _STRATEGY_PAPER_ACCOUNTING_TOTAL.labels(
        result=result,
        reason=(reason or "unknown")[:80],
    ).inc()


def record_execution_source_event(*, source_type: str, result: str) -> None:
    _EXECUTION_SOURCE_EVENT_TOTAL.labels(
        source_type=(source_type or "unknown")[:80],
        result=(result or "unknown")[:80],
    ).inc()


def record_execution_intent(*, source_type: str, result: str, reason: str) -> None:
    _EXECUTION_INTENT_TOTAL.labels(
        source_type=(source_type or "unknown")[:80],
        result=(result or "unknown")[:80],
        reason=(reason or "unknown")[:80],
    ).inc()


def record_execution_order_model_rejected(*, source_type: str, reason: str) -> None:
    _EXECUTION_ORDER_MODEL_REJECTED_TOTAL.labels(
        source_type=(source_type or "unknown")[:80],
        reason=(reason or "unknown")[:80],
    ).inc()


def record_execution_risk_gate(
    *, source_type: str, result: str, reason: str, latency_seconds: float
) -> None:
    bounded_source = (source_type or "unknown")[:80]
    bounded_result = (result or "unknown")[:80]
    _EXECUTION_RISK_GATE_TOTAL.labels(
        source_type=bounded_source,
        result=bounded_result,
        reason=(reason or "unknown")[:80],
    ).inc()
    _EXECUTION_RISK_GATE_LATENCY_SECONDS.labels(
        source_type=bounded_source,
        result=bounded_result,
    ).observe(max(0.0, latency_seconds))


def record_execution_dispatch(*, result: str, reason: str) -> None:
    _EXECUTION_DISPATCH_TOTAL.labels(
        result=(result or "unknown")[:80],
        reason=(reason or "unknown")[:80],
    ).inc()


def record_execution_dispatch_retry(*, reason: str) -> None:
    _EXECUTION_DISPATCH_RETRY_TOTAL.labels(reason=(reason or "unknown")[:80]).inc()


def record_execution_dispatch_dlq(*, reason: str) -> None:
    _EXECUTION_DISPATCH_DLQ_TOTAL.labels(reason=(reason or "unknown")[:80]).inc()


def record_execution_dispatch_backpressure(*, reason: str) -> None:
    _EXECUTION_DISPATCH_BACKPRESSURE_TOTAL.labels(reason=(reason or "unknown")[:80]).inc()


def record_execution_dispatch_redis_error(*, reason: str) -> None:
    _EXECUTION_DISPATCH_REDIS_ERRORS_TOTAL.labels(reason=(reason or "unknown")[:80]).inc()


def record_execution_notification_outbox(
    *, event_type: str, source_type: str, severity: str
) -> None:
    _EXECUTION_NOTIFICATION_OUTBOX_TOTAL.labels(
        event_type=(event_type or "unknown")[:80],
        source_type=(source_type or "unknown")[:80],
        severity=(severity or "unknown")[:80],
    ).inc()


def record_admin_notification(*, category: str, severity: str, status: str) -> None:
    _ADMIN_NOTIFICATIONS_TOTAL.labels(
        category=(category or "unknown")[:80],
        severity=(severity or "unknown")[:80],
        status=(status or "unknown")[:80],
    ).inc()


def record_notification_delivery_unknown(
    *, provider: str, channel: str, category: str
) -> None:
    _NOTIFICATIONS_DELIVERY_UNKNOWN_TOTAL.labels(
        provider=(provider or "unknown")[:80],
        channel=(channel or "unknown")[:80],
        category=(category or "unknown")[:80],
    ).inc()


def set_notifications_pending_oldest_age_seconds(
    *, provider: str, channel: str, severity: str, seconds: float
) -> None:
    _NOTIFICATIONS_PENDING_OLDEST_AGE_SECONDS.labels(
        provider=(provider or "unknown")[:80],
        channel=(channel or "unknown")[:80],
        severity=(severity or "unknown")[:80],
    ).set(max(0.0, seconds))


def record_notifications_delivery_retry(
    *, provider: str, channel: str, reason: str
) -> None:
    _NOTIFICATIONS_DELIVERIES_RETRY_TOTAL.labels(
        provider=(provider or "unknown")[:80],
        channel=(channel or "unknown")[:80],
        reason=(reason or "unknown")[:80],
    ).inc()


def set_notification_worker_up(*, worker: str, up: bool) -> None:
    _NOTIFICATIONS_WORKER_UP.labels(worker=(worker or "unknown")[:80]).set(1 if up else 0)


def record_notifications_report_schedule_missed(
    *, report_type: str, timezone: str
) -> None:
    _NOTIFICATIONS_REPORT_SCHEDULE_MISSED_TOTAL.labels(
        report_type=(report_type or "unknown")[:80],
        timezone=(timezone or "unknown")[:80],
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
    "record_exchange_account_state_sync",
    "record_exchange_config_guard",
    "record_live_strategy_profile_readiness",
    "record_market_data_readiness",
    "record_strategy_position_ownership",
    "record_strategy_capital_reservation",
    "record_strategy_paper_accounting",
    "record_strategy_variant_launch",
    "record_strategy_variant_compatibility",
    "record_execution_intent",
    "record_execution_dispatch",
    "record_execution_dispatch_backpressure",
    "record_execution_dispatch_dlq",
    "record_execution_dispatch_redis_error",
    "record_execution_dispatch_retry",
    "record_execution_order_model_rejected",
    "record_execution_notification_outbox",
    "record_execution_risk_gate",
    "record_execution_source_event",
    "record_admin_notification",
    "record_notification_delivery_unknown",
    "record_notifications_delivery_retry",
    "record_notifications_report_schedule_missed",
    "set_notification_worker_up",
    "set_notifications_pending_oldest_age_seconds",
]
