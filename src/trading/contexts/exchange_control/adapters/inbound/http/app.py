from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

from trading.contexts.exchange_control.application.readiness import (
    ExchangeControlReadinessProbe,
)
from trading.contexts.exchange_control.application.service_identity import (
    EXCHANGE_CONTROL_SERVICE_IDENTITY,
    build_exchange_control_service_identity,
)

EXCHANGE_CONTROL_DEFAULT_HOST = "127.0.0.1"
EXCHANGE_CONTROL_METRICS_PORT = 9205


@dataclass(frozen=True)
class ExchangeControlRuntimeConfig:
    service_identity_name: str
    bind_host: str
    metrics_port: int
    real_exchange_validation_enabled: bool = False

    @classmethod
    def from_environ(
        cls,
        *,
        environ: Mapping[str, str],
        bind_host: str | None = None,
        metrics_port: int | None = None,
    ) -> "ExchangeControlRuntimeConfig":
        environment_name = environ.get("ROEHUB_ENV", "dev").strip().lower()
        service_identity_name = environ.get(
            "ROEHUB_EXCHANGE_CONTROL_SERVICE_IDENTITY",
            EXCHANGE_CONTROL_SERVICE_IDENTITY,
        )
        resolved_host = bind_host or environ.get(
            "ROEHUB_EXCHANGE_CONTROL_BIND_HOST",
            EXCHANGE_CONTROL_DEFAULT_HOST,
        )
        resolved_port = metrics_port or _read_int(
            value=environ.get("ROEHUB_EXCHANGE_CONTROL_METRICS_PORT"),
            default=EXCHANGE_CONTROL_METRICS_PORT,
            name="ROEHUB_EXCHANGE_CONTROL_METRICS_PORT",
        )
        validation_enabled = _read_bool(
            value=environ.get("ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED"),
            default=False,
        )
        config = cls(
            service_identity_name=service_identity_name,
            bind_host=resolved_host,
            metrics_port=resolved_port,
            real_exchange_validation_enabled=validation_enabled,
        )
        config.validate(environment_name=environment_name)
        return config

    def validate(self, *, environment_name: str) -> None:
        build_exchange_control_service_identity(name=self.service_identity_name)
        if self.metrics_port <= 0:
            raise ValueError("exchange-control metrics_port must be > 0")
        if self.real_exchange_validation_enabled:
            raise ValueError("real exchange validation must remain disabled before Stage 5")
        if environment_name == "prod":
            if self.bind_host != EXCHANGE_CONTROL_DEFAULT_HOST:
                raise ValueError("prod exchange-control must bind to 127.0.0.1")
            if self.metrics_port != EXCHANGE_CONTROL_METRICS_PORT:
                raise ValueError("prod exchange-control must use metrics port 9205")


class ExchangeControlMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.active = Gauge(
            "exchange_control_active",
            "Whether exchange-control runtime is active.",
            registry=self.registry,
        )
        self.connection_validation_total = Counter(
            "exchange_connection_validation_total",
            "Exchange connection validation attempts by exchange, result, and reason.",
            ("exchange", "result", "reason"),
            registry=self.registry,
        )
        self.connection_status = Gauge(
            "exchange_connection_status",
            "Current exchange connection status by exchange and status.",
            ("exchange", "status"),
            registry=self.registry,
        )

    def mark_active(self) -> None:
        self.active.set(1)
        self.connection_validation_total.labels(
            exchange="none",
            result="disabled",
            reason="stage_2_no_real_exchange_calls",
        ).inc(0)
        self.connection_status.labels(exchange="none", status="validation_disabled").set(0)


def create_exchange_control_app(*, config: ExchangeControlRuntimeConfig) -> FastAPI:
    service_identity = build_exchange_control_service_identity(name=config.service_identity_name)
    readiness_probe = ExchangeControlReadinessProbe(service_identity=service_identity)
    metrics = ExchangeControlMetrics()
    metrics.mark_active()

    app = FastAPI(title="Roehub Exchange Control", version="1.0.0")

    @app.get("/health/ready")
    def get_readiness() -> dict[str, object]:
        return readiness_probe.check().as_response_payload()

    @app.get("/metrics", include_in_schema=False)
    def get_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


def _read_int(*, value: str | None, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _read_bool(*, value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "EXCHANGE_CONTROL_DEFAULT_HOST",
    "EXCHANGE_CONTROL_METRICS_PORT",
    "ExchangeControlRuntimeConfig",
    "create_exchange_control_app",
]
