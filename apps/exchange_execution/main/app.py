from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExchangeExecutionProcessRepository,
    InMemoryExecutionIntentRepository,
    PostgresExchangeExecutionProcessRepository,
    PostgresExecutionIntentRepository,
    RedisExchangeExecutionConsumer,
    RedisExecutionDispatchTransportConfig,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import (
    ExchangeExecutionProcessConfig,
    ExchangeExecutionProcessService,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionConsumer,
    ExchangeExecutionProcessRepository,
    ExecutionIntentRepository,
)
from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway

EXCHANGE_EXECUTION_DEFAULT_HOST = "127.0.0.1"
EXCHANGE_EXECUTION_METRICS_PORT = 9206
_ENV_NAME_KEY = "ROEHUB_ENV"
_CONFIG_PATH_KEY = "ROEHUB_EXCHANGE_EXECUTION_CONFIG"
_STRATEGY_FAIL_FAST_KEY = "STRATEGY_FAIL_FAST"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"


@dataclass(frozen=True, slots=True)
class ExchangeExecutionRuntimeSettings:
    env_name: str
    bind_host: str
    metrics_port: int
    postgres_dsn: str
    redis_enabled: bool
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password_env: str | None
    redis_socket_timeout_s: float
    redis_connect_timeout_s: float
    process_config: ExchangeExecutionProcessConfig
    poll_interval_seconds: float


class ExchangeExecutionMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.ready = Gauge(
            "exchange_execution_ready",
            "Readiness rollup for exchange-execution process.",
            ("status", "reason"),
            registry=self.registry,
        )
        self.dependency_ready = Gauge(
            "exchange_execution_dependency_ready",
            "Dependency readiness by dependency, status and reason.",
            ("dependency", "status", "reason"),
            registry=self.registry,
        )
        self.redis_stream_length = Gauge(
            "exchange_execution_redis_stream_length",
            "Redis stream lengths observed by exchange-execution.",
            ("stream",),
            registry=self.registry,
        )
        self.redis_pending = Gauge(
            "exchange_execution_redis_pending",
            "Pending Redis request messages for the exchange-execution consumer group.",
            registry=self.registry,
        )
        self.clock_drift_ms = Gauge(
            "exchange_execution_clock_drift_ms",
            "Clock drift in milliseconds between local process and Redis server time.",
            registry=self.registry,
        )
        self.observations_total = Counter(
            "exchange_execution_observations_total",
            "Request observations by disabled exchange-execution process.",
            ("status", "reason"),
            registry=self.registry,
        )
        self.dlq_total = Counter(
            "exchange_execution_dlq_total",
            "Requests moved to exchange-execution DLQ by reason.",
            ("reason",),
            registry=self.registry,
        )
        self.ack_total = Counter(
            "exchange_execution_ack_total",
            "Redis request acknowledgements after durable state changes.",
            ("reason",),
            registry=self.registry,
        )
        self.adapter_disabled = Gauge(
            "exchange_execution_adapter_disabled",
            "1 when order adapters are disabled.",
            registry=self.registry,
        )
        self.adapter_disabled.set(1)

    def update_readiness(self, snapshot: Any) -> None:
        self.ready.labels(status=snapshot.status, reason=snapshot.status_reason).set(1)
        for dependency in snapshot.dependencies:
            value = 1 if dependency.status == "ready" else 0
            self.dependency_ready.labels(
                dependency=dependency.name,
                status=dependency.status,
                reason=dependency.reason,
            ).set(value)
            if dependency.name == "redis":
                request_length = dependency.metadata.get("request_stream_length")
                pending_count = dependency.metadata.get("pending_count")
                if request_length is not None:
                    self.redis_stream_length.labels(stream="execution.requests.v1").set(
                        float(request_length)
                    )
                if pending_count is not None:
                    self.redis_pending.set(float(pending_count))
            if dependency.name == "dlq":
                dlq_length = dependency.metadata.get("dlq_stream_length")
                if dlq_length is not None:
                    self.redis_stream_length.labels(stream="execution.requests.dlq.v1").set(
                        float(dlq_length)
                    )
            if dependency.name == "clock_drift":
                drift = dependency.metadata.get("clock_drift_ms")
                if drift is not None:
                    self.clock_drift_ms.set(float(drift))

    def record_observation(self, status: str, reason: str) -> None:
        self.observations_total.labels(status=status, reason=reason).inc()

    def record_dlq(self, reason: str) -> None:
        self.dlq_total.labels(reason=reason).inc()

    def record_ack(self, reason: str) -> None:
        self.ack_total.labels(reason=reason).inc()


def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    effective_environ = os.environ if environ is None else environ
    settings = resolve_runtime_settings(environ=effective_environ)
    service, metrics = build_runtime(environ=effective_environ, settings=settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop_event = asyncio.Event()
        task: asyncio.Task[None] | None = None
        if settings.process_config.consumer_enabled:
            task = asyncio.create_task(
                _consumer_loop(
                    service=service,
                    stop_event=stop_event,
                    poll_interval_seconds=settings.poll_interval_seconds,
                )
            )
        try:
            yield
        finally:
            stop_event.set()
            if task is not None:
                await task

    app = FastAPI(
        title="Roehub Exchange Execution",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.exchange_execution_service = service
    app.state.exchange_execution_metrics = metrics

    @app.get("/health")
    def get_health() -> dict[str, object]:
        return {"status": "ok", "service": settings.process_config.service_id}

    @app.get("/health/ready")
    def get_readiness(response: Response) -> dict[str, object]:
        snapshot = service.readiness()
        metrics.update_readiness(snapshot)
        if snapshot.status == "not_ready":
            response.status_code = 503
        return {
            "service": snapshot.service_id,
            "status": snapshot.status,
            "status_reason": snapshot.status_reason,
            "adapter_mode": snapshot.adapter_mode,
            "checked_at": snapshot.checked_at.isoformat(),
            "dependencies": [
                {
                    "name": item.name,
                    "status": item.status,
                    "reason": item.reason,
                    "metadata": dict(item.metadata),
                }
                for item in snapshot.dependencies
            ],
        }

    @app.post("/internal/v1/run-once", include_in_schema=False)
    def post_run_once(response: Response) -> dict[str, object]:
        result = service.run_once()
        if result.reason == "consumer_disabled":
            response.status_code = 409
        return {
            "read_count": result.read_count,
            "observed_count": result.observed_count,
            "quarantined_count": result.quarantined_count,
            "acked_count": result.acked_count,
            "reason": result.reason,
        }

    @app.get("/metrics", include_in_schema=False)
    def get_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


def build_runtime(
    *,
    environ: Mapping[str, str],
    settings: ExchangeExecutionRuntimeSettings,
) -> tuple[ExchangeExecutionProcessService, ExchangeExecutionMetrics]:
    clock = SystemLiveExecutionClock()
    if settings.postgres_dsn:
        gateway = PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        process_repository: ExchangeExecutionProcessRepository = (
            PostgresExchangeExecutionProcessRepository(gateway=gateway)
        )
        intent_repository: ExecutionIntentRepository = PostgresExecutionIntentRepository(
            gateway=gateway
        )
    else:
        process_repository = InMemoryExchangeExecutionProcessRepository()
        intent_repository = InMemoryExecutionIntentRepository()
    consumer: ExchangeExecutionConsumer | None = None
    if settings.redis_enabled:
        consumer = RedisExchangeExecutionConsumer(
            config=RedisExecutionDispatchTransportConfig(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password_env=settings.redis_password_env,
                socket_timeout_s=settings.redis_socket_timeout_s,
                connect_timeout_s=settings.redis_connect_timeout_s,
                request_stream=settings.process_config.request_stream,
                retry_stream=settings.process_config.retry_stream,
                dlq_stream=settings.process_config.dlq_stream,
                consumer_group=settings.process_config.consumer_group,
            ),
            consumer_name=settings.process_config.consumer_name,
            environ=environ,
        )
    metrics = ExchangeExecutionMetrics()
    service = ExchangeExecutionProcessService(
        config=settings.process_config,
        repository=process_repository,
        intent_repository=intent_repository,
        consumer=consumer,
        clock=clock,
        on_observation=metrics.record_observation,
        on_dlq=metrics.record_dlq,
        on_ack=metrics.record_ack,
    )
    return service, metrics


def resolve_runtime_settings(
    *, environ: Mapping[str, str]
) -> ExchangeExecutionRuntimeSettings:
    env_name = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    config_path = Path(
        environ.get(
            _CONFIG_PATH_KEY,
            f"configs/{env_name}/exchange_execution.yaml",
        )
    )
    payload = _load_yaml(path=config_path)
    exchange_execution = _mapping(payload.get("exchange_execution"), "exchange_execution")
    http = _mapping(exchange_execution.get("http", {}), "exchange_execution.http")
    redis = _mapping(exchange_execution.get("redis_streams", {}), "redis_streams")
    process = _mapping(exchange_execution.get("process", {}), "exchange_execution.process")
    limiter = _mapping(exchange_execution.get("rate_limit", {}), "exchange_execution.rate_limit")
    clock = _mapping(exchange_execution.get("clock", {}), "exchange_execution.clock")

    raw_fail_fast = environ.get(_STRATEGY_FAIL_FAST_KEY)
    fail_fast = env_name == "prod" if raw_fail_fast is None else _parse_bool(raw_fail_fast)
    redis_enabled = _parse_bool(str(redis.get("enabled", "false")))
    bind_host = str(http.get("host", EXCHANGE_EXECUTION_DEFAULT_HOST))
    metrics_port = _positive_int(
        environ.get(
            "ROEHUB_EXCHANGE_EXECUTION_METRICS_PORT",
            str(http.get("port", EXCHANGE_EXECUTION_METRICS_PORT)),
        ),
        "ROEHUB_EXCHANGE_EXECUTION_METRICS_PORT",
    )
    process_config = ExchangeExecutionProcessConfig(
        service_id=str(process.get("service_id", "exchange-execution")),
        adapter_mode=str(process.get("adapter_mode", "disabled")),
        request_stream=str(redis.get("request_stream", "execution.requests.v1")),
        retry_stream=str(redis.get("retry_stream", "execution.requests.retry.v1")),
        dlq_stream=str(redis.get("dlq_stream", "execution.requests.dlq.v1")),
        consumer_group=str(redis.get("consumer_group", "exchange-execution.v1")),
        consumer_name=str(process.get("consumer_name", "exchange-execution-local")),
        consumer_enabled=_parse_bool(str(process.get("consumer_enabled", "false"))),
        read_count=_positive_int(str(process.get("read_count", "10")), "read_count"),
        block_ms=_non_negative_int(str(process.get("block_ms", "100")), "block_ms"),
        backpressure_max_stream_length=_positive_int(
            str(process.get("backpressure_max_stream_length", "10000")),
            "backpressure_max_stream_length",
        ),
        max_clock_drift_ms=_positive_float(
            str(clock.get("max_drift_ms", "1000")),
            "max_drift_ms",
        ),
        rate_limit_per_second=_positive_float(
            str(limiter.get("per_second", "5")),
            "rate_limit.per_second",
        ),
        rate_limit_burst=_positive_int(str(limiter.get("burst", "10")), "rate_limit.burst"),
        fail_fast=fail_fast,
    )
    if env_name == "prod":
        if bind_host != EXCHANGE_EXECUTION_DEFAULT_HOST:
            raise ValueError("prod exchange-execution must bind to 127.0.0.1")
        if metrics_port != EXCHANGE_EXECUTION_METRICS_PORT:
            raise ValueError("prod exchange-execution must use metrics port 9206")
    return ExchangeExecutionRuntimeSettings(
        env_name=env_name,
        bind_host=bind_host,
        metrics_port=metrics_port,
        postgres_dsn=environ.get(_STRATEGY_PG_DSN_KEY, "").strip(),
        redis_enabled=redis_enabled,
        redis_host=str(redis.get("host", "127.0.0.1" if env_name == "prod" else "redis")),
        redis_port=_positive_int(str(redis.get("port", "6379")), "redis.port"),
        redis_db=_non_negative_int(str(redis.get("db", "0")), "redis.db"),
        redis_password_env=_optional_str(redis.get("password_env")),
        redis_socket_timeout_s=_positive_float(
            str(redis.get("socket_timeout_s", "2.0")),
            "redis.socket_timeout_s",
        ),
        redis_connect_timeout_s=_positive_float(
            str(redis.get("connect_timeout_s", "2.0")),
            "redis.connect_timeout_s",
        ),
        process_config=process_config,
        poll_interval_seconds=_positive_float(
            str(process.get("poll_interval_seconds", "1.0")),
            "poll_interval_seconds",
        ),
    )


async def _consumer_loop(
    *,
    service: ExchangeExecutionProcessService,
    stop_event: asyncio.Event,
    poll_interval_seconds: float,
) -> None:
    while not stop_event.is_set():
        service.run_once()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_seconds)
        except TimeoutError:
            continue


def _load_yaml(*, path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing exchange-execution config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _mapping(raw, str(path))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"boolean-like value required, got {raw_value!r}")


def _positive_int(raw_value: str, key: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _non_negative_int(raw_value: str, key: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _positive_float(raw_value: str, key: str) -> float:
    value = float(raw_value)
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


app = create_app()
