from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, cast

import yaml
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from apps.exchange_execution.adapters import BinanceTestnetOrderAdapter, BybitTestnetOrderAdapter
from trading.contexts.exchange_control.adapters.outbound import (
    OpenBaoTransitExchangeSecretCipher,
    PostgresExchangeConnectionRepository,
)
from trading.contexts.live_execution.adapters.outbound import (
    ExchangeControlCredentialResolver,
    InMemoryExchangeExecutionOrderRepository,
    InMemoryExchangeExecutionProcessRepository,
    InMemoryExecutionGatewayPolicyRepository,
    InMemoryExecutionIntentRepository,
    PostgresExchangeExecutionOrderRepository,
    PostgresExchangeExecutionProcessRepository,
    PostgresExecutionGatewayPolicyRepository,
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
    ExchangeOrderAdapter,
    ExecutionDispatchUnavailableError,
    ExecutionIntentRepository,
)
from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway

EXCHANGE_EXECUTION_DEFAULT_HOST = "127.0.0.1"
EXCHANGE_EXECUTION_METRICS_PORT = 9206
_ENV_NAME_KEY = "ROEHUB_ENV"
_CONFIG_PATH_KEY = "ROEHUB_EXCHANGE_EXECUTION_CONFIG"
_CANCEL_AFTER_SUBMIT_KEY = "ROEHUB_EXCHANGE_EXECUTION_CANCEL_AFTER_SUBMIT"
_STRATEGY_FAIL_FAST_KEY = "STRATEGY_FAIL_FAST"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_PITR_VERIFIED_KEY = "ROEHUB_EXECUTION_PITR_VERIFIED"
_CONTAINER_BIND_KEY = "ROEHUB_EXCHANGE_EXECUTION_CONTAINER_BIND"


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
    adapter_timeout_seconds: float
    transit_address: str
    transit_token: str
    pitr_verified_env: str


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
        self.testnet_order_total = Counter(
            "exchange_execution_testnet_order_total",
            "Testnet order adapter outcomes by exchange and reason.",
            ("exchange", "reason"),
            registry=self.registry,
        )
        self.private_stream_total = Counter(
            "exchange_execution_private_stream_total",
            "Private stream lifecycle outcomes by exchange and reason.",
            ("exchange", "reason"),
            registry=self.registry,
        )
        self.submit_latency_ms = Histogram(
            "exchange_execution_submit_latency_ms",
            "Native exchange adapter order/cancel/status latency in milliseconds.",
            ("exchange",),
            registry=self.registry,
        )
        self.rate_limit_wait_total = Counter(
            "exchange_execution_rate_limit_wait_total",
            "Exchange-execution limiter waits before native adapter operations.",
            ("exchange", "operation"),
            registry=self.registry,
        )
        self.rate_limit_wait_seconds = Histogram(
            "exchange_execution_rate_limit_wait_seconds",
            "Exchange-execution limiter wait duration before native adapter operations.",
            ("exchange", "operation"),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry,
        )
        self.reconciliation_total = Counter(
            "execution_reconciliation_total",
            "Execution order reconciliation outcomes by status and reason.",
            ("status", "reason"),
            registry=self.registry,
        )
        self.notification_outbox_total = Counter(
            "execution_notification_outbox_total",
            "Execution notification outbox events by type and producer source.",
            ("event_type", "source_type", "severity"),
            registry=self.registry,
        )
        self.ledger_backup_restore_total = Counter(
            "execution_ledger_backup_restore_total",
            "Money-ledger backup/PITR restore drill outcomes.",
            ("result", "reason"),
            registry=self.registry,
        )

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
            if dependency.name == "adapter":
                submit_enabled = dependency.metadata.get("submit_enabled")
                if submit_enabled is not None:
                    self.adapter_disabled.set(0 if int(submit_enabled) == 1 else 1)

    def record_observation(self, status: str, reason: str) -> None:
        self.observations_total.labels(status=status, reason=reason).inc()

    def record_dlq(self, reason: str) -> None:
        self.dlq_total.labels(reason=reason).inc()

    def record_ack(self, reason: str) -> None:
        self.ack_total.labels(reason=reason).inc()

    def record_order_submit(self, exchange: str, reason: str) -> None:
        self.testnet_order_total.labels(exchange=exchange, reason=reason).inc()

    def record_private_stream(self, exchange: str, reason: str) -> None:
        self.private_stream_total.labels(exchange=exchange, reason=reason).inc()

    def record_order_latency(self, exchange: str, latency_ms: float) -> None:
        self.submit_latency_ms.labels(exchange=exchange).observe(latency_ms)

    def record_rate_limit_wait(self, exchange: str, operation: str, wait_seconds: float) -> None:
        self.rate_limit_wait_total.labels(exchange=exchange, operation=operation).inc()
        self.rate_limit_wait_seconds.labels(exchange=exchange, operation=operation).observe(
            wait_seconds
        )

    def record_reconciliation(self, status: str, reason: str) -> None:
        self.reconciliation_total.labels(status=status, reason=reason).inc()

    def record_notification(self, event_type: str, source_type: str, severity: str) -> None:
        self.notification_outbox_total.labels(
            event_type=event_type,
            source_type=source_type,
            severity=severity,
        ).inc()

    def record_pitr_drill(self, result: str, reason: str) -> None:
        self.ledger_backup_restore_total.labels(result=result, reason=reason).inc()


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

    @app.get("/health/live")
    def get_liveness() -> dict[str, object]:
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
        try:
            result = service.run_once()
        except ExecutionDispatchUnavailableError as error:
            response.status_code = 503
            return {
                "read_count": 0,
                "observed_count": 0,
                "submitted_count": 0,
                "guard_rejected_count": 0,
                "adapter_error_count": 0,
                "quarantined_count": 0,
                "acked_count": 0,
                "reason": error.reason,
            }
        if result.reason == "consumer_disabled":
            response.status_code = 409
        return {
            "read_count": result.read_count,
            "observed_count": result.observed_count,
            "submitted_count": result.submitted_count,
            "guard_rejected_count": result.guard_rejected_count,
            "adapter_error_count": result.adapter_error_count,
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
        order_repository = PostgresExchangeExecutionOrderRepository(gateway=gateway)
        gateway_policy_repository = PostgresExecutionGatewayPolicyRepository(gateway=gateway)
    else:
        process_repository = InMemoryExchangeExecutionProcessRepository()
        intent_repository = InMemoryExecutionIntentRepository()
        order_repository = InMemoryExchangeExecutionOrderRepository()
        gateway_policy_repository = InMemoryExecutionGatewayPolicyRepository()
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
    credential_resolver = None
    order_adapters: tuple[ExchangeOrderAdapter, ...] = ()
    if settings.process_config.adapter_mode == "testnet":
        if settings.postgres_dsn and settings.transit_address and settings.transit_token:
            credential_resolver = ExchangeControlCredentialResolver(
                connection_repository=PostgresExchangeConnectionRepository(
                    dsn=settings.postgres_dsn
                ),
                secret_cipher=OpenBaoTransitExchangeSecretCipher(
                    address=settings.transit_address,
                    token=settings.transit_token,
                    timeout_seconds=settings.adapter_timeout_seconds,
                ),
            )
        order_adapters = cast(
            tuple[ExchangeOrderAdapter, ...],
            (
                BinanceTestnetOrderAdapter(timeout_seconds=settings.adapter_timeout_seconds),
                BybitTestnetOrderAdapter(timeout_seconds=settings.adapter_timeout_seconds),
            ),
        )
    service = ExchangeExecutionProcessService(
        config=settings.process_config,
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=credential_resolver,
        order_adapters=order_adapters,
        gateway_policy_repository=gateway_policy_repository,
        consumer=consumer,
        clock=clock,
        on_observation=metrics.record_observation,
        on_dlq=metrics.record_dlq,
        on_ack=metrics.record_ack,
        on_order_submit=metrics.record_order_submit,
        on_private_stream=metrics.record_private_stream,
        on_order_latency=metrics.record_order_latency,
        on_rate_limit_wait=metrics.record_rate_limit_wait,
        on_reconciliation=metrics.record_reconciliation,
        on_notification=metrics.record_notification,
    )
    return service, metrics


def resolve_runtime_settings(*, environ: Mapping[str, str]) -> ExchangeExecutionRuntimeSettings:
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
    adapter = _mapping(exchange_execution.get("adapter", {}), "exchange_execution.adapter")
    ledger = _mapping(exchange_execution.get("ledger", {}), "exchange_execution.ledger")

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
    raw_cancel_after_submit = environ.get(_CANCEL_AFTER_SUBMIT_KEY)
    cancel_after_submit = (
        _parse_bool(raw_cancel_after_submit)
        if raw_cancel_after_submit is not None
        else _parse_bool(str(adapter.get("cancel_after_submit", "true")))
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
        enabled_exchanges=_string_tuple(adapter.get("enabled_exchanges", ("binance", "bybit"))),
        cancel_after_submit=cancel_after_submit,
        ledger_pitr_required=_parse_bool(
            str(ledger.get("pitr_required", "true" if env_name == "prod" else "false"))
        ),
        ledger_pitr_verified=_parse_bool(
            environ.get(str(ledger.get("pitr_verified_env", _PITR_VERIFIED_KEY)), "false")
        ),
        fail_fast=fail_fast,
    )
    if env_name == "prod":
        container_bind_enabled = _parse_bool(environ.get(_CONTAINER_BIND_KEY, "false"))
        allowed_container_bind = container_bind_enabled and bind_host == "0.0.0.0"
        if bind_host != EXCHANGE_EXECUTION_DEFAULT_HOST and not allowed_container_bind:
            raise ValueError(
                "prod exchange-execution must bind to 127.0.0.1 or use the "
                "explicit container bind"
            )
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
        adapter_timeout_seconds=_positive_float(
            str(adapter.get("timeout_seconds", "5.0")),
            "adapter.timeout_seconds",
        ),
        transit_address=environ.get("OPENBAO_ADDR", "").strip(),
        transit_token=environ.get("ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN_FILE", "").strip(),
        pitr_verified_env=str(ledger.get("pitr_verified_env", _PITR_VERIFIED_KEY)),
    )


async def _consumer_loop(
    *,
    service: ExchangeExecutionProcessService,
    stop_event: asyncio.Event,
    poll_interval_seconds: float,
) -> None:
    while not stop_event.is_set():
        try:
            service.run_once()
        except ExecutionDispatchUnavailableError:
            pass
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


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list | tuple):
        items = [str(item) for item in value]
    else:
        raise ValueError("string list required")
    normalized = tuple(item.strip().lower() for item in items if item.strip())
    if not normalized:
        raise ValueError("at least one exchange must be enabled")
    return normalized


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
