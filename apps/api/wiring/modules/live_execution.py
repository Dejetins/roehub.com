from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fastapi import APIRouter

from apps.api.monitoring import (
    record_execution_dispatch,
    record_execution_dispatch_backpressure,
    record_execution_dispatch_dlq,
    record_execution_dispatch_redis_error,
    record_execution_dispatch_retry,
    record_execution_intent,
    record_execution_order_model_rejected,
    record_execution_risk_gate,
    record_execution_source_event,
)
from apps.api.routes import build_ui_execution_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    PostgresExecutionIntentRepository,
    RedisExecutionDispatchTransport,
    RedisExecutionDispatchTransportConfig,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import (
    ExecutionDispatchConfig,
    ExecutionDispatchService,
    ExecutionIngressService,
)
from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway

_ENV_NAME_KEY = "ROEHUB_ENV"
_STRATEGY_FAIL_FAST_KEY = "STRATEGY_FAIL_FAST"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_DISPATCH_ENABLED_KEY = "ROEHUB_EXECUTION_DISPATCH_REDIS_ENABLED"
_DISPATCH_REDIS_HOST_KEY = "ROEHUB_EXECUTION_DISPATCH_REDIS_HOST"
_DISPATCH_REDIS_PORT_KEY = "ROEHUB_EXECUTION_DISPATCH_REDIS_PORT"
_DISPATCH_REDIS_DB_KEY = "ROEHUB_EXECUTION_DISPATCH_REDIS_DB"
_DISPATCH_REDIS_PASSWORD_ENV_KEY = "ROEHUB_EXECUTION_DISPATCH_REDIS_PASSWORD_ENV"
_DISPATCH_REDIS_SOCKET_TIMEOUT_KEY = "ROEHUB_EXECUTION_DISPATCH_REDIS_SOCKET_TIMEOUT_S"
_DISPATCH_REDIS_CONNECT_TIMEOUT_KEY = "ROEHUB_EXECUTION_DISPATCH_REDIS_CONNECT_TIMEOUT_S"
_DISPATCH_REQUEST_STREAM_KEY = "ROEHUB_EXECUTION_DISPATCH_REQUEST_STREAM"
_DISPATCH_RETRY_STREAM_KEY = "ROEHUB_EXECUTION_DISPATCH_RETRY_STREAM"
_DISPATCH_DLQ_STREAM_KEY = "ROEHUB_EXECUTION_DISPATCH_DLQ_STREAM"
_DISPATCH_CONSUMER_GROUP_KEY = "ROEHUB_EXECUTION_DISPATCH_CONSUMER_GROUP"
_DISPATCH_RETRY_BUDGET_KEY = "ROEHUB_EXECUTION_DISPATCH_RETRY_BUDGET"
_DISPATCH_BACKPRESSURE_LENGTH_KEY = "ROEHUB_EXECUTION_DISPATCH_BACKPRESSURE_LENGTH"
_ALLOWED_ENVS = ("dev", "prod", "test")


@dataclass(frozen=True, slots=True)
class LiveExecutionRuntimeSettings:
    env_name: str
    fail_fast: bool
    postgres_dsn: str
    dispatch_enabled: bool
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password_env: str | None
    redis_socket_timeout_s: float
    redis_connect_timeout_s: float
    request_stream: str
    retry_stream: str
    dlq_stream: str
    consumer_group: str
    retry_budget: int
    backpressure_length: int


def build_ui_execution_router_module(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    settings = _resolve_runtime_settings(environ=environ)
    if settings.postgres_dsn:
        repository = PostgresExecutionIntentRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)
        )
    elif settings.fail_fast:
        raise ValueError(
            f"{_STRATEGY_PG_DSN_KEY} is required when live-execution fail-fast mode is enabled"
        )
    else:
        repository = InMemoryExecutionIntentRepository()
    clock = SystemLiveExecutionClock()
    ingress_service = ExecutionIngressService(
        repository=repository,
        clock=clock,
        on_source_event=lambda source_type, result: record_execution_source_event(
            source_type=source_type,
            result=result,
        ),
        on_intent=lambda source_type, result, reason: record_execution_intent(
            source_type=source_type,
            result=result,
            reason=reason,
        ),
        on_order_model_rejected=lambda source_type, reason: record_execution_order_model_rejected(
            source_type=source_type,
            reason=reason,
        ),
        on_risk_decision=lambda source_type, result, reason, latency_seconds: (
            record_execution_risk_gate(
                source_type=source_type,
                result=result,
                reason=reason,
                latency_seconds=latency_seconds,
            )
        ),
    )
    dispatch_service = None
    if settings.dispatch_enabled:
        dispatch_service = ExecutionDispatchService(
            repository=repository,
            transport=RedisExecutionDispatchTransport(
                config=RedisExecutionDispatchTransportConfig(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    password_env=settings.redis_password_env,
                    socket_timeout_s=settings.redis_socket_timeout_s,
                    connect_timeout_s=settings.redis_connect_timeout_s,
                    request_stream=settings.request_stream,
                    retry_stream=settings.retry_stream,
                    dlq_stream=settings.dlq_stream,
                    consumer_group=settings.consumer_group,
                ),
                environ=environ,
            ),
            clock=clock,
            config=ExecutionDispatchConfig(
                retry_budget=settings.retry_budget,
                backpressure_max_stream_length=settings.backpressure_length,
            ),
            on_dispatch=lambda result, reason: record_execution_dispatch(
                result=result,
                reason=reason,
            ),
            on_retry=lambda reason: record_execution_dispatch_retry(reason=reason),
            on_dlq=lambda reason: record_execution_dispatch_dlq(reason=reason),
            on_backpressure=lambda reason: record_execution_dispatch_backpressure(
                reason=reason
            ),
            on_redis_error=lambda reason: record_execution_dispatch_redis_error(
                reason=reason
            ),
        )
    return build_ui_execution_router(
        ingress_service=ingress_service,
        dispatch_service=dispatch_service,
        current_user_dependency=current_user_dependency,
    )


def _resolve_runtime_settings(*, environ: Mapping[str, str]) -> LiveExecutionRuntimeSettings:
    env_name = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if env_name not in _ALLOWED_ENVS:
        raise ValueError(f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {env_name!r}")
    raw_fail_fast = environ.get(_STRATEGY_FAIL_FAST_KEY)
    fail_fast = env_name == "prod" if raw_fail_fast is None else _parse_bool(raw_fail_fast)
    dispatch_enabled = _resolve_dispatch_enabled(environ=environ, env_name=env_name)
    return LiveExecutionRuntimeSettings(
        env_name=env_name,
        fail_fast=fail_fast,
        postgres_dsn=environ.get(_STRATEGY_PG_DSN_KEY, "").strip(),
        dispatch_enabled=dispatch_enabled,
        redis_host=environ.get(_DISPATCH_REDIS_HOST_KEY, _default_redis_host(env_name)).strip(),
        redis_port=_parse_positive_int(
            environ.get(_DISPATCH_REDIS_PORT_KEY, "6379"),
            key=_DISPATCH_REDIS_PORT_KEY,
        ),
        redis_db=_parse_non_negative_int(
            environ.get(_DISPATCH_REDIS_DB_KEY, "0"),
            key=_DISPATCH_REDIS_DB_KEY,
        ),
        redis_password_env=_optional_str(environ.get(_DISPATCH_REDIS_PASSWORD_ENV_KEY)),
        redis_socket_timeout_s=_parse_positive_float(
            environ.get(_DISPATCH_REDIS_SOCKET_TIMEOUT_KEY, "2.0"),
            key=_DISPATCH_REDIS_SOCKET_TIMEOUT_KEY,
        ),
        redis_connect_timeout_s=_parse_positive_float(
            environ.get(_DISPATCH_REDIS_CONNECT_TIMEOUT_KEY, "2.0"),
            key=_DISPATCH_REDIS_CONNECT_TIMEOUT_KEY,
        ),
        request_stream=environ.get(
            _DISPATCH_REQUEST_STREAM_KEY, "execution.requests.v1"
        ).strip(),
        retry_stream=environ.get(
            _DISPATCH_RETRY_STREAM_KEY, "execution.requests.retry.v1"
        ).strip(),
        dlq_stream=environ.get(_DISPATCH_DLQ_STREAM_KEY, "execution.requests.dlq.v1").strip(),
        consumer_group=environ.get(
            _DISPATCH_CONSUMER_GROUP_KEY, "exchange-execution.v1"
        ).strip(),
        retry_budget=_parse_positive_int(
            environ.get(_DISPATCH_RETRY_BUDGET_KEY, "3"),
            key=_DISPATCH_RETRY_BUDGET_KEY,
        ),
        backpressure_length=_parse_positive_int(
            environ.get(_DISPATCH_BACKPRESSURE_LENGTH_KEY, "10000"),
            key=_DISPATCH_BACKPRESSURE_LENGTH_KEY,
        ),
    )


def _parse_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_STRATEGY_FAIL_FAST_KEY} must be boolean-like value, got {raw_value!r}")


def _resolve_dispatch_enabled(*, environ: Mapping[str, str], env_name: str) -> bool:
    raw = environ.get(_DISPATCH_ENABLED_KEY)
    if raw is None:
        return env_name == "prod"
    return _parse_bool_for_key(raw, key=_DISPATCH_ENABLED_KEY)


def _default_redis_host(env_name: str) -> str:
    return "127.0.0.1" if env_name == "prod" else "redis"


def _parse_bool_for_key(raw_value: str, *, key: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be boolean-like value, got {raw_value!r}")


def _parse_positive_int(raw_value: str, *, key: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _parse_non_negative_int(raw_value: str, *, key: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _parse_positive_float(raw_value: str, *, key: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{key} must be numeric") from error
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _optional_str(raw_value: str | None) -> str | None:
    if raw_value is None:
        return "ROEHUB_REDIS_PASSWORD"
    value = raw_value.strip()
    return value if value else None


__all__ = ["build_ui_execution_router_module"]
