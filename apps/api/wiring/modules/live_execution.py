from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fastapi import APIRouter

from apps.api.monitoring import (
    record_execution_intent,
    record_execution_order_model_rejected,
    record_execution_source_event,
)
from apps.api.routes import build_ui_execution_router
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExecutionIntentRepository,
    PostgresExecutionIntentRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import ExecutionIngressService
from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway

_ENV_NAME_KEY = "ROEHUB_ENV"
_STRATEGY_FAIL_FAST_KEY = "STRATEGY_FAIL_FAST"
_STRATEGY_PG_DSN_KEY = "STRATEGY_PG_DSN"
_ALLOWED_ENVS = ("dev", "prod", "test")


@dataclass(frozen=True, slots=True)
class LiveExecutionRuntimeSettings:
    env_name: str
    fail_fast: bool
    postgres_dsn: str


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
    ingress_service = ExecutionIngressService(
        repository=repository,
        clock=SystemLiveExecutionClock(),
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
    )
    return build_ui_execution_router(
        ingress_service=ingress_service,
        current_user_dependency=current_user_dependency,
    )


def _resolve_runtime_settings(*, environ: Mapping[str, str]) -> LiveExecutionRuntimeSettings:
    env_name = environ.get(_ENV_NAME_KEY, "dev").strip().lower()
    if env_name not in _ALLOWED_ENVS:
        raise ValueError(f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}, got {env_name!r}")
    raw_fail_fast = environ.get(_STRATEGY_FAIL_FAST_KEY)
    fail_fast = env_name == "prod" if raw_fail_fast is None else _parse_bool(raw_fail_fast)
    return LiveExecutionRuntimeSettings(
        env_name=env_name,
        fail_fast=fail_fast,
        postgres_dsn=environ.get(_STRATEGY_PG_DSN_KEY, "").strip(),
    )


def _parse_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_STRATEGY_FAIL_FAST_KEY} must be boolean-like value, got {raw_value!r}")


__all__ = ["build_ui_execution_router_module"]
