from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, cast

from redis import Redis
from redis.exceptions import RedisError, ResponseError

from trading.contexts.live_execution.application.ports import (
    ExecutionDispatchPoisonMessageError,
    ExecutionDispatchPublishResult,
    ExecutionDispatchTransport,
    ExecutionDispatchUnavailableError,
)
from trading.contexts.live_execution.domain import ExecutionIntent


@dataclass(frozen=True, slots=True)
class RedisExecutionDispatchTransportConfig:
    host: str
    port: int
    db: int
    password_env: str | None
    socket_timeout_s: float
    connect_timeout_s: float
    request_stream: str = "execution.requests.v1"
    retry_stream: str = "execution.requests.retry.v1"
    dlq_stream: str = "execution.requests.dlq.v1"
    consumer_group: str = "exchange-execution.v1"

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("RedisExecutionDispatchTransportConfig.host must be non-empty")
        if self.port <= 0:
            raise ValueError("RedisExecutionDispatchTransportConfig.port must be > 0")
        if self.db < 0:
            raise ValueError("RedisExecutionDispatchTransportConfig.db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError(
                "RedisExecutionDispatchTransportConfig.socket_timeout_s must be > 0"
            )
        if self.connect_timeout_s <= 0:
            raise ValueError(
                "RedisExecutionDispatchTransportConfig.connect_timeout_s must be > 0"
            )
        for field_name in ("request_stream", "retry_stream", "dlq_stream", "consumer_group"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(
                    f"RedisExecutionDispatchTransportConfig.{field_name} must be non-empty"
                )


class RedisExecutionDispatchTransport(ExecutionDispatchTransport):
    def __init__(
        self,
        *,
        config: RedisExecutionDispatchTransportConfig,
        environ: Mapping[str, str],
        redis_client: Redis | None = None,
    ) -> None:
        if config is None:  # type: ignore[truthy-bool]
            raise ValueError("RedisExecutionDispatchTransport requires config")
        self._config = config
        self._redis = (
            redis_client if redis_client is not None else self._build_redis_client(environ)
        )
        self._request_group_ready = False

    def ensure_request_group(self) -> None:
        if self._request_group_ready:
            return
        try:
            self._redis.xgroup_create(
                name=self._config.request_stream,
                groupname=self._config.consumer_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise _unavailable(error) from error
        except RedisError as error:
            raise _unavailable(error) from error
        self._request_group_ready = True

    def request_stream_length(self) -> int:
        try:
            return int(cast(Any, self._redis.xlen(self._config.request_stream)))
        except RedisError as error:
            raise _unavailable(error) from error

    def publish_request(
        self, *, intent: ExecutionIntent, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        return self._publish(
            stream_name=self._config.request_stream,
            payload=_request_payload(intent=intent, attempt_count=attempt_count),
        )

    def publish_retry(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        return self._publish(
            stream_name=self._config.retry_stream,
            payload={
                **_request_payload(intent=intent, attempt_count=attempt_count),
                "retry_reason": _bounded(reason),
            },
        )

    def publish_dlq(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult:
        return self._publish(
            stream_name=self._config.dlq_stream,
            payload={
                **_request_payload(intent=intent, attempt_count=attempt_count),
                "quarantine_reason": _bounded(reason),
            },
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        try:
            self._redis.xack(stream_name, self._config.consumer_group, message_id)
        except RedisError as error:
            raise _unavailable(error) from error

    def _publish(
        self, *, stream_name: str, payload: Mapping[str, str]
    ) -> ExecutionDispatchPublishResult:
        _validate_payload(payload=payload)
        try:
            redis_fields = cast(dict[Any, Any], dict(payload))
            message_id = str(self._redis.xadd(name=stream_name, fields=redis_fields))
        except RedisError as error:
            raise _unavailable(error) from error
        return ExecutionDispatchPublishResult(stream_name=stream_name, message_id=message_id)

    def _build_redis_client(self, environ: Mapping[str, str]) -> Redis:
        password = _resolve_password(
            environ=environ,
            password_env=self._config.password_env,
        )
        return Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=password,
            socket_timeout=self._config.socket_timeout_s,
            socket_connect_timeout=self._config.connect_timeout_s,
            decode_responses=True,
        )


def _request_payload(*, intent: ExecutionIntent, attempt_count: int) -> dict[str, str]:
    return {
        "schema_version": "io.roehub.execution-request/v1",
        "intent_id": str(intent.intent_id),
        "source_event_id": str(intent.source_event_id),
        "organization_id": str(intent.organization_id),
        "owner_user_id": str(intent.owner_user_id),
        "source_type": intent.source_type,
        "exchange_connection_id": str(intent.exchange_connection_id),
        "market_type": intent.market_type,
        "instrument_key": intent.instrument_key,
        "side": intent.side,
        "order_type": intent.order_type,
        "quantity": _decimal_field(intent.quantity),
        "quote_notional": _decimal_field(intent.quote_notional),
        "limit_price": _decimal_field(intent.limit_price),
        "risk_status": intent.risk_status,
        "risk_reason": intent.risk_reason,
        "created_at": intent.created_at.isoformat(),
        "dispatch_attempt": str(max(0, attempt_count)),
        "idempotency_key_hash": intent.idempotency_key_hash,
    }


def _validate_payload(*, payload: Mapping[str, str]) -> None:
    nullable_fields = frozenset({"quantity", "quote_notional", "limit_price"})
    missing = [
        key
        for key, value in payload.items()
        if key not in nullable_fields and (value is None or not str(value))
    ]
    if missing:
        raise ExecutionDispatchPoisonMessageError(
            reason=f"dispatch_payload_missing_{missing[0]}"
        )
    for key, value in payload.items():
        if len(str(value)) > 512:
            raise ExecutionDispatchPoisonMessageError(
                reason=f"dispatch_payload_too_large_{key}"
            )


def _decimal_field(value: Decimal | None) -> str:
    return "" if value is None else str(value)


def _bounded(reason: str) -> str:
    text = reason.strip() or "unknown"
    return text[:80]


def _resolve_password(*, environ: Mapping[str, str], password_env: str | None) -> str | None:
    if password_env is None:
        return None
    raw = environ.get(password_env)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return value


def _unavailable(error: BaseException) -> ExecutionDispatchUnavailableError:
    return ExecutionDispatchUnavailableError(reason=error.__class__.__name__)
