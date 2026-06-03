from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping, cast

from redis import Redis
from redis.exceptions import RedisError, ResponseError

from trading.contexts.live_execution.adapters.outbound.redis.execution_dispatch_transport import (
    RedisExecutionDispatchTransportConfig,
    _resolve_password,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionConsumer,
    ExchangeExecutionRedisHealth,
    ExchangeExecutionRedisMessage,
    ExecutionDispatchPublishResult,
    ExecutionDispatchUnavailableError,
)


class RedisExchangeExecutionConsumer(ExchangeExecutionConsumer):
    def __init__(
        self,
        *,
        config: RedisExecutionDispatchTransportConfig,
        consumer_name: str,
        environ: Mapping[str, str],
        redis_client: Redis | None = None,
    ) -> None:
        if not consumer_name.strip():
            raise ValueError("RedisExchangeExecutionConsumer.consumer_name must be non-empty")
        self._config = config
        self._consumer_name = consumer_name.strip()
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

    def health_snapshot(self) -> ExchangeExecutionRedisHealth:
        try:
            request_length = int(cast(Any, self._redis.xlen(self._config.request_stream)))
            retry_length = int(cast(Any, self._redis.xlen(self._config.retry_stream)))
            dlq_length = int(cast(Any, self._redis.xlen(self._config.dlq_stream)))
            pending = self._pending_count()
            drift_ms = self._clock_drift_ms()
        except RedisError as error:
            raise _unavailable(error) from error
        return ExchangeExecutionRedisHealth(
            request_stream_length=request_length,
            retry_stream_length=retry_length,
            dlq_stream_length=dlq_length,
            pending_count=pending,
            clock_drift_ms=drift_ms,
        )

    def read_new_requests(
        self, *, count: int, block_ms: int
    ) -> tuple[ExchangeExecutionRedisMessage, ...]:
        try:
            raw_response = self._redis.xreadgroup(
                groupname=self._config.consumer_group,
                consumername=self._consumer_name,
                streams={self._config.request_stream: ">"},
                count=count,
                block=block_ms,
            )
        except RedisError as error:
            raise _unavailable(error) from error
        return _messages_from_raw_response(raw_response=raw_response)

    def read_pending_requests(self, *, count: int) -> tuple[ExchangeExecutionRedisMessage, ...]:
        try:
            raw_response = self._redis.xreadgroup(
                groupname=self._config.consumer_group,
                consumername=self._consumer_name,
                streams={self._config.request_stream: "0"},
                count=count,
                block=0,
            )
        except RedisError as error:
            raise _unavailable(error) from error
        return _messages_from_raw_response(raw_response=raw_response)

    def publish_dlq(
        self, *, message: ExchangeExecutionRedisMessage, reason: str
    ) -> ExecutionDispatchPublishResult:
        payload = {
            "schema_version": "1",
            "source_stream": message.stream_name,
            "source_message_id": message.message_id,
            "quarantine_reason": _bounded(reason),
        }
        intent_id = message.payload.get("intent_id")
        if intent_id is not None:
            payload["intent_id"] = intent_id
        try:
            message_id = str(
                self._redis.xadd(
                    name=self._config.dlq_stream,
                    fields=cast(dict[Any, Any], payload),
                )
            )
        except RedisError as error:
            raise _unavailable(error) from error
        return ExecutionDispatchPublishResult(
            stream_name=self._config.dlq_stream,
            message_id=message_id,
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        try:
            self._redis.xack(stream_name, self._config.consumer_group, message_id)
        except RedisError as error:
            raise _unavailable(error) from error

    def _pending_count(self) -> int:
        try:
            raw = self._redis.xpending(self._config.request_stream, self._config.consumer_group)
        except ResponseError as error:
            if "NOGROUP" not in str(error):
                raise
            return 0
        if isinstance(raw, Mapping):
            return int(raw.get("pending", raw.get("count", 0)) or 0)
        return 0

    def _clock_drift_ms(self) -> float:
        redis_time = cast(tuple[int, int], self._redis.time())
        seconds = int(redis_time[0])
        microseconds = int(redis_time[1])
        redis_now = datetime.fromtimestamp(seconds + microseconds / 1_000_000, tz=UTC)
        local_now = datetime.now(tz=UTC)
        return (local_now - redis_now).total_seconds() * 1000

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


def _bounded(reason: str) -> str:
    text = reason.strip() or "unknown"
    return text[:80]


def _messages_from_raw_response(raw_response: object) -> tuple[ExchangeExecutionRedisMessage, ...]:
    messages: list[ExchangeExecutionRedisMessage] = []
    for stream_name, stream_messages in cast(Any, raw_response):
        for message_id, payload in stream_messages:
            messages.append(
                ExchangeExecutionRedisMessage(
                    stream_name=str(stream_name),
                    message_id=str(message_id),
                    payload={str(key): str(value) for key, value in dict(payload).items()},
                )
            )
    return tuple(messages)


def _unavailable(error: BaseException) -> ExecutionDispatchUnavailableError:
    return ExecutionDispatchUnavailableError(reason=error.__class__.__name__)
