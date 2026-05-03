from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, cast
from uuid import UUID

from redis import Redis

from trading.contexts.strategy.application.ports import (
    StrategyRealtimeOutputReader,
    StrategyRealtimeStreamMessageV1,
    StrategyRealtimeStreamUnavailableError,
)
from trading.shared_kernel.primitives import UserId

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedisStrategyRealtimeOutputReaderConfig:
    """
    Redis config for read-only browser-facing Strategy realtime stream bridge.
    """

    host: str
    port: int
    db: int
    password_env: str | None
    socket_timeout_s: float
    connect_timeout_s: float
    metrics_stream_prefix: str
    events_stream_prefix: str

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("Redis realtime output reader host must be non-empty")
        if self.port <= 0:
            raise ValueError("Redis realtime output reader port must be > 0")
        if self.db < 0:
            raise ValueError("Redis realtime output reader db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError("Redis realtime output reader socket_timeout_s must be > 0")
        if self.connect_timeout_s <= 0:
            raise ValueError("Redis realtime output reader connect_timeout_s must be > 0")
        if not self.metrics_stream_prefix.strip():
            raise ValueError("Redis realtime output reader metrics prefix must be non-empty")
        if not self.events_stream_prefix.strip():
            raise ValueError("Redis realtime output reader events prefix must be non-empty")


class RedisStrategyRealtimeOutputReader(StrategyRealtimeOutputReader):
    """
    Read-only XREAD adapter for per-user Strategy realtime Redis Streams v1.
    """

    def __init__(
        self,
        *,
        config: RedisStrategyRealtimeOutputReaderConfig,
        environ: Mapping[str, str],
        redis_client: Redis | None = None,
    ) -> None:
        if config is None:  # type: ignore[truthy-bool]
            raise ValueError("RedisStrategyRealtimeOutputReader requires config")
        self._config = config
        self._redis = (
            redis_client if redis_client is not None else self._build_redis_client(environ)
        )

    def read_for_user(
        self,
        *,
        user_id: UserId,
        strategy_id: UUID | None,
        last_event_id: str,
        count: int,
        block_ms: int,
    ) -> tuple[StrategyRealtimeStreamMessageV1, ...]:
        if count <= 0:
            raise ValueError("Redis realtime output reader count must be > 0")
        if block_ms < 0:
            raise ValueError("Redis realtime output reader block_ms must be >= 0")

        normalized_last_id = _normalize_last_event_id(last_event_id=last_event_id)
        streams = {
            self._metrics_stream_name(user_id=user_id): normalized_last_id,
            self._events_stream_name(user_id=user_id): normalized_last_id,
        }
        redis_streams = cast(dict[Any, Any], streams)
        try:
            raw_events = cast(
                list[tuple[Any, list[tuple[Any, Mapping[Any, Any]]]]],
                self._redis.xread(streams=redis_streams, count=count, block=block_ms),
            )
        except Exception as error:  # noqa: BLE001
            log.warning("strategy realtime output xread unavailable: %s", error)
            raise StrategyRealtimeStreamUnavailableError(
                "Redis realtime stream unavailable"
            ) from error

        out: list[StrategyRealtimeStreamMessageV1] = []
        for raw_stream, entries in raw_events:
            stream_name = _decode_scalar(raw_stream)
            stream_kind = (
                "metric"
                if stream_name.startswith(self._config.metrics_stream_prefix)
                else "event"
            )
            for raw_message_id, raw_payload in entries:
                payload = _decode_payload(raw_payload)
                if strategy_id is not None and payload.get("strategy_id") != str(strategy_id):
                    continue
                out.append(
                    StrategyRealtimeStreamMessageV1(
                        stream=stream_name,
                        stream_kind=stream_kind,
                        message_id=_decode_scalar(raw_message_id),
                        payload=payload,
                    )
                )
        return tuple(out)

    def _metrics_stream_name(self, *, user_id: UserId) -> str:
        return f"{self._config.metrics_stream_prefix}.{user_id}"

    def _events_stream_name(self, *, user_id: UserId) -> str:
        return f"{self._config.events_stream_prefix}.{user_id}"

    def _build_redis_client(self, environ: Mapping[str, str]) -> Redis:
        password = None
        if self._config.password_env is not None:
            password = environ.get(self._config.password_env) or os.environ.get(
                self._config.password_env
            )
        return Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=password,
            socket_timeout=self._config.socket_timeout_s,
            socket_connect_timeout=self._config.connect_timeout_s,
            decode_responses=False,
        )


def _normalize_last_event_id(*, last_event_id: str) -> str:
    normalized = last_event_id.strip()
    if not normalized:
        return "$"
    return normalized


def _decode_payload(raw_payload: Mapping[Any, Any]) -> dict[str, str]:
    return {
        _decode_scalar(raw_key): _decode_scalar(raw_value)
        for raw_key, raw_value in raw_payload.items()
    }


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


__all__ = ["RedisStrategyRealtimeOutputReader", "RedisStrategyRealtimeOutputReaderConfig"]
