from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, cast
from uuid import UUID

from redis import Redis
from redis.exceptions import ResponseError

from trading.contexts.market_data.adapters.outbound.config import (
    MarketDataRuntimeConfig,
    RedisStreamsConfig,
    build_instrument_key,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.feeds import LiveCandlePublisher

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedisLiveCandlePublisherHooks:
    """
    Optional callbacks used to expose publisher runtime metrics.

    Parameters:
    - on_publish_success: callback for successful `XADD`.
    - on_publish_error: callback for non-duplicate publish failures.
    - on_publish_duplicate: callback for duplicate/out-of-order stream IDs.
    - on_publish_duration: callback observing publish call duration in seconds.
    """

    on_publish_success: Callable[[], None] | None = None
    on_publish_error: Callable[[], None] | None = None
    on_publish_duplicate: Callable[[], None] | None = None
    on_publish_duration: Callable[[float], None] | None = None


class RedisStreamsLiveCandlePublisher(LiveCandlePublisher):
    """
    Best-effort WS closed candle publisher backed by Redis Streams.
    """

    def __init__(
        self,
        *,
        config: RedisStreamsConfig,
        runtime_config: MarketDataRuntimeConfig,
        process_ingest_id: UUID,
        environ: Mapping[str, str],
        hooks: RedisLiveCandlePublisherHooks | None = None,
        redis_client: Redis | None = None,
    ) -> None:
        """
        Initialize Redis Streams publisher dependencies.

        Parameters:
        - config: parsed Redis streams runtime config.
        - runtime_config: full runtime config for instrument-key fallback.
        - process_ingest_id: process-level ingest session UUID fallback.
        - environ: environment mapping for optional Redis password lookup.
        - hooks: optional callbacks for metrics integration.
        - redis_client: optional prebuilt Redis client (tests/custom wiring).

        Returns:
        - None.

        Assumptions/Invariants:
        - Publisher is created only when `config.enabled` is true.
        - `config.maxlen_approx` is validated and non-null after config parsing.

        Errors/Exceptions:
        - Raises `ValueError` when feature is disabled or maxlen is unresolved.

        Side effects:
        - Creates Redis client when `redis_client` is not provided.
        """
        if not config.enabled:
            raise ValueError(
                "RedisStreamsLiveCandlePublisher requires enabled redis_streams config"
            )
        if config.maxlen_approx is None:
            raise ValueError("live_feed.redis_streams.maxlen_approx must be resolved")

        self._config = config
        self._runtime_config = runtime_config
        self._process_ingest_id = process_ingest_id
        self._hooks = hooks if hooks is not None else RedisLiveCandlePublisherHooks()
        self._maxlen = config.maxlen_approx
        self._redis = (
            redis_client if redis_client is not None else self._build_redis_client(environ)
        )

    def publish_1m_closed(self, candle: CandleWithMeta) -> None:
        """
        Publish one WS closed candle to per-instrument Redis stream.

        Parameters:
        - candle: closed 1m candle normalized by WS adapters.

        Returns:
        - None.

        Assumptions/Invariants:
        - Publish path is best-effort and must not raise upstream.
        - Redis Stream ID is deterministic: `<ts_open_epoch_ms>-0`.

        Errors/Exceptions:
        - Exceptions are captured, logged, and transformed into metric callbacks.

        Side effects:
        - Appends one message to Redis stream with approximate maxlen trimming.
        """
        started_at = time.perf_counter()
        stream_name = self._stream_name(candle)
        stream_id = self._stream_id(candle)
        fields = self._payload(candle)

        try:
            redis_fields = cast(dict[Any, Any], fields)
            self._redis.xadd(
                name=stream_name,
                fields=redis_fields,
                id=stream_id,
                maxlen=self._maxlen,
                approximate=True,
            )
            _emit_counter(self._hooks.on_publish_success)
        except ResponseError as exc:
            if _is_duplicate_or_out_of_order(exc):
                _emit_counter(self._hooks.on_publish_duplicate)
                return
            _emit_counter(self._hooks.on_publish_error)
            log.exception(
                "redis publish failed for stream=%s id=%s instrument_key=%s",
                stream_name,
                stream_id,
                fields["instrument_key"],
            )
        except Exception:  # noqa: BLE001
            _emit_counter(self._hooks.on_publish_error)
            log.exception(
                "redis publish failed for stream=%s id=%s instrument_key=%s",
                stream_name,
                stream_id,
                fields["instrument_key"],
            )
        finally:
            _emit_duration(self._hooks.on_publish_duration, time.perf_counter() - started_at)

    def _build_redis_client(self, environ: Mapping[str, str]) -> Redis:
        """
        Build Redis client from runtime config and environment mapping.

        Parameters:
        - environ: environment mapping used for optional password lookup.

        Returns:
        - Configured `redis.Redis` client.

        Assumptions/Invariants:
        - Password is optional; missing env variable means no password.

        Errors/Exceptions:
        - None.

        Side effects:
        - Allocates Redis client with connection pool internals.
        """
        password = self._password_from_environment(environ)
        return Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=password,
            socket_timeout=self._config.socket_timeout_s,
            socket_connect_timeout=self._config.connect_timeout_s,
            decode_responses=True,
        )

    def _password_from_environment(self, environ: Mapping[str, str]) -> str | None:
        """
        Resolve optional Redis password from configured environment variable.

        Parameters:
        - environ: environment mapping.

        Returns:
        - Password value or `None` when not configured/missing.

        Assumptions/Invariants:
        - Empty env values are treated as absent.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        key = self._config.password_env
        if key is None:
            return None
        raw_value = environ.get(key)
        if raw_value is None:
            return None
        value = raw_value.strip()
        if not value:
            return None
        return value

    def _stream_name(self, candle: CandleWithMeta) -> str:
        """
        Build stream name for one candle event.

        Parameters:
        - candle: closed candle event.

        Returns:
        - Stream name in format `<prefix>.<instrument_key>`.

        Assumptions/Invariants:
        - `stream_mode` is `per_instrument`.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return f"{self._config.stream_prefix}.{self._instrument_key(candle)}"

    def _instrument_key(self, candle: CandleWithMeta) -> str:
        """
        Resolve canonical instrument key for stream routing.

        Parameters:
        - candle: closed candle event.

        Returns:
        - Canonical instrument key.

        Assumptions/Invariants:
        - Prefer key from candle metadata.
        - Falls back to runtime-config-based builder when metadata is empty.

        Errors/Exceptions:
        - Propagates `KeyError` from runtime config when fallback market is unknown.

        Side effects:
        - None.
        """
        from_meta = candle.meta.instrument_key.strip()
        if from_meta:
            return from_meta
        return build_instrument_key(
            cfg=self._runtime_config,
            instrument_id=candle.candle.instrument_id,
        )

    def _stream_id(self, candle: CandleWithMeta) -> str:
        """
        Build deterministic Redis Stream ID from candle open timestamp.

        Parameters:
        - candle: closed candle event.

        Returns:
        - Stream entry id formatted as `<epoch_ms>-0`.

        Assumptions/Invariants:
        - Candle timestamps are UTC and millisecond-normalized.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        open_epoch_ms = int(candle.candle.ts_open.value.timestamp() * 1000)
        return f"{open_epoch_ms}-0"

    def _payload(self, candle: CandleWithMeta) -> dict[str, str]:
        """
        Convert candle event to Redis Streams wire payload schema v1.

        Parameters:
        - candle: closed candle event.

        Returns:
        - Dictionary with string values only.

        Assumptions/Invariants:
        - Schema version is fixed to `"1"` for v1 contract.
        - Source is fixed to `"ws"` for this publisher.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        row = candle.candle
        return {
            "schema_version": "1",
            "market_id": str(row.instrument_id.market_id.value),
            "symbol": str(row.instrument_id.symbol),
            "instrument_key": self._instrument_key(candle),
            "ts_open": str(row.ts_open),
            "ts_close": str(row.ts_close),
            "open": str(row.open),
            "high": str(row.high),
            "low": str(row.low),
            "close": str(row.close),
            "volume_base": str(row.volume_base),
            "volume_quote": "" if row.volume_quote is None else str(row.volume_quote),
            "source": "ws",
            "ingested_at": str(candle.meta.ingested_at),
            "ingest_id": self._ingest_id(candle),
        }

    def _ingest_id(self, candle: CandleWithMeta) -> str:
        """
        Resolve ingest identifier for output payload.

        Parameters:
        - candle: closed candle event.

        Returns:
        - UUID string from candle metadata or process-level fallback.

        Assumptions/Invariants:
        - Process ingest id is stable for worker lifetime.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        value = (
            candle.meta.ingest_id
            if candle.meta.ingest_id is not None
            else self._process_ingest_id
        )
        return str(value)


def _is_duplicate_or_out_of_order(exc: ResponseError) -> bool:
    """
    Check whether Redis XADD failure means duplicate or out-of-order stream ID.

    Parameters:
    - exc: Redis response error raised by `xadd`.

    Returns:
    - `True` when error indicates duplicate/out-of-order ID and should be ignored.

    Assumptions/Invariants:
    - Redis uses textual diagnostics for WRONGID/equal-or-smaller ID failures.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    message = str(exc).lower()
    return "equal or smaller than the target stream top item" in message or "wrongid" in message


def _emit_counter(callback: Callable[[], None] | None) -> None:
    """
    Call optional counter callback.

    Parameters:
    - callback: callback to invoke.

    Returns:
    - None.

    Assumptions/Invariants:
    - Callback has no arguments.

    Errors/Exceptions:
    - None.

    Side effects:
    - Executes callback when provided.
    """
    if callback is not None:
        callback()


def _emit_duration(callback: Callable[[float], None] | None, value: float) -> None:
    """
    Call optional duration callback with observed seconds.

    Parameters:
    - callback: callback to invoke.
    - value: observed duration in seconds.

    Returns:
    - None.

    Assumptions/Invariants:
    - Duration is non-negative.

    Errors/Exceptions:
    - None.

    Side effects:
    - Executes callback when provided.
    """
    if callback is not None:
        callback(value)
