from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, cast
from uuid import UUID

from redis import Redis

from trading.contexts.market_data.adapters.outbound.config import (
    RedisHotCacheConfig,
    RedisStreamsConfig,
)
from trading.contexts.market_data.application.dto import CandleWithMeta, ClosedCandleTailRow
from trading.contexts.market_data.application.ports.feeds import LiveCandlePublisher
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedisCandleHotCacheHooks:
    """Optional callbacks used to expose hot-cache runtime metrics."""

    on_write_success: Callable[[], None] | None = None
    on_write_error: Callable[[], None] | None = None
    on_write_duration: Callable[[float], None] | None = None
    on_read_hit: Callable[[], None] | None = None
    on_read_miss: Callable[[], None] | None = None
    on_read_error: Callable[[], None] | None = None
    on_read_duration: Callable[[float], None] | None = None


class RedisCandleHotCache:
    """Redis range-store for short-tail closed 1m candles."""

    def __init__(
        self,
        *,
        connection_config: RedisStreamsConfig,
        config: RedisHotCacheConfig,
        environ: Mapping[str, str],
        hooks: RedisCandleHotCacheHooks | None = None,
        redis_client: Redis | None = None,
    ) -> None:
        """
        Initialize Redis hot cache dependencies.

        Parameters:
        - connection_config: Redis connection settings shared with live feed.
        - config: hot-cache key and retention settings.
        - environ: environment mapping for optional Redis auth lookup.
        - hooks: optional callbacks for metrics integration.
        - redis_client: optional prebuilt Redis client for tests/custom wiring.
        """
        if not config.enabled:
            raise ValueError("RedisCandleHotCache requires enabled redis_hot_cache config")

        self._connection_config = connection_config
        self._config = config
        self._hooks = hooks if hooks is not None else RedisCandleHotCacheHooks()
        self._redis = (
            redis_client if redis_client is not None else self._build_redis_client(environ)
        )

    def write_closed_1m(self, candle: CandleWithMeta) -> bool:
        """
        Store one closed 1m candle in Redis hot cache.

        Duplicate writes overwrite the same hash field and zset member identified by
        `ts_open_epoch_ms`, so one minute cannot produce ambiguous rows.
        """
        started_at = time.perf_counter()
        instrument_key = candle.meta.instrument_key
        z_key = self._z_key(instrument_key)
        h_key = self._h_key(instrument_key)
        ts_open_epoch_ms = _epoch_ms(candle.candle.ts_open)
        member = str(ts_open_epoch_ms)

        try:
            payload = self._payload(candle, ts_open_epoch_ms=ts_open_epoch_ms)
            self._redis.hset(h_key, member, payload)
            self._redis.zadd(z_key, {member: ts_open_epoch_ms})
            self._prune_expired(z_key=z_key, h_key=h_key, ts_open_epoch_ms=ts_open_epoch_ms)
            _emit_counter(self._hooks.on_write_success)
            return True
        except Exception:  # noqa: BLE001
            _emit_counter(self._hooks.on_write_error)
            log.exception(
                "redis hot cache write failed for instrument_key=%s ts_open_epoch_ms=%s",
                instrument_key,
                ts_open_epoch_ms,
            )
            return False
        finally:
            _emit_duration(self._hooks.on_write_duration, time.perf_counter() - started_at)

    def read_range(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        start: UtcTimestamp,
        end: UtcTimestamp,
    ) -> tuple[ClosedCandleTailRow, ...]:
        """Read hot-cache candles for half-open range `[start,end)`."""
        if start.value >= end.value:
            raise ValueError("redis hot cache read_range requires start < end")

        started_at = time.perf_counter()
        key = instrument_key.strip()
        if not key:
            raise ValueError("redis hot cache read_range requires non-empty instrument_key")
        z_key = self._z_key(key)
        h_key = self._h_key(key)

        try:
            raw_members = cast(
                list[object],
                self._redis.zrangebyscore(
                    z_key,
                    min=_epoch_ms(start),
                    max=f"({_epoch_ms(end)}",
                ),
            )
            members = tuple(
                _as_text(value)
                for value in raw_members
            )
            if not members:
                _emit_counter(self._hooks.on_read_miss)
                return ()

            payloads = cast(list[object | None], self._redis.hmget(h_key, list(members)))
            rows = tuple(
                self._row_from_payload(
                    payload,
                    expected_instrument_id=instrument_id,
                    expected_instrument_key=key,
                )
                for payload in payloads
                if payload is not None
            )
            sorted_rows = tuple(sorted(rows, key=lambda row: row.ts_open.value))
            if sorted_rows:
                _emit_counter(self._hooks.on_read_hit)
                return sorted_rows
            _emit_counter(self._hooks.on_read_miss)
            return ()
        except Exception:
            _emit_counter(self._hooks.on_read_error)
            log.exception(
                "redis hot cache read failed for instrument_key=%s start=%s end=%s",
                key,
                start,
                end,
            )
            raise
        finally:
            _emit_duration(self._hooks.on_read_duration, time.perf_counter() - started_at)

    def _build_redis_client(self, environ: Mapping[str, str]) -> Redis:
        redis_kwargs: dict[str, Any] = {
            "host": self._connection_config.host,
            "port": self._connection_config.port,
            "db": self._connection_config.db,
            "socket_timeout": self._connection_config.socket_timeout_s,
            "socket_connect_timeout": self._connection_config.connect_timeout_s,
            "decode_responses": True,
        }
        redis_auth = _redis_auth_from_environment(self._connection_config, environ)
        if redis_auth is not None:
            redis_kwargs.update({"password": redis_auth})
        return Redis(**cast(Any, redis_kwargs))

    def _z_key(self, instrument_key: str) -> str:
        return f"{self._config.key_prefix}:{instrument_key}:z"

    def _h_key(self, instrument_key: str) -> str:
        return f"{self._config.key_prefix}:{instrument_key}:h"

    def _payload(self, candle: CandleWithMeta, *, ts_open_epoch_ms: int) -> str:
        row = candle.candle
        meta = candle.meta
        payload: dict[str, object] = {
            "schema_version": 1,
            "market_id": row.instrument_id.market_id.value,
            "symbol": str(row.instrument_id.symbol),
            "instrument_key": meta.instrument_key,
            "ts_open": str(row.ts_open),
            "ts_close": str(row.ts_close),
            "ts_open_epoch_ms": ts_open_epoch_ms,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume_base": row.volume_base,
            "volume_quote": row.volume_quote,
            "source": meta.source,
            "ingested_at": str(meta.ingested_at),
            "ingest_id": None if meta.ingest_id is None else str(meta.ingest_id),
            "trades_count": meta.trades_count,
            "taker_buy_volume_base": meta.taker_buy_volume_base,
            "taker_buy_volume_quote": meta.taker_buy_volume_quote,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _row_from_payload(
        self,
        payload: object,
        *,
        expected_instrument_id: InstrumentId,
        expected_instrument_key: str,
    ) -> ClosedCandleTailRow:
        parsed = json.loads(_as_text(payload))
        if not isinstance(parsed, dict):
            raise ValueError("redis hot cache payload must be a JSON object")

        instrument_key = _required_text(parsed, "instrument_key")
        if instrument_key != expected_instrument_key:
            raise ValueError("redis hot cache payload instrument_key mismatch")
        instrument_id = InstrumentId(
            market_id=MarketId(_required_int(parsed, "market_id")),
            symbol=Symbol(_required_text(parsed, "symbol")),
        )
        if instrument_id != expected_instrument_id:
            raise ValueError("redis hot cache payload instrument_id mismatch")

        candle = Candle(
            instrument_id=instrument_id,
            ts_open=_parse_utc_timestamp(_required_text(parsed, "ts_open")),
            ts_close=_parse_utc_timestamp(_required_text(parsed, "ts_close")),
            open=_required_float(parsed, "open"),
            high=_required_float(parsed, "high"),
            low=_required_float(parsed, "low"),
            close=_required_float(parsed, "close"),
            volume_base=_required_float(parsed, "volume_base"),
            volume_quote=_optional_float(parsed.get("volume_quote")),
        )
        meta = CandleMeta(
            source=_required_text(parsed, "source"),
            ingested_at=_parse_utc_timestamp(_required_text(parsed, "ingested_at")),
            ingest_id=_optional_uuid(parsed.get("ingest_id")),
            instrument_key=instrument_key,
            trades_count=_optional_int(parsed.get("trades_count")),
            taker_buy_volume_base=_optional_float(parsed.get("taker_buy_volume_base")),
            taker_buy_volume_quote=_optional_float(parsed.get("taker_buy_volume_quote")),
        )
        return ClosedCandleTailRow(
            candle=CandleWithMeta(candle=candle, meta=meta),
            source="redis_hot_cache",
        )

    def _prune_expired(self, *, z_key: str, h_key: str, ts_open_epoch_ms: int) -> None:
        cutoff_ms = ts_open_epoch_ms - self._config.retention_ms
        raw_expired_members = cast(
            list[object],
            self._redis.zrangebyscore(z_key, min="-inf", max=f"({cutoff_ms}"),
        )
        expired_members = tuple(
            _as_text(value)
            for value in raw_expired_members
        )
        if not expired_members:
            return
        self._redis.zremrangebyscore(z_key, min="-inf", max=f"({cutoff_ms}")
        self._redis.hdel(h_key, *expired_members)


class RedisHotCacheLiveCandlePublisher(LiveCandlePublisher):
    """Live publisher adapter that writes WS closed candles into Redis hot cache."""

    def __init__(self, hot_cache: RedisCandleHotCache) -> None:
        if hot_cache is None:  # type: ignore[truthy-bool]
            raise ValueError("RedisHotCacheLiveCandlePublisher requires hot_cache")
        self._hot_cache = hot_cache

    def publish_1m_closed(self, candle: CandleWithMeta) -> None:
        """Store one closed 1m candle in the hot cache."""
        self._hot_cache.write_closed_1m(candle)


def _redis_auth_from_environment(
    config: RedisStreamsConfig,
    environ: Mapping[str, str],
) -> str | None:
    key = config.password_env
    if key is None:
        return None
    env_value = environ.get(key)
    if env_value is None:
        return None
    value = env_value.strip()
    if not value:
        return None
    return value


def _epoch_ms(value: UtcTimestamp) -> int:
    return int(value.value.timestamp() * 1000)


def _parse_utc_timestamp(value: str) -> UtcTimestamp:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return UtcTimestamp(datetime.fromisoformat(normalized))


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"redis hot cache payload requires non-empty {key}")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"redis hot cache payload requires int {key}")
    return value


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"redis hot cache payload requires numeric {key}")
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("redis hot cache payload optional float must be numeric")
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("redis hot cache payload optional int must be int")
    return value


def _optional_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("redis hot cache payload ingest_id must be string")
    return UUID(value)


def _emit_counter(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _emit_duration(callback: Callable[[float], None] | None, value: float) -> None:
    if callback is not None:
        callback(value)
