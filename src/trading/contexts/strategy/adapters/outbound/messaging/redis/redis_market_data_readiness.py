from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping, cast

from redis import Redis
from redis.exceptions import RedisError, ResponseError

from trading.contexts.strategy.application.ports.market_data_readiness import (
    MarketDataReadinessReader,
    MarketDataReadinessSnapshot,
)

from .redis_streams_live_candle_stream import (
    RedisStrategyLiveCandleStreamConfig,
    _resolve_password,
)

_DEFAULT_STALE_AFTER_SECONDS = 180


class RedisMarketDataReadinessReader(MarketDataReadinessReader):
    def __init__(
        self,
        *,
        config: RedisStrategyLiveCandleStreamConfig,
        environ: Mapping[str, str],
        stale_after_seconds: int = _DEFAULT_STALE_AFTER_SECONDS,
        redis_client: Redis | None = None,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be > 0")
        self._config = config
        self._stale_after_seconds = stale_after_seconds
        self._redis = (
            redis_client if redis_client is not None else self._build_redis_client(environ)
        )

    def check(
        self,
        *,
        instrument_key: str,
        timeframe: str,
        observed_at: datetime,
    ) -> MarketDataReadinessSnapshot:
        stream_name = f"{self._config.stream_prefix}.{instrument_key.strip()}"
        if not instrument_key.strip():
            raise ValueError("instrument_key must be non-empty")
        if timeframe.strip() != "1m":
            required_stream_name = stream_name
        else:
            required_stream_name = stream_name
        try:
            info = cast(Mapping[str, Any], self._redis.xinfo_stream(required_stream_name))
        except ResponseError as error:
            if "no such key" in str(error).lower():
                return MarketDataReadinessSnapshot(
                    state="missing",
                    reason_code="market_data_stream_missing",
                    stream_name=required_stream_name,
                    stream_length=0,
                    last_message_id=None,
                    last_observed_at=None,
                    age_seconds=None,
                )
            raise
        except RedisError:
            return MarketDataReadinessSnapshot(
                state="pending",
                reason_code="market_data_readiness_probe_unavailable",
                stream_name=required_stream_name,
                stream_length=None,
                last_message_id=None,
                last_observed_at=None,
                age_seconds=None,
            )

        length = _int_or_none(info.get("length"))
        last_id = str(info.get("last-generated-id") or "").strip() or None
        if not length or last_id is None:
            return MarketDataReadinessSnapshot(
                state="pending",
                reason_code="market_data_stream_empty",
                stream_name=required_stream_name,
                stream_length=length or 0,
                last_message_id=last_id,
                last_observed_at=None,
                age_seconds=None,
            )
        last_at = _timestamp_from_redis_id(last_id)
        age_seconds = max(0, int((observed_at - last_at).total_seconds()))
        if age_seconds > self._stale_after_seconds:
            return MarketDataReadinessSnapshot(
                state="stale",
                reason_code="market_data_stream_stale",
                stream_name=required_stream_name,
                stream_length=length,
                last_message_id=last_id,
                last_observed_at=last_at,
                age_seconds=age_seconds,
            )
        return MarketDataReadinessSnapshot(
            state="ready",
            reason_code=(
                "market_data_stream_ready"
                if timeframe.strip() == "1m"
                else "market_data_stream_ready_for_rollup"
            ),
            stream_name=required_stream_name,
            stream_length=length,
            last_message_id=last_id,
            last_observed_at=last_at,
            age_seconds=age_seconds,
        )

    def _build_redis_client(self, environ: Mapping[str, str]) -> Redis:
        return Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            password=_resolve_password(environ=environ, password_env=self._config.password_env),
            socket_timeout=self._config.socket_timeout_s,
            socket_connect_timeout=self._config.connect_timeout_s,
            decode_responses=True,
        )


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _timestamp_from_redis_id(message_id: str) -> datetime:
    milliseconds = int(message_id.split("-", 1)[0])
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
