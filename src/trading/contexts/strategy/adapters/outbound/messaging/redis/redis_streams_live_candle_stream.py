from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, cast
from uuid import UUID

from redis import Redis
from redis.exceptions import ResponseError

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.strategy.application.ports import (
    StrategyLiveCandleMessage,
    StrategyLiveCandleStream,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)

log = logging.getLogger(__name__)

_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "market_id",
        "symbol",
        "instrument_key",
        "ts_open",
        "ts_close",
        "open",
        "high",
        "low",
        "close",
        "volume_base",
        "volume_quote",
        "source",
        "ingested_at",
        "ingest_id",
    }
)


@dataclass(frozen=True, slots=True)
class RedisStrategyLiveCandleStreamConfig:
    """
    RedisStrategyLiveCandleStreamConfig — runtime config for Strategy live-runner Redis consumer.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
      - src/trading/contexts/strategy/application/ports/live_candle_stream.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
    """

    host: str
    port: int
    db: int
    password_env: str | None
    socket_timeout_s: float
    connect_timeout_s: float
    stream_prefix: str
    consumer_group: str
    consumer_name: str
    read_count: int
    block_ms: int

    def __post_init__(self) -> None:
        """
        Validate Redis stream consumer config invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Strategy v1 uses deterministic consumer group and stream prefix.
        Raises:
            ValueError: If one of required fields is invalid.
        Side Effects:
            None.
        """
        if not self.host.strip():
            raise ValueError("RedisStrategyLiveCandleStreamConfig.host must be non-empty")
        if self.port <= 0:
            raise ValueError("RedisStrategyLiveCandleStreamConfig.port must be > 0")
        if self.db < 0:
            raise ValueError("RedisStrategyLiveCandleStreamConfig.db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError(
                "RedisStrategyLiveCandleStreamConfig.socket_timeout_s must be > 0"
            )
        if self.connect_timeout_s <= 0:
            raise ValueError(
                "RedisStrategyLiveCandleStreamConfig.connect_timeout_s must be > 0"
            )
        if not self.stream_prefix.strip():
            raise ValueError("RedisStrategyLiveCandleStreamConfig.stream_prefix must be non-empty")
        if not self.consumer_group.strip():
            raise ValueError(
                "RedisStrategyLiveCandleStreamConfig.consumer_group must be non-empty"
            )
        if not self.consumer_name.strip():
            raise ValueError(
                "RedisStrategyLiveCandleStreamConfig.consumer_name must be non-empty"
            )
        if self.read_count <= 0:
            raise ValueError("RedisStrategyLiveCandleStreamConfig.read_count must be > 0")
        if self.block_ms < 0:
            raise ValueError("RedisStrategyLiveCandleStreamConfig.block_ms must be >= 0")


class RedisStrategyLiveCandleStream(StrategyLiveCandleStream):
    """
    RedisStrategyLiveCandleStream — Redis Streams consumer for Strategy live-runner v1.

    Consumer-group contract for v1:
    - group default: `strategy.live_runner.v1`
    - consumer name: deterministic instance id (`<hostname>-<pid>`)

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
      - src/trading/contexts/strategy/application/ports/live_candle_stream.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
    """

    def __init__(
        self,
        *,
        config: RedisStrategyLiveCandleStreamConfig,
        environ: Mapping[str, str],
        redis_client: Redis | None = None,
    ) -> None:
        """
        Initialize stream consumer dependencies.

        Args:
            config: Validated Redis consumer runtime config.
            environ: Environment mapping for optional password lookup.
            redis_client: Optional injected Redis client for tests.
        Returns:
            None.
        Assumptions:
            Runtime wiring provides deterministic consumer group and consumer name.
        Raises:
            ValueError: If required config is missing.
        Side Effects:
            Creates Redis client when `redis_client` is not injected.
        """
        if config is None:  # type: ignore[truthy-bool]
            raise ValueError("RedisStrategyLiveCandleStream requires config")
        self._config = config
        self._redis = (
            redis_client if redis_client is not None else self._build_redis_client(environ)
        )
        self._created_groups: set[str] = set()

    def read_closed_1m(self, *, instrument_key: str) -> tuple[StrategyLiveCandleMessage, ...]:
        """
        Read and parse closed 1m candle messages for one instrument stream.

        Args:
            instrument_key: Canonical instrument key.
        Returns:
            tuple[StrategyLiveCandleMessage, ...]: Parsed message list.
        Assumptions:
            Stream is published by market-data WS worker with schema version `1`.
        Raises:
            Exception: Redis/network errors while reading stream.
        Side Effects:
            May auto-create Redis consumer group for stream.
            Invalid payloads are acknowledged and dropped to avoid poison-message loops.
        """
        stream_name = self._stream_name(instrument_key=instrument_key)
        self._ensure_group(stream_name=stream_name)

        raw_events = cast(
            list[tuple[str, list[tuple[str, Mapping[str, Any]]]]],
            self._redis.xreadgroup(
                groupname=self._config.consumer_group,
                consumername=self._config.consumer_name,
                streams={stream_name: ">"},
                count=self._config.read_count,
                block=self._config.block_ms,
            ),
        )

        out: list[StrategyLiveCandleMessage] = []
        for _, entries in raw_events:
            for message_id, fields in entries:
                try:
                    candle = _parse_candle_payload(fields=fields)
                except Exception as error:  # noqa: BLE001
                    log.exception(
                        "strategy live-runner drop invalid redis candle payload stream=%s id=%s",
                        stream_name,
                        message_id,
                    )
                    self.ack(instrument_key=instrument_key, message_id=message_id)
                    _ = error
                    continue
                out.append(StrategyLiveCandleMessage(message_id=message_id, candle=candle))
        return tuple(out)

    def ack(self, *, instrument_key: str, message_id: str) -> None:
        """
        Acknowledge one processed message in configured consumer group.

        Args:
            instrument_key: Canonical instrument key.
            message_id: Stream message id.
        Returns:
            None.
        Assumptions:
            Ack is called only after successful persistence in Strategy run storage.
        Raises:
            Exception: Redis/network errors while acknowledging message.
        Side Effects:
            Removes one entry from consumer-group pending list.
        """
        stream_name = self._stream_name(instrument_key=instrument_key)
        self._redis.xack(stream_name, self._config.consumer_group, message_id)

    def _stream_name(self, *, instrument_key: str) -> str:
        """
        Build deterministic stream name for one instrument.

        Args:
            instrument_key: Canonical instrument key.
        Returns:
            str: Stream name `md.candles.1m.<instrument_key>`.
        Assumptions:
            Prefix is configured as `md.candles.1m` in live feed contract.
        Raises:
            ValueError: If instrument key is blank.
        Side Effects:
            None.
        """
        normalized = instrument_key.strip()
        if not normalized:
            raise ValueError("Redis stream instrument_key must be non-empty")
        return f"{self._config.stream_prefix}.{normalized}"

    def _ensure_group(self, *, stream_name: str) -> None:
        """
        Create Redis consumer group lazily for one stream when absent.

        Args:
            stream_name: Target Redis stream name.
        Returns:
            None.
        Assumptions:
            Group creation is idempotent with BUSYGROUP response handling.
        Raises:
            Exception: Redis/network errors except BUSYGROUP conflict.
        Side Effects:
            Creates stream/group with `MKSTREAM` when missing.
        """
        if stream_name in self._created_groups:
            return
        try:
            self._redis.xgroup_create(
                name=stream_name,
                groupname=self._config.consumer_group,
                id="$",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise
        self._created_groups.add(stream_name)

    def _build_redis_client(self, environ: Mapping[str, str]) -> Redis:
        """
        Build Redis client from runtime config and environment mapping.

        Args:
            environ: Environment mapping for optional password lookup.
        Returns:
            Redis: Configured redis-py client.
        Assumptions:
            Missing password env value means no password.
        Raises:
            None.
        Side Effects:
            Allocates redis client instance.
        """
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


def _resolve_password(*, environ: Mapping[str, str], password_env: str | None) -> str | None:
    """
    Resolve optional Redis password from environment mapping.

    Args:
        environ: Environment mapping.
        password_env: Password env variable name.
    Returns:
        str | None: Password value or `None`.
    Assumptions:
        Empty env values are treated as missing.
    Raises:
        None.
    Side Effects:
        None.
    """
    if password_env is None:
        return None
    raw = environ.get(password_env)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return value


def _parse_candle_payload(*, fields: Mapping[str, Any]) -> CandleWithMeta:
    """
    Parse Redis stream payload into canonical CandleWithMeta object.

    Args:
        fields: Redis message field mapping.
    Returns:
        CandleWithMeta: Parsed candle payload.
    Assumptions:
        Payload follows market-data live feed schema version `1`.
    Raises:
        ValueError: If payload is malformed or violates candle invariants.
    Side Effects:
        None.
    """
    missing = sorted(_REQUIRED_FIELDS.difference(fields.keys()))
    if missing:
        raise ValueError(f"Redis candle payload misses required fields: {missing}")

    schema_version = str(fields["schema_version"]).strip()
    if schema_version != "1":
        raise ValueError(f"Unsupported Redis candle schema_version={schema_version!r}")

    market_id = _parse_int(value=fields["market_id"], field_name="market_id")
    symbol = str(fields["symbol"]).strip()
    if not symbol:
        raise ValueError("Redis candle payload field 'symbol' must be non-empty")
    instrument_key = str(fields["instrument_key"]).strip()
    if not instrument_key:
        raise ValueError("Redis candle payload field 'instrument_key' must be non-empty")

    ts_open = UtcTimestamp(_parse_iso_utc(value=fields["ts_open"], field_name="ts_open"))
    ts_close = UtcTimestamp(_parse_iso_utc(value=fields["ts_close"], field_name="ts_close"))
    candle = Candle(
        instrument_id=InstrumentId(market_id=MarketId(market_id), symbol=Symbol(symbol)),
        ts_open=ts_open,
        ts_close=ts_close,
        open=_parse_float(value=fields["open"], field_name="open"),
        high=_parse_float(value=fields["high"], field_name="high"),
        low=_parse_float(value=fields["low"], field_name="low"),
        close=_parse_float(value=fields["close"], field_name="close"),
        volume_base=_parse_float(value=fields["volume_base"], field_name="volume_base"),
        volume_quote=_parse_optional_float(
            value=fields["volume_quote"],
            field_name="volume_quote",
        ),
    )

    ingest_id = _parse_optional_uuid(value=fields["ingest_id"], field_name="ingest_id")
    meta = CandleMeta(
        source=str(fields["source"]).strip(),
        ingested_at=UtcTimestamp(
            _parse_iso_utc(value=fields["ingested_at"], field_name="ingested_at")
        ),
        ingest_id=ingest_id,
        instrument_key=instrument_key,
        trades_count=None,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def _parse_int(*, value: Any, field_name: str) -> int:
    """
    Parse integer payload field with deterministic error messages.

    Args:
        value: Raw payload field value.
        field_name: Field name for validation errors.
    Returns:
        int: Parsed integer value.
    Assumptions:
        Boolean values are rejected explicitly.
    Raises:
        ValueError: If value cannot be parsed as integer.
    Side Effects:
        None.
    """
    if isinstance(value, bool):
        raise ValueError(f"Redis candle payload field {field_name!r} must be integer, got bool")
    text = str(value).strip()
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(
            f"Redis candle payload field {field_name!r} must be integer, got {text!r}"
        ) from error


def _parse_float(*, value: Any, field_name: str) -> float:
    """
    Parse float payload field with deterministic error messages.

    Args:
        value: Raw payload field value.
        field_name: Field name for validation errors.
    Returns:
        float: Parsed floating-point value.
    Assumptions:
        Payload uses string numeric encoding.
    Raises:
        ValueError: If value cannot be parsed as float.
    Side Effects:
        None.
    """
    text = str(value).strip()
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(
            f"Redis candle payload field {field_name!r} must be float, got {text!r}"
        ) from error


def _parse_optional_float(*, value: Any, field_name: str) -> float | None:
    """
    Parse optional float payload field where empty string means null.

    Args:
        value: Raw payload field value.
        field_name: Field name for validation errors.
    Returns:
        float | None: Parsed value or `None`.
    Assumptions:
        Empty strings represent null values in Redis schema v1.
    Raises:
        ValueError: If non-empty value cannot be parsed as float.
    Side Effects:
        None.
    """
    text = str(value).strip()
    if not text:
        return None
    return _parse_float(value=text, field_name=field_name)


def _parse_optional_uuid(*, value: Any, field_name: str) -> UUID | None:
    """
    Parse optional UUID payload field where empty string means null.

    Args:
        value: Raw payload field value.
        field_name: Field name for validation errors.
    Returns:
        UUID | None: Parsed UUID value or `None`.
    Assumptions:
        Empty strings represent null values for optional UUID fields.
    Raises:
        ValueError: If non-empty value cannot be parsed as UUID.
    Side Effects:
        None.
    """
    text = str(value).strip()
    if not text:
        return None
    try:
        return UUID(text)
    except ValueError as error:
        raise ValueError(
            f"Redis candle payload field {field_name!r} must be UUID, got {text!r}"
        ) from error


def _parse_iso_utc(*, value: Any, field_name: str) -> datetime:
    """
    Parse ISO-8601 datetime payload value with `Z` suffix support.

    Args:
        value: Raw payload field value.
        field_name: Field name for validation errors.
    Returns:
        datetime: Parsed timezone-aware datetime.
    Assumptions:
        UtcTimestamp validation performs final UTC and precision normalization.
    Raises:
        ValueError: If value is not valid ISO datetime.
    Side Effects:
        None.
    """
    text = str(value).strip()
    if not text:
        raise ValueError(f"Redis candle payload field {field_name!r} must be non-empty")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"Redis candle payload field {field_name!r} must be ISO datetime, got {text!r}"
        ) from error
