from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

from redis import Redis
from redis.exceptions import ResponseError

from trading.contexts.strategy.application.ports import (
    SCHEMA_VERSION_V1,
    StrategyRealtimeEventV1,
    StrategyRealtimeMetricV1,
    StrategyRealtimeOutputPublisher,
    StrategyRealtimeOutputRecordV1,
)
from trading.shared_kernel.primitives import UtcTimestamp

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedisStrategyRealtimeOutputPublisherConfig:
    """
    RedisStrategyRealtimeOutputPublisherConfig — Redis Streams runtime config
    for realtime output v1.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - configs/dev/strategy_live_runner.yaml
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
        """
        Validate realtime Redis config invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stream prefixes map to per-user streams with final `.user.<user_id>` suffix.
        Raises:
            ValueError: If one of required config values is invalid.
        Side Effects:
            None.
        """
        if not self.host.strip():
            raise ValueError("Redis realtime output host must be non-empty")
        if self.port <= 0:
            raise ValueError("Redis realtime output port must be > 0")
        if self.db < 0:
            raise ValueError("Redis realtime output db must be >= 0")
        if self.socket_timeout_s <= 0:
            raise ValueError("Redis realtime output socket_timeout_s must be > 0")
        if self.connect_timeout_s <= 0:
            raise ValueError("Redis realtime output connect_timeout_s must be > 0")
        if not self.metrics_stream_prefix.strip():
            raise ValueError("Redis realtime output metrics_stream_prefix must be non-empty")
        if not self.events_stream_prefix.strip():
            raise ValueError("Redis realtime output events_stream_prefix must be non-empty")


@dataclass(frozen=True, slots=True)
class RedisStrategyRealtimeOutputPublisherHooks:
    """
    RedisStrategyRealtimeOutputPublisherHooks — optional callbacks for publish counters.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - tests/unit/contexts/strategy/adapters/test_redis_strategy_realtime_output_publisher.py
    """

    on_publish_success: Callable[[], None] | None = None
    on_publish_error: Callable[[], None] | None = None
    on_publish_duplicate: Callable[[], None] | None = None


class RedisStrategyRealtimeOutputPublisher(StrategyRealtimeOutputPublisher):
    """
    RedisStrategyRealtimeOutputPublisher — best-effort Redis Streams publisher
    for Strategy realtime v1.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/realtime_output_publisher.py
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
    """

    def __init__(
        self,
        *,
        config: RedisStrategyRealtimeOutputPublisherConfig,
        environ: Mapping[str, str],
        hooks: RedisStrategyRealtimeOutputPublisherHooks | None = None,
        redis_client: Redis | None = None,
    ) -> None:
        """
        Initialize realtime output publisher dependencies.

        Args:
            config: Validated realtime output redis config.
            environ: Environment mapping for optional password lookup.
            hooks: Optional callbacks for publish counters.
            redis_client: Optional prebuilt Redis client for tests/wiring.
        Returns:
            None.
        Assumptions:
            Publisher is used from single live-runner process in Strategy v1.
        Raises:
            ValueError: If config is missing.
        Side Effects:
            Creates Redis client when `redis_client` is not injected.
        """
        if config is None:  # type: ignore[truthy-bool]
            raise ValueError("RedisStrategyRealtimeOutputPublisher requires config")

        self._config = config
        self._hooks = hooks if hooks is not None else RedisStrategyRealtimeOutputPublisherHooks()
        self._redis = (
            redis_client if redis_client is not None else self._build_redis_client(environ)
        )
        self._next_seq_by_stream_ts: dict[tuple[str, int], int] = {}

    def publish_records_v1(self, *, records: Sequence[StrategyRealtimeOutputRecordV1]) -> None:
        """
        Publish metric/event records to per-user Redis streams with deterministic IDs.

        Args:
            records: Realtime output records to publish.
        Returns:
            None.
        Assumptions:
            Redis failures are best-effort and never propagate to live-runner loop.
        Raises:
            None.
        Side Effects:
            Writes records into Redis streams and updates publish hooks.
        """
        if not records:
            return

        ordered_records = tuple(sorted(records, key=_record_sort_key))
        for record in ordered_records:
            stream_name = self._stream_name(record=record)
            ts_epoch_ms = _ts_epoch_ms(record=record)
            stream_id = self._stream_id(stream_name=stream_name, ts_epoch_ms=ts_epoch_ms)
            fields = self._payload(record=record)
            try:
                redis_fields = cast(dict[Any, Any], fields)
                self._redis.xadd(
                    name=stream_name,
                    fields=redis_fields,
                    id=stream_id,
                )
                _emit_counter(self._hooks.on_publish_success)
            except ResponseError as error:
                if _is_duplicate_or_out_of_order(error):
                    _emit_counter(self._hooks.on_publish_duplicate)
                    continue
                _emit_counter(self._hooks.on_publish_error)
                log.exception(
                    (
                        "strategy realtime output publish failed "
                        "stream=%s id=%s kind=%s strategy_id=%s run_id=%s"
                    ),
                    stream_name,
                    stream_id,
                    _record_kind(record=record),
                    record.strategy_id,
                    record.run_id,
                )
            except Exception:  # noqa: BLE001
                _emit_counter(self._hooks.on_publish_error)
                log.exception(
                    (
                        "strategy realtime output publish failed "
                        "stream=%s id=%s kind=%s strategy_id=%s run_id=%s"
                    ),
                    stream_name,
                    stream_id,
                    _record_kind(record=record),
                    record.strategy_id,
                    record.run_id,
                )

    def _stream_id(self, *, stream_name: str, ts_epoch_ms: int) -> str:
        """
        Build deterministic stream id `<ts_epoch_ms>-<seq>` per `(stream, ts_epoch_ms)`.

        Args:
            stream_name: Target redis stream name.
            ts_epoch_ms: Event timestamp in epoch milliseconds.
        Returns:
            str: Deterministic Redis stream id.
        Assumptions:
            Sequence starts from `0` and is monotonic per `(stream, ts_epoch_ms)` key.
        Raises:
            None.
        Side Effects:
            Increments internal sequence cursor map.
        """
        key = (stream_name, ts_epoch_ms)
        seq = self._next_seq_by_stream_ts.get(key, 0)
        self._next_seq_by_stream_ts[key] = seq + 1
        return f"{ts_epoch_ms}-{seq}"

    def _stream_name(self, *, record: StrategyRealtimeOutputRecordV1) -> str:
        """
        Build per-user stream name for metric/event record.

        Args:
            record: Realtime output record.
        Returns:
            str: Per-user stream name.
        Assumptions:
            Prefixes are `strategy.metrics.v1.user` and `strategy.events.v1.user` in v1.
        Raises:
            ValueError: If record kind is unsupported.
        Side Effects:
            None.
        """
        user_suffix = str(record.user_id)
        if isinstance(record, StrategyRealtimeMetricV1):
            return f"{self._config.metrics_stream_prefix}.{user_suffix}"
        if isinstance(record, StrategyRealtimeEventV1):
            return f"{self._config.events_stream_prefix}.{user_suffix}"
        raise ValueError("Unsupported realtime record kind")

    def _payload(self, *, record: StrategyRealtimeOutputRecordV1) -> dict[str, str]:
        """
        Convert realtime record into Redis wire payload with string-only values.

        Args:
            record: Realtime output record.
        Returns:
            dict[str, str]: Redis payload mapping.
        Assumptions:
            Schema version is fixed to `1`.
        Raises:
            ValueError: If record kind is unsupported.
        Side Effects:
            None.
        """
        base_fields = {
            "schema_version": SCHEMA_VERSION_V1,
            "ts": str(UtcTimestamp(record.ts)),
            "strategy_id": str(record.strategy_id),
            "run_id": str(record.run_id),
            "instrument_key": record.instrument_key,
            "timeframe": record.timeframe,
        }
        if isinstance(record, StrategyRealtimeMetricV1):
            payload = {
                **base_fields,
                "metric_type": record.metric_type,
                "value": record.value,
            }
            _assert_required_keys(
                payload=payload,
                required=(
                    "schema_version",
                    "ts",
                    "strategy_id",
                    "run_id",
                    "metric_type",
                    "value",
                    "instrument_key",
                    "timeframe",
                ),
            )
            return payload

        if isinstance(record, StrategyRealtimeEventV1):
            payload = {
                **base_fields,
                "event_type": record.event_type,
                "payload_json": record.payload_json,
            }
            _assert_required_keys(
                payload=payload,
                required=(
                    "schema_version",
                    "ts",
                    "strategy_id",
                    "run_id",
                    "event_type",
                    "payload_json",
                    "instrument_key",
                    "timeframe",
                ),
            )
            return payload

        raise ValueError("Unsupported realtime record kind")

    def _build_redis_client(self, environ: Mapping[str, str]) -> Redis:
        """
        Build Redis client from config and environment mapping.

        Args:
            environ: Environment mapping for optional password lookup.
        Returns:
            Redis: Configured redis-py client.
        Assumptions:
            Missing/blank password env means no password.
        Raises:
            None.
        Side Effects:
            Allocates Redis client.
        """
        password = _resolve_password(environ=environ, password_env=self._config.password_env)
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
        password_env: Password variable name.
    Returns:
        str | None: Password value or `None`.
    Assumptions:
        Empty environment value is treated as absent password.
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


def _record_sort_key(record: StrategyRealtimeOutputRecordV1) -> tuple[Any, ...]:
    """
    Build deterministic record sorting key for sequence assignment.

    Args:
        record: Realtime output record.
    Returns:
        tuple[Any, ...]: Sort key preserving deterministic order.
    Assumptions:
        Sort key matches architecture contract ordering dimensions for v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    kind = _record_kind(record=record)
    type_key = _record_type_key(record=record)
    return (
        _ts_epoch_ms(record=record),
        kind,
        str(record.strategy_id),
        str(record.run_id),
        record.instrument_key,
        record.timeframe,
        type_key,
        str(record.user_id),
    )


def _record_kind(*, record: StrategyRealtimeOutputRecordV1) -> str:
    """
    Resolve record kind literal used for deterministic sorting and logs.

    Args:
        record: Realtime output record.
    Returns:
        str: `metric` or `event`.
    Assumptions:
        Runtime values are valid dataclass instances from v1 port module.
    Raises:
        ValueError: If record class is unsupported.
    Side Effects:
        None.
    """
    if isinstance(record, StrategyRealtimeMetricV1):
        return "metric"
    if isinstance(record, StrategyRealtimeEventV1):
        return "event"
    raise ValueError("Unsupported realtime record kind")


def _record_type_key(*, record: StrategyRealtimeOutputRecordV1) -> str:
    """
    Resolve metric/event type string from record.

    Args:
        record: Realtime output record.
    Returns:
        str: `metric_type` or `event_type` value.
    Assumptions:
        Record payload is pre-validated by dataclass invariants.
    Raises:
        ValueError: If record class is unsupported.
    Side Effects:
        None.
    """
    if isinstance(record, StrategyRealtimeMetricV1):
        return record.metric_type
    if isinstance(record, StrategyRealtimeEventV1):
        return record.event_type
    raise ValueError("Unsupported realtime record kind")


def _ts_epoch_ms(*, record: StrategyRealtimeOutputRecordV1) -> int:
    """
    Convert record timestamp to epoch milliseconds for deterministic stream IDs.

    Args:
        record: Realtime output record.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Record timestamp is timezone-aware and normalized by `UtcTimestamp`.
    Raises:
        ValueError: If timestamp cannot be normalized.
    Side Effects:
        None.
    """
    return int(UtcTimestamp(record.ts).value.timestamp() * 1000)


def _assert_required_keys(*, payload: Mapping[str, str], required: tuple[str, ...]) -> None:
    """
    Validate required payload fields and string-only values.

    Args:
        payload: Generated payload mapping.
        required: Required field names.
    Returns:
        None.
    Assumptions:
        Payload is generated by deterministic adapter conversion.
    Raises:
        ValueError: If required keys are missing or non-string value is found.
    Side Effects:
        None.
    """
    for key in required:
        if key not in payload:
            raise ValueError(f"Realtime payload misses required key: {key}")
    for key, value in payload.items():
        if type(value) is not str:
            raise ValueError(f"Realtime payload value at key {key!r} must be string")


def _is_duplicate_or_out_of_order(exc: ResponseError) -> bool:
    """
    Check whether Redis response error indicates duplicate/out-of-order stream ID.

    Args:
        exc: Redis response error from `xadd`.
    Returns:
        bool: `True` when error should be counted as duplicate, not as hard failure.
    Assumptions:
        Redis uses textual diagnostics for WRONGID/equal-or-smaller stream id failures.
    Raises:
        None.
    Side Effects:
        None.
    """
    message = str(exc).lower()
    return "equal or smaller than the target stream top item" in message or "wrongid" in message


def _emit_counter(callback: Callable[[], None] | None) -> None:
    """
    Execute optional counter callback.

    Args:
        callback: Counter callback.
    Returns:
        None.
    Assumptions:
        Callback has no arguments.
    Raises:
        None.
    Side Effects:
        Executes callback when provided.
    """
    if callback is not None:
        callback()
