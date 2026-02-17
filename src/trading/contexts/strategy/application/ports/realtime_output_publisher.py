from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Protocol, Sequence
from uuid import UUID

from trading.shared_kernel.primitives import UserId, UtcTimestamp

MetricTypeV1 = Literal[
    "warmup_processed_bars",
    "checkpoint_ts_open",
    "lag_seconds",
    "candles_processed_total",
    "rollup_bucket_closed",
    "gap_detected",
    "repair_missing_bars",
    "warmup_required_bars",
    "warmup_satisfied",
    "run_state",
    "rollup_bucket_count_1m",
    "rollup_bucket_open_ts",
    "repair_attempt",
    "repair_continuous",
    "dropped_non_contiguous_total",
]
EventTypeV1 = Literal[
    "run_state_changed",
    "run_stopped",
    "run_failed",
]
RealtimeOutputKindV1 = Literal["metric", "event"]

METRIC_TYPES_V1: tuple[str, ...] = (
    "warmup_processed_bars",
    "checkpoint_ts_open",
    "lag_seconds",
    "candles_processed_total",
    "rollup_bucket_closed",
    "gap_detected",
    "repair_missing_bars",
    "warmup_required_bars",
    "warmup_satisfied",
    "run_state",
    "rollup_bucket_count_1m",
    "rollup_bucket_open_ts",
    "repair_attempt",
    "repair_continuous",
    "dropped_non_contiguous_total",
)
EVENT_TYPES_V1: tuple[str, ...] = (
    "run_state_changed",
    "run_stopped",
    "run_failed",
)
SCHEMA_VERSION_V1 = "1"


@dataclass(frozen=True, slots=True)
class StrategyRealtimeMetricV1:
    """
    StrategyRealtimeMetricV1 — one v1 realtime metric record for per-user Redis stream publish.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_realtime_output_publisher.py
      - tests/unit/contexts/strategy/adapters/test_redis_strategy_realtime_output_publisher.py
    """

    user_id: UserId
    ts: datetime
    strategy_id: UUID
    run_id: UUID
    metric_type: MetricTypeV1
    value: str
    instrument_key: str
    timeframe: str

    def __post_init__(self) -> None:
        """
        Validate metric payload invariants and normalize timestamp precision.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Metric type list is fixed for realtime output schema version `1`.
        Raises:
            ValueError: If metric type or required string fields are invalid.
        Side Effects:
            Normalizes `ts` to UTC with millisecond precision.
        """
        if self.metric_type not in METRIC_TYPES_V1:
            raise ValueError(
                "StrategyRealtimeMetricV1.metric_type must be one of fixed v1 values"
            )
        _require_text(value=self.value, field_name="value", allow_empty=True)
        _require_text(value=self.instrument_key, field_name="instrument_key", allow_empty=False)
        _require_text(value=self.timeframe, field_name="timeframe", allow_empty=False)

        normalized_ts = UtcTimestamp(self.ts).value
        object.__setattr__(self, "ts", normalized_ts)


@dataclass(frozen=True, slots=True)
class StrategyRealtimeEventV1:
    """
    StrategyRealtimeEventV1 — one v1 realtime event record for per-user Redis stream publish.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_realtime_output_publisher.py
      - tests/unit/contexts/strategy/adapters/test_redis_strategy_realtime_output_publisher.py
    """

    user_id: UserId
    ts: datetime
    strategy_id: UUID
    run_id: UUID
    event_type: EventTypeV1
    payload_json: str
    instrument_key: str
    timeframe: str

    def __post_init__(self) -> None:
        """
        Validate event payload invariants and normalize payload JSON deterministically.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Event type list is fixed for realtime output schema version `1`.
        Raises:
            ValueError: If event type, payload JSON, or required fields are invalid.
        Side Effects:
            Normalizes `ts` to UTC and rewrites `payload_json` into canonical JSON object string.
        """
        if self.event_type not in EVENT_TYPES_V1:
            raise ValueError(
                "StrategyRealtimeEventV1.event_type must be one of fixed v1 values"
            )
        _require_text(value=self.payload_json, field_name="payload_json", allow_empty=False)
        _require_text(value=self.instrument_key, field_name="instrument_key", allow_empty=False)
        _require_text(value=self.timeframe, field_name="timeframe", allow_empty=False)

        normalized_payload = _normalize_payload_json_object(text=self.payload_json)
        normalized_ts = UtcTimestamp(self.ts).value
        object.__setattr__(self, "payload_json", normalized_payload)
        object.__setattr__(self, "ts", normalized_ts)


StrategyRealtimeOutputRecordV1 = StrategyRealtimeMetricV1 | StrategyRealtimeEventV1


class StrategyRealtimeOutputPublisher(Protocol):
    """
    StrategyRealtimeOutputPublisher — application port for Strategy realtime output v1 publish.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_realtime_output_publisher.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
    """

    def publish_records_v1(self, *, records: Sequence[StrategyRealtimeOutputRecordV1]) -> None:
        """
        Publish metric/event records using realtime output v1 schema.

        Args:
            records: Ordered record collection to publish.
        Returns:
            None.
        Assumptions:
            Implementation enforces deterministic stream IDs and best-effort semantics.
        Raises:
            Exception: Adapter implementations may raise only on configuration/runtime misuse.
        Side Effects:
            Performs outbound IO (Redis Streams, logs, metrics hooks) in concrete adapters.
        """
        ...


@dataclass(frozen=True, slots=True)
class NoOpStrategyRealtimeOutputPublisher(StrategyRealtimeOutputPublisher):
    """
    NoOpStrategyRealtimeOutputPublisher — disabled realtime output adapter for safe defaults.

    Docs:
      - docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      - tests/unit/contexts/strategy/application/test_strategy_live_runner_realtime_output.py
    """

    def publish_records_v1(self, *, records: Sequence[StrategyRealtimeOutputRecordV1]) -> None:
        """
        Ignore publish requests and keep live-runner behavior unchanged.

        Args:
            records: Realtime records ignored by no-op adapter.
        Returns:
            None.
        Assumptions:
            No-op mode is used when realtime output is disabled in runtime config.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = records


def serialize_realtime_event_payload_json(*, payload: Mapping[str, Any]) -> str:
    """
    Serialize event payload mapping into deterministic JSON object string.

    Args:
        payload: Source payload mapping.
    Returns:
        str: Canonical JSON object string with sorted keys and ASCII-safe output.
    Assumptions:
        Payload keys/values are JSON-compatible and converted to strings for stable UI contract.
    Raises:
        ValueError: If payload cannot be represented as JSON object.
    Side Effects:
        None.
    """
    normalized: dict[str, str] = {}
    for key in sorted(payload, key=lambda row: str(row)):
        raw_key = str(key)
        raw_value = payload[key]
        if raw_value is None:
            normalized[raw_key] = ""
            continue
        if type(raw_value) is str:
            normalized[raw_key] = raw_value
            continue
        normalized[raw_key] = str(raw_value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_payload_json_object(*, text: str) -> str:
    """
    Validate and canonicalize payload JSON string into deterministic object JSON.

    Args:
        text: Raw JSON object string.
    Returns:
        str: Canonical JSON object string with sorted keys and ASCII-safe output.
    Assumptions:
        Only JSON object payloads are allowed for realtime v1 events.
    Raises:
        ValueError: If input is not valid JSON object.
    Side Effects:
        None.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("payload_json must be valid JSON object string") from error
    if type(parsed) is not dict:
        raise ValueError("payload_json must encode JSON object")
    return serialize_realtime_event_payload_json(payload=parsed)


def _require_text(*, value: Any, field_name: str, allow_empty: bool) -> None:
    """
    Validate strict string value contract for realtime record fields.

    Args:
        value: Candidate value.
        field_name: Field name for deterministic error text.
        allow_empty: Whether empty string value is accepted.
    Returns:
        None.
    Assumptions:
        Realtime wire payload schema requires string values only.
    Raises:
        ValueError: If value is non-string or empty when disallowed.
    Side Effects:
        None.
    """
    if type(value) is not str:
        raise ValueError(f"{field_name} must be string")
    if allow_empty:
        return
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty string")
