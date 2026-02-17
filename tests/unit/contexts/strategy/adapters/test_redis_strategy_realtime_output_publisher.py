from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from uuid import UUID

import pytest
from redis.exceptions import ResponseError

from trading.contexts.strategy.adapters.outbound.messaging.redis import (
    RedisStrategyRealtimeOutputPublisher,
    RedisStrategyRealtimeOutputPublisherConfig,
    RedisStrategyRealtimeOutputPublisherHooks,
)
from trading.contexts.strategy.application.ports import (
    StrategyRealtimeEventV1,
    StrategyRealtimeMetricV1,
)
from trading.shared_kernel.primitives import UserId


class _FakeRedis:
    """
    Fake Redis client recording `xadd` calls and optionally raising configured error.
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        """
        Initialize fake redis client state.

        Args:
            error: Optional exception raised by every `xadd` call.
        Returns:
            None.
        Assumptions:
            Tests inspect captured call arguments for deterministic behavior.
        Raises:
            None.
        Side Effects:
            Stores mutable call log.
        """
        self._error = error
        self.calls: list[dict[str, object]] = []

    def xadd(self, *, name: str, fields: dict[str, str], id: str) -> str:
        """
        Record one `xadd` call and optionally raise configured error.

        Args:
            name: Stream name.
            fields: Payload mapping.
            id: Stream entry id.
        Returns:
            str: Stream id when no error is configured.
        Assumptions:
            Publisher always sends string-only payload values.
        Raises:
            Exception: Configured fake exception.
        Side Effects:
            Appends call payload to internal log.
        """
        self.calls.append({"name": name, "fields": fields, "id": id})
        if self._error is not None:
            raise self._error
        return id


class _HooksProbe:
    """
    Probe callbacks for publish success/error/duplicate hooks.
    """

    def __init__(self) -> None:
        """
        Initialize zeroed callback counters.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Each publish attempt triggers at most one counter callback.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.success_count = 0
        self.error_count = 0
        self.duplicate_count = 0

    def on_success(self) -> None:
        """
        Increase successful publish callback counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Callback is invoked by publisher on successful `xadd`.
        Raises:
            None.
        Side Effects:
            Mutates probe state.
        """
        self.success_count += 1

    def on_error(self) -> None:
        """
        Increase publish error callback counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Callback is invoked only for non-duplicate failures.
        Raises:
            None.
        Side Effects:
            Mutates probe state.
        """
        self.error_count += 1

    def on_duplicate(self) -> None:
        """
        Increase duplicate publish callback counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Callback is invoked for duplicate/out-of-order stream IDs.
        Raises:
            None.
        Side Effects:
            Mutates probe state.
        """
        self.duplicate_count += 1


def test_redis_realtime_output_publisher_assigns_deterministic_ids_per_stream_and_ts() -> None:
    """
    Ensure publisher assigns deterministic `<ts_epoch_ms>-<seq>` IDs per stream and timestamp.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001004")
    ts = datetime(2026, 2, 17, 18, 0, tzinfo=timezone.utc)
    run_id = UUID("00000000-0000-0000-0000-00000000A104")
    strategy_id = UUID("00000000-0000-0000-0000-00000000B104")
    redis_client = _FakeRedis()
    publisher = _publisher(redis_client=redis_client)

    publisher.publish_records_v1(
        records=(
            StrategyRealtimeMetricV1(
                user_id=user_id,
                ts=ts,
                strategy_id=strategy_id,
                run_id=run_id,
                metric_type="warmup_processed_bars",
                value="10",
                instrument_key="binance:spot:BTCUSDT",
                timeframe="1m",
            ),
            StrategyRealtimeEventV1(
                user_id=user_id,
                ts=ts,
                strategy_id=strategy_id,
                run_id=run_id,
                event_type="run_state_changed",
                payload_json='{"to":"running","from":"warming_up"}',
                instrument_key="binance:spot:BTCUSDT",
                timeframe="1m",
            ),
            StrategyRealtimeMetricV1(
                user_id=user_id,
                ts=ts,
                strategy_id=strategy_id,
                run_id=run_id,
                metric_type="checkpoint_ts_open",
                value="2026-02-17T17:59:00.000Z",
                instrument_key="binance:spot:BTCUSDT",
                timeframe="1m",
            ),
        )
    )

    ts_epoch_ms = int(ts.timestamp() * 1000)
    metrics_calls = [
        row
        for row in redis_client.calls
        if row["name"] == f"strategy.metrics.v1.user.{user_id}"
    ]
    events_calls = [
        row for row in redis_client.calls if row["name"] == f"strategy.events.v1.user.{user_id}"
    ]

    assert [row["id"] for row in metrics_calls] == [f"{ts_epoch_ms}-0", f"{ts_epoch_ms}-1"]
    metrics_metric_types = [
        cast(dict[str, str], row["fields"])["metric_type"]
        for row in metrics_calls
    ]
    assert metrics_metric_types == ["checkpoint_ts_open", "warmup_processed_bars"]
    assert [row["id"] for row in events_calls] == [f"{ts_epoch_ms}-0"]


def test_redis_realtime_output_publisher_maps_required_string_payload_fields() -> None:
    """
    Ensure publisher maps payload schemas and keeps all values as strings.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001005")
    ts = datetime(2026, 2, 17, 18, 1, tzinfo=timezone.utc)
    redis_client = _FakeRedis()
    publisher = _publisher(redis_client=redis_client)

    publisher.publish_records_v1(
        records=(
            StrategyRealtimeMetricV1(
                user_id=user_id,
                ts=ts,
                strategy_id=UUID("00000000-0000-0000-0000-00000000B105"),
                run_id=UUID("00000000-0000-0000-0000-00000000A105"),
                metric_type="lag_seconds",
                value="15",
                instrument_key="binance:spot:BTCUSDT",
                timeframe="1m",
            ),
            StrategyRealtimeEventV1(
                user_id=user_id,
                ts=ts,
                strategy_id=UUID("00000000-0000-0000-0000-00000000B105"),
                run_id=UUID("00000000-0000-0000-0000-00000000A105"),
                event_type="run_failed",
                payload_json='{"error":"boom"}',
                instrument_key="binance:spot:BTCUSDT",
                timeframe="1m",
            ),
        )
    )

    payloads = [
        cast(dict[str, str], row["fields"])
        for row in redis_client.calls
    ]
    metric_fields = next(fields for fields in payloads if "metric_type" in fields)
    event_fields = next(fields for fields in payloads if "event_type" in fields)

    assert isinstance(metric_fields, dict)
    assert isinstance(event_fields, dict)
    assert sorted(metric_fields) == [
        "instrument_key",
        "metric_type",
        "run_id",
        "schema_version",
        "strategy_id",
        "timeframe",
        "ts",
        "value",
    ]
    assert sorted(event_fields) == [
        "event_type",
        "instrument_key",
        "payload_json",
        "run_id",
        "schema_version",
        "strategy_id",
        "timeframe",
        "ts",
    ]
    assert all(type(value) is str for value in metric_fields.values())
    assert all(type(value) is str for value in event_fields.values())


def test_redis_realtime_output_publisher_counts_duplicate_response_error() -> None:
    """
    Ensure duplicate Redis response errors are counted via duplicate hook and do not raise.
    """
    hooks_probe = _HooksProbe()
    redis_client = _FakeRedis(
        error=ResponseError(
            "The ID specified in XADD is equal or smaller than the target stream top item"
        )
    )
    publisher = _publisher(redis_client=redis_client, hooks_probe=hooks_probe)

    publisher.publish_records_v1(records=(_metric_record(),))

    assert hooks_probe.success_count == 0
    assert hooks_probe.error_count == 0
    assert hooks_probe.duplicate_count == 1


def test_redis_realtime_output_publisher_counts_non_duplicate_errors() -> None:
    """
    Ensure non-duplicate publish errors are counted via error hook and remain best-effort.
    """
    hooks_probe = _HooksProbe()
    redis_client = _FakeRedis(error=RuntimeError("redis unavailable"))
    publisher = _publisher(redis_client=redis_client, hooks_probe=hooks_probe)

    publisher.publish_records_v1(records=(_metric_record(),))

    assert hooks_probe.success_count == 0
    assert hooks_probe.error_count == 1
    assert hooks_probe.duplicate_count == 0


def test_realtime_output_port_validates_fixed_metric_and_event_enums() -> None:
    """
    Ensure metric_type and event_type are validated against fixed v1 enumerations.
    """
    user_id = UserId.from_string("00000000-0000-0000-0000-000000001006")
    ts = datetime(2026, 2, 17, 18, 2, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        StrategyRealtimeMetricV1(
            user_id=user_id,
            ts=ts,
            strategy_id=UUID("00000000-0000-0000-0000-00000000B106"),
            run_id=UUID("00000000-0000-0000-0000-00000000A106"),
            metric_type="unknown_metric",  # type: ignore[arg-type]
            value="1",
            instrument_key="binance:spot:BTCUSDT",
            timeframe="1m",
        )

    with pytest.raises(ValueError):
        StrategyRealtimeEventV1(
            user_id=user_id,
            ts=ts,
            strategy_id=UUID("00000000-0000-0000-0000-00000000B106"),
            run_id=UUID("00000000-0000-0000-0000-00000000A106"),
            event_type="unknown_event",  # type: ignore[arg-type]
            payload_json='{"error":"boom"}',
            instrument_key="binance:spot:BTCUSDT",
            timeframe="1m",
        )


def test_realtime_output_port_rejects_non_object_event_payload_json() -> None:
    """
    Ensure event payload_json accepts only JSON object strings.
    """
    with pytest.raises(ValueError):
        StrategyRealtimeEventV1(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000001008"),
            ts=datetime(2026, 2, 17, 18, 4, tzinfo=timezone.utc),
            strategy_id=UUID("00000000-0000-0000-0000-00000000B108"),
            run_id=UUID("00000000-0000-0000-0000-00000000A108"),
            event_type="run_stopped",
            payload_json="[]",
            instrument_key="binance:spot:BTCUSDT",
            timeframe="1m",
        )


def _publisher(
    redis_client: _FakeRedis,
    hooks_probe: _HooksProbe | None = None,
) -> RedisStrategyRealtimeOutputPublisher:
    """
    Build realtime output publisher with deterministic test doubles.

    Args:
        redis_client: Fake redis client.
        hooks_probe: Optional callbacks probe.
    Returns:
        RedisStrategyRealtimeOutputPublisher: Publisher bound to fake dependencies.
    Assumptions:
        Config stream prefixes match v1 architecture contract.
    Raises:
        ValueError: If config violates runtime invariants.
    Side Effects:
        None.
    """
    hooks = RedisStrategyRealtimeOutputPublisherHooks()
    if hooks_probe is not None:
        hooks = RedisStrategyRealtimeOutputPublisherHooks(
            on_publish_success=hooks_probe.on_success,
            on_publish_error=hooks_probe.on_error,
            on_publish_duplicate=hooks_probe.on_duplicate,
        )

    return RedisStrategyRealtimeOutputPublisher(
        config=RedisStrategyRealtimeOutputPublisherConfig(
            host="redis",
            port=6379,
            db=0,
            password_env=None,
            socket_timeout_s=2.0,
            connect_timeout_s=2.0,
            metrics_stream_prefix="strategy.metrics.v1.user",
            events_stream_prefix="strategy.events.v1.user",
        ),
        environ={},
        hooks=hooks,
        redis_client=redis_client,  # type: ignore[arg-type]
    )


def _metric_record() -> StrategyRealtimeMetricV1:
    """
    Build deterministic metric record fixture for error-path tests.

    Args:
        None.
    Returns:
        StrategyRealtimeMetricV1: Metric record fixture.
    Assumptions:
        Fixture values satisfy realtime metric v1 schema.
    Raises:
        ValueError: If fixture violates record invariants.
    Side Effects:
        None.
    """
    return StrategyRealtimeMetricV1(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000001007"),
        ts=datetime(2026, 2, 17, 18, 3, tzinfo=timezone.utc),
        strategy_id=UUID("00000000-0000-0000-0000-00000000B107"),
        run_id=UUID("00000000-0000-0000-0000-00000000A107"),
        metric_type="lag_seconds",
        value="42",
        instrument_key="binance:spot:BTCUSDT",
        timeframe="1m",
    )
