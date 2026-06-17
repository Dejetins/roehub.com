from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from trading.contexts.strategy.adapters.outbound.messaging.redis import (
    RedisStrategyLiveCandleStream,
    RedisStrategyLiveCandleStreamConfig,
)


class _FakeRedis:
    """
    Fake Redis client for strategy live candle stream adapter tests.
    """

    def __init__(self, *, read_batches: list[Any]) -> None:
        """
        Initialize fake client with queued read responses.

        Args:
            read_batches: Queued `xreadgroup` return values.
        Returns:
            None.
        Assumptions:
            Tests provide deterministic Redis batch payload shape.
        Raises:
            None.
        Side Effects:
            Stores mutable call logs and response queue.
        """
        self._read_batches = list(read_batches)
        self.group_calls: list[dict[str, Any]] = []
        self.ack_calls: list[tuple[str, str, str]] = []

    def xgroup_create(self, *, name: str, groupname: str, id: str, mkstream: bool) -> None:
        """
        Record consumer-group creation calls.

        Args:
            name: Stream name.
            groupname: Consumer group name.
            id: Start id.
            mkstream: Whether stream creation is enabled.
        Returns:
            None.
        Assumptions:
            Test verifies deterministic stream/group naming.
        Raises:
            None.
        Side Effects:
            Appends call payload to `group_calls`.
        """
        self.group_calls.append(
            {
                "name": name,
                "groupname": groupname,
                "id": id,
                "mkstream": mkstream,
            }
        )

    def xreadgroup(
        self,
        *,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int,
        block: int,
    ) -> Any:
        """
        Return next queued read batch.

        Args:
            groupname: Consumer group name.
            consumername: Consumer name.
            streams: Stream map.
            count: Batch size.
            block: Block timeout.
        Returns:
            Any: Queued Redis read batch payload.
        Assumptions:
            Adapter tests validate parsing logic and ack behavior.
        Raises:
            AssertionError: If queue is exhausted unexpectedly.
        Side Effects:
            Pops one queued read batch.
        """
        _ = groupname
        _ = consumername
        _ = streams
        _ = count
        _ = block
        if not self._read_batches:
            raise AssertionError("xreadgroup queue is exhausted")
        return self._read_batches.pop(0)

    def xack(self, stream: str, group: str, message_id: str) -> None:
        """
        Record ack calls for assertions.

        Args:
            stream: Stream name.
            group: Consumer group.
            message_id: Message id.
        Returns:
            None.
        Assumptions:
            Ack call ordering is deterministic in tests.
        Raises:
            None.
        Side Effects:
            Appends call tuple to `ack_calls`.
        """
        self.ack_calls.append((stream, group, message_id))


def test_redis_strategy_live_candle_stream_reads_and_parses_payload() -> None:
    """
    Ensure adapter parses one valid Redis payload into `StrategyLiveCandleMessage`.
    """
    instrument_key = "binance:spot:BTCUSDT"
    stream_name = f"md.candles.1m.{instrument_key}"
    payload = _build_valid_payload()
    redis_client = _FakeRedis(read_batches=[[(stream_name, [("1000-0", payload)])]])
    adapter = RedisStrategyLiveCandleStream(
        config=_build_config(),
        environ={},
        redis_client=redis_client,  # type: ignore[arg-type]
    )

    rows = adapter.read_closed_1m(instrument_key=instrument_key)

    assert len(rows) == 1
    assert rows[0].message_id == "1000-0"
    assert rows[0].candle.candle.ts_open.value == datetime(2026, 2, 17, 12, 0, tzinfo=timezone.utc)
    assert rows[0].candle.meta.instrument_key == instrument_key
    assert redis_client.group_calls[0]["name"] == stream_name

    adapter.ack(instrument_key=instrument_key, message_id="1000-0")
    assert redis_client.ack_calls == [(stream_name, "strategy.live_runner.v1", "1000-0")]


def test_redis_strategy_live_candle_stream_drops_invalid_payload_and_acks() -> None:
    """
    Ensure invalid payload is dropped and acknowledged to avoid poison-message loops.
    """
    instrument_key = "binance:spot:BTCUSDT"
    stream_name = f"md.candles.1m.{instrument_key}"
    invalid_payload = _build_valid_payload()
    invalid_payload.pop("open")
    redis_client = _FakeRedis(read_batches=[[(stream_name, [("1001-0", invalid_payload)])]])
    adapter = RedisStrategyLiveCandleStream(
        config=_build_config(),
        environ={},
        redis_client=redis_client,  # type: ignore[arg-type]
    )

    rows = adapter.read_closed_1m(instrument_key=instrument_key)

    assert rows == ()
    assert redis_client.ack_calls == [(stream_name, "strategy.live_runner.v1", "1001-0")]


def _build_config() -> RedisStrategyLiveCandleStreamConfig:
    """
    Build deterministic Redis consumer config fixture.

    Args:
        None.
    Returns:
        RedisStrategyLiveCandleStreamConfig: Valid runtime config fixture.
    Assumptions:
        Consumer group literal for v1 is `strategy.live_runner.v1`.
    Raises:
        ValueError: If config literals violate dataclass invariants.
    Side Effects:
        None.
    """
    return RedisStrategyLiveCandleStreamConfig(
        host="redis",
        port=6379,
        db=0,
        password_env=None,
        socket_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_prefix="md.candles.1m",
        consumer_group="strategy.live_runner.v1",
        consumer_name="host-1",
        read_count=100,
        block_ms=100,
    )


def _build_valid_payload() -> dict[str, str]:
    """
    Build deterministic valid Redis payload fixture for schema version 1.

    Args:
        None.
    Returns:
        dict[str, str]: Valid payload mapping.
    Assumptions:
        Payload matches market-data live feed contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "schema_version": "1",
        "market_id": "1",
        "symbol": "BTCUSDT",
        "instrument_key": "binance:spot:BTCUSDT",
        "ts_open": "2026-02-17T12:00:00.000Z",
        "ts_close": "2026-02-17T12:01:00.000Z",
        "open": "100.0",
        "high": "101.0",
        "low": "99.0",
        "close": "100.5",
        "volume_base": "10.0",
        "volume_quote": "1005.0",
        "source": "ws",
        "ingested_at": "2026-02-17T12:01:00.000Z",
        "ingest_id": str(UUID("00000000-0000-0000-0000-00000000ABCD")),
    }
