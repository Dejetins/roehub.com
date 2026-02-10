from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from redis.exceptions import ResponseError

from trading.contexts.market_data.adapters.outbound.config.runtime_config import RedisStreamsConfig
from trading.contexts.market_data.adapters.outbound.messaging.redis import (
    RedisLiveCandlePublisherHooks,
    RedisStreamsLiveCandlePublisher,
)
from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)


class _RuntimeConfigStub:
    """Runtime config stub guarding against unexpected instrument-key fallback calls."""

    def market_by_id(self, market_id) -> None:
        """
        Fail fast when fallback path is unexpectedly used.

        Parameters:
        - market_id: market identifier lookup argument.

        Returns:
        - None.
        """
        _ = market_id
        raise AssertionError("runtime config fallback must not be called in this test")


class _FakeRedis:
    """Fake Redis client recording `xadd` calls and optionally raising errors."""

    def __init__(self, error: Exception | None = None) -> None:
        """
        Initialize fake client state.

        Parameters:
        - error: optional exception raised by every `xadd` call.

        Returns:
        - None.
        """
        self._error = error
        self.calls: list[dict[str, object]] = []

    def xadd(
        self,
        *,
        name: str,
        fields: dict[str, str],
        id: str,
        maxlen: int | None,
        approximate: bool,
    ) -> str:
        """
        Record call arguments and optionally raise configured error.

        Parameters:
        - name: stream name.
        - fields: payload fields.
        - id: stream entry id.
        - maxlen: max length trimming setting.
        - approximate: approximate trimming mode flag.

        Returns:
        - Entry ID string when no error is configured.
        """
        self.calls.append(
            {
                "name": name,
                "fields": fields,
                "id": id,
                "maxlen": maxlen,
                "approximate": approximate,
            }
        )
        if self._error is not None:
            raise self._error
        return id


class _HooksProbe:
    """Hooks recorder for publish success/error/duplicate/duration callbacks."""

    def __init__(self) -> None:
        """
        Initialize zeroed counters and duration capture storage.

        Parameters:
        - None.

        Returns:
        - None.
        """
        self.success_count = 0
        self.error_count = 0
        self.duplicate_count = 0
        self.duration_values: list[float] = []

    def on_success(self) -> None:
        """
        Record one success callback invocation.

        Parameters:
        - None.

        Returns:
        - None.
        """
        self.success_count += 1

    def on_error(self) -> None:
        """
        Record one error callback invocation.

        Parameters:
        - None.

        Returns:
        - None.
        """
        self.error_count += 1

    def on_duplicate(self) -> None:
        """
        Record one duplicate callback invocation.

        Parameters:
        - None.

        Returns:
        - None.
        """
        self.duplicate_count += 1

    def on_duration(self, seconds: float) -> None:
        """
        Record one duration observation.

        Parameters:
        - seconds: publish call duration.

        Returns:
        - None.
        """
        self.duration_values.append(seconds)


def _publisher(
    fake_redis: _FakeRedis,
    hooks_probe: _HooksProbe,
    *,
    process_ingest_id: UUID,
) -> RedisStreamsLiveCandlePublisher:
    """
    Build Redis streams publisher with test doubles.

    Parameters:
    - fake_redis: fake redis client implementation.
    - hooks_probe: callback recorder.
    - process_ingest_id: process-level ingest id fallback for payload.

    Returns:
    - Configured `RedisStreamsLiveCandlePublisher`.
    """
    config = RedisStreamsConfig(
        enabled=True,
        host="redis",
        port=6379,
        db=0,
        password_env="ROEHUB_REDIS_PASSWORD",
        socket_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_mode="per_instrument",
        stream_prefix="md.candles.1m",
        retention_days=7,
        maxlen_approx=999,
    )
    hooks = RedisLiveCandlePublisherHooks(
        on_publish_success=hooks_probe.on_success,
        on_publish_error=hooks_probe.on_error,
        on_publish_duplicate=hooks_probe.on_duplicate,
        on_publish_duration=hooks_probe.on_duration,
    )
    return RedisStreamsLiveCandlePublisher(
        config=config,
        runtime_config=_RuntimeConfigStub(),  # type: ignore[arg-type]
        process_ingest_id=process_ingest_id,
        environ={},
        hooks=hooks,
        redis_client=fake_redis,  # type: ignore[arg-type]
    )


def _row(
    *,
    ts_open: datetime,
    ingested_at: datetime,
    meta_ingest_id: UUID | None,
    volume_quote: float | None,
) -> CandleWithMeta:
    """
    Build deterministic closed WS candle row for publisher tests.

    Parameters:
    - ts_open: candle open timestamp in UTC.
    - ingested_at: candle metadata ingestion timestamp.
    - meta_ingest_id: optional ingest id stored in candle metadata.
    - volume_quote: quote volume value or `None`.

    Returns:
    - CandleWithMeta instance.
    """
    instrument_id = InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))
    candle = Candle(
        instrument_id=instrument_id,
        ts_open=UtcTimestamp(ts_open),
        ts_close=UtcTimestamp(ts_open + timedelta(minutes=1)),
        open=100.1,
        high=101.2,
        low=99.9,
        close=100.8,
        volume_base=12.34,
        volume_quote=volume_quote,
    )
    meta = CandleMeta(
        source="ws",
        ingested_at=UtcTimestamp(ingested_at),
        ingest_id=meta_ingest_id,
        instrument_key="binance:spot:BTCUSDT",
        trades_count=7,
        taker_buy_volume_base=1.1,
        taker_buy_volume_quote=110.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def test_redis_publisher_maps_stream_payload_and_deterministic_id() -> None:
    """
    Ensure publisher writes expected stream name, payload fields, and deterministic ID.
    """
    fake_redis = _FakeRedis()
    hooks_probe = _HooksProbe()
    process_ingest_id = UUID("00000000-0000-0000-0000-000000000099")
    publisher = _publisher(fake_redis, hooks_probe, process_ingest_id=process_ingest_id)
    ts_open = datetime(2026, 2, 10, 12, 34, tzinfo=timezone.utc)
    row = _row(
        ts_open=ts_open,
        ingested_at=datetime(2026, 2, 10, 12, 35, 0, 120000, tzinfo=timezone.utc),
        meta_ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        volume_quote=1234.5,
    )

    publisher.publish_1m_closed(row)

    assert len(fake_redis.calls) == 1
    call = fake_redis.calls[0]
    expected_id = f"{int(ts_open.timestamp() * 1000)}-0"
    assert call["name"] == "md.candles.1m.binance:spot:BTCUSDT"
    assert call["id"] == expected_id
    assert call["maxlen"] == 999
    assert call["approximate"] is True

    fields = call["fields"]
    assert isinstance(fields, dict)
    assert fields["schema_version"] == "1"
    assert fields["market_id"] == "1"
    assert fields["symbol"] == "BTCUSDT"
    assert fields["instrument_key"] == "binance:spot:BTCUSDT"
    assert fields["ts_open"] == "2026-02-10T12:34:00.000Z"
    assert fields["ts_close"] == "2026-02-10T12:35:00.000Z"
    assert fields["open"] == "100.1"
    assert fields["high"] == "101.2"
    assert fields["low"] == "99.9"
    assert fields["close"] == "100.8"
    assert fields["volume_base"] == "12.34"
    assert fields["volume_quote"] == "1234.5"
    assert fields["source"] == "ws"
    assert fields["ingested_at"] == "2026-02-10T12:35:00.120Z"
    assert fields["ingest_id"] == "00000000-0000-0000-0000-000000000001"

    assert hooks_probe.success_count == 1
    assert hooks_probe.error_count == 0
    assert hooks_probe.duplicate_count == 0
    assert len(hooks_probe.duration_values) == 1
    assert hooks_probe.duration_values[0] >= 0.0


def test_redis_publisher_uses_process_ingest_id_when_meta_ingest_id_is_missing() -> None:
    """
    Ensure payload uses process-level ingest id fallback when metadata value is absent.
    """
    fake_redis = _FakeRedis()
    hooks_probe = _HooksProbe()
    process_ingest_id = UUID("00000000-0000-0000-0000-000000000777")
    publisher = _publisher(fake_redis, hooks_probe, process_ingest_id=process_ingest_id)
    row = _row(
        ts_open=datetime(2026, 2, 10, 12, 40, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 2, 10, 12, 41, tzinfo=timezone.utc),
        meta_ingest_id=None,
        volume_quote=None,
    )

    publisher.publish_1m_closed(row)

    fields = fake_redis.calls[0]["fields"]
    assert isinstance(fields, dict)
    assert fields["ingest_id"] == "00000000-0000-0000-0000-000000000777"
    assert fields["volume_quote"] == ""


def test_redis_publisher_treats_duplicate_id_error_as_noop() -> None:
    """
    Ensure duplicate/out-of-order XADD errors are counted and do not raise exceptions.
    """
    fake_redis = _FakeRedis(
        error=ResponseError(
            "The ID specified in XADD is equal or smaller than the target stream top item"
        )
    )
    hooks_probe = _HooksProbe()
    publisher = _publisher(
        fake_redis,
        hooks_probe,
        process_ingest_id=UUID("00000000-0000-0000-0000-000000000111"),
    )
    row = _row(
        ts_open=datetime(2026, 2, 10, 12, 50, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 2, 10, 12, 51, tzinfo=timezone.utc),
        meta_ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        volume_quote=1.0,
    )

    publisher.publish_1m_closed(row)

    assert hooks_probe.success_count == 0
    assert hooks_probe.error_count == 0
    assert hooks_probe.duplicate_count == 1
    assert len(hooks_probe.duration_values) == 1


def test_redis_publisher_counts_non_duplicate_failures() -> None:
    """
    Ensure non-duplicate redis failures increment error counter and stay best-effort.
    """
    fake_redis = _FakeRedis(error=RuntimeError("redis unavailable"))
    hooks_probe = _HooksProbe()
    publisher = _publisher(
        fake_redis,
        hooks_probe,
        process_ingest_id=UUID("00000000-0000-0000-0000-000000000222"),
    )
    row = _row(
        ts_open=datetime(2026, 2, 10, 13, 0, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 2, 10, 13, 1, tzinfo=timezone.utc),
        meta_ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        volume_quote=2.0,
    )

    publisher.publish_1m_closed(row)

    assert hooks_probe.success_count == 0
    assert hooks_probe.error_count == 1
    assert hooks_probe.duplicate_count == 0
    assert len(hooks_probe.duration_values) == 1
