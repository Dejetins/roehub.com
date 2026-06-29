from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    RedisHotCacheConfig,
    RedisStreamsConfig,
)
from trading.contexts.market_data.adapters.outbound.messaging.redis import (
    FanoutLiveCandlePublisher,
    RedisCandleHotCache,
    RedisCandleHotCacheHooks,
    RedisHotCacheLiveCandlePublisher,
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


class _FakeRedis:
    """Minimal Redis hash/zset fake for hot-cache unit tests."""

    def __init__(self, *, fail_writes: bool = False) -> None:
        self._fail_writes = fail_writes
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, int]] = {}

    def hset(self, name: str, key: str, value: str) -> int:
        if self._fail_writes:
            raise RuntimeError("redis unavailable")
        is_new = key not in self.hashes.setdefault(name, {})
        self.hashes[name][key] = value
        return 1 if is_new else 0

    def zadd(self, name: str, mapping: dict[str, int]) -> int:
        if self._fail_writes:
            raise RuntimeError("redis unavailable")
        zset = self.zsets.setdefault(name, {})
        added = 0
        for member, score in mapping.items():
            if member not in zset:
                added += 1
            zset[member] = score
        return added

    def zrangebyscore(self, name: str, min, max) -> list[str]:  # noqa: ANN001, A002
        zset = self.zsets.get(name, {})
        min_value = float("-inf") if min == "-inf" else int(min)
        if isinstance(max, str) and max.startswith("("):
            max_value = int(max[1:])
            inclusive_max = False
        else:
            max_value = int(max)
            inclusive_max = True

        rows = []
        for member, score in zset.items():
            max_matches = score <= max_value if inclusive_max else score < max_value
            if score >= min_value and max_matches:
                rows.append((score, member))
        return [member for _, member in sorted(rows)]

    def hmget(self, name: str, keys: tuple[str, ...]) -> list[str | None]:
        hash_values = self.hashes.get(name, {})
        return [hash_values.get(key) for key in keys]

    def zremrangebyscore(self, name: str, min, max) -> int:  # noqa: ANN001, A002
        members = self.zrangebyscore(name, min=min, max=max)
        zset = self.zsets.get(name, {})
        for member in members:
            zset.pop(member, None)
        return len(members)

    def hdel(self, name: str, *keys: str) -> int:
        hash_values = self.hashes.get(name, {})
        deleted = 0
        for key in keys:
            if key in hash_values:
                deleted += 1
                hash_values.pop(key)
        return deleted


class _HooksProbe:
    """Hook recorder for Redis hot-cache metric callbacks."""

    def __init__(self) -> None:
        self.write_success_count = 0
        self.write_error_count = 0
        self.read_hit_count = 0
        self.read_miss_count = 0
        self.read_error_count = 0
        self.write_durations: list[float] = []
        self.read_durations: list[float] = []

    def on_write_success(self) -> None:
        self.write_success_count += 1

    def on_write_error(self) -> None:
        self.write_error_count += 1

    def on_read_hit(self) -> None:
        self.read_hit_count += 1

    def on_read_miss(self) -> None:
        self.read_miss_count += 1

    def on_read_error(self) -> None:
        self.read_error_count += 1

    def on_write_duration(self, seconds: float) -> None:
        self.write_durations.append(seconds)

    def on_read_duration(self, seconds: float) -> None:
        self.read_durations.append(seconds)


class _PublisherProbe:
    """Live publisher recorder for fan-out behavior."""

    def __init__(self) -> None:
        self.calls: list[CandleWithMeta] = []

    def publish_1m_closed(self, candle: CandleWithMeta) -> None:
        self.calls.append(candle)


def _connection_config() -> RedisStreamsConfig:
    return RedisStreamsConfig(
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


def _hot_cache(
    fake_redis: _FakeRedis,
    hooks_probe: _HooksProbe,
    *,
    retention_hours: int = 24,
) -> RedisCandleHotCache:
    hooks = RedisCandleHotCacheHooks(
        on_write_success=hooks_probe.on_write_success,
        on_write_error=hooks_probe.on_write_error,
        on_write_duration=hooks_probe.on_write_duration,
        on_read_hit=hooks_probe.on_read_hit,
        on_read_miss=hooks_probe.on_read_miss,
        on_read_error=hooks_probe.on_read_error,
        on_read_duration=hooks_probe.on_read_duration,
    )
    return RedisCandleHotCache(
        connection_config=_connection_config(),
        config=RedisHotCacheConfig(
            enabled=True,
            key_prefix="md:hot:1m",
            retention_hours=retention_hours,
        ),
        environ={},
        hooks=hooks,
        redis_client=fake_redis,  # type: ignore[arg-type]
    )


def _row(ts_open: datetime, *, close: float = 100.8) -> CandleWithMeta:
    instrument_id = InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))
    candle = Candle(
        instrument_id=instrument_id,
        ts_open=UtcTimestamp(ts_open),
        ts_close=UtcTimestamp(ts_open + timedelta(minutes=1)),
        open=100.1,
        high=101.2,
        low=99.9,
        close=close,
        volume_base=12.34,
        volume_quote=1234.5,
    )
    meta = CandleMeta(
        source="ws",
        ingested_at=UtcTimestamp(ts_open + timedelta(minutes=1, milliseconds=120)),
        ingest_id=UUID("00000000-0000-0000-0000-000000000001"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=7,
        taker_buy_volume_base=1.1,
        taker_buy_volume_quote=110.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def test_hot_cache_writes_duplicate_without_ambiguity_and_reads_sorted_range() -> None:
    fake_redis = _FakeRedis()
    hooks_probe = _HooksProbe()
    cache = _hot_cache(fake_redis, hooks_probe)
    base = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)

    assert cache.write_closed_1m(_row(base + timedelta(minutes=2))) is True
    assert cache.write_closed_1m(_row(base)) is True
    assert cache.write_closed_1m(_row(base + timedelta(minutes=1))) is True
    assert cache.write_closed_1m(_row(base + timedelta(minutes=1))) is True

    rows = cache.read_range(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        instrument_key="binance:spot:BTCUSDT",
        start=UtcTimestamp(base),
        end=UtcTimestamp(base + timedelta(minutes=3)),
    )

    assert [str(row.ts_open) for row in rows] == [
        "2026-02-10T12:00:00.000Z",
        "2026-02-10T12:01:00.000Z",
        "2026-02-10T12:02:00.000Z",
    ]
    assert {row.source for row in rows} == {"redis_hot_cache"}
    assert rows[1].candle.candle.close == 100.8

    z_key = "md:hot:1m:binance:spot:BTCUSDT:z"
    h_key = "md:hot:1m:binance:spot:BTCUSDT:h"
    assert len(fake_redis.zsets[z_key]) == 3
    assert len(fake_redis.hashes[h_key]) == 3
    assert hooks_probe.write_success_count == 4
    assert hooks_probe.write_error_count == 0
    assert hooks_probe.read_hit_count == 1
    assert hooks_probe.read_miss_count == 0
    assert len(hooks_probe.write_durations) == 4
    assert len(hooks_probe.read_durations) == 1


def test_hot_cache_prunes_rows_older_than_retention_window() -> None:
    fake_redis = _FakeRedis()
    hooks_probe = _HooksProbe()
    cache = _hot_cache(fake_redis, hooks_probe, retention_hours=1)
    old_ts = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
    current_ts = datetime(2026, 2, 10, 11, 1, tzinfo=timezone.utc)

    assert cache.write_closed_1m(_row(old_ts)) is True
    assert cache.write_closed_1m(_row(current_ts, close=101.0)) is True

    rows = cache.read_range(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        instrument_key="binance:spot:BTCUSDT",
        start=UtcTimestamp(old_ts),
        end=UtcTimestamp(current_ts + timedelta(minutes=1)),
    )

    assert [str(row.ts_open) for row in rows] == ["2026-02-10T11:01:00.000Z"]
    assert rows[0].candle.candle.close == 101.0


def test_hot_cache_emits_miss_and_write_error_hooks() -> None:
    miss_redis = _FakeRedis()
    hooks_probe = _HooksProbe()
    cache = _hot_cache(miss_redis, hooks_probe)
    base = datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc)

    rows = cache.read_range(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        instrument_key="binance:spot:BTCUSDT",
        start=UtcTimestamp(base),
        end=UtcTimestamp(base + timedelta(minutes=1)),
    )

    assert rows == ()
    assert hooks_probe.read_miss_count == 1

    failing_redis = _FakeRedis(fail_writes=True)
    failing_hooks = _HooksProbe()
    failing_cache = _hot_cache(failing_redis, failing_hooks)

    assert failing_cache.write_closed_1m(_row(base)) is False
    assert failing_hooks.write_success_count == 0
    assert failing_hooks.write_error_count == 1
    assert len(failing_hooks.write_durations) == 1


def test_hot_cache_live_publisher_and_fanout_publishers_are_best_effort() -> None:
    fake_redis = _FakeRedis()
    hooks_probe = _HooksProbe()
    cache = _hot_cache(fake_redis, hooks_probe)
    hot_cache_publisher = RedisHotCacheLiveCandlePublisher(cache)
    second_publisher = _PublisherProbe()
    fanout = FanoutLiveCandlePublisher((hot_cache_publisher, second_publisher))
    row = _row(datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc))

    fanout.publish_1m_closed(row)

    assert hooks_probe.write_success_count == 1
    assert second_publisher.calls == [row]
