from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresCandleRepairAuditRepository,
)
from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    CanonicalCandleBatch1m,
    ClosedCandleTailRepairPolicy,
    ClosedCandleTailRow,
    MarketDataCandleRepairAuditEvent,
)
from trading.contexts.market_data.application.ports.stores import CandleRepairAuditRepository
from trading.contexts.market_data.application.services import (
    ClosedCandleTailProviderHooks,
    MarketDataClosedCandleTailProvider,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class _Clock:
    def __init__(self, now: datetime) -> None:
        self._now = UtcTimestamp(now)

    def now(self) -> UtcTimestamp:
        return self._now


class _HotCache:
    def __init__(self) -> None:
        self.rows: dict[datetime, CandleWithMeta] = {}
        self.read_calls = 0
        self.write_calls = 0

    def read_range(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        start: UtcTimestamp,
        end: UtcTimestamp,
    ) -> tuple[ClosedCandleTailRow, ...]:
        self.read_calls += 1
        rows = []
        for ts_open, row in sorted(self.rows.items()):
            if start.value <= ts_open < end.value:
                assert row.candle.instrument_id == instrument_id
                assert row.meta.instrument_key == instrument_key
                rows.append(ClosedCandleTailRow(candle=row, source="redis_hot_cache"))
        return tuple(rows)

    def write_closed_1m(self, candle: CandleWithMeta) -> bool:
        self.write_calls += 1
        self.rows[candle.candle.ts_open.value] = candle
        return True


class _LeakyHotCache(_HotCache):
    def read_range(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        start: UtcTimestamp,
        end: UtcTimestamp,
    ) -> tuple[ClosedCandleTailRow, ...]:
        self.read_calls += 1
        _ = start
        _ = end
        rows = []
        for _ts_open, row in sorted(self.rows.items()):
            assert row.candle.instrument_id == instrument_id
            assert row.meta.instrument_key == instrument_key
            rows.append(ClosedCandleTailRow(candle=row, source="redis_hot_cache"))
        return tuple(rows)


class _CanonicalReader:
    def __init__(
        self,
        *,
        rows: tuple[CandleWithMeta, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self._rows = rows
        self._error = error
        self.read_calls = 0

    def read_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ):
        self.read_calls += 1
        if self._error is not None:
            raise self._error
        for row in self._rows:
            if row.candle.instrument_id == instrument_id and time_range.contains(
                row.candle.ts_open
            ):
                yield row

    def read_1m_arrays(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> CanonicalCandleBatch1m:
        _ = instrument_id
        _ = time_range
        raise NotImplementedError


class _RestSource:
    def __init__(self, rows: tuple[CandleWithMeta, ...]) -> None:
        self._rows = rows
        self.calls: list[TimeRange] = []

    def stream_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ):
        self.calls.append(time_range)
        for row in self._rows:
            if row.candle.instrument_id == instrument_id and time_range.contains(
                row.candle.ts_open
            ):
                yield row


class _AuditRepository:
    def __init__(self) -> None:
        self.records: list[MarketDataCandleRepairAuditEvent] = []

    def record(
        self,
        *,
        event: MarketDataCandleRepairAuditEvent,
    ) -> MarketDataCandleRepairAuditEvent:
        self.records.append(event)
        return event

    def get_by_id(self, *, event_id: UUID) -> MarketDataCandleRepairAuditEvent | None:
        for event in self.records:
            if event.event_id == event_id:
                return event
        return None

    def list_for_correlation(
        self,
        *,
        correlation_id: str,
    ) -> tuple[MarketDataCandleRepairAuditEvent, ...]:
        return tuple(event for event in self.records if event.correlation_id == correlation_id)


class _PostgresAuditGateway:
    def __init__(self) -> None:
        self.rows: dict[str, Mapping[str, Any]] = {}
        self.queries: list[str] = []
        self.insert_parameters: Mapping[str, Any] | None = None

    def fetch_one(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        self.queries.append(query)
        if "INSERT INTO market_data_candle_repair_events" in query:
            row = dict(parameters)
            self.rows[str(parameters["event_id"])] = row
            self.insert_parameters = row
            return row
        if "WHERE event_id = %(event_id)s" in query:
            return self.rows.get(str(parameters["event_id"]))
        raise AssertionError(f"unexpected query: {query}")

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        self.queries.append(query)
        if "WHERE correlation_id = %(correlation_id)s" not in query:
            raise AssertionError(f"unexpected query: {query}")
        correlation_id = str(parameters["correlation_id"])
        return tuple(
            row for row in self.rows.values() if row["correlation_id"] == correlation_id
        )

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        _ = query
        _ = parameters
        raise AssertionError("audit repository should not call execute")


def _instrument_id() -> InstrumentId:
    return InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))


def _row(ts_open: datetime, *, source: str = "rest") -> CandleWithMeta:
    instrument_id = _instrument_id()
    candle = Candle(
        instrument_id=instrument_id,
        ts_open=UtcTimestamp(ts_open),
        ts_close=UtcTimestamp(ts_open + timedelta(minutes=1)),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume_base=2.0,
        volume_quote=200.0,
    )
    meta = CandleMeta(
        source=source,
        ingested_at=UtcTimestamp(ts_open + timedelta(minutes=1, milliseconds=120)),
        ingest_id=UUID("00000000-0000-0000-0000-000000000303"),
        instrument_key="binance:spot:BTCUSDT",
        trades_count=10,
        taker_buy_volume_base=0.5,
        taker_buy_volume_quote=50.0,
    )
    return CandleWithMeta(candle=candle, meta=meta)


def _provider(
    *,
    hot_cache: _HotCache,
    canonical_reader: _CanonicalReader,
    rest_source: _RestSource,
    audit_repository: CandleRepairAuditRepository,
    now: datetime,
    rest_tail_limit_minutes: int = 15,
    hooks: ClosedCandleTailProviderHooks | None = None,
) -> MarketDataClosedCandleTailProvider:
    return MarketDataClosedCandleTailProvider(
        hot_cache=hot_cache,
        canonical_reader=canonical_reader,
        rest_source=rest_source,
        audit_repository=audit_repository,
        clock=_Clock(now),
        policy=ClosedCandleTailRepairPolicy(
            rest_tail_limit_minutes=rest_tail_limit_minutes,
        ),
        clickhouse_circuit_open_seconds=30.0,
        hooks=hooks,
    )


def test_provider_falls_back_from_clickhouse_failure_to_rest_and_writes_hot_cache() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    hot_cache = _HotCache()
    canonical_reader = _CanonicalReader(error=RuntimeError("clickhouse unavailable"))
    rest_source = _RestSource((_row(base),))
    audit_repository = _AuditRepository()
    provider = _provider(
        hot_cache=hot_cache,
        canonical_reader=canonical_reader,
        rest_source=rest_source,
        audit_repository=audit_repository,
        now=base + timedelta(minutes=5, seconds=30),
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=1)),
        correlation_id="stage03-provider-proof",
    )

    assert result.continuous is True
    assert [row.source for row in result.candles] == ["rest"]
    assert [(item.source, item.status) for item in result.sources_attempted] == [
        ("redis_hot_cache", "miss"),
        ("clickhouse", "failed"),
        ("rest", "succeeded"),
    ]
    assert hot_cache.write_calls == 1
    assert canonical_reader.read_calls == 1
    assert len(rest_source.calls) == 1
    assert len(audit_repository.records) == 1
    assert audit_repository.records[0].status == "succeeded"
    assert [str(item) for item in audit_repository.records[0].restored_ts_opens] == [
        "2026-06-30T12:00:00.000Z"
    ]

    second = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=1)),
        correlation_id="stage03-provider-proof-redis-hit",
    )

    assert second.continuous is True
    assert [row.source for row in second.candles] == ["redis_hot_cache"]
    assert [(item.source, item.status) for item in second.sources_attempted] == [
        ("redis_hot_cache", "succeeded")
    ]
    assert canonical_reader.read_calls == 1
    assert len(rest_source.calls) == 1
    assert len(audit_repository.records) == 2


def test_provider_ignores_source_rows_outside_requested_range() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    hot_cache = _LeakyHotCache()
    hot_cache.rows[base - timedelta(minutes=1)] = _row(base - timedelta(minutes=1))
    hot_cache.rows[base] = _row(base)
    hot_cache.rows[base + timedelta(minutes=1)] = _row(base + timedelta(minutes=1))
    hot_cache.rows[base + timedelta(minutes=2)] = _row(base + timedelta(minutes=2))
    canonical_reader = _CanonicalReader(error=RuntimeError("clickhouse should not be called"))
    rest_source = _RestSource(())
    audit_repository = _AuditRepository()
    provider = _provider(
        hot_cache=hot_cache,
        canonical_reader=canonical_reader,
        rest_source=rest_source,
        audit_repository=audit_repository,
        now=base + timedelta(minutes=5),
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=2)),
        correlation_id="stage07-ignore-out-of-range-source-rows",
    )

    assert result.continuous is True
    assert [row.ts_open.value for row in result.candles] == [
        base,
        base + timedelta(minutes=1),
    ]
    assert [(item.source, item.status) for item in result.sources_attempted] == [
        ("redis_hot_cache", "succeeded")
    ]
    assert hot_cache.read_calls == 1
    assert canonical_reader.read_calls == 0
    assert rest_source.calls == []


def test_provider_emits_bounded_repair_metrics_hooks() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    attempts: list[tuple[str, str]] = []
    latencies: list[tuple[str, str, float]] = []
    circuit_states: list[int] = []
    provider = _provider(
        hot_cache=_HotCache(),
        canonical_reader=_CanonicalReader(error=RuntimeError("clickhouse unavailable")),
        rest_source=_RestSource((_row(base),)),
        audit_repository=_AuditRepository(),
        now=base + timedelta(minutes=5, seconds=30),
        hooks=ClosedCandleTailProviderHooks(
            on_repair_attempt=lambda source, status: attempts.append((source, status)),
            on_repair_latency=lambda source, status, duration: latencies.append(
                (source, status, duration)
            ),
            on_clickhouse_circuit_state=circuit_states.append,
        ),
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=1)),
        correlation_id="stage05-provider-metrics-proof",
    )

    assert result.continuous is True
    assert attempts == [
        ("redis_hot_cache", "miss"),
        ("clickhouse", "failed"),
        ("rest", "succeeded"),
    ]
    assert [(source, status) for source, status, _duration in latencies] == attempts
    assert all(duration >= 0.0 for _source, _status, duration in latencies)
    assert circuit_states == [1]


def test_provider_returns_sorted_rows_when_sources_restore_out_of_order() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    hot_cache = _HotCache()
    hot_cache.rows[base + timedelta(minutes=2)] = _row(base + timedelta(minutes=2))
    canonical_reader = _CanonicalReader(error=RuntimeError("clickhouse unavailable"))
    rest_source = _RestSource((_row(base), _row(base + timedelta(minutes=1))))
    audit_repository = _AuditRepository()
    provider = _provider(
        hot_cache=hot_cache,
        canonical_reader=canonical_reader,
        rest_source=rest_source,
        audit_repository=audit_repository,
        now=base + timedelta(minutes=5),
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=3)),
        correlation_id="stage03-sorted-tail",
    )

    assert result.continuous is True
    assert [row.ts_open.value for row in result.candles] == [
        base,
        base + timedelta(minutes=1),
        base + timedelta(minutes=2),
    ]


def test_provider_returns_miss_and_audit_when_rest_tail_is_missing() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    audit_repository = _AuditRepository()
    provider = _provider(
        hot_cache=_HotCache(),
        canonical_reader=_CanonicalReader(rows=()),
        rest_source=_RestSource(()),
        audit_repository=audit_repository,
        now=base + timedelta(minutes=5),
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=1)),
        correlation_id="stage03-rest-miss",
    )

    assert result.continuous is False
    assert [str(item) for item in result.missing_ts_opens] == [
        "2026-06-30T12:00:00.000Z"
    ]
    assert [(item.source, item.status) for item in result.sources_attempted] == [
        ("redis_hot_cache", "miss"),
        ("clickhouse", "miss"),
        ("rest", "miss"),
    ]
    assert audit_repository.records[0].status == "miss"
    assert audit_repository.records[0].error_code == "missing_closed_tail"


def test_provider_rejects_current_open_range_without_source_side_effects() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    hot_cache = _HotCache()
    canonical_reader = _CanonicalReader(rows=())
    rest_source = _RestSource((_row(base + timedelta(minutes=2)),))
    audit_repository = _AuditRepository()
    provider = _provider(
        hot_cache=hot_cache,
        canonical_reader=canonical_reader,
        rest_source=rest_source,
        audit_repository=audit_repository,
        now=base + timedelta(minutes=2, seconds=30),
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base + timedelta(minutes=2)),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=3)),
        correlation_id="stage03-current-open",
    )

    assert result.continuous is False
    assert result.error_code == "non_closed_range"
    assert hot_cache.read_calls == 0
    assert canonical_reader.read_calls == 0
    assert rest_source.calls == []
    assert audit_repository.records[0].status == "miss"
    assert audit_repository.records[0].error_code == "non_closed_range"


def test_provider_skips_rest_when_range_is_older_than_tail_limit() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    rest_source = _RestSource((_row(base),))
    audit_repository = _AuditRepository()
    provider = _provider(
        hot_cache=_HotCache(),
        canonical_reader=_CanonicalReader(rows=()),
        rest_source=rest_source,
        audit_repository=audit_repository,
        now=base + timedelta(minutes=20),
        rest_tail_limit_minutes=5,
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=1)),
        correlation_id="stage03-tail-limit",
    )

    assert result.continuous is False
    assert rest_source.calls == []
    assert result.sources_attempted[-1].source == "rest"
    assert result.sources_attempted[-1].status == "miss"
    assert result.sources_attempted[-1].error_code == "rest_tail_limit_exceeded"
    assert audit_repository.records[0].status == "miss"


def test_provider_records_audit_through_postgres_repository_boundary() -> None:
    base = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    gateway = _PostgresAuditGateway()
    audit_repository = PostgresCandleRepairAuditRepository(gateway=gateway)
    provider = _provider(
        hot_cache=_HotCache(),
        canonical_reader=_CanonicalReader(error=RuntimeError("clickhouse unavailable")),
        rest_source=_RestSource((_row(base),)),
        audit_repository=audit_repository,
        now=base + timedelta(minutes=5),
    )

    result = provider.get_closed_1m_tail(
        instrument_id=_instrument_id(),
        instrument_key="binance:spot:BTCUSDT",
        start_ts_open=UtcTimestamp(base),
        end_ts_open=UtcTimestamp(base + timedelta(minutes=1)),
        correlation_id="stage03-postgres-audit-proof",
    )
    listed = audit_repository.list_for_correlation(
        correlation_id="stage03-postgres-audit-proof"
    )

    assert result.continuous is True
    assert len(listed) == 1
    assert listed[0].status == "succeeded"
    assert tuple(attempt.source for attempt in listed[0].sources_attempted) == (
        "redis_hot_cache",
        "clickhouse",
        "rest",
    )
    assert gateway.insert_parameters is not None
    assert json.loads(str(gateway.insert_parameters["sources_attempted_json"])) == [
        {"error_code": None, "source": "redis_hot_cache", "status": "miss"},
        {"error_code": "clickhouse_exception", "source": "clickhouse", "status": "failed"},
        {"error_code": None, "source": "rest", "status": "succeeded"},
    ]
    joined_queries = "\n".join(gateway.queries)
    assert "canonical_candles" not in joined_queries
    assert "ClickHouse" not in joined_queries
