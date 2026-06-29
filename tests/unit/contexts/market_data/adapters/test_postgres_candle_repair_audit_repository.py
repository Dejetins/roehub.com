from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    PostgresCandleRepairAuditRepository,
)
from trading.contexts.market_data.application.dto import (
    CandleRepairSourceAttempt,
    MarketDataCandleRepairAuditEvent,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)


class _StatefulGateway:
    def __init__(self) -> None:
        self.rows: dict[str, Mapping[str, Any]] = {}
        self.queries: list[str] = []
        self.insert_parameters: Mapping[str, Any] | None = None

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
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
        raise AssertionError("audit repository should not call execute")


def test_postgres_candle_repair_audit_repository_inserts_and_reads_without_clickhouse() -> None:
    gateway = _StatefulGateway()
    repository = PostgresCandleRepairAuditRepository(gateway=gateway)
    event = _event()

    persisted = repository.record(event=event)
    fetched = repository.get_by_id(event_id=event.event_id)
    listed = repository.list_for_correlation(correlation_id=event.correlation_id)

    assert persisted.event_id == event.event_id
    assert fetched == persisted
    assert listed == (persisted,)
    assert persisted.status == "failed"
    assert tuple(attempt.source for attempt in persisted.sources_attempted) == (
        "redis_hot_cache",
        "clickhouse",
        "rest",
    )
    assert tuple(str(ts) for ts in persisted.restored_ts_opens) == (
        "2026-06-29T12:00:00.000Z",
    )
    assert tuple(str(ts) for ts in persisted.missing_ts_opens) == (
        "2026-06-29T12:01:00.000Z",
    )
    joined_queries = "\n".join(gateway.queries)
    assert "canonical_candles" not in joined_queries
    assert "ClickHouse" not in joined_queries

    assert gateway.insert_parameters is not None
    assert json.loads(str(gateway.insert_parameters["sources_attempted_json"])) == [
        {"error_code": None, "source": "redis_hot_cache", "status": "miss"},
        {"error_code": "http_connection_reset", "source": "clickhouse", "status": "failed"},
        {"error_code": None, "source": "rest", "status": "miss"},
    ]


def _event() -> MarketDataCandleRepairAuditEvent:
    return MarketDataCandleRepairAuditEvent(
        event_id=UUID("00000000-0000-0000-0000-000000003801"),
        correlation_id="stage01-audit-proof",
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        instrument_key="binance:spot:BTCUSDT",
        time_range=TimeRange(start=_ts(0), end=_ts(2)),
        status="failed",
        sources_attempted=(
            CandleRepairSourceAttempt(source="redis_hot_cache", status="miss"),
            CandleRepairSourceAttempt(
                source="clickhouse",
                status="failed",
                error_code="http_connection_reset",
            ),
            CandleRepairSourceAttempt(source="rest", status="miss"),
        ),
        restored_ts_opens=(_ts(0),),
        missing_ts_opens=(_ts(1),),
        error_code="short_tail_missing",
        error_summary="bounded repair sources exhausted",
        created_at=_ts(3),
    )


def _ts(minute: int) -> UtcTimestamp:
    return UtcTimestamp(datetime(2026, 6, 29, 12, minute, tzinfo=timezone.utc))
