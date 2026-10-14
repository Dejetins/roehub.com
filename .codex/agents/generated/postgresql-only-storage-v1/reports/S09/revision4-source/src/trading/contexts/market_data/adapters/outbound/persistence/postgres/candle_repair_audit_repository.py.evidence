from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, cast
from uuid import UUID

from trading.contexts.market_data.application.dto import (
    CandleRepairSourceAttempt,
    MarketDataCandleRepairAuditEvent,
)
from trading.contexts.market_data.application.ports.stores.candle_repair_audit_repository import (
    CandleRepairAuditRepository,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

from .gateway import MarketDataPostgresGateway


class PostgresCandleRepairAuditRepository(CandleRepairAuditRepository):
    """
    Postgres-backed audit repository for Market Data live-tail repair attempts.
    """

    def __init__(self, *, gateway: MarketDataPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresCandleRepairAuditRepository requires gateway")
        self._gateway = gateway

    def record(
        self,
        *,
        event: MarketDataCandleRepairAuditEvent,
    ) -> MarketDataCandleRepairAuditEvent:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO market_data_candle_repair_events
            (
                event_id, correlation_id, market_id, symbol, instrument_key,
                range_start_ts_open, range_end_ts_open, status,
                sources_attempted_json, restored_ts_opens_json, missing_ts_opens_json,
                error_code, error_summary, created_at
            )
            VALUES
            (
                %(event_id)s, %(correlation_id)s, %(market_id)s, %(symbol)s,
                %(instrument_key)s, %(range_start_ts_open)s, %(range_end_ts_open)s,
                %(status)s, %(sources_attempted_json)s::jsonb,
                %(restored_ts_opens_json)s::jsonb, %(missing_ts_opens_json)s::jsonb,
                %(error_code)s, %(error_summary)s, %(created_at)s
            )
            ON CONFLICT (event_id) DO NOTHING
            RETURNING *
            """,
            parameters=_event_params(event),
        )
        if row is None:
            return self.get_by_id(event_id=event.event_id) or event
        return _map_event(row=row)

    def get_by_id(self, *, event_id: UUID) -> MarketDataCandleRepairAuditEvent | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM market_data_candle_repair_events
            WHERE event_id = %(event_id)s
            """,
            parameters={"event_id": str(event_id)},
        )
        return _map_event(row=row) if row is not None else None

    def list_for_correlation(
        self,
        *,
        correlation_id: str,
    ) -> tuple[MarketDataCandleRepairAuditEvent, ...]:
        row_values = self._gateway.fetch_all(
            query="""
            SELECT *
            FROM market_data_candle_repair_events
            WHERE correlation_id = %(correlation_id)s
            ORDER BY created_at ASC, event_id ASC
            """,
            parameters={"correlation_id": correlation_id.strip()},
        )
        return tuple(_map_event(row=row) for row in row_values)


def _event_params(event: MarketDataCandleRepairAuditEvent) -> Mapping[str, Any]:
    return {
        "event_id": str(event.event_id),
        "correlation_id": event.correlation_id,
        "market_id": int(event.instrument_id.market_id.value),
        "symbol": str(event.instrument_id.symbol),
        "instrument_key": event.instrument_key,
        "range_start_ts_open": event.time_range.start.value,
        "range_end_ts_open": event.time_range.end.value,
        "status": event.status,
        "sources_attempted_json": json.dumps(
            [
                {
                    "source": attempt.source,
                    "status": attempt.status,
                    "error_code": attempt.error_code,
                }
                for attempt in event.sources_attempted
            ],
            sort_keys=True,
        ),
        "restored_ts_opens_json": json.dumps(
            [str(ts_open) for ts_open in event.restored_ts_opens],
            sort_keys=True,
        ),
        "missing_ts_opens_json": json.dumps(
            [str(ts_open) for ts_open in event.missing_ts_opens],
            sort_keys=True,
        ),
        "error_code": event.error_code,
        "error_summary": event.error_summary,
        "created_at": event.created_at.value,
    }


def _map_event(*, row: Mapping[str, Any]) -> MarketDataCandleRepairAuditEvent:
    instrument_id = InstrumentId(
        market_id=MarketId(int(row["market_id"])),
        symbol=Symbol(str(row["symbol"])),
    )
    return MarketDataCandleRepairAuditEvent(
        event_id=UUID(str(row["event_id"])),
        correlation_id=str(row["correlation_id"]),
        instrument_id=instrument_id,
        instrument_key=str(row["instrument_key"]),
        time_range=TimeRange(
            start=UtcTimestamp(_coerce_datetime(row["range_start_ts_open"])),
            end=UtcTimestamp(_coerce_datetime(row["range_end_ts_open"])),
        ),
        status=cast(Any, str(row["status"])),
        sources_attempted=_map_source_attempts(row["sources_attempted_json"]),
        restored_ts_opens=_map_ts_open_array(row["restored_ts_opens_json"]),
        missing_ts_opens=_map_ts_open_array(row["missing_ts_opens_json"]),
        error_code=str(row["error_code"]) if row.get("error_code") is not None else None,
        error_summary=(
            str(row["error_summary"]) if row.get("error_summary") is not None else None
        ),
        created_at=UtcTimestamp(_coerce_datetime(row["created_at"])),
    )


def _map_source_attempts(value: Any) -> tuple[CandleRepairSourceAttempt, ...]:
    payload = _json_value(value)
    if not isinstance(payload, list):
        raise ValueError("sources_attempted_json must map to a list")
    attempts: list[CandleRepairSourceAttempt] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("sources_attempted_json items must be objects")
        attempts.append(
            CandleRepairSourceAttempt(
                source=cast(Any, str(item["source"])),
                status=cast(Any, str(item["status"])),
                error_code=(
                    str(item["error_code"]) if item.get("error_code") is not None else None
                ),
            )
        )
    return tuple(attempts)


def _map_ts_open_array(value: Any) -> tuple[UtcTimestamp, ...]:
    payload = _json_value(value)
    if not isinstance(payload, list):
        raise ValueError("ts_open JSON payload must map to a list")
    return tuple(UtcTimestamp(_parse_ts_open(str(item))) for item in payload)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _parse_ts_open(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    return _parse_ts_open(str(value))
