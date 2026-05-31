from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import ExchangeExecutionOrderRepository
from trading.contexts.live_execution.domain import (
    ExchangeExecutionOrderRecord,
    ExchangeOrderCancelResult,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExchangeOrderSubmitResult,
    ExchangePrivateStreamSession,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import UserId


class PostgresExchangeExecutionOrderRepository(ExchangeExecutionOrderRepository):
    def __init__(self, *, gateway: StrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresExchangeExecutionOrderRepository requires gateway")
        self._gateway = gateway

    def get_by_intent(self, *, intent_id: UUID) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query="SELECT * FROM execution_orders WHERE intent_id = %(intent_id)s",
            parameters={"intent_id": str(intent_id)},
        )
        return _map_order(row) if row is not None else None

    def record_guard_rejection(
        self, *, command: ExchangeOrderCommand, reason: str
    ) -> ExchangeExecutionOrderRecord:
        return self._insert_or_update_base(
            command=command,
            status="guard_rejected",
            reason=reason,
            metadata={"guard_reason": reason},
        )

    def record_submit_pending(
        self, *, command: ExchangeOrderCommand
    ) -> ExchangeExecutionOrderRecord:
        return self._insert_or_update_base(
            command=command,
            status="submit_pending",
            reason="submit_pending",
            metadata={},
        )

    def record_submit_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderSubmitResult,
    ) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET exchange_order_id = %(exchange_order_id)s,
                status = 'submitted',
                status_reason = %(status_reason)s,
                submitted_at = %(submitted_at)s,
                adapter_attempt_count = adapter_attempt_count + 1,
                latency_ms = %(latency_ms)s,
                metadata_json = %(metadata_json)s::jsonb,
                updated_at = %(updated_at)s
            WHERE intent_id = %(intent_id)s
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "exchange_order_id": result.exchange_order_id,
                "status_reason": result.exchange_status,
                "submitted_at": result.submitted_at,
                "latency_ms": result.latency_ms,
                "metadata_json": _metadata_json(result.metadata),
                "updated_at": result.submitted_at,
            },
        )
        return _map_order(row) if row is not None else None

    def record_status_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderStatusResult,
    ) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET exchange_order_id = %(exchange_order_id)s,
                status = 'status_checked',
                status_reason = %(status_reason)s,
                last_checked_at = %(checked_at)s,
                latency_ms = %(latency_ms)s,
                metadata_json = %(metadata_json)s::jsonb,
                updated_at = %(updated_at)s
            WHERE intent_id = %(intent_id)s
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "exchange_order_id": result.exchange_order_id,
                "status_reason": result.exchange_status,
                "checked_at": result.checked_at,
                "latency_ms": result.latency_ms,
                "metadata_json": _metadata_json(result.metadata),
                "updated_at": result.checked_at,
            },
        )
        return _map_order(row) if row is not None else None

    def record_cancel_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderCancelResult,
    ) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET exchange_order_id = %(exchange_order_id)s,
                status = 'cancelled',
                status_reason = %(status_reason)s,
                cancel_requested_at = COALESCE(cancel_requested_at, %(cancelled_at)s),
                cancelled_at = %(cancelled_at)s,
                latency_ms = %(latency_ms)s,
                metadata_json = %(metadata_json)s::jsonb,
                updated_at = %(updated_at)s
            WHERE intent_id = %(intent_id)s
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "exchange_order_id": result.exchange_order_id,
                "status_reason": result.exchange_status,
                "cancelled_at": result.cancelled_at,
                "latency_ms": result.latency_ms,
                "metadata_json": _metadata_json(result.metadata),
                "updated_at": result.cancelled_at,
            },
        )
        return _map_order(row) if row is not None else None

    def record_adapter_error(
        self, *, intent_id: UUID, reason: str
    ) -> ExchangeExecutionOrderRecord | None:
        now = datetime.now(tz=UTC)
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET status = 'adapter_error',
                status_reason = %(reason)s,
                updated_at = %(updated_at)s
            WHERE intent_id = %(intent_id)s
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "reason": reason,
                "updated_at": now,
            },
        )
        return _map_order(row) if row is not None else None

    def record_private_stream_session(
        self,
        *,
        connection_id: UUID,
        session: ExchangePrivateStreamSession,
    ) -> ExchangePrivateStreamSession:
        self._gateway.fetch_one(
            query="""
            INSERT INTO exchange_private_stream_sessions
            (
                session_id, exchange_connection_id, exchange_name, environment,
                market_type, status, status_reason, opened_at, keepalive_at,
                expires_at, metadata_json, updated_at
            )
            VALUES
            (
                %(session_id)s, %(connection_id)s, %(exchange_name)s, %(environment)s,
                %(market_type)s, %(status)s, %(status_reason)s, %(opened_at)s,
                %(keepalive_at)s, %(expires_at)s, %(metadata_json)s::jsonb,
                %(updated_at)s
            )
            ON CONFLICT (exchange_connection_id, exchange_name, market_type, environment)
            DO UPDATE SET
                session_id = EXCLUDED.session_id,
                status = EXCLUDED.status,
                status_reason = EXCLUDED.status_reason,
                opened_at = EXCLUDED.opened_at,
                keepalive_at = EXCLUDED.keepalive_at,
                expires_at = EXCLUDED.expires_at,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = EXCLUDED.updated_at
            RETURNING session_id
            """,
            parameters={
                "session_id": str(session.session_id),
                "connection_id": str(connection_id),
                "exchange_name": session.exchange_name,
                "environment": session.environment,
                "market_type": session.market_type,
                "status": session.status,
                "status_reason": session.status_reason,
                "opened_at": session.opened_at,
                "keepalive_at": session.keepalive_at,
                "expires_at": session.expires_at,
                "metadata_json": _metadata_json(session.metadata),
                "updated_at": session.keepalive_at or session.opened_at,
            },
        )
        return session

    def _insert_or_update_base(
        self,
        *,
        command: ExchangeOrderCommand,
        status: str,
        reason: str,
        metadata: Mapping[str, int | float | str],
    ) -> ExchangeExecutionOrderRecord:
        now = datetime.now(tz=UTC)
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_orders
            (
                order_id, intent_id, owner_user_id, exchange_connection_id,
                exchange_name, environment, market_type, instrument_key, side,
                order_type, quantity, quote_notional, limit_price, client_order_id,
                exchange_order_id, status, status_reason, submitted_at,
                cancel_requested_at, cancelled_at, last_checked_at,
                adapter_attempt_count, latency_ms, metadata_json, created_at, updated_at
            )
            VALUES
            (
                %(order_id)s, %(intent_id)s, %(owner_user_id)s,
                %(exchange_connection_id)s, %(exchange_name)s, %(environment)s,
                %(market_type)s, %(instrument_key)s, %(side)s, %(order_type)s,
                %(quantity)s, %(quote_notional)s, %(limit_price)s, %(client_order_id)s,
                NULL, %(status)s, %(status_reason)s, NULL, NULL, NULL, NULL,
                0, NULL, %(metadata_json)s::jsonb, %(created_at)s, %(updated_at)s
            )
            ON CONFLICT (intent_id) DO UPDATE
            SET status = EXCLUDED.status,
                status_reason = EXCLUDED.status_reason,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = EXCLUDED.updated_at
            RETURNING *
            """,
            parameters={
                "order_id": str(uuid4()),
                "intent_id": str(command.intent_id),
                "owner_user_id": str(command.owner_user_id),
                "exchange_connection_id": str(command.exchange_connection_id),
                "exchange_name": command.exchange_name,
                "environment": command.environment,
                "market_type": command.market_type,
                "instrument_key": command.instrument_key,
                "side": command.side,
                "order_type": command.order_type,
                "quantity": command.quantity,
                "quote_notional": command.quote_notional,
                "limit_price": command.limit_price,
                "client_order_id": command.client_order_id,
                "status": status,
                "status_reason": reason,
                "metadata_json": _metadata_json(metadata),
                "created_at": now,
                "updated_at": now,
            },
        )
        if row is None:
            raise RuntimeError("execution order write returned no row")
        return _map_order(row)


def _map_order(row: Mapping[str, Any]) -> ExchangeExecutionOrderRecord:
    return ExchangeExecutionOrderRecord(
        order_id=UUID(str(row["order_id"])),
        intent_id=UUID(str(row["intent_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        exchange_connection_id=UUID(str(row["exchange_connection_id"])),
        exchange_name=str(row["exchange_name"]),
        environment=str(row["environment"]),
        market_type=str(row["market_type"]),
        instrument_key=str(row["instrument_key"]),
        side=str(row["side"]),
        order_type=str(row["order_type"]),
        quantity=_decimal_or_none(row.get("quantity")),
        quote_notional=_decimal_or_none(row.get("quote_notional")),
        limit_price=_decimal_or_none(row.get("limit_price")),
        client_order_id=str(row["client_order_id"]),
        exchange_order_id=_str_or_none(row.get("exchange_order_id")),
        status=str(row["status"]),  # type: ignore[arg-type]
        status_reason=str(row["status_reason"]),
        submitted_at=_datetime_or_none(row.get("submitted_at")),
        cancel_requested_at=_datetime_or_none(row.get("cancel_requested_at")),
        cancelled_at=_datetime_or_none(row.get("cancelled_at")),
        last_checked_at=_datetime_or_none(row.get("last_checked_at")),
        adapter_attempt_count=int(row.get("adapter_attempt_count") or 0),
        latency_ms=(
            float(row["latency_ms"]) if row.get("latency_ms") is not None else None
        ),
        metadata=_metadata_mapping(row.get("metadata_json")),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _metadata_json(metadata: Mapping[str, int | float | str]) -> str:
    return json.dumps(dict(metadata), sort_keys=True)


def _metadata_mapping(value: object) -> Mapping[str, int | float | str]:
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(item, (int, float, str))
        }
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, Mapping):
            return {
                str(key): item
                for key, item in payload.items()
                if isinstance(item, (int, float, str))
            }
    return {}


def _decimal_or_none(value: object) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value))


def _datetime_or_none(value: object) -> datetime | None:
    return _datetime(value) if value is not None else None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
