from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.live_execution.application.ports import ExecutionIntentRepository
from trading.contexts.live_execution.domain import (
    ExecutionIntent,
    ExecutionNotificationOutboxEvent,
    ExecutionProducerOutcomeLink,
    ExecutionRiskAuditEvent,
    ExecutionSourceEvent,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import UserId


class PostgresExecutionIntentRepository(ExecutionIntentRepository):
    def __init__(self, *, gateway: StrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresExecutionIntentRepository requires gateway")
        self._gateway = gateway

    def record_source_event(self, *, event: ExecutionSourceEvent) -> ExecutionSourceEvent:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_source_events
            (
                source_event_id, owner_user_id, source_type, source_event_ref,
                source_ref_json, strategy_signal_id, idempotency_key_hash,
                outcome, outcome_reason, intent_id, received_at
            )
            VALUES
            (
                %(source_event_id)s, %(owner_user_id)s, %(source_type)s,
                %(source_event_ref)s, %(source_ref_json)s::jsonb,
                %(strategy_signal_id)s, %(idempotency_key_hash)s, %(outcome)s,
                %(outcome_reason)s, %(intent_id)s, %(received_at)s
            )
            ON CONFLICT (owner_user_id, source_type, idempotency_key_hash) DO NOTHING
            RETURNING *
            """,
            parameters=_source_event_params(event),
        )
        if row is None:
            return self.get_source_event_by_idempotency(
                owner_user_id=event.owner_user_id,
                source_type=event.source_type,
                idempotency_key_hash=event.idempotency_key_hash,
            ) or event
        return _map_source_event(row)

    def get_source_event_by_id(
        self, *, owner_user_id: UserId, source_event_id: UUID
    ) -> ExecutionSourceEvent | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM execution_source_events
            WHERE owner_user_id = %(owner_user_id)s
              AND source_event_id = %(source_event_id)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "source_event_id": str(source_event_id),
            },
        )
        return _map_source_event(row) if row is not None else None

    def get_source_event_by_idempotency(
        self,
        *,
        owner_user_id: UserId,
        source_type: str,
        idempotency_key_hash: str,
    ) -> ExecutionSourceEvent | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM execution_source_events
            WHERE owner_user_id = %(owner_user_id)s
              AND source_type = %(source_type)s
              AND idempotency_key_hash = %(idempotency_key_hash)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "source_type": source_type,
                "idempotency_key_hash": idempotency_key_hash,
            },
        )
        return _map_source_event(row) if row is not None else None

    def update_source_event_outcome(
        self,
        *,
        owner_user_id: UserId,
        source_event_id: UUID,
        outcome: str,
        outcome_reason: str,
        intent_id: UUID | None,
    ) -> ExecutionSourceEvent | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_source_events
            SET outcome = %(outcome)s,
                outcome_reason = %(outcome_reason)s,
                intent_id = %(intent_id)s
            WHERE owner_user_id = %(owner_user_id)s
              AND source_event_id = %(source_event_id)s
            RETURNING *
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "source_event_id": str(source_event_id),
                "outcome": outcome,
                "outcome_reason": outcome_reason,
                "intent_id": str(intent_id) if intent_id is not None else None,
            },
        )
        return _map_source_event(row) if row is not None else None

    def record_intent(self, *, intent: ExecutionIntent) -> ExecutionIntent:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_intents
            (
                intent_id, source_event_id, owner_user_id, source_type,
                strategy_signal_id, exchange_connection_id, market_type,
                instrument_key, side, order_type, quantity, quote_notional,
                limit_price, status, status_reason, risk_status, risk_reason,
                idempotency_key_hash, created_at, dispatch_attempt_count,
                dispatch_stream_name, dispatch_redis_message_id, dispatch_last_error,
                dispatch_updated_at
            )
            VALUES
            (
                %(intent_id)s, %(source_event_id)s, %(owner_user_id)s,
                %(source_type)s, %(strategy_signal_id)s, %(exchange_connection_id)s,
                %(market_type)s, %(instrument_key)s, %(side)s, %(order_type)s,
                %(quantity)s, %(quote_notional)s, %(limit_price)s, %(status)s,
                %(status_reason)s, %(risk_status)s, %(risk_reason)s,
                %(idempotency_key_hash)s, %(created_at)s, %(dispatch_attempt_count)s,
                %(dispatch_stream_name)s, %(dispatch_redis_message_id)s,
                %(dispatch_last_error)s, %(dispatch_updated_at)s
            )
            ON CONFLICT (owner_user_id, idempotency_key_hash) DO NOTHING
            RETURNING *
            """,
            parameters=_intent_params(intent),
        )
        if row is None:
            return self.get_intent_by_idempotency(
                owner_user_id=intent.owner_user_id,
                idempotency_key_hash=intent.idempotency_key_hash,
            ) or intent
        return _map_intent(row)

    def get_intent_by_idempotency(
        self, *, owner_user_id: UserId, idempotency_key_hash: str
    ) -> ExecutionIntent | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM execution_intents
            WHERE owner_user_id = %(owner_user_id)s
              AND idempotency_key_hash = %(idempotency_key_hash)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "idempotency_key_hash": idempotency_key_hash,
            },
        )
        return _map_intent(row) if row is not None else None

    def get_intent_by_id(
        self, *, owner_user_id: UserId, intent_id: UUID
    ) -> ExecutionIntent | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT *
            FROM execution_intents
            WHERE owner_user_id = %(owner_user_id)s
              AND intent_id = %(intent_id)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "intent_id": str(intent_id),
            },
        )
        return _map_intent(row) if row is not None else None

    def claim_intent_for_dispatch(
        self, *, intent_id: UUID, now: datetime, retry_budget: int
    ) -> ExecutionIntent | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_intents
            SET status = 'dispatching',
                status_reason = 'dispatch_publish_pending',
                dispatch_attempt_count = dispatch_attempt_count + 1,
                dispatch_last_error = NULL,
                dispatch_updated_at = %(now)s
            WHERE intent_id = %(intent_id)s
              AND status IN ('accepted', 'retry')
              AND risk_status = 'accepted'
              AND dispatch_attempt_count < %(retry_budget)s
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "now": now,
                "retry_budget": retry_budget,
            },
        )
        return _map_intent(row) if row is not None else None

    def mark_intent_dispatched(
        self,
        *,
        intent_id: UUID,
        stream_name: str,
        redis_message_id: str,
        now: datetime,
    ) -> ExecutionIntent | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_intents
            SET status = 'dispatched',
                status_reason = 'redis_xadd_ok',
                dispatch_stream_name = %(stream_name)s,
                dispatch_redis_message_id = %(redis_message_id)s,
                dispatch_last_error = NULL,
                dispatch_updated_at = %(now)s
            WHERE intent_id = %(intent_id)s
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "stream_name": stream_name,
                "redis_message_id": redis_message_id,
                "now": now,
            },
        )
        return _map_intent(row) if row is not None else None

    def mark_intent_dispatch_retry(
        self, *, intent_id: UUID, reason: str, now: datetime
    ) -> ExecutionIntent | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_intents
            SET status = 'retry',
                status_reason = %(reason)s,
                dispatch_last_error = %(reason)s,
                dispatch_updated_at = %(now)s
            WHERE intent_id = %(intent_id)s
              AND status <> 'dispatched'
              AND risk_status = 'accepted'
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "reason": reason,
                "now": now,
            },
        )
        return _map_intent(row) if row is not None else None

    def mark_intent_quarantined(
        self, *, intent_id: UUID, reason: str, stream_name: str | None, now: datetime
    ) -> ExecutionIntent | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_intents
            SET status = 'quarantined',
                status_reason = %(reason)s,
                dispatch_stream_name = COALESCE(%(stream_name)s, dispatch_stream_name),
                dispatch_last_error = %(reason)s,
                dispatch_updated_at = %(now)s
            WHERE intent_id = %(intent_id)s
              AND status <> 'dispatched'
              AND risk_status = 'accepted'
            RETURNING *
            """,
            parameters={
                "intent_id": str(intent_id),
                "reason": reason,
                "stream_name": stream_name,
                "now": now,
            },
        )
        return _map_intent(row) if row is not None else None

    def record_risk_audit_event(
        self, *, event: ExecutionRiskAuditEvent
    ) -> ExecutionRiskAuditEvent:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_risk_audit_events
            (
                event_id, intent_id, source_event_id, owner_user_id, source_type,
                event_type, risk_status, risk_reason, check_name, metadata_json, created_at
            )
            VALUES
            (
                %(event_id)s, %(intent_id)s, %(source_event_id)s, %(owner_user_id)s,
                %(source_type)s, %(event_type)s, %(risk_status)s, %(risk_reason)s,
                %(check_name)s, %(metadata_json)s::jsonb, %(created_at)s
            )
            ON CONFLICT (event_id) DO NOTHING
            RETURNING *
            """,
            parameters=_risk_audit_params(event),
        )
        return _map_risk_audit_event(row) if row is not None else event

    def record_notification_outbox(
        self, *, event: ExecutionNotificationOutboxEvent
    ) -> ExecutionNotificationOutboxEvent:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_notification_outbox
            (
                notification_id, owner_user_id, source_type, event_type, severity,
                reason, source_event_id, intent_id, order_id, strategy_signal_id,
                labels_json, status, created_at, sent_at
            )
            VALUES
            (
                %(notification_id)s, %(owner_user_id)s, %(source_type)s,
                %(event_type)s, %(severity)s, %(reason)s, %(source_event_id)s,
                %(intent_id)s, %(order_id)s, %(strategy_signal_id)s,
                %(labels_json)s::jsonb, %(status)s, %(created_at)s, %(sent_at)s
            )
            ON CONFLICT (
                owner_user_id, event_type, source_event_key, intent_key, order_key, reason
            ) DO NOTHING
            RETURNING *
            """,
            parameters=_notification_params(event),
        )
        if row is None:
            existing = self._gateway.fetch_one(
                query="""
                SELECT *
                FROM execution_notification_outbox
                WHERE owner_user_id = %(owner_user_id)s
                  AND event_type = %(event_type)s
                  AND source_event_key = COALESCE(
                    %(source_event_id)s,
                    '00000000-0000-0000-0000-000000000000'::uuid
                  )
                  AND intent_key = COALESCE(
                    %(intent_id)s,
                    '00000000-0000-0000-0000-000000000000'::uuid
                  )
                  AND order_key = COALESCE(
                    %(order_id)s,
                    '00000000-0000-0000-0000-000000000000'::uuid
                  )
                  AND reason = %(reason)s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                parameters=_notification_params(event),
            )
            return _map_notification(existing) if existing is not None else event
        return _map_notification(row)

    def list_recent_notifications(
        self,
        *,
        owner_user_id: UserId,
        limit: int,
        strategy_id: UUID | None = None,
    ) -> tuple[ExecutionNotificationOutboxEvent, ...]:
        if strategy_id is None:
            rows = self._gateway.fetch_all(
                query="""
                SELECT *
                FROM execution_notification_outbox
                WHERE owner_user_id = %(owner_user_id)s
                ORDER BY created_at DESC
                LIMIT %(limit)s
                """,
                parameters={"owner_user_id": str(owner_user_id), "limit": max(1, limit)},
            )
        else:
            rows = self._gateway.fetch_all(
                query="""
                SELECT n.*
                FROM execution_notification_outbox n
                LEFT JOIN execution_source_events e
                  ON e.source_event_id = n.source_event_id
                WHERE n.owner_user_id = %(owner_user_id)s
                  AND (
                    n.strategy_signal_id IS NOT NULL
                    OR e.source_ref_json ->> 'strategy_id' = %(strategy_id)s
                  )
                ORDER BY n.created_at DESC
                LIMIT %(limit)s
                """,
                parameters={
                    "owner_user_id": str(owner_user_id),
                    "strategy_id": str(strategy_id),
                    "limit": max(1, limit),
                },
            )
        return tuple(_map_notification(row) for row in rows)

    def list_producer_outcome_links_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID, limit: int
    ) -> tuple[ExecutionProducerOutcomeLink, ...]:
        rows = self._gateway.fetch_all(
            query="""
            SELECT
                e.source_event_id,
                e.owner_user_id,
                e.source_type,
                e.source_event_ref,
                e.strategy_signal_id,
                e.outcome,
                e.outcome_reason,
                e.intent_id,
                i.status AS intent_status,
                i.status_reason AS intent_status_reason,
                i.risk_status,
                i.risk_reason,
                o.status AS order_status,
                o.status_reason AS order_status_reason,
                n.event_type AS notification_event_type,
                n.reason AS notification_reason,
                COALESCE(o.updated_at, i.dispatch_updated_at, i.created_at, e.received_at)
                    AS updated_at
            FROM execution_source_events e
            LEFT JOIN execution_intents i
              ON i.intent_id = e.intent_id
            LEFT JOIN execution_orders o
              ON o.intent_id = i.intent_id
            LEFT JOIN LATERAL (
                SELECT event_type, reason
                FROM execution_notification_outbox n
                WHERE n.source_event_id = e.source_event_id
                   OR n.intent_id = i.intent_id
                   OR n.order_id = o.order_id
                ORDER BY n.created_at DESC
                LIMIT 1
            ) n ON TRUE
            WHERE e.owner_user_id = %(owner_user_id)s
              AND e.source_ref_json ->> 'strategy_id' = %(strategy_id)s
            ORDER BY updated_at DESC
            LIMIT %(limit)s
            """,
            parameters={
                "owner_user_id": str(owner_user_id),
                "strategy_id": str(strategy_id),
                "limit": max(1, limit),
            },
        )
        return tuple(_map_producer_link(row) for row in rows)


def _source_event_params(event: ExecutionSourceEvent) -> dict[str, object]:
    return {
        "source_event_id": str(event.source_event_id),
        "owner_user_id": str(event.owner_user_id),
        "source_type": event.source_type,
        "source_event_ref": event.source_event_ref,
        "source_ref_json": json.dumps(dict(event.source_ref_json), sort_keys=True),
        "strategy_signal_id": (
            str(event.strategy_signal_id) if event.strategy_signal_id is not None else None
        ),
        "idempotency_key_hash": event.idempotency_key_hash,
        "outcome": event.outcome,
        "outcome_reason": event.outcome_reason,
        "intent_id": str(event.intent_id) if event.intent_id is not None else None,
        "received_at": event.received_at,
    }


def _intent_params(intent: ExecutionIntent) -> dict[str, object]:
    return {
        "intent_id": str(intent.intent_id),
        "source_event_id": str(intent.source_event_id),
        "owner_user_id": str(intent.owner_user_id),
        "source_type": intent.source_type,
        "strategy_signal_id": (
            str(intent.strategy_signal_id) if intent.strategy_signal_id is not None else None
        ),
        "exchange_connection_id": str(intent.exchange_connection_id),
        "market_type": intent.market_type,
        "instrument_key": intent.instrument_key,
        "side": intent.side,
        "order_type": intent.order_type,
        "quantity": intent.quantity,
        "quote_notional": intent.quote_notional,
        "limit_price": intent.limit_price,
        "status": intent.status,
        "status_reason": intent.status_reason,
        "risk_status": intent.risk_status,
        "risk_reason": intent.risk_reason,
        "idempotency_key_hash": intent.idempotency_key_hash,
        "created_at": intent.created_at,
        "dispatch_attempt_count": intent.dispatch_attempt_count,
        "dispatch_stream_name": intent.dispatch_stream_name,
        "dispatch_redis_message_id": intent.dispatch_redis_message_id,
        "dispatch_last_error": intent.dispatch_last_error,
        "dispatch_updated_at": intent.dispatch_updated_at,
    }


def _risk_audit_params(event: ExecutionRiskAuditEvent) -> dict[str, object]:
    return {
        "event_id": str(event.event_id),
        "intent_id": str(event.intent_id),
        "source_event_id": str(event.source_event_id),
        "owner_user_id": str(event.owner_user_id),
        "source_type": event.source_type,
        "event_type": event.event_type,
        "risk_status": event.risk_status,
        "risk_reason": event.risk_reason,
        "check_name": event.check_name,
        "metadata_json": json.dumps(dict(event.metadata_json), sort_keys=True),
        "created_at": event.created_at,
    }


def _notification_params(event: ExecutionNotificationOutboxEvent) -> dict[str, object]:
    return {
        "notification_id": str(event.notification_id),
        "owner_user_id": str(event.owner_user_id),
        "source_type": event.source_type,
        "event_type": event.event_type,
        "severity": event.severity,
        "reason": event.reason,
        "source_event_id": (
            str(event.source_event_id) if event.source_event_id is not None else None
        ),
        "intent_id": str(event.intent_id) if event.intent_id is not None else None,
        "order_id": str(event.order_id) if event.order_id is not None else None,
        "strategy_signal_id": (
            str(event.strategy_signal_id) if event.strategy_signal_id is not None else None
        ),
        "labels_json": json.dumps(dict(event.labels_json), sort_keys=True),
        "status": event.status,
        "created_at": event.created_at,
        "sent_at": event.sent_at,
    }


def _map_source_event(row: Mapping[str, Any]) -> ExecutionSourceEvent:
    return ExecutionSourceEvent(
        source_event_id=UUID(str(row["source_event_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        source_type=str(row["source_type"]),  # type: ignore[arg-type]
        source_event_ref=str(row["source_event_ref"]),
        source_ref_json=_json_mapping(row.get("source_ref_json")),
        strategy_signal_id=_uuid_or_none(row.get("strategy_signal_id")),
        idempotency_key_hash=str(row["idempotency_key_hash"]),
        outcome=str(row["outcome"]),  # type: ignore[arg-type]
        outcome_reason=str(row["outcome_reason"]),
        intent_id=_uuid_or_none(row.get("intent_id")),
        received_at=_datetime(row["received_at"]),
    )


def _map_intent(row: Mapping[str, Any]) -> ExecutionIntent:
    return ExecutionIntent(
        intent_id=UUID(str(row["intent_id"])),
        source_event_id=UUID(str(row["source_event_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        source_type=str(row["source_type"]),  # type: ignore[arg-type]
        strategy_signal_id=_uuid_or_none(row.get("strategy_signal_id")),
        exchange_connection_id=UUID(str(row["exchange_connection_id"])),
        market_type=str(row["market_type"]),
        instrument_key=str(row["instrument_key"]),
        side=str(row["side"]),  # type: ignore[arg-type]
        order_type=str(row["order_type"]),  # type: ignore[arg-type]
        quantity=_decimal_or_none(row.get("quantity")),
        quote_notional=_decimal_or_none(row.get("quote_notional")),
        limit_price=_decimal_or_none(row.get("limit_price")),
        status=str(row["status"]),  # type: ignore[arg-type]
        status_reason=str(row["status_reason"]),
        risk_status=str(row["risk_status"]),
        risk_reason=str(row["risk_reason"]),
        idempotency_key_hash=str(row["idempotency_key_hash"]),
        created_at=_datetime(row["created_at"]),
        dispatch_attempt_count=int(row.get("dispatch_attempt_count") or 0),
        dispatch_stream_name=_str_or_none(row.get("dispatch_stream_name")),
        dispatch_redis_message_id=_str_or_none(row.get("dispatch_redis_message_id")),
        dispatch_last_error=_str_or_none(row.get("dispatch_last_error")),
        dispatch_updated_at=(
            _datetime(row["dispatch_updated_at"])
            if row.get("dispatch_updated_at") is not None
            else None
        ),
    )


def _map_risk_audit_event(row: Mapping[str, Any]) -> ExecutionRiskAuditEvent:
    return ExecutionRiskAuditEvent(
        event_id=UUID(str(row["event_id"])),
        intent_id=UUID(str(row["intent_id"])),
        source_event_id=UUID(str(row["source_event_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        source_type=str(row["source_type"]),  # type: ignore[arg-type]
        event_type=str(row["event_type"]),
        risk_status=str(row["risk_status"]),  # type: ignore[arg-type]
        risk_reason=str(row["risk_reason"]),
        check_name=str(row["check_name"]),
        metadata_json=_json_mapping(row.get("metadata_json")),
        created_at=_datetime(row["created_at"]),
    )


def _map_notification(row: Mapping[str, Any]) -> ExecutionNotificationOutboxEvent:
    return ExecutionNotificationOutboxEvent(
        notification_id=UUID(str(row["notification_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        source_type=str(row["source_type"]),  # type: ignore[arg-type]
        event_type=str(row["event_type"]),  # type: ignore[arg-type]
        severity=str(row["severity"]),  # type: ignore[arg-type]
        reason=str(row["reason"]),
        source_event_id=_uuid_or_none(row.get("source_event_id")),
        intent_id=_uuid_or_none(row.get("intent_id")),
        order_id=_uuid_or_none(row.get("order_id")),
        strategy_signal_id=_uuid_or_none(row.get("strategy_signal_id")),
        labels_json=_json_mapping(row.get("labels_json")),
        status=str(row["status"]),  # type: ignore[arg-type]
        created_at=_datetime(row["created_at"]),
        sent_at=_datetime(row["sent_at"]) if row.get("sent_at") is not None else None,
    )


def _map_producer_link(row: Mapping[str, Any]) -> ExecutionProducerOutcomeLink:
    return ExecutionProducerOutcomeLink(
        source_event_id=UUID(str(row["source_event_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        source_type=str(row["source_type"]),  # type: ignore[arg-type]
        source_event_ref=str(row["source_event_ref"]),
        strategy_signal_id=_uuid_or_none(row.get("strategy_signal_id")),
        outcome=str(row["outcome"]),
        outcome_reason=str(row["outcome_reason"]),
        intent_id=_uuid_or_none(row.get("intent_id")),
        intent_status=_str_or_none(row.get("intent_status")),
        intent_status_reason=_str_or_none(row.get("intent_status_reason")),
        risk_status=_str_or_none(row.get("risk_status")),
        risk_reason=_str_or_none(row.get("risk_reason")),
        order_status=_str_or_none(row.get("order_status")),
        order_status_reason=_str_or_none(row.get("order_status_reason")),
        notification_event_type=_str_or_none(row.get("notification_event_type")),
        notification_reason=_str_or_none(row.get("notification_reason")),
        updated_at=_datetime(row["updated_at"]),
    )


def _json_mapping(value: object) -> Mapping[str, str]:
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, Mapping):
            return {str(key): str(item) for key, item in payload.items()}
    return {}


def _uuid_or_none(value: object) -> UUID | None:
    return UUID(str(value)) if value is not None else None


def _decimal_or_none(value: object) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value))


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
