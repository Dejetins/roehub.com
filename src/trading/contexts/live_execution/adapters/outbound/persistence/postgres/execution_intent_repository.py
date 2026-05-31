from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.live_execution.application.ports import ExecutionIntentRepository
from trading.contexts.live_execution.domain import (
    ExecutionIntent,
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
                idempotency_key_hash, created_at
            )
            VALUES
            (
                %(intent_id)s, %(source_event_id)s, %(owner_user_id)s,
                %(source_type)s, %(strategy_signal_id)s, %(exchange_connection_id)s,
                %(market_type)s, %(instrument_key)s, %(side)s, %(order_type)s,
                %(quantity)s, %(quote_notional)s, %(limit_price)s, %(status)s,
                %(status_reason)s, %(risk_status)s, %(risk_reason)s,
                %(idempotency_key_hash)s, %(created_at)s
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
