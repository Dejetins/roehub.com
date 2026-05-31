from __future__ import annotations

from dataclasses import replace
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


class InMemoryExchangeExecutionOrderRepository(ExchangeExecutionOrderRepository):
    def __init__(self) -> None:
        self.orders: dict[UUID, ExchangeExecutionOrderRecord] = {}
        self.private_stream_sessions: dict[UUID, ExchangePrivateStreamSession] = {}

    def get_by_intent(self, *, intent_id: UUID) -> ExchangeExecutionOrderRecord | None:
        return self.orders.get(intent_id)

    def record_guard_rejection(
        self, *, command: ExchangeOrderCommand, reason: str
    ) -> ExchangeExecutionOrderRecord:
        existing = self.orders.get(command.intent_id)
        now = existing.created_at if existing is not None else _now_from_command(command)
        record = _base_record(command=command, status="guard_rejected", reason=reason, now=now)
        self.orders[command.intent_id] = record
        return record

    def record_submit_pending(
        self, *, command: ExchangeOrderCommand
    ) -> ExchangeExecutionOrderRecord:
        existing = self.orders.get(command.intent_id)
        if existing is not None:
            return existing
        record = _base_record(
            command=command,
            status="submit_pending",
            reason="submit_pending",
            now=_now_from_command(command),
        )
        self.orders[command.intent_id] = record
        return record

    def record_submit_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderSubmitResult,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if existing is None:
            return None
        updated = replace(
            existing,
            exchange_order_id=result.exchange_order_id,
            status="submitted",
            status_reason=result.exchange_status,
            submitted_at=result.submitted_at,
            adapter_attempt_count=existing.adapter_attempt_count + 1,
            latency_ms=result.latency_ms,
            metadata=dict(result.metadata),
            updated_at=result.submitted_at,
        )
        self.orders[intent_id] = updated
        return updated

    def record_status_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderStatusResult,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if existing is None:
            return None
        updated = replace(
            existing,
            exchange_order_id=result.exchange_order_id,
            status="status_checked",
            status_reason=result.exchange_status,
            last_checked_at=result.checked_at,
            latency_ms=result.latency_ms,
            metadata=dict(result.metadata),
            updated_at=result.checked_at,
        )
        self.orders[intent_id] = updated
        return updated

    def record_cancel_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderCancelResult,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if existing is None:
            return None
        updated = replace(
            existing,
            exchange_order_id=result.exchange_order_id,
            status="cancelled",
            status_reason=result.exchange_status,
            cancel_requested_at=result.cancelled_at,
            cancelled_at=result.cancelled_at,
            latency_ms=result.latency_ms,
            metadata=dict(result.metadata),
            updated_at=result.cancelled_at,
        )
        self.orders[intent_id] = updated
        return updated

    def record_adapter_error(
        self, *, intent_id: UUID, reason: str
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if existing is None:
            return None
        updated = replace(existing, status="adapter_error", status_reason=reason)
        self.orders[intent_id] = updated
        return updated

    def record_private_stream_session(
        self,
        *,
        connection_id: UUID,
        session: ExchangePrivateStreamSession,
    ) -> ExchangePrivateStreamSession:
        self.private_stream_sessions[connection_id] = session
        return session


def _base_record(
    *,
    command: ExchangeOrderCommand,
    status: str,
    reason: str,
    now: object,
) -> ExchangeExecutionOrderRecord:
    return ExchangeExecutionOrderRecord(
        order_id=uuid4(),
        intent_id=command.intent_id,
        owner_user_id=command.owner_user_id,
        exchange_connection_id=command.exchange_connection_id,
        exchange_name=command.exchange_name,
        environment=command.environment,
        market_type=command.market_type,
        instrument_key=command.instrument_key,
        side=command.side,
        order_type=command.order_type,
        quantity=command.quantity,
        quote_notional=command.quote_notional,
        limit_price=command.limit_price,
        client_order_id=command.client_order_id,
        exchange_order_id=None,
        status=status,  # type: ignore[arg-type]
        status_reason=reason,
        submitted_at=None,
        cancel_requested_at=None,
        cancelled_at=None,
        last_checked_at=None,
        adapter_attempt_count=0,
        latency_ms=None,
        metadata={},
        created_at=now,  # type: ignore[arg-type]
        updated_at=now,  # type: ignore[arg-type]
    )


def _now_from_command(command: ExchangeOrderCommand) -> object:
    _ = command
    from datetime import UTC, datetime

    return datetime.now(tz=UTC)
