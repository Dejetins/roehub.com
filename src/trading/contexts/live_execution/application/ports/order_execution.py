from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionOrderRecord,
    ExchangeOrderCancelResult,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExchangeOrderSubmitResult,
    ExchangePrivateStreamSession,
    ExecutionFill,
    ExecutionFillFact,
    ExecutionFundingEvent,
    ExecutionFundingFact,
    ExecutionLedgerPitrDrill,
    ExecutionLedgerRetentionPolicy,
    ExecutionOrderEvent,
    ExecutionReconciliationRun,
)
from trading.shared_kernel.primitives import UserId


class ExchangeExecutionCredentialUnavailable(RuntimeError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ExchangeOrderAdapterError(RuntimeError):
    def __init__(self, *, reason: str, unknown_state: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.unknown_state = unknown_state


class ExchangeExecutionCredentialResolver(Protocol):
    def resolve(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeExecutionConnection: ...


class ExchangeOrderAdapter(Protocol):
    exchange_name: str

    def server_time_ms(self) -> int: ...

    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult: ...

    def get_order_status(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult: ...

    def cancel_order(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderCancelResult: ...

    def ensure_private_stream_session(
        self,
        *,
        connection: ExchangeExecutionConnection,
    ) -> ExchangePrivateStreamSession: ...


class ExchangeExecutionOrderRepository(Protocol):
    def get_by_intent(self, *, intent_id: UUID) -> ExchangeExecutionOrderRecord | None: ...

    def record_guard_rejection(
        self, *, command: ExchangeOrderCommand, reason: str
    ) -> ExchangeExecutionOrderRecord: ...

    def record_submit_pending(
        self, *, command: ExchangeOrderCommand
    ) -> ExchangeExecutionOrderRecord: ...

    def record_submit_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderSubmitResult,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_status_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderStatusResult,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_cancel_result(
        self,
        *,
        intent_id: UUID,
        result: ExchangeOrderCancelResult,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_adapter_error(
        self, *, intent_id: UUID, reason: str
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_private_stream_session(
        self,
        *,
        connection_id: UUID,
        session: ExchangePrivateStreamSession,
    ) -> ExchangePrivateStreamSession: ...

    def record_order_event(self, *, event: ExecutionOrderEvent) -> ExecutionOrderEvent: ...

    def record_fill(
        self,
        *,
        order: ExchangeExecutionOrderRecord,
        fill: ExecutionFillFact,
    ) -> ExecutionFill: ...

    def record_funding_event(
        self,
        *,
        order: ExchangeExecutionOrderRecord,
        funding_event: ExecutionFundingFact,
    ) -> ExecutionFundingEvent: ...

    def record_reconciliation_run(
        self, *, run: ExecutionReconciliationRun
    ) -> ExecutionReconciliationRun: ...

    def record_retention_policy(
        self, *, policy: ExecutionLedgerRetentionPolicy
    ) -> ExecutionLedgerRetentionPolicy: ...

    def record_pitr_drill(self, *, drill: ExecutionLedgerPitrDrill) -> ExecutionLedgerPitrDrill: ...
