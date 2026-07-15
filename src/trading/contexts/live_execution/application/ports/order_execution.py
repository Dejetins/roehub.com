from __future__ import annotations

from datetime import datetime
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
    ExecutionSubmitClaim,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


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
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
    ) -> ExchangeExecutionConnection: ...


class ExchangeOrderAdapter(Protocol):
    exchange_name: str
    provider_id: str
    provider_version: str
    provider_kind: str
    revision_hash: str

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

    def get_order_status_by_client_order_id(
        self,
        *,
        command: ExchangeOrderCommand,
        client_order_id: str,
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
    def get_by_intent(
        self, *, organization_id: OrganizationId, intent_id: UUID
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_guard_rejection(
        self, *, command: ExchangeOrderCommand, reason: str
    ) -> ExchangeExecutionOrderRecord: ...

    def record_claim_guard_rejection(
        self,
        *,
        command: ExchangeOrderCommand,
        claim_id: UUID,
        rejected_at: datetime,
        reason: str,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_submit_pending(
        self, *, command: ExchangeOrderCommand
    ) -> ExchangeExecutionOrderRecord: ...

    def claim_submit(
        self,
        *,
        command: ExchangeOrderCommand,
        claim_id: UUID,
        claimed_at: datetime,
        expires_at: datetime,
        submit_guard_audit_event_id: UUID,
        mainnet_approval_id: UUID | None,
    ) -> ExecutionSubmitClaim: ...

    def record_submit_result(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        finalized_at: datetime,
        result: ExchangeOrderSubmitResult,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def renew_submit_claim(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        renewed_at: datetime,
        expires_at: datetime,
    ) -> bool: ...

    def record_status_result(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        result: ExchangeOrderStatusResult,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_cancel_result(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        result: ExchangeOrderCancelResult,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_adapter_error(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        occurred_at: datetime,
        reason: str,
    ) -> ExchangeExecutionOrderRecord | None: ...

    def record_private_stream_session(
        self,
        *,
        organization_id: OrganizationId,
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
