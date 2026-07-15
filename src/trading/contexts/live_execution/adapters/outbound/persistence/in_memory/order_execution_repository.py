from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import ExchangeExecutionOrderRepository
from trading.contexts.live_execution.domain import (
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
from trading.shared_kernel.primitives import OrganizationId


class InMemoryExchangeExecutionOrderRepository(ExchangeExecutionOrderRepository):
    def __init__(self) -> None:
        self.orders: dict[UUID, ExchangeExecutionOrderRecord] = {}
        self.private_stream_sessions: dict[
            tuple[OrganizationId, UUID], ExchangePrivateStreamSession
        ] = {}
        self.order_events: list[ExecutionOrderEvent] = []
        self.fills: dict[tuple[UUID, str], ExecutionFill] = {}
        self.funding_events: dict[tuple[UUID, str], ExecutionFundingEvent] = {}
        self.reconciliation_runs: list[ExecutionReconciliationRun] = []
        self.retention_policies: dict[str, ExecutionLedgerRetentionPolicy] = {}
        self.pitr_drills: list[ExecutionLedgerPitrDrill] = []

    def get_by_intent(
        self, *, organization_id: OrganizationId, intent_id: UUID
    ) -> ExchangeExecutionOrderRecord | None:
        order = self.orders.get(intent_id)
        if order is None or order.organization_id != organization_id:
            return None
        return order

    def record_guard_rejection(
        self, *, command: ExchangeOrderCommand, reason: str
    ) -> ExchangeExecutionOrderRecord:
        existing = self.orders.get(command.intent_id)
        now = existing.created_at if existing is not None else _now_from_command(command)
        record = _base_record(command=command, status="guard_rejected", reason=reason, now=now)
        self.orders[command.intent_id] = record
        return record

    def record_claim_guard_rejection(
        self,
        *,
        command: ExchangeOrderCommand,
        claim_id: UUID,
        rejected_at: datetime,
        reason: str,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(command.intent_id)
        if (
            existing is None
            or existing.organization_id != command.organization_id
            or existing.status != "submit_pending"
            or existing.submit_claim_id != claim_id
            or existing.submit_claim_expires_at is None
            or existing.submit_claim_expires_at <= rejected_at
        ):
            return None
        rejected = replace(
            existing,
            status="guard_rejected",
            status_reason=reason,
            metadata={"guard_reason": reason},
            submit_claim_id=None,
            submit_claimed_at=None,
            submit_claim_expires_at=None,
            updated_at=rejected_at,
        )
        self.orders[command.intent_id] = rejected
        return rejected

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

    def claim_submit(
        self,
        *,
        command: ExchangeOrderCommand,
        claim_id: UUID,
        claimed_at: datetime,
        expires_at: datetime,
        submit_guard_audit_event_id: UUID,
        mainnet_approval_id: UUID | None,
    ) -> ExecutionSubmitClaim:
        existing = self.orders.get(command.intent_id)
        if existing is not None:
            if existing.exchange_order_id is not None:
                return ExecutionSubmitClaim(
                    order=existing,
                    claim_id=claim_id,
                    acquired=False,
                    reason="order_already_processed",
                )
            if existing.status in {"unknown", "submit_pending"}:
                return ExecutionSubmitClaim(
                    order=existing,
                    claim_id=claim_id,
                    acquired=False,
                    reason=(
                        "submission_in_flight"
                        if existing.status == "submit_pending"
                        and existing.submit_claim_expires_at is not None
                        and existing.submit_claim_expires_at > claimed_at
                        else "unknown_state_reconciliation_required"
                    ),
                )
        base = _base_record(
            command=command,
            status="submit_pending",
            reason="submit_claim_acquired",
            now=claimed_at,
        )
        claimed = replace(
            base if existing is None else existing,
            status="submit_pending",
            status_reason="submit_claim_acquired",
            submit_claim_id=claim_id,
            submit_claimed_at=claimed_at,
            submit_claim_expires_at=expires_at,
            submit_guard_audit_event_id=submit_guard_audit_event_id,
            mainnet_approval_id=mainnet_approval_id,
            updated_at=claimed_at,
        )
        self.orders[command.intent_id] = claimed
        return ExecutionSubmitClaim(
            order=claimed,
            claim_id=claim_id,
            acquired=True,
            reason="submit_claim_acquired",
        )

    def record_submit_result(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        finalized_at: datetime,
        result: ExchangeOrderSubmitResult,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if (
            existing is None
            or existing.organization_id != organization_id
            or existing.submit_claim_id != claim_id
            or existing.status != "submit_pending"
            or existing.submit_claim_expires_at is None
            or existing.submit_claim_expires_at <= finalized_at
        ):
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
            submit_claim_id=None,
            submit_claimed_at=None,
            submit_claim_expires_at=None,
        )
        self.orders[intent_id] = updated
        return updated

    def renew_submit_claim(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        renewed_at: datetime,
        expires_at: datetime,
    ) -> bool:
        existing = self.orders.get(intent_id)
        if (
            existing is None
            or existing.organization_id != organization_id
            or existing.status != "submit_pending"
            or existing.submit_claim_id != claim_id
            or existing.submit_claim_expires_at is None
            or existing.submit_claim_expires_at <= renewed_at
            or expires_at <= renewed_at
        ):
            return False
        self.orders[intent_id] = replace(
            existing,
            submit_claimed_at=renewed_at,
            submit_claim_expires_at=expires_at,
            updated_at=renewed_at,
        )
        return True

    def record_status_result(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        result: ExchangeOrderStatusResult,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if existing is None or existing.organization_id != organization_id:
            return None
        updated = replace(
            existing,
            exchange_order_id=result.exchange_order_id or None,
            status="status_checked",
            status_reason=result.exchange_status,
            last_checked_at=result.checked_at,
            latency_ms=result.latency_ms,
            metadata=dict(result.metadata),
            updated_at=result.checked_at,
            submit_claim_id=None,
            submit_claimed_at=None,
            submit_claim_expires_at=None,
        )
        self.orders[intent_id] = updated
        return updated

    def record_cancel_result(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        result: ExchangeOrderCancelResult,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if existing is None or existing.organization_id != organization_id:
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
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        occurred_at: datetime,
        reason: str,
    ) -> ExchangeExecutionOrderRecord | None:
        existing = self.orders.get(intent_id)
        if (
            existing is None
            or existing.organization_id != organization_id
            or existing.submit_claim_id != claim_id
            or existing.status != "submit_pending"
            or existing.submit_claim_expires_at is None
            or existing.submit_claim_expires_at <= occurred_at
        ):
            return None
        updated = replace(
            existing,
            status="unknown" if "unknown_state" in reason else "adapter_error",
            status_reason=reason,
            submit_claim_id=None,
            submit_claimed_at=None,
            submit_claim_expires_at=None,
            updated_at=occurred_at,
        )
        self.orders[intent_id] = updated
        return updated

    def record_private_stream_session(
        self,
        *,
        organization_id: OrganizationId,
        connection_id: UUID,
        session: ExchangePrivateStreamSession,
    ) -> ExchangePrivateStreamSession:
        if session.organization_id != organization_id:
            raise ValueError("private stream organization mismatch")
        self.private_stream_sessions[(organization_id, connection_id)] = session
        return session

    def record_order_event(self, *, event: ExecutionOrderEvent) -> ExecutionOrderEvent:
        existing = next(
            (
                item
                for item in self.order_events
                if item.order_id == event.order_id
                and item.event_type == event.event_type
                and item.provider_event_id == event.provider_event_id
            ),
            None,
        )
        if existing is not None:
            return existing
        self.order_events.append(event)
        return event

    def record_fill(
        self,
        *,
        order: ExchangeExecutionOrderRecord,
        fill: ExecutionFillFact,
    ) -> ExecutionFill:
        key = (order.order_id, fill.provider_trade_id)
        existing = self.fills.get(key)
        if existing is not None:
            return existing
        record = ExecutionFill(
            fill_id=uuid4(),
            order_id=order.order_id,
            intent_id=order.intent_id,
            organization_id=order.organization_id,
            owner_user_id=order.owner_user_id,
            provider_trade_id=fill.provider_trade_id,
            price=fill.price,
            quantity=fill.quantity,
            fee_amount=fill.fee_amount,
            fee_asset=fill.fee_asset,
            filled_at=fill.filled_at,
            liquidity=fill.liquidity,
            metadata=dict(fill.metadata),
        )
        self.fills[key] = record
        return record

    def record_funding_event(
        self,
        *,
        order: ExchangeExecutionOrderRecord,
        funding_event: ExecutionFundingFact,
    ) -> ExecutionFundingEvent:
        key = (order.order_id, funding_event.provider_event_id)
        existing = self.funding_events.get(key)
        if existing is not None:
            return existing
        record = ExecutionFundingEvent(
            funding_event_id=uuid4(),
            order_id=order.order_id,
            intent_id=order.intent_id,
            organization_id=order.organization_id,
            owner_user_id=order.owner_user_id,
            provider_event_id=funding_event.provider_event_id,
            amount=funding_event.amount,
            asset=funding_event.asset,
            funding_at=funding_event.funding_at,
            reason=funding_event.reason,
            metadata=dict(funding_event.metadata),
        )
        self.funding_events[key] = record
        return record

    def record_reconciliation_run(
        self, *, run: ExecutionReconciliationRun
    ) -> ExecutionReconciliationRun:
        self.reconciliation_runs.append(run)
        return run

    def record_retention_policy(
        self, *, policy: ExecutionLedgerRetentionPolicy
    ) -> ExecutionLedgerRetentionPolicy:
        self.retention_policies[policy.policy_name] = policy
        return policy

    def record_pitr_drill(self, *, drill: ExecutionLedgerPitrDrill) -> ExecutionLedgerPitrDrill:
        self.pitr_drills.append(drill)
        return drill


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
        organization_id=command.organization_id,
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
