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
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresGateway,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class PostgresExchangeExecutionOrderRepository(ExchangeExecutionOrderRepository):
    def __init__(self, *, gateway: StrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresExchangeExecutionOrderRepository requires gateway")
        self._gateway = gateway

    def get_by_intent(
        self, *, organization_id: OrganizationId, intent_id: UUID
    ) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query=(
                "SELECT * FROM execution_orders "
                "WHERE organization_id = %(organization_id)s "
                "AND intent_id = %(intent_id)s"
            ),
            parameters={
                "organization_id": str(organization_id),
                "intent_id": str(intent_id),
            },
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

    def record_claim_guard_rejection(
        self,
        *,
        command: ExchangeOrderCommand,
        claim_id: UUID,
        rejected_at: datetime,
        reason: str,
    ) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET status = 'guard_rejected',
                status_reason = %(reason)s,
                metadata_json = %(metadata_json)s::jsonb,
                submit_claim_id = NULL,
                submit_claimed_at = NULL,
                submit_claim_expires_at = NULL,
                updated_at = %(rejected_at)s
            WHERE organization_id = %(organization_id)s
              AND intent_id = %(intent_id)s
              AND status = 'submit_pending'
              AND submit_claim_id = %(claim_id)s
              AND submit_claim_expires_at > %(rejected_at)s
            RETURNING *
            """,
            parameters={
                "organization_id": str(command.organization_id),
                "intent_id": str(command.intent_id),
                "claim_id": str(claim_id),
                "reason": reason,
                "metadata_json": _metadata_json({"guard_reason": reason}),
                "rejected_at": rejected_at,
            },
        )
        return _map_order(row) if row is not None else None

    def record_submit_pending(
        self, *, command: ExchangeOrderCommand
    ) -> ExchangeExecutionOrderRecord:
        return self._insert_or_update_base(
            command=command,
            status="submit_pending",
            reason="submit_pending",
            metadata={},
        )

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
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_orders
            (
                order_id, intent_id, organization_id, owner_user_id,
                exchange_connection_id, exchange_name, environment, market_type,
                instrument_key, side, order_type, quantity, quote_notional,
                limit_price, client_order_id, exchange_order_id, status,
                status_reason, adapter_attempt_count, metadata_json, created_at,
                updated_at, submit_claim_id, submit_claimed_at,
                submit_claim_expires_at, submit_guard_audit_event_id,
                mainnet_approval_id
            )
            VALUES
            (
                %(order_id)s, %(intent_id)s, %(organization_id)s, %(owner_user_id)s,
                %(exchange_connection_id)s, %(exchange_name)s, %(environment)s,
                %(market_type)s, %(instrument_key)s, %(side)s, %(order_type)s,
                %(quantity)s, %(quote_notional)s, %(limit_price)s, %(client_order_id)s,
                NULL, 'submit_pending', 'submit_claim_acquired', 0, '{}'::jsonb,
                %(claimed_at)s, %(claimed_at)s, %(claim_id)s, %(claimed_at)s,
                %(expires_at)s, %(submit_guard_audit_event_id)s,
                %(mainnet_approval_id)s
            )
            ON CONFLICT (organization_id, intent_id) DO UPDATE
            SET status = 'submit_pending',
                status_reason = 'submit_claim_acquired',
                submit_claim_id = EXCLUDED.submit_claim_id,
                submit_claimed_at = EXCLUDED.submit_claimed_at,
                submit_claim_expires_at = EXCLUDED.submit_claim_expires_at,
                submit_guard_audit_event_id = EXCLUDED.submit_guard_audit_event_id,
                mainnet_approval_id = EXCLUDED.mainnet_approval_id,
                updated_at = EXCLUDED.updated_at
            WHERE execution_orders.exchange_order_id IS NULL
              AND execution_orders.status NOT IN ('submit_pending', 'unknown')
            RETURNING *
            """,
            parameters={
                "order_id": str(uuid4()),
                "intent_id": str(command.intent_id),
                "organization_id": str(command.organization_id),
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
                "claim_id": str(claim_id),
                "claimed_at": claimed_at,
                "expires_at": expires_at,
                "submit_guard_audit_event_id": str(submit_guard_audit_event_id),
                "mainnet_approval_id": (
                    str(mainnet_approval_id) if mainnet_approval_id is not None else None
                ),
            },
        )
        if row is not None:
            return ExecutionSubmitClaim(
                order=_map_order(row),
                claim_id=claim_id,
                acquired=True,
                reason="submit_claim_acquired",
            )
        existing = self.get_by_intent(
            organization_id=command.organization_id,
            intent_id=command.intent_id,
        )
        if existing is None:
            raise RuntimeError("execution submit claim returned no order")
        reason = (
            "order_already_processed"
            if existing.exchange_order_id is not None
            else "submission_in_flight"
            if existing.status == "submit_pending"
            and existing.submit_claim_expires_at is not None
            and existing.submit_claim_expires_at > claimed_at
            else "unknown_state_reconciliation_required"
        )
        return ExecutionSubmitClaim(
            order=existing,
            claim_id=claim_id,
            acquired=False,
            reason=reason,
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
                submit_claim_id = NULL,
                submit_claimed_at = NULL,
                submit_claim_expires_at = NULL,
                updated_at = %(updated_at)s
            WHERE organization_id = %(organization_id)s
              AND intent_id = %(intent_id)s
              AND status = 'submit_pending'
              AND submit_claim_id = %(claim_id)s
              AND submit_claim_expires_at > %(finalized_at)s
            RETURNING *
            """,
            parameters={
                "organization_id": str(organization_id),
                "intent_id": str(intent_id),
                "claim_id": str(claim_id),
                "finalized_at": finalized_at,
                "exchange_order_id": result.exchange_order_id,
                "status_reason": result.exchange_status,
                "submitted_at": result.submitted_at,
                "latency_ms": result.latency_ms,
                "metadata_json": _metadata_json(result.metadata),
                "updated_at": result.submitted_at,
            },
        )
        return _map_order(row) if row is not None else None

    def renew_submit_claim(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        renewed_at: datetime,
        expires_at: datetime,
    ) -> bool:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET submit_claimed_at = %(renewed_at)s,
                submit_claim_expires_at = %(expires_at)s,
                updated_at = %(renewed_at)s
            WHERE organization_id = %(organization_id)s
              AND intent_id = %(intent_id)s
              AND status = 'submit_pending'
              AND submit_claim_id = %(claim_id)s
              AND submit_claim_expires_at > %(renewed_at)s
              AND %(expires_at)s > %(renewed_at)s
            RETURNING order_id
            """,
            parameters={
                "organization_id": str(organization_id),
                "intent_id": str(intent_id),
                "claim_id": str(claim_id),
                "renewed_at": renewed_at,
                "expires_at": expires_at,
            },
        )
        return row is not None

    def record_status_result(
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        result: ExchangeOrderStatusResult,
    ) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET exchange_order_id = NULLIF(%(exchange_order_id)s, ''),
                status = 'status_checked',
                status_reason = %(status_reason)s,
                last_checked_at = %(checked_at)s,
                latency_ms = %(latency_ms)s,
                metadata_json = %(metadata_json)s::jsonb,
                submit_claim_id = NULL,
                submit_claimed_at = NULL,
                submit_claim_expires_at = NULL,
                updated_at = %(updated_at)s
            WHERE organization_id = %(organization_id)s
              AND intent_id = %(intent_id)s
            RETURNING *
            """,
            parameters={
                "organization_id": str(organization_id),
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
        organization_id: OrganizationId,
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
            WHERE organization_id = %(organization_id)s
              AND intent_id = %(intent_id)s
            RETURNING *
            """,
            parameters={
                "organization_id": str(organization_id),
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
        self,
        *,
        organization_id: OrganizationId,
        intent_id: UUID,
        claim_id: UUID,
        occurred_at: datetime,
        reason: str,
    ) -> ExchangeExecutionOrderRecord | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE execution_orders
            SET status = CASE
                    WHEN %(reason)s LIKE '%%unknown_state%%' THEN 'unknown'
                    ELSE 'adapter_error'
                END,
                status_reason = %(reason)s,
                submit_claim_id = NULL,
                submit_claimed_at = NULL,
                submit_claim_expires_at = NULL,
                updated_at = %(updated_at)s
            WHERE organization_id = %(organization_id)s
              AND intent_id = %(intent_id)s
              AND status = 'submit_pending'
              AND submit_claim_id = %(claim_id)s
              AND submit_claim_expires_at > %(updated_at)s
            RETURNING *
            """,
            parameters={
                "organization_id": str(organization_id),
                "intent_id": str(intent_id),
                "claim_id": str(claim_id),
                "reason": reason,
                "updated_at": occurred_at,
            },
        )
        return _map_order(row) if row is not None else None

    def record_private_stream_session(
        self,
        *,
        organization_id: OrganizationId,
        connection_id: UUID,
        session: ExchangePrivateStreamSession,
    ) -> ExchangePrivateStreamSession:
        if session.organization_id != organization_id:
            raise ValueError("private stream organization mismatch")
        self._gateway.fetch_one(
            query="""
            INSERT INTO exchange_private_stream_sessions
            (
                session_id, organization_id, exchange_connection_id,
                exchange_name, environment,
                market_type, status, status_reason, opened_at, keepalive_at,
                expires_at, metadata_json, updated_at
            )
            VALUES
            (
                %(session_id)s, %(organization_id)s, %(connection_id)s,
                %(exchange_name)s, %(environment)s,
                %(market_type)s, %(status)s, %(status_reason)s, %(opened_at)s,
                %(keepalive_at)s, %(expires_at)s, %(metadata_json)s::jsonb,
                %(updated_at)s
            )
            ON CONFLICT (
                organization_id, exchange_connection_id, exchange_name,
                market_type, environment
            )
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
                "organization_id": str(organization_id),
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

    def record_order_event(self, *, event: ExecutionOrderEvent) -> ExecutionOrderEvent:
        self._gateway.fetch_one(
            query="""
            INSERT INTO execution_order_events
            (
                event_id, order_id, intent_id, organization_id, owner_user_id,
                event_type, status,
                reason, provider_order_id, provider_event_id, observed_at, metadata_json
            )
            VALUES
            (
                %(event_id)s, %(order_id)s, %(intent_id)s, %(organization_id)s,
                %(owner_user_id)s,
                %(event_type)s, %(status)s, %(reason)s, %(provider_order_id)s,
                %(provider_event_id)s, %(observed_at)s, %(metadata_json)s::jsonb
            )
            ON CONFLICT (order_id, event_type, provider_event_id_key) DO NOTHING
            RETURNING event_id
            """,
            parameters={
                "event_id": str(event.event_id),
                "order_id": str(event.order_id),
                "intent_id": str(event.intent_id),
                "organization_id": str(event.organization_id),
                "owner_user_id": str(event.owner_user_id),
                "event_type": event.event_type,
                "status": event.status,
                "reason": event.reason,
                "provider_order_id": event.provider_order_id,
                "provider_event_id": event.provider_event_id,
                "observed_at": event.observed_at,
                "metadata_json": _metadata_json(event.metadata),
            },
        )
        return event

    def record_fill(
        self,
        *,
        order: ExchangeExecutionOrderRecord,
        fill: ExecutionFillFact,
    ) -> ExecutionFill:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_fills
            (
                fill_id, order_id, intent_id, organization_id, owner_user_id,
                provider_trade_id,
                price, quantity, fee_amount, fee_asset, filled_at, liquidity,
                metadata_json
            )
            VALUES
            (
                %(fill_id)s, %(order_id)s, %(intent_id)s, %(organization_id)s,
                %(owner_user_id)s,
                %(provider_trade_id)s, %(price)s, %(quantity)s, %(fee_amount)s,
                %(fee_asset)s, %(filled_at)s, %(liquidity)s, %(metadata_json)s::jsonb
            )
            ON CONFLICT (order_id, provider_trade_id) DO UPDATE
            SET price = EXCLUDED.price,
                quantity = EXCLUDED.quantity,
                fee_amount = EXCLUDED.fee_amount,
                fee_asset = EXCLUDED.fee_asset,
                filled_at = EXCLUDED.filled_at,
                liquidity = EXCLUDED.liquidity,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            parameters={
                "fill_id": str(uuid4()),
                "order_id": str(order.order_id),
                "intent_id": str(order.intent_id),
                "organization_id": str(order.organization_id),
                "owner_user_id": str(order.owner_user_id),
                "provider_trade_id": fill.provider_trade_id,
                "price": fill.price,
                "quantity": fill.quantity,
                "fee_amount": fill.fee_amount,
                "fee_asset": fill.fee_asset,
                "filled_at": fill.filled_at,
                "liquidity": fill.liquidity,
                "metadata_json": _metadata_json(fill.metadata),
            },
        )
        if row is None:
            raise RuntimeError("execution fill write returned no row")
        return _map_fill(row)

    def record_funding_event(
        self,
        *,
        order: ExchangeExecutionOrderRecord,
        funding_event: ExecutionFundingFact,
    ) -> ExecutionFundingEvent:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO execution_funding_events
            (
                funding_event_id, order_id, intent_id, organization_id, owner_user_id,
                provider_event_id, amount, asset, funding_at, reason, metadata_json
            )
            VALUES
            (
                %(funding_event_id)s, %(order_id)s, %(intent_id)s,
                %(organization_id)s, %(owner_user_id)s, %(provider_event_id)s,
                %(amount)s, %(asset)s,
                %(funding_at)s, %(reason)s, %(metadata_json)s::jsonb
            )
            ON CONFLICT (order_id, provider_event_id) DO UPDATE
            SET amount = EXCLUDED.amount,
                asset = EXCLUDED.asset,
                funding_at = EXCLUDED.funding_at,
                reason = EXCLUDED.reason,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            parameters={
                "funding_event_id": str(uuid4()),
                "order_id": str(order.order_id),
                "intent_id": str(order.intent_id),
                "organization_id": str(order.organization_id),
                "owner_user_id": str(order.owner_user_id),
                "provider_event_id": funding_event.provider_event_id,
                "amount": funding_event.amount,
                "asset": funding_event.asset,
                "funding_at": funding_event.funding_at,
                "reason": funding_event.reason,
                "metadata_json": _metadata_json(funding_event.metadata),
            },
        )
        if row is None:
            raise RuntimeError("execution funding event write returned no row")
        return _map_funding_event(row)

    def record_reconciliation_run(
        self, *, run: ExecutionReconciliationRun
    ) -> ExecutionReconciliationRun:
        self._gateway.fetch_one(
            query="""
            INSERT INTO execution_reconciliation_runs
            (
                reconciliation_run_id, order_id, intent_id, organization_id,
                owner_user_id,
                exchange_name, environment, status, reason, local_status,
                provider_status, fill_count, funding_event_count, started_at,
                completed_at, metadata_json
            )
            VALUES
            (
                %(reconciliation_run_id)s, %(order_id)s, %(intent_id)s,
                %(organization_id)s, %(owner_user_id)s, %(exchange_name)s,
                %(environment)s, %(status)s,
                %(reason)s, %(local_status)s, %(provider_status)s, %(fill_count)s,
                %(funding_event_count)s, %(started_at)s, %(completed_at)s,
                %(metadata_json)s::jsonb
            )
            RETURNING reconciliation_run_id
            """,
            parameters={
                "reconciliation_run_id": str(run.reconciliation_run_id),
                "order_id": str(run.order_id),
                "intent_id": str(run.intent_id),
                "organization_id": str(run.organization_id),
                "owner_user_id": str(run.owner_user_id),
                "exchange_name": run.exchange_name,
                "environment": run.environment,
                "status": run.status,
                "reason": run.reason,
                "local_status": run.local_status,
                "provider_status": run.provider_status,
                "fill_count": run.fill_count,
                "funding_event_count": run.funding_event_count,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "metadata_json": _metadata_json(run.metadata),
            },
        )
        return run

    def record_retention_policy(
        self, *, policy: ExecutionLedgerRetentionPolicy
    ) -> ExecutionLedgerRetentionPolicy:
        self._gateway.fetch_one(
            query="""
            INSERT INTO execution_ledger_retention_policies
            (
                policy_name, table_name, partition_key, retention_days,
                archive_before_purge, pitr_required, checked_at, status, reason
            )
            VALUES
            (
                %(policy_name)s, %(table_name)s, %(partition_key)s,
                %(retention_days)s, %(archive_before_purge)s, %(pitr_required)s,
                %(checked_at)s, %(status)s, %(reason)s
            )
            ON CONFLICT (policy_name) DO UPDATE
            SET table_name = EXCLUDED.table_name,
                partition_key = EXCLUDED.partition_key,
                retention_days = EXCLUDED.retention_days,
                archive_before_purge = EXCLUDED.archive_before_purge,
                pitr_required = EXCLUDED.pitr_required,
                checked_at = EXCLUDED.checked_at,
                status = EXCLUDED.status,
                reason = EXCLUDED.reason
            RETURNING policy_name
            """,
            parameters={
                "policy_name": policy.policy_name,
                "table_name": policy.table_name,
                "partition_key": policy.partition_key,
                "retention_days": policy.retention_days,
                "archive_before_purge": policy.archive_before_purge,
                "pitr_required": policy.pitr_required,
                "checked_at": policy.checked_at,
                "status": policy.status,
                "reason": policy.reason,
            },
        )
        return policy

    def record_pitr_drill(self, *, drill: ExecutionLedgerPitrDrill) -> ExecutionLedgerPitrDrill:
        self._gateway.fetch_one(
            query="""
            INSERT INTO execution_ledger_pitr_drills
            (
                drill_id, target_time, status, reason, verified_at,
                row_counts_json, metadata_json
            )
            VALUES
            (
                %(drill_id)s, %(target_time)s, %(status)s, %(reason)s,
                %(verified_at)s, %(row_counts_json)s::jsonb, %(metadata_json)s::jsonb
            )
            ON CONFLICT (drill_id) DO NOTHING
            RETURNING drill_id
            """,
            parameters={
                "drill_id": str(drill.drill_id),
                "target_time": drill.target_time,
                "status": drill.status,
                "reason": drill.reason,
                "verified_at": drill.verified_at,
                "row_counts_json": json.dumps(dict(drill.row_counts), sort_keys=True),
                "metadata_json": _metadata_json(drill.metadata),
            },
        )
        return drill

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
                order_id, intent_id, organization_id, owner_user_id,
                exchange_connection_id,
                exchange_name, environment, market_type, instrument_key, side,
                order_type, quantity, quote_notional, limit_price, client_order_id,
                exchange_order_id, status, status_reason, submitted_at,
                cancel_requested_at, cancelled_at, last_checked_at,
                adapter_attempt_count, latency_ms, metadata_json, created_at, updated_at
            )
            VALUES
            (
                %(order_id)s, %(intent_id)s, %(organization_id)s, %(owner_user_id)s,
                %(exchange_connection_id)s, %(exchange_name)s, %(environment)s,
                %(market_type)s, %(instrument_key)s, %(side)s, %(order_type)s,
                %(quantity)s, %(quote_notional)s, %(limit_price)s, %(client_order_id)s,
                NULL, %(status)s, %(status_reason)s, NULL, NULL, NULL, NULL,
                0, NULL, %(metadata_json)s::jsonb, %(created_at)s, %(updated_at)s
            )
            ON CONFLICT (organization_id, intent_id) DO UPDATE
            SET status = EXCLUDED.status,
                status_reason = EXCLUDED.status_reason,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = EXCLUDED.updated_at
            RETURNING *
            """,
            parameters={
                "order_id": str(uuid4()),
                "intent_id": str(command.intent_id),
                "organization_id": str(command.organization_id),
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
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
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
        submit_claim_id=(
            UUID(str(row["submit_claim_id"]))
            if row.get("submit_claim_id") is not None
            else None
        ),
        submit_claimed_at=_datetime_or_none(row.get("submit_claimed_at")),
        submit_claim_expires_at=_datetime_or_none(row.get("submit_claim_expires_at")),
        submit_guard_audit_event_id=(
            UUID(str(row["submit_guard_audit_event_id"]))
            if row.get("submit_guard_audit_event_id") is not None
            else None
        ),
        mainnet_approval_id=(
            UUID(str(row["mainnet_approval_id"]))
            if row.get("mainnet_approval_id") is not None
            else None
        ),
    )


def _map_fill(row: Mapping[str, Any]) -> ExecutionFill:
    return ExecutionFill(
        fill_id=UUID(str(row["fill_id"])),
        order_id=UUID(str(row["order_id"])),
        intent_id=UUID(str(row["intent_id"])),
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        provider_trade_id=str(row["provider_trade_id"]),
        price=Decimal(str(row["price"])),
        quantity=Decimal(str(row["quantity"])),
        fee_amount=Decimal(str(row["fee_amount"])),
        fee_asset=str(row["fee_asset"]),
        filled_at=_datetime(row["filled_at"]),
        liquidity=_str_or_none(row.get("liquidity")),
        metadata=_metadata_mapping(row.get("metadata_json")),
    )


def _map_funding_event(row: Mapping[str, Any]) -> ExecutionFundingEvent:
    return ExecutionFundingEvent(
        funding_event_id=UUID(str(row["funding_event_id"])),
        order_id=UUID(str(row["order_id"])),
        intent_id=UUID(str(row["intent_id"])),
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        provider_event_id=str(row["provider_event_id"]),
        amount=Decimal(str(row["amount"])),
        asset=str(row["asset"]),
        funding_at=_datetime(row["funding_at"]),
        reason=str(row["reason"]),
        metadata=_metadata_mapping(row.get("metadata_json")),
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
