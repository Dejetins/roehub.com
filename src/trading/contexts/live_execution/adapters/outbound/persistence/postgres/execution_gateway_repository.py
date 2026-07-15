from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import ExecutionGatewayPolicyRepository
from trading.contexts.live_execution.domain import (
    ExecutionAccountSafetyState,
    ExecutionCurrentRiskSnapshot,
    ExecutionGatewayAuditEvent,
    ExecutionKillSwitchState,
    ExecutionMainnetApproval,
    ExecutionPolicyPrincipal,
    ExecutionProviderRegistration,
    ExecutionSubmitGuardDecision,
    ExecutionSubmitGuardQuery,
    evaluate_execution_submit_guard,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres.gateway import (
    StrategyPostgresTransaction,
    TransactionalStrategyPostgresGateway,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_SENSITIVE_PARTS = frozenset(
    {"secret", "token", "password", "credential", "cookie", "authorization", "payload"}
)


class PostgresExecutionGatewayPolicyRepository(ExecutionGatewayPolicyRepository):
    def __init__(self, *, gateway: TransactionalStrategyPostgresGateway) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresExecutionGatewayPolicyRepository requires gateway")
        self._gateway = gateway

    def register_provider(
        self,
        *,
        provider: ExecutionProviderRegistration,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionProviderRegistration:
        with self._gateway.transaction() as transaction:
            self._require_authority(
                transaction=transaction,
                principal=principal,
                evaluated_at=provider.updated_at,
                organization_id=None,
                roles=(),
            )
            existing = transaction.fetch_one(
                query=(
                    "SELECT * FROM execution_provider_allowlist "
                    "WHERE provider_id = %(provider_id)s"
                ),
                parameters={"provider_id": provider.provider_id},
            )
            if existing is not None and _map_provider(existing) == provider:
                return provider
            self._record_audit(transaction=transaction, audit=audit)
            row = transaction.fetch_one(
                query="""
            INSERT INTO execution_provider_allowlist
            (
                provider_id, provider_version, provider_kind, exchange_name,
                revision_hash, order_submit_capability, enabled,
                approved_by_user_id, updated_at, audit_event_id
            )
            VALUES
            (
                %(provider_id)s, %(provider_version)s, %(provider_kind)s,
                %(exchange_name)s, %(revision_hash)s, %(order_submit_capability)s,
                %(enabled)s, %(approved_by_user_id)s, %(updated_at)s,
                %(audit_event_id)s
            )
            ON CONFLICT (provider_id) DO UPDATE
            SET provider_version = EXCLUDED.provider_version,
                provider_kind = EXCLUDED.provider_kind,
                exchange_name = EXCLUDED.exchange_name,
                revision_hash = EXCLUDED.revision_hash,
                order_submit_capability = EXCLUDED.order_submit_capability,
                enabled = EXCLUDED.enabled,
                approved_by_user_id = EXCLUDED.approved_by_user_id,
                updated_at = EXCLUDED.updated_at,
                audit_event_id = EXCLUDED.audit_event_id
            RETURNING *
            """,
                parameters={
                "provider_id": provider.provider_id,
                "provider_version": provider.provider_version,
                "provider_kind": provider.provider_kind,
                "exchange_name": provider.exchange_name,
                "revision_hash": provider.revision_hash,
                "order_submit_capability": provider.order_submit_capability,
                "enabled": provider.enabled,
                "approved_by_user_id": str(provider.approved_by_user_id),
                "updated_at": provider.updated_at,
                "audit_event_id": str(audit.event_id),
                },
            )
            if row is None:
                raise RuntimeError("execution provider registration returned no row")
        return _map_provider(row)

    def set_account_safety_state(
        self,
        *,
        state: ExecutionAccountSafetyState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionAccountSafetyState:
        with self._gateway.transaction() as transaction:
            self._require_authority(
                transaction=transaction,
                principal=principal,
                evaluated_at=state.updated_at,
                organization_id=state.organization_id,
                roles=("owner", "admin"),
            )
            existing = transaction.fetch_one(
                query="""
                SELECT * FROM execution_account_safety_state
                WHERE organization_id = %(organization_id)s
                  AND exchange_connection_id = %(exchange_connection_id)s
                """,
                parameters={
                    "organization_id": str(state.organization_id),
                    "exchange_connection_id": str(state.exchange_connection_id),
                },
            )
            if existing is not None and _map_safety(existing) == state:
                return state
            self._record_audit(transaction=transaction, audit=audit)
            row = transaction.fetch_one(
                query="""
            INSERT INTO execution_account_safety_state
            (
                organization_id, owner_user_id, exchange_connection_id, mode,
                risk_revision_hash, account_revision_hash, secret_reference_hash,
                risk_allows_submit, max_order_notional, daily_notional_limit,
                max_account_exposure_notional, risk_valid_until,
                updated_by_user_id, updated_at, audit_event_id
            )
            VALUES
            (
                %(organization_id)s, %(owner_user_id)s, %(exchange_connection_id)s,
                %(mode)s, %(risk_revision_hash)s, %(account_revision_hash)s,
                %(secret_reference_hash)s, %(risk_allows_submit)s,
                %(max_order_notional)s, %(daily_notional_limit)s,
                %(max_account_exposure_notional)s, %(risk_valid_until)s,
                %(updated_by_user_id)s, %(updated_at)s, %(audit_event_id)s
            )
            ON CONFLICT (organization_id, exchange_connection_id) DO UPDATE
            SET owner_user_id = EXCLUDED.owner_user_id,
                mode = EXCLUDED.mode,
                risk_revision_hash = EXCLUDED.risk_revision_hash,
                account_revision_hash = EXCLUDED.account_revision_hash,
                secret_reference_hash = EXCLUDED.secret_reference_hash,
                risk_allows_submit = EXCLUDED.risk_allows_submit,
                max_order_notional = EXCLUDED.max_order_notional,
                daily_notional_limit = EXCLUDED.daily_notional_limit,
                max_account_exposure_notional = EXCLUDED.max_account_exposure_notional,
                risk_valid_until = EXCLUDED.risk_valid_until,
                updated_by_user_id = EXCLUDED.updated_by_user_id,
                updated_at = EXCLUDED.updated_at,
                audit_event_id = EXCLUDED.audit_event_id
            RETURNING *
            """,
                parameters={
                "organization_id": str(state.organization_id),
                "owner_user_id": str(state.owner_user_id),
                "exchange_connection_id": str(state.exchange_connection_id),
                "mode": state.mode,
                "risk_revision_hash": state.risk_revision_hash,
                "account_revision_hash": state.account_revision_hash,
                "secret_reference_hash": state.secret_reference_hash,
                "risk_allows_submit": state.risk_allows_submit,
                "max_order_notional": state.max_order_notional,
                "daily_notional_limit": state.daily_notional_limit,
                "max_account_exposure_notional": state.max_account_exposure_notional,
                "risk_valid_until": state.risk_valid_until,
                "updated_by_user_id": str(state.updated_by_user_id),
                "updated_at": state.updated_at,
                "audit_event_id": str(audit.event_id),
                },
            )
            if row is None:
                raise RuntimeError("execution account safety write returned no row")
        return _map_safety(row)

    def set_kill_switch(
        self,
        *,
        state: ExecutionKillSwitchState,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionKillSwitchState:
        with self._gateway.transaction() as transaction:
            self._require_authority(
                transaction=transaction,
                principal=principal,
                evaluated_at=state.updated_at,
                organization_id=state.organization_id,
                roles=("owner", "admin") if state.organization_id is not None else (),
            )
            self._record_audit(transaction=transaction, audit=audit)
            row = transaction.fetch_one(
                query="""
            INSERT INTO execution_kill_switch_state
            (
                scope_type, organization_id, exchange_connection_id, active,
                reason, updated_by_user_id, updated_at, audit_event_id
            )
            VALUES
            (
                %(scope_type)s, %(organization_id)s, %(exchange_connection_id)s,
                %(active)s, %(reason)s, %(updated_by_user_id)s, %(updated_at)s,
                %(audit_event_id)s
            )
            ON CONFLICT (scope_type, organization_id, exchange_connection_id)
            DO UPDATE SET active = EXCLUDED.active,
                reason = EXCLUDED.reason,
                updated_by_user_id = EXCLUDED.updated_by_user_id,
                updated_at = EXCLUDED.updated_at,
                audit_event_id = EXCLUDED.audit_event_id
            RETURNING *
            """,
                parameters={
                "scope_type": state.scope_type,
                "organization_id": (
                    str(state.organization_id) if state.organization_id is not None else None
                ),
                "exchange_connection_id": (
                    str(state.exchange_connection_id)
                    if state.exchange_connection_id is not None
                    else None
                ),
                "active": state.active,
                "reason": state.reason,
                "updated_by_user_id": str(state.updated_by_user_id),
                "updated_at": state.updated_at,
                "audit_event_id": str(audit.event_id),
                },
            )
            if row is None:
                raise RuntimeError("execution kill switch write returned no row")
        return _map_kill_switch(row)

    def approve_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval:
        with self._gateway.transaction() as transaction:
            self._require_authority(
                transaction=transaction,
                principal=principal,
                evaluated_at=approval.approved_at,
                organization_id=approval.organization_id,
                roles=("owner",),
            )
            self._record_audit(transaction=transaction, audit=audit)
            row = transaction.fetch_one(
                query="""
            INSERT INTO execution_mainnet_approvals
            (
                approval_id, organization_id, owner_user_id,
                exchange_connection_id, exchange_name, market_type, provider_id,
                risk_revision_hash, account_revision_hash, provider_revision_hash,
                recent_auth_session_id, recent_auth_at, approved_at, expires_at,
                audit_event_id
            )
            SELECT
                %(approval_id)s, %(organization_id)s, %(owner_user_id)s,
                %(exchange_connection_id)s, %(exchange_name)s, %(market_type)s,
                %(provider_id)s, %(risk_revision_hash)s, %(account_revision_hash)s,
                %(provider_revision_hash)s, %(recent_auth_session_id)s,
                %(recent_auth_at)s, %(approved_at)s, %(expires_at)s,
                %(audit_event_id)s
            FROM identity_memberships AS membership
            JOIN identity_sessions AS session
              ON session.session_id = %(recent_auth_session_id)s
             AND session.user_id = membership.user_id
            JOIN execution_account_safety_state AS safety
              ON safety.organization_id = membership.organization_id
             AND safety.owner_user_id = membership.user_id
             AND safety.exchange_connection_id = %(exchange_connection_id)s
            JOIN execution_provider_allowlist AS provider
              ON provider.provider_id = %(provider_id)s
            WHERE membership.organization_id = %(organization_id)s
              AND membership.user_id = %(owner_user_id)s
              AND membership.role = 'owner'
              AND membership.status = 'active'
              AND session.created_at = %(recent_auth_at)s
              AND session.revoked_at IS NULL
              AND session.idle_expires_at > %(approved_at)s
              AND session.absolute_expires_at > %(approved_at)s
              AND %(approved_at)s - %(recent_auth_at)s <= INTERVAL '10 minutes'
              AND safety.mode = 'mainnet'
              AND safety.risk_revision_hash = %(risk_revision_hash)s
              AND safety.account_revision_hash = %(account_revision_hash)s
              AND provider.revision_hash = %(provider_revision_hash)s
              AND provider.enabled
              AND provider.provider_kind IN ('core', 'verified')
              AND provider.order_submit_capability
            RETURNING *
            """,
                parameters=_approval_params(approval),
            )
            if row is None:
                raise RuntimeError("mainnet approval authority check failed")
        return _map_approval(row)

    def revoke_mainnet(
        self,
        *,
        approval: ExecutionMainnetApproval,
        audit: ExecutionGatewayAuditEvent,
        principal: ExecutionPolicyPrincipal,
    ) -> ExecutionMainnetApproval:
        if approval.revoked_at is None:
            raise RuntimeError("mainnet approval revoke timestamp missing")
        with self._gateway.transaction() as transaction:
            self._require_authority(
                transaction=transaction,
                principal=principal,
                evaluated_at=approval.revoked_at,
                organization_id=approval.organization_id,
                roles=("owner",),
            )
            self._record_audit(transaction=transaction, audit=audit)
            row = transaction.fetch_one(
                query="""
            UPDATE execution_mainnet_approvals
            SET revoked_at = %(revoked_at)s,
                revocation_reason = %(revocation_reason)s,
                revocation_audit_event_id = %(audit_event_id)s
            WHERE approval_id = %(approval_id)s
              AND organization_id = %(organization_id)s
              AND revoked_at IS NULL
            RETURNING *
            """,
                parameters={
                "approval_id": str(approval.approval_id),
                "organization_id": str(approval.organization_id),
                "revoked_at": approval.revoked_at,
                "revocation_reason": approval.revocation_reason,
                "audit_event_id": str(audit.event_id),
                },
            )
            if row is None:
                raise RuntimeError("mainnet approval revoke returned no row")
        return _map_approval(row)

    def evaluate_and_record(
        self, *, query: ExecutionSubmitGuardQuery
    ) -> ExecutionSubmitGuardDecision:
        scope = {
            **_query_scope(query),
            "exchange_name": query.connection.exchange_name,
            "environment": query.connection.environment,
            "instrument_key": query.intent.instrument_key,
            "market_type": query.intent.market_type,
            "provider_id": query.adapter.provider_id,
            "evaluated_at": query.evaluated_at,
            "intent_id": str(query.intent.intent_id),
        }
        with self._gateway.serialized_transaction(
            lock_key=(
                "execution-risk:"
                f"{query.intent.organization_id}:"
                f"{query.intent.exchange_connection_id}"
            )
        ) as transaction:
            safety_row = transaction.fetch_one(
                query="""
                SELECT * FROM execution_account_safety_state
                WHERE organization_id = %(organization_id)s
                  AND owner_user_id = %(owner_user_id)s
                  AND exchange_connection_id = %(exchange_connection_id)s
                FOR UPDATE
                """,
                parameters=scope,
            )
            claim_row = None
            if query.phase == "pre_submit":
                claim_row = transaction.fetch_one(
                    query="""
                    SELECT submit_claim_id, submit_claim_expires_at
                    FROM execution_orders
                    WHERE organization_id = %(organization_id)s
                      AND intent_id = %(intent_id)s
                      AND status = 'submit_pending'
                    FOR UPDATE
                    """,
                    parameters=scope,
                )
            provider_row = transaction.fetch_one(
                query=(
                    "SELECT * FROM execution_provider_allowlist "
                    "WHERE provider_id = %(provider_id)s"
                ),
                parameters=scope,
            )
            approval_row = transaction.fetch_one(
                query="""
                SELECT approval.*
                FROM execution_mainnet_approvals AS approval
                JOIN identity_memberships AS membership
                  ON membership.organization_id = approval.organization_id
                 AND membership.user_id = approval.owner_user_id
                 AND membership.role = 'owner'
                 AND membership.status = 'active'
                JOIN identity_sessions AS session
                  ON session.session_id = approval.recent_auth_session_id
                 AND session.user_id = approval.owner_user_id
                 AND session.created_at = approval.recent_auth_at
                 AND session.revoked_at IS NULL
                 AND session.idle_expires_at > %(evaluated_at)s
                 AND session.absolute_expires_at > %(evaluated_at)s
                WHERE approval.organization_id = %(organization_id)s
                  AND approval.owner_user_id = %(owner_user_id)s
                  AND approval.exchange_connection_id = %(exchange_connection_id)s
                  AND approval.exchange_name = %(exchange_name)s
                  AND approval.market_type = %(market_type)s
                  AND approval.provider_id = %(provider_id)s
                ORDER BY approval.approved_at DESC
                LIMIT 1
                """,
                parameters=scope,
            )
            kill_row = transaction.fetch_one(
                query="""
                SELECT EXISTS (
                    SELECT 1 FROM execution_kill_switch_state
                    WHERE active
                      AND (
                        scope_type = 'installation'
                        OR (scope_type = 'organization'
                            AND organization_id = %(organization_id)s)
                        OR (scope_type = 'account'
                            AND organization_id = %(organization_id)s
                            AND exchange_connection_id = %(exchange_connection_id)s)
                      )
                ) AS active
                """,
                parameters=scope,
            )
            risk_row = transaction.fetch_one(
                query="""
                WITH latest_snapshot AS MATERIALIZED (
                    SELECT *
                    FROM exchange_account_snapshots
                    WHERE organization_id = %(organization_id)s
                      AND owner_user_id = %(owner_user_id)s
                      AND exchange_connection_id = %(exchange_connection_id)s
                      AND exchange_name = %(exchange_name)s
                      AND market_type = %(market_type)s
                      AND environment = %(environment)s
                    ORDER BY observed_at DESC
                    LIMIT 1
                ), latest_guard AS MATERIALIZED (
                    SELECT guard.*
                    FROM exchange_account_config_guard_results AS guard
                    JOIN latest_snapshot AS snapshot
                      ON snapshot.account_snapshot_id = guard.account_snapshot_id
                    WHERE guard.organization_id = %(organization_id)s
                      AND guard.owner_user_id = %(owner_user_id)s
                      AND guard.exchange_connection_id = %(exchange_connection_id)s
                      AND guard.instrument_key = %(instrument_key)s
                      AND guard.market_type = %(market_type)s
                    ORDER BY guard.checked_at DESC
                    LIMIT 1
                ), daily AS MATERIALIZED (
                    SELECT
                        COALESCE(SUM(COALESCE(
                            quote_notional,
                            quantity * limit_price
                        )), 0) AS notional,
                        COUNT(*) FILTER (
                            WHERE quote_notional IS NULL
                              AND (quantity IS NULL OR limit_price IS NULL)
                        ) AS incomplete_count
                    FROM execution_orders
                    WHERE organization_id = %(organization_id)s
                      AND owner_user_id = %(owner_user_id)s
                      AND exchange_connection_id = %(exchange_connection_id)s
                      AND intent_id <> %(intent_id)s
                      AND created_at >= date_trunc('day', %(evaluated_at)s)
                      AND created_at < date_trunc('day', %(evaluated_at)s)
                          + INTERVAL '1 day'
                      AND (
                          status IN ('submitted', 'cancelled', 'unknown')
                          OR (
                              status = 'status_checked'
                              AND exchange_order_id IS NOT NULL
                          )
                          OR (
                              status = 'submit_pending'
                              AND EXISTS (
                                  SELECT 1
                                  FROM execution_gateway_audit_events AS reservation_audit
                                  WHERE reservation_audit.event_id =
                                      execution_orders.submit_guard_audit_event_id
                                    AND reservation_audit.event_type =
                                      'execution_submit_guard_pre_submit'
                                    AND reservation_audit.decision = 'accepted'
                              )
                          )
                      )
                ), exposure AS MATERIALIZED (
                    SELECT
                        COALESCE((
                            SELECT SUM(ABS(position.quantity * position.entry_price))
                            FROM exchange_position_snapshots AS position
                            JOIN latest_snapshot AS snapshot
                              ON snapshot.account_snapshot_id = position.account_snapshot_id
                            WHERE position.organization_id = %(organization_id)s
                              AND position.owner_user_id = %(owner_user_id)s
                              AND position.exchange_connection_id = %(exchange_connection_id)s
                              AND position.entry_price IS NOT NULL
                        ), 0) + COALESCE((
                            SELECT SUM(ABS(open_order.quantity * open_order.price))
                            FROM exchange_open_order_snapshots AS open_order
                            JOIN latest_snapshot AS snapshot
                              ON snapshot.account_snapshot_id = open_order.account_snapshot_id
                            WHERE open_order.organization_id = %(organization_id)s
                              AND open_order.owner_user_id = %(owner_user_id)s
                              AND open_order.exchange_connection_id = %(exchange_connection_id)s
                              AND open_order.price IS NOT NULL
                        ), 0) + COALESCE((
                            SELECT SUM(ABS(COALESCE(
                                reserved.quote_notional,
                                reserved.quantity * reserved.limit_price
                            )))
                            FROM execution_orders AS reserved
                            WHERE reserved.organization_id = %(organization_id)s
                              AND reserved.owner_user_id = %(owner_user_id)s
                              AND reserved.exchange_connection_id = %(exchange_connection_id)s
                              AND reserved.intent_id <> %(intent_id)s
                              AND (
                                  reserved.status IN ('unknown', 'submitted')
                                  OR (
                                      reserved.status = 'status_checked'
                                      AND reserved.exchange_order_id IS NOT NULL
                                  )
                                  OR (
                                      reserved.status = 'submit_pending'
                                      AND EXISTS (
                                          SELECT 1
                                          FROM execution_gateway_audit_events
                                              AS reservation_audit
                                          WHERE reservation_audit.event_id =
                                              reserved.submit_guard_audit_event_id
                                            AND reservation_audit.event_type =
                                              'execution_submit_guard_pre_submit'
                                            AND reservation_audit.decision = 'accepted'
                                      )
                                  )
                              )
                        ), 0) AS notional,
                        COALESCE((
                            SELECT COUNT(*)
                            FROM exchange_position_snapshots AS position
                            JOIN latest_snapshot AS snapshot
                              ON snapshot.account_snapshot_id = position.account_snapshot_id
                            WHERE position.organization_id = %(organization_id)s
                              AND position.owner_user_id = %(owner_user_id)s
                              AND position.exchange_connection_id = %(exchange_connection_id)s
                              AND position.entry_price IS NULL
                        ), 0) + COALESCE((
                            SELECT COUNT(*)
                            FROM exchange_open_order_snapshots AS open_order
                            JOIN latest_snapshot AS snapshot
                              ON snapshot.account_snapshot_id = open_order.account_snapshot_id
                            WHERE open_order.organization_id = %(organization_id)s
                              AND open_order.owner_user_id = %(owner_user_id)s
                              AND open_order.exchange_connection_id = %(exchange_connection_id)s
                              AND open_order.price IS NULL
                        ), 0) + COALESCE((
                            SELECT COUNT(*)
                            FROM execution_orders AS reserved
                            WHERE reserved.organization_id = %(organization_id)s
                              AND reserved.owner_user_id = %(owner_user_id)s
                              AND reserved.exchange_connection_id = %(exchange_connection_id)s
                              AND reserved.intent_id <> %(intent_id)s
                              AND (
                                  reserved.status IN ('unknown', 'submitted')
                                  OR (
                                      reserved.status = 'status_checked'
                                      AND reserved.exchange_order_id IS NOT NULL
                                  )
                                  OR (
                                      reserved.status = 'submit_pending'
                                      AND EXISTS (
                                          SELECT 1
                                          FROM execution_gateway_audit_events
                                              AS reservation_audit
                                          WHERE reservation_audit.event_id =
                                              reserved.submit_guard_audit_event_id
                                            AND reservation_audit.event_type =
                                              'execution_submit_guard_pre_submit'
                                            AND reservation_audit.decision = 'accepted'
                                      )
                                  )
                              )
                              AND reserved.quote_notional IS NULL
                              AND (
                                  reserved.quantity IS NULL
                                  OR reserved.limit_price IS NULL
                              )
                        ), 0) AS incomplete_count
                )
                SELECT
                    COALESCE(snapshot.sync_status = 'fresh'
                        AND snapshot.observed_at >= %(evaluated_at)s - INTERVAL '60 seconds',
                        FALSE) AS account_snapshot_fresh,
                    COALESCE(guard.status = 'verified'
                        AND guard.checked_at >= snapshot.observed_at,
                        FALSE) AS config_guard_verified,
                    daily.notional AS daily_notional_used,
                    daily.incomplete_count = 0 AS daily_notional_complete,
                    exposure.notional AS account_exposure_notional,
                    exposure.incomplete_count = 0 AS account_exposure_complete,
                    snapshot.observed_at
                FROM daily
                CROSS JOIN exposure
                LEFT JOIN latest_snapshot AS snapshot ON TRUE
                LEFT JOIN latest_guard AS guard ON TRUE
                """,
                parameters=scope,
            )
            risk_snapshot = _map_risk_snapshot(risk_row)
            if risk_row is None or not bool(risk_row["daily_notional_complete"]):
                risk_snapshot = replace(risk_snapshot, account_snapshot_fresh=False)
            decision = evaluate_execution_submit_guard(
                query=query,
                safety=_map_safety(safety_row) if safety_row is not None else None,
                provider=_map_provider(provider_row) if provider_row is not None else None,
                kill_switch_active=bool(kill_row and kill_row["active"]),
                approval=_map_approval(approval_row) if approval_row is not None else None,
                risk_snapshot=risk_snapshot,
            )
            if query.phase == "pre_submit" and (
                claim_row is None
                or UUID(str(claim_row["submit_claim_id"]))
                != query.submission_attempt_id
                or claim_row["submit_claim_expires_at"] <= query.evaluated_at
            ):
                decision = ExecutionSubmitGuardDecision(
                    status="rejected",
                    reason="execution_submit_claim_invalid",
                    check_name="submit_claim",
                    phase=query.phase,
                    evaluated_at=query.evaluated_at,
                )
            audit_event_id = uuid4()
            decision = replace(decision, audit_event_id=audit_event_id)
            self._record_audit(
                transaction=transaction,
                audit=ExecutionGatewayAuditEvent(
                    event_id=audit_event_id,
                    organization_id=query.intent.organization_id,
                    owner_user_id=query.intent.owner_user_id,
                    exchange_connection_id=query.intent.exchange_connection_id,
                    intent_id=query.intent.intent_id,
                    approval_id=decision.approval_id,
                    event_type=f"execution_submit_guard_{query.phase}",
                    decision=decision.status,
                    reason=decision.reason,
                    actor_user_id=None,
                    created_at=query.evaluated_at,
                    metadata=decision.metadata,
                ),
            )
            if query.phase == "pre_submit" and decision.accepted:
                reservation_row = transaction.fetch_one(
                    query="""
                    UPDATE execution_orders
                    SET submit_guard_audit_event_id = %(audit_event_id)s,
                        updated_at = %(evaluated_at)s
                    WHERE organization_id = %(organization_id)s
                      AND intent_id = %(intent_id)s
                      AND status = 'submit_pending'
                      AND submit_claim_id = %(submission_attempt_id)s
                      AND submit_claim_expires_at > %(evaluated_at)s
                    RETURNING order_id
                    """,
                    parameters={
                        **scope,
                        "audit_event_id": str(audit_event_id),
                        "submission_attempt_id": str(query.submission_attempt_id),
                    },
                )
                if reservation_row is None:
                    raise RuntimeError("execution risk reservation claim fence lost")
        return decision

    def _record_audit(
        self,
        *,
        transaction: StrategyPostgresTransaction,
        audit: ExecutionGatewayAuditEvent,
    ) -> None:
        row = transaction.fetch_one(
            query="""
            INSERT INTO execution_gateway_audit_events
            (
                event_id, organization_id, owner_user_id,
                exchange_connection_id, intent_id, approval_id, event_type,
                decision, reason, actor_user_id, created_at, metadata_json
            )
            VALUES
            (
                %(event_id)s, %(organization_id)s, %(owner_user_id)s,
                %(exchange_connection_id)s, %(intent_id)s, %(approval_id)s,
                %(event_type)s, %(decision)s, %(reason)s, %(actor_user_id)s,
                %(created_at)s, %(metadata_json)s::jsonb
            )
            RETURNING event_id
            """,
            parameters={
                "event_id": str(audit.event_id),
                "organization_id": (
                    str(audit.organization_id) if audit.organization_id is not None else None
                ),
                "owner_user_id": (
                    str(audit.owner_user_id) if audit.owner_user_id is not None else None
                ),
                "exchange_connection_id": (
                    str(audit.exchange_connection_id)
                    if audit.exchange_connection_id is not None
                    else None
                ),
                "intent_id": str(audit.intent_id) if audit.intent_id is not None else None,
                "approval_id": (
                    str(audit.approval_id) if audit.approval_id is not None else None
                ),
                "event_type": audit.event_type,
                "decision": audit.decision,
                "reason": audit.reason,
                "actor_user_id": (
                    str(audit.actor_user_id) if audit.actor_user_id is not None else None
                ),
                "created_at": audit.created_at,
                "metadata_json": _metadata_json(audit.metadata),
            },
        )
        if row is None:
            raise RuntimeError("execution gateway audit write returned no row")

    @staticmethod
    def _require_authority(
        *,
        transaction: StrategyPostgresTransaction,
        principal: ExecutionPolicyPrincipal,
        evaluated_at: datetime,
        organization_id: OrganizationId | None,
        roles: tuple[str, ...],
    ) -> None:
        parameters = {
            "user_id": str(principal.user_id),
            "session_id": str(principal.session_id),
            "authenticated_at": principal.authenticated_at,
            "evaluated_at": evaluated_at,
            "organization_id": str(organization_id) if organization_id is not None else None,
            "roles": list(roles),
        }
        row = transaction.fetch_one(
            query="""
            SELECT EXISTS (
                SELECT 1
                FROM identity_sessions AS session
                WHERE session.session_id = %(session_id)s
                  AND session.user_id = %(user_id)s
                  AND session.created_at = %(authenticated_at)s
                  AND session.revoked_at IS NULL
                  AND session.idle_expires_at > %(evaluated_at)s
                  AND session.absolute_expires_at > %(evaluated_at)s
                  AND (
                    (CAST(%(organization_id)s AS UUID) IS NULL AND EXISTS (
                        SELECT 1
                        FROM identity_installation_owners AS installation_owner
                        WHERE installation_owner.user_id = %(user_id)s
                    ))
                    OR (CAST(%(organization_id)s AS UUID) IS NOT NULL AND EXISTS (
                        SELECT 1
                        FROM identity_memberships AS membership
                        WHERE membership.organization_id = CAST(%(organization_id)s AS UUID)
                          AND membership.user_id = %(user_id)s
                          AND membership.status = 'active'
                          AND membership.role = ANY(CAST(%(roles)s AS TEXT[]))
                    ))
                  )
            ) AS authorized
            """,
            parameters=parameters,
        )
        if row is None or not bool(row["authorized"]):
            raise RuntimeError("execution policy authority check failed")


def _query_scope(query: ExecutionSubmitGuardQuery) -> dict[str, str]:
    return {
        "organization_id": str(query.intent.organization_id),
        "owner_user_id": str(query.intent.owner_user_id),
        "exchange_connection_id": str(query.intent.exchange_connection_id),
    }


def _approval_params(approval: ExecutionMainnetApproval) -> dict[str, object]:
    return {
        "approval_id": str(approval.approval_id),
        "organization_id": str(approval.organization_id),
        "owner_user_id": str(approval.owner_user_id),
        "exchange_connection_id": str(approval.exchange_connection_id),
        "exchange_name": approval.exchange_name,
        "market_type": approval.market_type,
        "provider_id": approval.provider_id,
        "risk_revision_hash": approval.risk_revision_hash,
        "account_revision_hash": approval.account_revision_hash,
        "provider_revision_hash": approval.provider_revision_hash,
        "recent_auth_session_id": str(approval.recent_auth_session_id),
        "recent_auth_at": approval.recent_auth_at,
        "approved_at": approval.approved_at,
        "expires_at": approval.expires_at,
        "audit_event_id": str(approval.audit_event_id),
    }


def _map_provider(row: Mapping[str, Any]) -> ExecutionProviderRegistration:
    return ExecutionProviderRegistration(
        provider_id=str(row["provider_id"]),
        provider_version=str(row["provider_version"]),
        provider_kind=str(row["provider_kind"]),  # type: ignore[arg-type]
        exchange_name=str(row["exchange_name"]),
        revision_hash=str(row["revision_hash"]),
        order_submit_capability=bool(row["order_submit_capability"]),
        enabled=bool(row["enabled"]),
        approved_by_user_id=UserId.from_string(str(row["approved_by_user_id"])),
        updated_at=_datetime(row["updated_at"]),
    )


def _map_safety(row: Mapping[str, Any]) -> ExecutionAccountSafetyState:
    return ExecutionAccountSafetyState(
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        exchange_connection_id=UUID(str(row["exchange_connection_id"])),
        mode=str(row["mode"]),  # type: ignore[arg-type]
        risk_revision_hash=str(row["risk_revision_hash"]),
        account_revision_hash=str(row["account_revision_hash"]),
        secret_reference_hash=str(row["secret_reference_hash"]),
        risk_allows_submit=bool(row["risk_allows_submit"]),
        max_order_notional=Decimal(str(row["max_order_notional"])),
        daily_notional_limit=Decimal(str(row["daily_notional_limit"])),
        max_account_exposure_notional=Decimal(
            str(row["max_account_exposure_notional"])
        ),
        risk_valid_until=_datetime(row["risk_valid_until"]),
        updated_by_user_id=UserId.from_string(str(row["updated_by_user_id"])),
        updated_at=_datetime(row["updated_at"]),
    )


def _map_kill_switch(row: Mapping[str, Any]) -> ExecutionKillSwitchState:
    return ExecutionKillSwitchState(
        scope_type=str(row["scope_type"]),  # type: ignore[arg-type]
        active=bool(row["active"]),
        reason=str(row["reason"]),
        updated_by_user_id=UserId.from_string(str(row["updated_by_user_id"])),
        updated_at=_datetime(row["updated_at"]),
        organization_id=(
            OrganizationId.from_string(str(row["organization_id"]))
            if row.get("organization_id") is not None
            else None
        ),
        exchange_connection_id=(
            UUID(str(row["exchange_connection_id"]))
            if row.get("exchange_connection_id") is not None
            else None
        ),
    )


def _map_approval(row: Mapping[str, Any]) -> ExecutionMainnetApproval:
    return ExecutionMainnetApproval(
        approval_id=UUID(str(row["approval_id"])),
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        exchange_connection_id=UUID(str(row["exchange_connection_id"])),
        exchange_name=str(row["exchange_name"]),
        market_type=str(row["market_type"]),
        provider_id=str(row["provider_id"]),
        risk_revision_hash=str(row["risk_revision_hash"]),
        account_revision_hash=str(row["account_revision_hash"]),
        provider_revision_hash=str(row["provider_revision_hash"]),
        recent_auth_session_id=UUID(str(row["recent_auth_session_id"])),
        recent_auth_at=_datetime(row["recent_auth_at"]),
        approved_at=_datetime(row["approved_at"]),
        expires_at=_datetime(row["expires_at"]),
        audit_event_id=UUID(str(row["audit_event_id"])),
        revoked_at=_datetime_or_none(row.get("revoked_at")),
        revocation_reason=(
            str(row["revocation_reason"]) if row.get("revocation_reason") is not None else None
        ),
    )


def _map_risk_snapshot(row: Mapping[str, Any] | None) -> ExecutionCurrentRiskSnapshot:
    if row is None:
        return ExecutionCurrentRiskSnapshot(
            account_snapshot_fresh=False,
            config_guard_verified=False,
            daily_notional_used=Decimal("0"),
            account_exposure_notional=Decimal("0"),
            account_exposure_complete=False,
            observed_at=None,
        )
    return ExecutionCurrentRiskSnapshot(
        account_snapshot_fresh=bool(row["account_snapshot_fresh"]),
        config_guard_verified=bool(row["config_guard_verified"]),
        daily_notional_used=Decimal(str(row["daily_notional_used"])),
        account_exposure_notional=Decimal(str(row["account_exposure_notional"])),
        account_exposure_complete=bool(row["account_exposure_complete"]),
        observed_at=_datetime_or_none(row.get("observed_at")),
    )


def _metadata_json(metadata: Mapping[str, str]) -> str:
    clean: dict[str, str] = {}
    for key, value in metadata.items():
        normalized_key = str(key).strip()
        if any(part in normalized_key.casefold() for part in _SENSITIVE_PARTS):
            raise ValueError("sensitive execution gateway audit metadata key")
        clean[normalized_key[:64]] = str(value)[:256]
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("expected datetime")
    return value


def _datetime_or_none(value: object) -> datetime | None:
    return _datetime(value) if value is not None else None
