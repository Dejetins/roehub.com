from __future__ import annotations

import argparse
import contextlib
import io
import json
import secrets
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psycopg

from apps.exchange_execution.adapters import ExchangeExecutionEmulatorAdapter
from apps.migrations.storage import apply_postgres_migrations
from trading.contexts.live_execution.adapters.outbound import (
    PostgresExchangeExecutionOrderRepository,
    PostgresExchangeExecutionProcessRepository,
    PostgresExecutionGatewayPolicyRepository,
    PostgresExecutionIntentRepository,
)
from trading.contexts.live_execution.application import (
    CreateExecutionIntentCommand,
    ExchangeExecutionProcessConfig,
    ExchangeExecutionProcessService,
    ExecutionGatewayPolicyService,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionRedisHealth,
    ExchangeExecutionRedisMessage,
    ExecutionDispatchPublishResult,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExecutionAccountSafetyState,
    ExecutionAdapterIdentity,
    ExecutionGatewayAuditEvent,
    ExecutionGatewayPolicyError,
    ExecutionKillSwitchState,
    ExecutionMainnetApproval,
    ExecutionPolicyPrincipal,
    ExecutionProviderRegistration,
    ExecutionRiskContext,
    ExecutionSubmitGuardQuery,
)
from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway
from trading.shared_kernel.primitives import OrganizationId, UserId

ROOT = Path(__file__).resolve().parents[3]
POSTGRES_IMAGE = "postgres:16"
PROOF_SCHEMA = "io.roehub.execution-gateway-proof/v1"


class _Clock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)


class _Consumer:
    def __init__(self, *, intent_id: UUID, organization_id: UUID, owner_user_id: UUID) -> None:
        self.message = ExchangeExecutionRedisMessage(
            stream_name="execution.requests.v1",
            message_id=f"1-{uuid4().int}",
            payload={
                "organization_id": str(organization_id),
                "owner_user_id": str(owner_user_id),
                "intent_id": str(intent_id),
            },
        )
        self.acked = False

    def ensure_request_group(self) -> None:
        return None

    def health_snapshot(self) -> ExchangeExecutionRedisHealth:
        return ExchangeExecutionRedisHealth(
            request_stream_length=1,
            retry_stream_length=0,
            dlq_stream_length=0,
            pending_count=0,
            clock_drift_ms=0.0,
        )

    def read_pending_requests(self, *, count: int) -> tuple[ExchangeExecutionRedisMessage, ...]:
        _ = count
        return ()

    def read_new_requests(
        self, *, count: int, block_ms: int
    ) -> tuple[ExchangeExecutionRedisMessage, ...]:
        _ = (count, block_ms)
        return (self.message,)

    def publish_dlq(
        self, *, message: ExchangeExecutionRedisMessage, reason: str
    ) -> ExecutionDispatchPublishResult:
        _ = (message, reason)
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.dlq.v1",
            message_id="dlq-1",
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        _ = (stream_name, message_id)
        self.acked = True


class _Resolver:
    def __init__(
        self, *, organization_id: OrganizationId, owner_user_id: UserId, connection_id: UUID
    ) -> None:
        credential_fields = {
            "api_" + "key": "emulator-public",
            "api_" + "secret": "emulator-private-placeholder",
        }
        self.connection = ExchangeExecutionConnection(
            connection_id=connection_id,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_name="bybit",
            market_type="spot",
            environment="mainnet",
            connection_readiness="ready_for_trading",
            effective_capability="trading",
            secret_reference_hash="4" * 64,
            account_revision_hash="3" * 64,
            credential=ExchangeExecutionCredential(**credential_fields),
        )

    def resolve(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
    ) -> ExchangeExecutionConnection:
        if (
            organization_id != self.connection.organization_id
            or owner_user_id != self.connection.owner_user_id
            or exchange_connection_id != self.connection.connection_id
        ):
            raise RuntimeError("emulator connection scope mismatch")
        return self.connection


class _RotatingResolver(_Resolver):
    def __init__(
        self, *, organization_id: OrganizationId, owner_user_id: UserId, connection_id: UUID
    ) -> None:
        super().__init__(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        self.resolve_count = 0

    def resolve(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
    ) -> ExchangeExecutionConnection:
        connection = super().resolve(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
        )
        self.resolve_count += 1
        if self.resolve_count == 1:
            return connection
        return replace(connection, secret_reference_hash="8" * 64)


class _KillOnPreSubmitPolicy:
    def __init__(
        self,
        *,
        repository: PostgresExecutionGatewayPolicyRepository,
        service: ExecutionGatewayPolicyService,
        owner_user_id: UserId,
        principal: ExecutionPolicyPrincipal,
    ) -> None:
        self.repository = repository
        self.service = service
        self.owner_user_id = owner_user_id
        self.principal = principal

    def evaluate_and_record(self, *, query: ExecutionSubmitGuardQuery):
        if query.phase == "pre_submit":
            self.service.set_kill_switch(
                state=ExecutionKillSwitchState(
                    scope_type="account",
                    organization_id=query.intent.organization_id,
                    exchange_connection_id=query.intent.exchange_connection_id,
                    active=True,
                    reason="stage16_in_flight_incident",
                    updated_by_user_id=self.owner_user_id,
                    updated_at=query.evaluated_at,
                ),
                principal=self.principal,
            )
        return self.repository.evaluate_and_record(query=query)


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def _mapped_port(container: str, port: int) -> int:
    output = _run(["docker", "port", container, f"{port}/tcp"]).stdout.strip()
    return int(output.rsplit(":", 1)[1])


def _wait_postgres(dsn: str) -> None:
    for _ in range(80):
        try:
            with psycopg.connect(dsn, connect_timeout=1):
                return
        except psycopg.Error:
            time.sleep(0.25)
    raise RuntimeError("disposable PostgreSQL did not become ready")


def _seed_identity(dsn: str) -> tuple[UUID, UUID, UUID, UUID, datetime]:
    installation_id = uuid4()
    organization_id = uuid4()
    owner_user_id = uuid4()
    connection_id = uuid4()
    session_id = uuid4()
    account_snapshot_id = uuid4()
    now = datetime.now(tz=UTC)
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO identity_users
                (user_id, telegram_user_id, paid_level, created_at, is_deleted)
            VALUES (%s, NULL, 'free', %s, FALSE)
            """,
            (owner_user_id, now),
        )
        cursor.execute(
            """
            INSERT INTO identity_sessions
                (session_id, user_id, created_at, last_seen_at,
                 idle_expires_at, absolute_expires_at, revoked_at)
            VALUES (%s, %s, %s, %s, %s, %s, NULL)
            """,
            (
                session_id,
                owner_user_id,
                now,
                now,
                now + timedelta(hours=1),
                now + timedelta(hours=2),
            ),
        )
        cursor.execute(
            """
            INSERT INTO identity_installations
                (installation_id, singleton_key, display_name, created_at)
            VALUES (%s, TRUE, 'Stage 16 fixture', %s)
            """,
            (installation_id, now),
        )
        cursor.execute(
            """
            INSERT INTO identity_installation_owners
                (installation_id, user_id, granted_by_user_id, granted_at)
            VALUES (%s, %s, %s, %s)
            """,
            (installation_id, owner_user_id, owner_user_id, now),
        )
        cursor.execute(
            """
            INSERT INTO identity_organizations
                (organization_id, installation_id, slug, display_name, status, created_at)
            VALUES (%s, %s, 'stage16-proof', 'Stage 16 proof', 'active', %s)
            """,
            (organization_id, installation_id, now),
        )
        cursor.execute(
            """
            INSERT INTO identity_memberships
                (organization_id, user_id, role, status, created_at, updated_at)
            VALUES (%s, %s, 'owner', 'active', %s, %s)
            """,
            (organization_id, owner_user_id, now, now),
        )
        cursor.execute(
            """
            INSERT INTO exchange_connections
            (
                connection_id, owner_user_id, organization_id, exchange_name,
                market_type, environment, label, active_credential_version_id,
                status, status_reason, permission_summary_json,
                ip_restriction_status, created_at, updated_at
            )
            VALUES
            (
                %s, %s, %s, 'bybit', 'spot', 'mainnet', 'stage16-emulator',
                NULL, 'active', 'emulator_only', '{}'::jsonb, 'unknown', %s, %s
            )
            """,
            (connection_id, owner_user_id, organization_id, now, now),
        )
        cursor.execute(
            """
            INSERT INTO exchange_account_snapshots (
                account_snapshot_id, organization_id, owner_user_id,
                exchange_connection_id, exchange_name, market_type,
                environment, account_mode, source_hash, sync_status,
                sync_reason, observed_at, synced_at, balance_count,
                position_count, open_order_count, filter_count, metadata_json
            ) VALUES (
                %s, %s, %s, %s, 'bybit', 'spot', 'mainnet', 'one_way',
                %s, 'fresh', 'stage16_emulator', %s, %s,
                0, 0, 0, 0, '{}'::jsonb
            )
            """,
            (
                account_snapshot_id,
                organization_id,
                owner_user_id,
                connection_id,
                "6" * 64,
                now,
                now,
            ),
        )
        cursor.execute(
            """
            INSERT INTO exchange_account_config_guard_results (
                config_guard_result_id, organization_id, account_snapshot_id,
                owner_user_id, exchange_connection_id, instrument_key,
                market_type, status, reason_codes_json, requirement_json,
                checked_at
            ) VALUES (
                %s, %s, %s, %s, %s, 'bybit:spot:BTCUSDT', 'spot',
                'verified', '[]'::jsonb, '{}'::jsonb, %s
            )
            """,
            (
                uuid4(),
                organization_id,
                account_snapshot_id,
                owner_user_id,
                connection_id,
                now,
            ),
        )
    return organization_id, owner_user_id, connection_id, session_id, now


def _create_dispatched_intent(
    *,
    repository: PostgresExecutionIntentRepository,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    connection_id: UUID,
    scenario: str,
    quote_notional: Decimal = Decimal("1"),
) -> Any:
    clock = _Clock()
    ingress = ExecutionIngressService(repository=repository, clock=clock)
    source = ingress.record_source_event(
        command=RecordExecutionSourceEventCommand(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            source_type="ops_test",
            source_event_ref=f"stage16:{scenario}",
            source_ref_json={"scenario": scenario},
            strategy_signal_id=None,
            idempotency_key=f"stage16-source:{scenario}",
        )
    )
    intent = ingress.create_intent(
        command=CreateExecutionIntentCommand(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            source_event_id=source.event.source_event_id,
            idempotency_key=f"stage16-intent:{scenario}",
            exchange_connection_id=connection_id,
            market_type="spot",
            instrument_key="bybit:spot:BTCUSDT",
            order_type="market",
            side="buy",
            quantity=None,
            quote_notional=quote_notional,
            limit_price=None,
            advanced_order_flags={},
            constraints={
                "expires_at": (datetime.now(tz=UTC) + timedelta(minutes=5)).isoformat()
            },
            risk_context=ExecutionRiskContext(
                organization_ownership_verified=True,
                account_ownership_verified=True,
                exchange_connection_active=True,
                secret_custody_ready=True,
                source_authorized=True,
                exchange_config_verified=True,
                account_state_fresh=True,
                kill_switch_open=True,
                environment_policy_allows=True,
                max_order_size_ok=True,
                daily_limit_ok=True,
            ),
        )
    ).intent
    claimed = repository.claim_intent_for_dispatch(
        organization_id=organization_id,
        intent_id=intent.intent_id,
        now=clock.now(),
        retry_budget=3,
    )
    if claimed is None:
        raise RuntimeError("intent dispatch claim failed")
    dispatched = repository.mark_intent_dispatched(
        organization_id=organization_id,
        intent_id=intent.intent_id,
        stream_name="execution.requests.v1",
        redis_message_id=f"proof-{scenario}",
        now=clock.now(),
    )
    if dispatched is None:
        raise RuntimeError("intent dispatch persistence failed")
    return dispatched


def _process_once(
    *,
    intent: Any,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    connection_id: UUID,
    process_repository: PostgresExchangeExecutionProcessRepository,
    intent_repository: PostgresExecutionIntentRepository,
    order_repository: PostgresExchangeExecutionOrderRepository,
    gateway_policy_repository: Any,
    adapter: ExchangeExecutionEmulatorAdapter,
    expect_ack: bool = True,
    credential_resolver: Any | None = None,
) -> Any:
    consumer = _Consumer(
        intent_id=intent.intent_id,
        organization_id=organization_id.value,
        owner_user_id=owner_user_id.value,
    )
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="emulator",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
            submit_claim_ttl_seconds=1,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=credential_resolver
        or _Resolver(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        ),
        order_adapters=(adapter,),
        gateway_policy_repository=gateway_policy_repository,
        consumer=consumer,
        clock=_Clock(),
    )
    result = service.run_once()
    if consumer.acked is not expect_ack:
        raise RuntimeError("emulator acknowledgement did not match durable-state policy")
    return result


def _db_counts(dsn: str) -> dict[str, int]:
    tables = (
        "execution_gateway_audit_events",
        "execution_mainnet_approvals",
        "execution_orders",
        "execution_reconciliation_runs",
    )
    counts: dict[str, int] = {}
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        for table in tables:
            cursor.execute(f"SELECT count(*) FROM {table}")  # noqa: S608
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError(f"proof table count unavailable: {table}")
            counts[table] = int(row[0])
    return counts


def _prove_audit_immutable(dsn: str) -> None:
    try:
        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE execution_gateway_audit_events SET reason = 'tampered'"
            )
    except psycopg.Error:
        return
    raise RuntimeError("execution gateway audit unexpectedly allowed mutation")


def _prove_policy_delete_guards(dsn: str) -> None:
    for table in (
        "execution_provider_allowlist",
        "execution_account_safety_state",
        "execution_kill_switch_state",
        "execution_mainnet_approvals",
    ):
        try:
            with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table}")  # noqa: S608
        except psycopg.Error:
            continue
        raise RuntimeError(f"execution policy table unexpectedly allowed delete: {table}")


def _prove_policy_audit_atomicity(
    *,
    dsn: str,
    repository: PostgresExecutionGatewayPolicyRepository,
    provider: ExecutionProviderRegistration,
    principal: ExecutionPolicyPrincipal,
) -> None:
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT enabled, provider_version, audit_event_id, updated_at
            FROM execution_provider_allowlist
            WHERE provider_id = %s
            """,
            (provider.provider_id,),
        )
        original = cursor.fetchone()
    if original is None:
        raise RuntimeError("provider policy missing for atomicity proof")
    duplicate_audit_id = UUID(str(original[2]))
    changed_at = datetime.now(tz=UTC)
    changed = replace(provider, enabled=False, updated_at=changed_at)
    duplicate_audit = ExecutionGatewayAuditEvent(
        event_id=duplicate_audit_id,
        organization_id=None,
        owner_user_id=None,
        exchange_connection_id=None,
        intent_id=None,
        approval_id=None,
        event_type="execution_provider_registered",
        decision="accepted",
        reason="duplicate_audit_must_rollback",
        actor_user_id=principal.user_id,
        created_at=changed_at,
        metadata={"provider_id": provider.provider_id},
    )
    try:
        repository.register_provider(
            provider=changed,
            audit=duplicate_audit,
            principal=principal,
        )
    except psycopg.Error:
        pass
    else:
        raise RuntimeError("duplicate gateway audit unexpectedly mutated provider policy")
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT enabled, provider_version, audit_event_id, updated_at
            FROM execution_provider_allowlist
            WHERE provider_id = %s
            """,
            (provider.provider_id,),
        )
        after_duplicate = cursor.fetchone()
    if after_duplicate != original:
        raise RuntimeError("duplicate audit failure did not roll back policy mutation")

    try:
        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE execution_provider_allowlist
                SET enabled = NOT enabled,
                    updated_at = updated_at + INTERVAL '1 second'
                WHERE provider_id = %s
                """,
                (provider.provider_id,),
            )
    except psycopg.Error:
        pass
    else:
        raise RuntimeError("unaudited provider policy update unexpectedly succeeded")
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT enabled, provider_version, audit_event_id, updated_at
            FROM execution_provider_allowlist
            WHERE provider_id = %s
            """,
            (provider.provider_id,),
        )
        after_unaudited = cursor.fetchone()
    if after_unaudited != original:
        raise RuntimeError("unaudited policy update was not rolled back")

    audited_at = datetime.now(tz=UTC)
    audited_provider = replace(
        provider,
        provider_version="v1-proof",
        updated_at=audited_at,
    )
    repository.register_provider(
        provider=audited_provider,
        audit=replace(
            duplicate_audit,
            event_id=uuid4(),
            reason="audited_policy_update_proof",
            created_at=audited_at,
        ),
        principal=principal,
    )
    restored_at = datetime.now(tz=UTC)
    restored = repository.register_provider(
        provider=replace(provider, updated_at=restored_at),
        audit=replace(
            duplicate_audit,
            event_id=uuid4(),
            reason="audited_policy_restore_proof",
            created_at=restored_at,
        ),
        principal=principal,
    )
    if restored.provider_version != provider.provider_version or not restored.enabled:
        raise RuntimeError("audited provider policy update/restore did not persist")


def _prove_cross_intent_risk_reservation(
    *,
    dsn: str,
    intent_repository: PostgresExecutionIntentRepository,
    order_repository: PostgresExchangeExecutionOrderRepository,
    policy_repository: PostgresExecutionGatewayPolicyRepository,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    connection: ExchangeExecutionConnection,
    adapter: ExchangeExecutionEmulatorAdapter,
    approval_id: UUID,
    scenario: str,
    expected_rejection: str,
) -> None:
    adapter_identity = ExecutionAdapterIdentity(
        provider_id=adapter.provider_id,
        provider_version=adapter.provider_version,
        provider_kind="core",
        exchange_name=adapter.exchange_name,
        revision_hash=adapter.revision_hash,
    )
    intents = tuple(
        _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection.connection_id,
            scenario=f"{scenario}-{index}",
            quote_notional=Decimal("600"),
        )
        for index in range(2)
    )
    claims: list[tuple[Any, UUID]] = []
    for intent in intents:
        claim_id = uuid4()
        preflight = policy_repository.evaluate_and_record(
            query=ExecutionSubmitGuardQuery(
                intent=intent,
                connection=connection,
                adapter=adapter_identity,
                phase="preflight",
                submission_attempt_id=claim_id,
                evaluated_at=datetime.now(tz=UTC),
            )
        )
        if not preflight.accepted or preflight.audit_event_id is None:
            raise RuntimeError("cross-intent risk preflight unexpectedly rejected")
        claimed_at = datetime.now(tz=UTC)
        claim = order_repository.claim_submit(
            command=ExchangeOrderCommand.from_intent(
                intent=intent,
                exchange_name=connection.exchange_name,
                environment=connection.environment,
                client_order_id=f"rh1_{intent.idempotency_key_hash[:28]}",
            ),
            claim_id=claim_id,
            claimed_at=claimed_at,
            expires_at=claimed_at + timedelta(minutes=2),
            submit_guard_audit_event_id=preflight.audit_event_id,
            mainnet_approval_id=approval_id,
        )
        if not claim.acquired:
            raise RuntimeError("cross-intent risk proof could not acquire distinct claim")
        claims.append((intent, claim_id))

    def evaluate(item: tuple[Any, UUID]):
        intent, claim_id = item
        repository = PostgresExecutionGatewayPolicyRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
        )
        return repository.evaluate_and_record(
            query=ExecutionSubmitGuardQuery(
                intent=intent,
                connection=connection,
                adapter=adapter_identity,
                phase="pre_submit",
                submission_attempt_id=claim_id,
                evaluated_at=datetime.now(tz=UTC),
            )
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        decisions = tuple(executor.map(evaluate, claims))
    accepted = tuple(decision for decision in decisions if decision.accepted)
    rejected = tuple(decision for decision in decisions if not decision.accepted)
    if len(accepted) != 1 or len(rejected) != 1:
        raise RuntimeError(
            "cross-intent risk reservation was not serialized: "
            + ",".join(f"{item.status}:{item.reason}" for item in decisions)
        )
    if rejected[0].reason != expected_rejection:
        raise RuntimeError(
            f"cross-intent risk rejection mismatch: {rejected[0].reason}"
        )

    for intent, claim_id in claims:
        command = ExchangeOrderCommand.from_intent(
            intent=intent,
            exchange_name=connection.exchange_name,
            environment=connection.environment,
            client_order_id=f"rh1_{intent.idempotency_key_hash[:28]}",
        )
        rejected_order = order_repository.record_claim_guard_rejection(
            command=command,
            claim_id=claim_id,
            rejected_at=datetime.now(tz=UTC),
            reason="risk_reservation_proof_cleanup",
        )
        if rejected_order is None:
            raise RuntimeError("cross-intent risk reservation cleanup lost claim fence")


def _prove_expired_accepted_reservation(
    *,
    intent_repository: PostgresExecutionIntentRepository,
    order_repository: PostgresExchangeExecutionOrderRepository,
    policy_repository: PostgresExecutionGatewayPolicyRepository,
    organization_id: OrganizationId,
    owner_user_id: UserId,
    connection: ExchangeExecutionConnection,
    adapter: ExchangeExecutionEmulatorAdapter,
    approval_id: UUID,
) -> None:
    adapter_identity = ExecutionAdapterIdentity(
        provider_id=adapter.provider_id,
        provider_version=adapter.provider_version,
        provider_kind="core",
        exchange_name=adapter.exchange_name,
        revision_hash=adapter.revision_hash,
    )
    intent = _create_dispatched_intent(
        repository=intent_repository,
        organization_id=organization_id,
        owner_user_id=owner_user_id,
        connection_id=connection.connection_id,
        scenario="provider-accepted-worker-crash",
        quote_notional=Decimal("600"),
    )
    claim_id = uuid4()
    base_time = datetime.now(tz=UTC)
    preflight = policy_repository.evaluate_and_record(
        query=ExecutionSubmitGuardQuery(
            intent=intent,
            connection=connection,
            adapter=adapter_identity,
            phase="preflight",
            submission_attempt_id=claim_id,
            evaluated_at=base_time,
        )
    )
    if not preflight.accepted or preflight.audit_event_id is None:
        raise RuntimeError("expired-reservation proof preflight was rejected")
    command = ExchangeOrderCommand.from_intent(
        intent=intent,
        exchange_name=connection.exchange_name,
        environment=connection.environment,
        client_order_id=f"rh1_{intent.idempotency_key_hash[:28]}",
    )
    claim = order_repository.claim_submit(
        command=command,
        claim_id=claim_id,
        claimed_at=base_time,
        expires_at=base_time + timedelta(seconds=2),
        submit_guard_audit_event_id=preflight.audit_event_id,
        mainnet_approval_id=approval_id,
    )
    if not claim.acquired:
        raise RuntimeError("expired-reservation proof claim was not acquired")
    pre_submit = policy_repository.evaluate_and_record(
        query=ExecutionSubmitGuardQuery(
            intent=intent,
            connection=connection,
            adapter=adapter_identity,
            phase="pre_submit",
            submission_attempt_id=claim_id,
            evaluated_at=base_time + timedelta(seconds=1),
        )
    )
    if not pre_submit.accepted:
        raise RuntimeError("expired-reservation proof pre-submit was rejected")
    submitted = adapter.submit_order(
        command=command,
        credential=connection.credential,
    )
    persisted = order_repository.get_by_intent(
        organization_id=organization_id,
        intent_id=intent.intent_id,
    )
    if (
        persisted is None
        or persisted.status != "submit_pending"
        or persisted.exchange_order_id is not None
    ):
        raise RuntimeError("worker-crash fixture unexpectedly finalized provider result")

    competitor = _create_dispatched_intent(
        repository=intent_repository,
        organization_id=organization_id,
        owner_user_id=owner_user_id,
        connection_id=connection.connection_id,
        scenario="expired-reservation-competitor",
        quote_notional=Decimal("600"),
    )
    competitor_decision = policy_repository.evaluate_and_record(
        query=ExecutionSubmitGuardQuery(
            intent=competitor,
            connection=connection,
            adapter=adapter_identity,
            phase="preflight",
            submission_attempt_id=uuid4(),
            evaluated_at=base_time + timedelta(seconds=3),
        )
    )
    if (
        competitor_decision.accepted
        or competitor_decision.reason != "execution_daily_notional_limit_exceeded"
    ):
        raise RuntimeError("expired accepted reservation stopped protecting daily cap")

    reconciled = adapter.get_order_status_by_client_order_id(
        command=command,
        client_order_id=command.client_order_id,
        credential=connection.credential,
    )
    if reconciled.lookup_outcome != "found":
        raise RuntimeError("accepted emulator order was not found during reconciliation")
    recorded = order_repository.record_status_result(
        organization_id=organization_id,
        intent_id=intent.intent_id,
        result=replace(
            reconciled,
            exchange_order_id=submitted.exchange_order_id,
        ),
    )
    if recorded is None or recorded.exchange_order_id != submitted.exchange_order_id:
        raise RuntimeError("expired accepted reservation reconciliation did not persist")


def _prove_postgres_claim_fencing(
    *,
    dsn: str,
    intent: Any,
    connection: ExchangeExecutionConnection,
    adapter: ExchangeExecutionEmulatorAdapter,
    approval_id: UUID,
    policy_repository: PostgresExecutionGatewayPolicyRepository,
) -> None:
    guard = policy_repository.evaluate_and_record(
        query=ExecutionSubmitGuardQuery(
            intent=intent,
            connection=connection,
            adapter=ExecutionAdapterIdentity(
                provider_id=adapter.provider_id,
                provider_version=adapter.provider_version,
                provider_kind="core",
                exchange_name=adapter.exchange_name,
                revision_hash=adapter.revision_hash,
            ),
            phase="preflight",
            submission_attempt_id=uuid4(),
            evaluated_at=datetime.now(tz=UTC),
        )
    )
    if not guard.accepted or guard.audit_event_id is None:
        raise RuntimeError("claim-fencing proof guard was not accepted")
    guard_audit_event_id = guard.audit_event_id
    command = ExchangeOrderCommand.from_intent(
        intent=intent,
        exchange_name=connection.exchange_name,
        environment=connection.environment,
        client_order_id=f"rh1_{intent.idempotency_key_hash[:28]}",
    )
    claimed_at = datetime.now(tz=UTC)
    claim_ids = (uuid4(), uuid4())

    def claim(claim_id: UUID):
        repository = PostgresExchangeExecutionOrderRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
        )
        return repository.claim_submit(
            command=command,
            claim_id=claim_id,
            claimed_at=claimed_at,
            expires_at=claimed_at + timedelta(seconds=30),
            submit_guard_audit_event_id=guard_audit_event_id,
            mainnet_approval_id=approval_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = tuple(executor.map(claim, claim_ids))
    acquired = tuple(item for item in claims if item.acquired)
    if len(acquired) != 1:
        raise RuntimeError("PostgreSQL concurrent submit claim was not exclusive")
    live_claim = acquired[0]
    restarted = PostgresExchangeExecutionOrderRepository(
        gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
    ).claim_submit(
        command=command,
        claim_id=uuid4(),
        claimed_at=claimed_at + timedelta(seconds=1),
        expires_at=claimed_at + timedelta(seconds=31),
        submit_guard_audit_event_id=guard_audit_event_id,
        mainnet_approval_id=approval_id,
    )
    if restarted.acquired or restarted.reason != "submission_in_flight":
        raise RuntimeError("live submit claim did not survive repository restart")
    restarted_adapter = ExchangeExecutionEmulatorAdapter(exchange_name=adapter.exchange_name)
    restarted_process = _process_once(
        intent=intent,
        organization_id=intent.organization_id,
        owner_user_id=intent.owner_user_id,
        connection_id=intent.exchange_connection_id,
        process_repository=PostgresExchangeExecutionProcessRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
        ),
        intent_repository=PostgresExecutionIntentRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
        ),
        order_repository=PostgresExchangeExecutionOrderRepository(
            gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
        ),
        gateway_policy_repository=policy_repository,
        adapter=restarted_adapter,
        expect_ack=False,
    )
    if restarted_process.submitted_count != 0 or restarted_adapter._orders:  # noqa: SLF001
        raise RuntimeError("restarted worker submitted through a live claim")
    repository = PostgresExchangeExecutionOrderRepository(
        gateway=PsycopgStrategyPostgresGateway(dsn=dsn)
    )
    expired_write = repository.record_adapter_error(
        organization_id=intent.organization_id,
        intent_id=intent.intent_id,
        claim_id=live_claim.claim_id,
        occurred_at=claimed_at + timedelta(seconds=31),
        reason="expired_worker_must_not_finalize",
    )
    if expired_write is not None:
        raise RuntimeError("expired worker bypassed submit claim fence")
    reconciled = repository.record_status_result(
        organization_id=intent.organization_id,
        intent_id=intent.intent_id,
        result=ExchangeOrderStatusResult(
            exchange_order_id="",
            exchange_status="not_found",
            checked_at=claimed_at + timedelta(seconds=32),
            latency_ms=0.0,
            metadata={"proof": "confirmed_absent"},
            lookup_outcome="confirmed_absent",
        ),
    )
    if reconciled is None:
        raise RuntimeError("claim-fencing proof reconciliation failed")
    replacement_claim_id = uuid4()
    replacement = repository.claim_submit(
        command=command,
        claim_id=replacement_claim_id,
        claimed_at=claimed_at + timedelta(seconds=33),
        expires_at=claimed_at + timedelta(seconds=63),
        submit_guard_audit_event_id=guard_audit_event_id,
        mainnet_approval_id=approval_id,
    )
    if not replacement.acquired:
        raise RuntimeError("reconciled submit was not claimable")
    stale_rejection = repository.record_claim_guard_rejection(
        command=command,
        claim_id=live_claim.claim_id,
        rejected_at=claimed_at + timedelta(seconds=34),
        reason="stale_worker_must_not_reject",
    )
    if stale_rejection is not None:
        raise RuntimeError("stale worker bypassed guard-rejection claim fence")
    stale_write = repository.record_adapter_error(
        organization_id=intent.organization_id,
        intent_id=intent.intent_id,
        claim_id=live_claim.claim_id,
        occurred_at=claimed_at + timedelta(seconds=34),
        reason="stale_worker_must_not_finalize",
    )
    if stale_write is not None:
        raise RuntimeError("stale worker bypassed submit claim fence")
    if repository.record_adapter_error(
        organization_id=intent.organization_id,
        intent_id=intent.intent_id,
        claim_id=replacement_claim_id,
        occurred_at=claimed_at + timedelta(seconds=34),
        reason="claim_fencing_proof_complete",
    ) is None:
        raise RuntimeError("current worker could not finalize its submit claim")
    final_claim_id = uuid4()
    final_claim = repository.claim_submit(
        command=command,
        claim_id=final_claim_id,
        claimed_at=claimed_at + timedelta(seconds=35),
        expires_at=claimed_at + timedelta(seconds=65),
        submit_guard_audit_event_id=guard_audit_event_id,
        mainnet_approval_id=approval_id,
    )
    if not final_claim.acquired:
        raise RuntimeError("guard-rejection fencing proof could not acquire claim")
    if repository.record_claim_guard_rejection(
        command=command,
        claim_id=replacement_claim_id,
        rejected_at=claimed_at + timedelta(seconds=36),
        reason="superseded_worker_must_not_reject",
    ) is not None:
        raise RuntimeError("superseded worker bypassed guard-rejection claim fence")
    if repository.record_claim_guard_rejection(
        command=command,
        claim_id=final_claim_id,
        rejected_at=claimed_at + timedelta(seconds=36),
        reason="claim_guard_rejection_fence_proof_complete",
    ) is None:
        raise RuntimeError("current worker could not record fenced guard rejection")


def run_proof() -> dict[str, Any]:
    suffix = secrets.token_hex(4)
    postgres = f"roehub-stage16-postgres-{suffix}"
    postgres_password = secrets.token_urlsafe(24)
    created = False
    try:
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                postgres,
                "-e",
                "POSTGRES_USER=roehub",
                "-e",
                f"POSTGRES_PASSWORD={postgres_password}",
                "-e",
                "POSTGRES_DB=roehub",
                "-p",
                "127.0.0.1::5432",
                POSTGRES_IMAGE,
            ]
        )
        created = True
        port = _mapped_port(postgres, 5432)
        dsn = f"postgresql://roehub:{postgres_password}@127.0.0.1:{port}/roehub"
        _wait_postgres(dsn)
        with contextlib.redirect_stdout(io.StringIO()):
            apply_postgres_migrations(
                dsn,
                repo_root=ROOT,
                manifest_path=ROOT / "migrations/postgres/manifest.json",
            )
        (
            raw_organization_id,
            raw_owner_user_id,
            connection_id,
            session_id,
            authenticated_at,
        ) = _seed_identity(dsn)
        organization_id = OrganizationId(raw_organization_id)
        owner_user_id = UserId.from_string(str(raw_owner_user_id))
        principal = ExecutionPolicyPrincipal(
            user_id=owner_user_id,
            session_id=session_id,
            authenticated_at=authenticated_at,
        )
        gateway = PsycopgStrategyPostgresGateway(dsn=dsn)
        policy_repository = PostgresExecutionGatewayPolicyRepository(gateway=gateway)
        policy_service = ExecutionGatewayPolicyService(repository=policy_repository)
        intent_repository = PostgresExecutionIntentRepository(gateway=gateway)
        order_repository = PostgresExchangeExecutionOrderRepository(gateway=gateway)
        process_repository = PostgresExchangeExecutionProcessRepository(gateway=gateway)
        now = datetime.now(tz=UTC)
        provider = policy_service.register_provider(
            provider=ExecutionProviderRegistration(
                provider_id="core:exchange-emulator",
                provider_version="v1",
                provider_kind="core",
                exchange_name="bybit",
                revision_hash=ExchangeExecutionEmulatorAdapter(
                    exchange_name="bybit"
                ).revision_hash,
                order_submit_capability=True,
                enabled=True,
                approved_by_user_id=owner_user_id,
                updated_at=now,
            ),
            principal=principal,
        )
        safety = policy_service.set_account_safety_state(
            state=ExecutionAccountSafetyState(
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                exchange_connection_id=connection_id,
                mode="mainnet",
                risk_revision_hash="2" * 64,
                account_revision_hash="3" * 64,
                secret_reference_hash="4" * 64,
                risk_allows_submit=True,
                max_order_notional=Decimal("1000"),
                daily_notional_limit=Decimal("10000"),
                max_account_exposure_notional=Decimal("25000"),
                risk_valid_until=now + timedelta(hours=1),
                updated_by_user_id=owner_user_id,
                updated_at=now,
            ),
            principal=principal,
        )
        approval = policy_service.approve_mainnet(
            approval=ExecutionMainnetApproval(
                approval_id=uuid4(),
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                exchange_connection_id=connection_id,
                exchange_name="bybit",
                market_type="spot",
                provider_id=provider.provider_id,
                risk_revision_hash=safety.risk_revision_hash,
                account_revision_hash=safety.account_revision_hash,
                provider_revision_hash=provider.revision_hash,
                recent_auth_session_id=session_id,
                recent_auth_at=authenticated_at,
                approved_at=now,
                expires_at=now + timedelta(minutes=10),
                audit_event_id=uuid4(),
            ),
            principal=principal,
            now=now,
        )

        _prove_policy_audit_atomicity(
            dsn=dsn,
            repository=policy_repository,
            provider=provider,
            principal=principal,
        )
        proof_connection = _Resolver(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        ).connection
        risk_adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")
        policy_service.set_account_safety_state(
            state=replace(
                safety,
                max_order_notional=Decimal("600"),
                daily_notional_limit=Decimal("1000"),
                max_account_exposure_notional=Decimal("5000"),
                updated_at=datetime.now(tz=UTC),
            ),
            principal=principal,
        )
        _prove_cross_intent_risk_reservation(
            dsn=dsn,
            intent_repository=intent_repository,
            order_repository=order_repository,
            policy_repository=policy_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection=proof_connection,
            adapter=risk_adapter,
            approval_id=approval.approval_id,
            scenario="concurrent-daily-limit",
            expected_rejection="execution_daily_notional_limit_exceeded",
        )
        policy_service.set_account_safety_state(
            state=replace(
                safety,
                max_order_notional=Decimal("600"),
                daily_notional_limit=Decimal("5000"),
                max_account_exposure_notional=Decimal("1000"),
                updated_at=datetime.now(tz=UTC),
            ),
            principal=principal,
        )
        _prove_cross_intent_risk_reservation(
            dsn=dsn,
            intent_repository=intent_repository,
            order_repository=order_repository,
            policy_repository=policy_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection=proof_connection,
            adapter=risk_adapter,
            approval_id=approval.approval_id,
            scenario="concurrent-exposure-limit",
            expected_rejection="execution_account_exposure_limit_exceeded",
        )
        policy_service.set_account_safety_state(
            state=replace(
                safety,
                max_order_notional=Decimal("600"),
                daily_notional_limit=Decimal("1000"),
                max_account_exposure_notional=Decimal("5000"),
                updated_at=datetime.now(tz=UTC),
            ),
            principal=principal,
        )
        _prove_expired_accepted_reservation(
            intent_repository=intent_repository,
            order_repository=order_repository,
            policy_repository=policy_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection=proof_connection,
            adapter=ExchangeExecutionEmulatorAdapter(exchange_name="bybit"),
            approval_id=approval.approval_id,
        )
        policy_service.set_account_safety_state(
            state=replace(safety, updated_at=datetime.now(tz=UTC)),
            principal=principal,
        )

        claim_intent = _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            scenario="postgres-claim-fencing",
        )
        claim_adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")
        _prove_postgres_claim_fencing(
            dsn=dsn,
            intent=claim_intent,
            connection=proof_connection,
            adapter=claim_adapter,
            approval_id=approval.approval_id,
            policy_repository=policy_repository,
        )

        normal = _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            scenario="normal",
        )
        normal_adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")
        normal_result = _process_once(
            intent=normal,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            process_repository=process_repository,
            intent_repository=intent_repository,
            order_repository=order_repository,
            gateway_policy_repository=policy_repository,
            adapter=normal_adapter,
        )
        duplicate_result = _process_once(
            intent=normal,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            process_repository=process_repository,
            intent_repository=intent_repository,
            order_repository=order_repository,
            gateway_policy_repository=policy_repository,
            adapter=normal_adapter,
        )
        if normal_result.submitted_count != 1 or duplicate_result.submitted_count != 0:
            raise RuntimeError("emulator normal/idempotent submit proof failed")
        if len(normal_adapter._orders) != 1:  # noqa: SLF001
            raise RuntimeError("duplicate intent produced a second emulator effect")

        credential_intent = _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            scenario="credential-rotated-during-flight",
        )
        credential_adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")
        rotating_resolver = _RotatingResolver(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        credential_result = _process_once(
            intent=credential_intent,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            process_repository=process_repository,
            intent_repository=intent_repository,
            order_repository=order_repository,
            gateway_policy_repository=policy_repository,
            adapter=credential_adapter,
            credential_resolver=rotating_resolver,
        )
        credential_order = order_repository.get_by_intent(
            organization_id=organization_id,
            intent_id=credential_intent.intent_id,
        )
        if (
            credential_result.guard_rejected_count != 1
            or rotating_resolver.resolve_count < 2
            or credential_adapter._orders  # noqa: SLF001
            or credential_order is None
            or credential_order.status_reason != "execution_secret_reference_unbound"
        ):
            raise RuntimeError("current credential was not re-resolved before submit")

        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE exchange_account_snapshots
                SET observed_at = %s, synced_at = %s
                WHERE organization_id = %s AND exchange_connection_id = %s
                """,
                (
                    datetime.now(tz=UTC) - timedelta(minutes=5),
                    datetime.now(tz=UTC) - timedelta(minutes=5),
                    organization_id.value,
                    connection_id,
                ),
            )
        stale_snapshot_intent = _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            scenario="stale-account-snapshot",
        )
        stale_snapshot_adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")
        stale_snapshot_result = _process_once(
            intent=stale_snapshot_intent,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            process_repository=process_repository,
            intent_repository=intent_repository,
            order_repository=order_repository,
            gateway_policy_repository=policy_repository,
            adapter=stale_snapshot_adapter,
        )
        stale_snapshot_order = order_repository.get_by_intent(
            organization_id=organization_id,
            intent_id=stale_snapshot_intent.intent_id,
        )
        if (
            stale_snapshot_result.guard_rejected_count != 1
            or stale_snapshot_adapter._orders  # noqa: SLF001
            or stale_snapshot_order is None
            or stale_snapshot_order.status_reason
            != "execution_account_snapshot_not_fresh"
        ):
            raise RuntimeError("stale account risk snapshot did not fail closed")
        refreshed_at = datetime.now(tz=UTC)
        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE exchange_account_snapshots
                SET observed_at = %s, synced_at = %s
                WHERE organization_id = %s AND exchange_connection_id = %s
                """,
                (refreshed_at, refreshed_at, organization_id.value, connection_id),
            )
            cursor.execute(
                """
                UPDATE exchange_account_config_guard_results
                SET checked_at = %s
                WHERE organization_id = %s AND exchange_connection_id = %s
                """,
                (refreshed_at, organization_id.value, connection_id),
            )

        unknown_checks: dict[str, bool] = {}
        for outcome in ("timeout_before_accept", "timeout_after_accept"):
            intent = _create_dispatched_intent(
                repository=intent_repository,
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                connection_id=connection_id,
                scenario=outcome,
            )
            client_order_id = f"rh1_{intent.idempotency_key_hash[:28]}"
            adapter = ExchangeExecutionEmulatorAdapter(
                exchange_name="bybit",
                scripted_outcomes={client_order_id: outcome},
            )
            first = _process_once(
                intent=intent,
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                connection_id=connection_id,
                process_repository=process_repository,
                intent_repository=intent_repository,
                order_repository=order_repository,
                gateway_policy_repository=policy_repository,
                adapter=adapter,
                expect_ack=False,
            )
            second = _process_once(
                intent=intent,
                organization_id=organization_id,
                owner_user_id=owner_user_id,
                connection_id=connection_id,
                process_repository=process_repository,
                intent_repository=intent_repository,
                order_repository=order_repository,
                gateway_policy_repository=policy_repository,
                adapter=adapter,
            )
            if first.adapter_error_count != 1 or second.submitted_count != 0:
                raise RuntimeError("unknown state did not reconcile before replay")
            if outcome == "timeout_before_accept":
                adapter.scripted_outcomes[client_order_id] = "accepted"
                third = _process_once(
                    intent=intent,
                    organization_id=organization_id,
                    owner_user_id=owner_user_id,
                    connection_id=connection_id,
                    process_repository=process_repository,
                    intent_repository=intent_repository,
                    order_repository=order_repository,
                    gateway_policy_repository=policy_repository,
                    adapter=adapter,
                )
                if third.submitted_count != 1:
                    raise RuntimeError("confirmed-absent order did not allow explicit replay")
            unknown_checks[outcome] = True

        kill_intent = _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            scenario="kill-during-flight",
        )
        kill_adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")
        kill_result = _process_once(
            intent=kill_intent,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            process_repository=process_repository,
            intent_repository=intent_repository,
            order_repository=order_repository,
            gateway_policy_repository=_KillOnPreSubmitPolicy(
                repository=policy_repository,
                service=policy_service,
                owner_user_id=owner_user_id,
                principal=principal,
            ),
            adapter=kill_adapter,
        )
        if kill_result.guard_rejected_count != 1 or kill_adapter._orders:  # noqa: SLF001
            raise RuntimeError("kill switch during flight did not block emulator submit")
        policy_service.set_kill_switch(
            state=ExecutionKillSwitchState(
                scope_type="account",
                organization_id=organization_id,
                exchange_connection_id=connection_id,
                active=False,
                reason="stage16_proof_resume",
                updated_by_user_id=owner_user_id,
                updated_at=datetime.now(tz=UTC),
            ),
            principal=principal,
        )

        stale_intent = _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            scenario="risk-revision-change",
        )
        policy_service.set_account_safety_state(
            state=replace(safety, risk_revision_hash="9" * 64, updated_at=datetime.now(tz=UTC)),
            principal=principal,
        )
        stale_result = _process_once(
            intent=stale_intent,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            process_repository=process_repository,
            intent_repository=intent_repository,
            order_repository=order_repository,
            gateway_policy_repository=policy_repository,
            adapter=ExchangeExecutionEmulatorAdapter(exchange_name="bybit"),
        )
        if stale_result.guard_rejected_count != 1:
            raise RuntimeError("material risk change did not invalidate mainnet approval")

        try:
            policy_service.register_provider(
                provider=replace(
                    provider,
                    provider_id="plugin:untrusted-execution",
                    provider_kind="plugin",
                ),
                principal=principal,
            )
        except ExecutionGatewayPolicyError:
            plugin_denied = True
        else:
            plugin_denied = False
        if not plugin_denied:
            raise RuntimeError("general plugin unexpectedly gained submit capability")

        policy_service.set_account_safety_state(
            state=replace(safety, updated_at=datetime.now(tz=UTC)),
            principal=principal,
        )
        approval = policy_service.revoke_mainnet(
            approval=approval,
            principal=principal,
            revoked_at=datetime.now(tz=UTC),
            reason="stage16_revocation_proof",
        )
        revoked_intent = _create_dispatched_intent(
            repository=intent_repository,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            scenario="revoked-approval",
        )
        revoked_adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")
        revoked_result = _process_once(
            intent=revoked_intent,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            process_repository=process_repository,
            intent_repository=intent_repository,
            order_repository=order_repository,
            gateway_policy_repository=policy_repository,
            adapter=revoked_adapter,
        )
        if revoked_result.guard_rejected_count != 1 or revoked_adapter._orders:  # noqa: SLF001
            raise RuntimeError("revoked mainnet approval did not block submit")

        _prove_audit_immutable(dsn)
        _prove_policy_delete_guards(dsn)
        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE identity_sessions SET revoked_at = %s WHERE session_id = %s",
                (datetime.now(tz=UTC), session_id),
            )
        try:
            policy_service.set_kill_switch(
                state=ExecutionKillSwitchState(
                    scope_type="account",
                    organization_id=organization_id,
                    exchange_connection_id=connection_id,
                    active=True,
                    reason="revoked_session_must_not_mutate_policy",
                    updated_by_user_id=owner_user_id,
                    updated_at=datetime.now(tz=UTC),
                ),
                principal=principal,
            )
        except RuntimeError:
            revoked_session_denied = True
        else:
            revoked_session_denied = False
        if not revoked_session_denied:
            raise RuntimeError("revoked identity session retained execution policy authority")
        counts = _db_counts(dsn)
        return {
            "schema": PROOF_SCHEMA,
            "status": "passed",
            "proof_boundary": "N/A",
            "mainnet_external_effects": False,
            "provider": "core:exchange-emulator",
            "checks": {
                "canonical_intent_and_idempotency": "passed",
                "persisted_owner_approval": "passed",
                "preflight_and_pre_submit_guard": "passed",
                "duplicate_submit": "passed",
                "postgres_concurrent_claim_and_restart": "passed",
                "claim_fenced_finalization": "passed",
                "claim_fenced_guard_rejection": "passed",
                "cross_intent_daily_limit_serialization": "passed",
                "cross_intent_exposure_serialization": "passed",
                "expired_accepted_reservation": "passed",
                "policy_audit_atomicity": "passed",
                "adapter_artifact_revision": "passed",
                "current_credential_reresolution": "passed",
                "current_risk_snapshot_freshness": "passed",
                "timeout_before_accept": (
                    "passed" if unknown_checks["timeout_before_accept"] else "failed"
                ),
                "timeout_after_accept": (
                    "passed" if unknown_checks["timeout_after_accept"] else "failed"
                ),
                "kill_switch_during_flight": "passed",
                "risk_revision_invalidation": "passed",
                "untrusted_plugin_denial": "passed",
                "approval_revocation": "passed",
                "audit_immutability": "passed",
                "policy_delete_guards": "passed",
                "identity_session_authority": "passed",
            },
            "database_counts": counts,
            "approval_revoked": approval.revoked_at is not None,
        }
    finally:
        if created:
            _run(["docker", "rm", "-f", "-v", postgres], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default="")
    args = parser.parse_args()
    proof = run_proof()
    rendered = json.dumps(proof, indent=2, sort_keys=True) + "\n"
    if args.evidence:
        destination = Path(args.evidence)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
