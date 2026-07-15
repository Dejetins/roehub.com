"""Disposable PostgreSQL proof for trading organization isolation."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from trading.contexts.live_execution.adapters.outbound.persistence.postgres import (
    PostgresExchangeExecutionOrderRepository,
    PostgresExchangeExecutionProcessRepository,
    PostgresExecutionGatewayPolicyRepository,
    PostgresExecutionIntentRepository,
    PostgresExecutionRiskContextResolver,
    PostgresPaperAccountingRepository,
    PostgresStrategyPositionOwnershipRepository,
)
from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    CreateExecutionIntentCommand,
    ExchangeExecutionProcessConfig,
    ExchangeExecutionProcessService,
    ExecutionGatewayPolicyService,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
    StrategyPositionOwnershipService,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionRedisHealth,
    ExchangeExecutionRedisMessage,
    ExecutionDispatchPublishResult,
    ExecutionRiskContextQuery,
    ExecutionRiskContextResolutionError,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
    ExchangeOrderCancelResult,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExchangeOrderSubmitResult,
    ExchangePrivateStreamSession,
    ExecutionAccountSafetyState,
    ExecutionIntent,
    ExecutionPolicyPrincipal,
    ExecutionProviderRegistration,
    ExecutionRiskContext,
)
from trading.contexts.strategy.adapters.outbound.persistence.postgres import (
    PostgresStrategyRepository,
    PostgresStrategyRunRepository,
    PsycopgStrategyPostgresGateway,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategyRun, StrategySpecV1
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    OrganizationId,
    Symbol,
    Timeframe,
    UserId,
)


class TradingRuntimeProofError(RuntimeError):
    """Raised when disposable trading-tenancy evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class _Fixture:
    organizations: tuple[OrganizationId, OrganizationId]
    users: tuple[UserId, UserId]
    connections: tuple[UUID, UUID]
    sessions: tuple[UUID, UUID]


class _Clock:
    def __init__(self, *, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


class _SingleMessageConsumer:
    def __init__(self, *, message: ExchangeExecutionRedisMessage) -> None:
        self._message = message
        self._read = False
        self.acked: list[tuple[str, str]] = []

    def ensure_request_group(self) -> None:
        return None

    def health_snapshot(self) -> ExchangeExecutionRedisHealth:
        return ExchangeExecutionRedisHealth(
            request_stream_length=0 if self._read else 1,
            retry_stream_length=0,
            dlq_stream_length=0,
            pending_count=0,
            clock_drift_ms=0.0,
        )

    def read_new_requests(
        self, *, count: int, block_ms: int
    ) -> tuple[ExchangeExecutionRedisMessage, ...]:
        _ = count, block_ms
        if self._read:
            return ()
        self._read = True
        return (self._message,)

    def read_pending_requests(self, *, count: int) -> tuple[ExchangeExecutionRedisMessage, ...]:
        _ = count
        return ()

    def publish_dlq(
        self, *, message: ExchangeExecutionRedisMessage, reason: str
    ) -> ExecutionDispatchPublishResult:
        _ = message, reason
        raise TradingRuntimeProofError("valid trading proof message reached the dead-letter queue")

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        self.acked.append((stream_name, message_id))


class _TestnetConnectionResolver:
    def __init__(self, *, environment: str = "testnet") -> None:
        self._environment = environment

    def resolve(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
    ) -> ExchangeExecutionConnection:
        return ExchangeExecutionConnection(
            connection_id=exchange_connection_id,
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            exchange_name="bybit",
            market_type="spot",
            environment=self._environment,
            connection_readiness="ready_for_trading",
            effective_capability="trading",
            secret_reference_hash="4" * 64,
            account_revision_hash="3" * 64,
            credential=ExchangeExecutionCredential(
                api_key="<redacted>",
                api_secret="<redacted>",
            ),
        )


class _ReconciliationOnlyAdapter:
    exchange_name = "bybit"
    provider_id = "core:controlled-bybit"
    provider_version = "v1"
    provider_kind = "core"
    revision_hash = "b" * 64

    def __init__(self, *, now: datetime) -> None:
        self._now = now
        self.submitted = 0

    def server_time_ms(self) -> int:
        return int(self._now.timestamp() * 1000)

    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult:
        _ = command, credential
        self.submitted += 1
        raise TradingRuntimeProofError("unknown order state was resubmitted")

    def get_order_status(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult:
        _ = command, credential
        return ExchangeOrderStatusResult(
            exchange_order_id=exchange_order_id,
            exchange_status="new",
            checked_at=self._now,
            latency_ms=0.1,
            metadata={"provider": self.exchange_name},
        )

    def get_order_status_by_client_order_id(
        self,
        *,
        command: ExchangeOrderCommand,
        client_order_id: str,
        credential: object,
    ) -> ExchangeOrderStatusResult:
        return self.get_order_status(
            command=command,
            exchange_order_id=f"reconciled-{client_order_id}",
            credential=credential,
        )

    def cancel_order(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderCancelResult:
        _ = command, credential
        return ExchangeOrderCancelResult(
            exchange_order_id=exchange_order_id,
            exchange_status="cancelled",
            cancelled_at=self._now,
            latency_ms=0.1,
            metadata={"provider": self.exchange_name},
        )

    def ensure_private_stream_session(
        self, *, connection: ExchangeExecutionConnection
    ) -> ExchangePrivateStreamSession:
        return ExchangePrivateStreamSession(
            session_id=uuid4(),
            organization_id=connection.organization_id,
            exchange_name=connection.exchange_name,
            environment=connection.environment,
            market_type=connection.market_type,  # type: ignore[arg-type]
            status="ready",
            status_reason="disposable_probe",
            opened_at=self._now,
            keepalive_at=self._now,
            expires_at=None,
            metadata={"provider": connection.exchange_name},
        )


class _ControlledSubmitAdapter(_ReconciliationOnlyAdapter):
    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult:
        _ = command, credential
        self.submitted += 1
        return ExchangeOrderSubmitResult(
            exchange_order_id=f"controlled-{self.submitted}",
            exchange_status="new",
            submitted_at=self._now,
            latency_ms=0.1,
            metadata={"provider": self.exchange_name, "controlled": 1},
        )


def _testnet_gateway_policy(
    *,
    gateway: PsycopgStrategyPostgresGateway,
    organization_id: OrganizationId,
    user_id: UserId,
    connection_id: UUID,
    now: datetime,
) -> PostgresExecutionGatewayPolicyRepository:
    repository = PostgresExecutionGatewayPolicyRepository(gateway=gateway)
    service = ExecutionGatewayPolicyService(repository=repository)
    session_row = gateway.fetch_one(
        query="""
        SELECT session_id, created_at
        FROM identity_sessions
        WHERE user_id = %(user_id)s
          AND revoked_at IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """,
        parameters={"user_id": str(user_id)},
    )
    if session_row is None or not isinstance(session_row["created_at"], datetime):
        raise TradingRuntimeProofError("execution policy session fixture is unavailable")
    principal = ExecutionPolicyPrincipal(
        user_id=user_id,
        session_id=UUID(str(session_row["session_id"])),
        authenticated_at=session_row["created_at"],
    )
    service.register_provider(
        provider=ExecutionProviderRegistration(
            provider_id=_ReconciliationOnlyAdapter.provider_id,
            provider_version=_ReconciliationOnlyAdapter.provider_version,
            provider_kind="core",
            exchange_name="bybit",
            revision_hash=_ReconciliationOnlyAdapter.revision_hash,
            order_submit_capability=True,
            enabled=True,
            approved_by_user_id=user_id,
            updated_at=now,
        ),
        principal=principal,
    )
    service.set_account_safety_state(
        state=ExecutionAccountSafetyState(
            organization_id=organization_id,
            owner_user_id=user_id,
            exchange_connection_id=connection_id,
            mode="testnet",
            risk_revision_hash="2" * 64,
            account_revision_hash="3" * 64,
            secret_reference_hash="4" * 64,
            risk_allows_submit=True,
            max_order_notional=Decimal("1000"),
            daily_notional_limit=Decimal("10000"),
            max_account_exposure_notional=Decimal("25000"),
            risk_valid_until=now + timedelta(hours=1),
            updated_by_user_id=user_id,
            updated_at=now,
        ),
        principal=principal,
    )
    return repository


def run_probe(*, postgres_dsn: str) -> dict[str, object]:
    now = datetime.now(UTC)
    fixture = _seed_fixture(dsn=postgres_dsn, now=now)
    gateway = PsycopgStrategyPostgresGateway(dsn=postgres_dsn)
    strategy_repository = PostgresStrategyRepository(gateway=gateway)
    run_repository = PostgresStrategyRunRepository(gateway=gateway)
    intent_repository = PostgresExecutionIntentRepository(gateway=gateway)
    risk_context_resolver = PostgresExecutionRiskContextResolver(gateway=gateway)
    paper_repository = PostgresPaperAccountingRepository(gateway=gateway)
    ownership_repository = PostgresStrategyPositionOwnershipRepository(gateway=gateway)

    strategies: list[Strategy] = []
    runs: list[StrategyRun] = []
    for organization_id, user_id in zip(
        fixture.organizations, fixture.users, strict=True
    ):
        strategy = strategy_repository.create(
            strategy=Strategy.create(
                organization_id=organization_id,
                user_id=user_id,
                spec=_strategy_spec(),
                created_at=now,
            )
        )
        run = run_repository.create(
            run=StrategyRun.start(
                run_id=uuid4(),
                organization_id=organization_id,
                user_id=user_id,
                strategy_id=strategy.strategy_id,
                started_at=now,
                metadata_json={"proof": "stage10"},
            )
        )
        strategies.append(strategy)
        runs.append(run)

    database_constraints = {
        "strategy_membership": _expect_strategy_membership_rejection(
            repository=strategy_repository,
            organization_id=fixture.organizations[1],
            user_id=fixture.users[0],
            now=now,
        )
    }

    ingress = ExecutionIngressService(repository=intent_repository, clock=_Clock(now=now))
    source_results = []
    intent_results = []
    accepted_risk_contexts: list[ExecutionRiskContext] = []
    for index, (organization_id, user_id, connection_id) in enumerate(
        zip(
            fixture.organizations,
            fixture.users,
            fixture.connections,
            strict=True,
        )
    ):
        source = ingress.record_source_event(
            command=RecordExecutionSourceEventCommand(
                organization_id=organization_id,
                owner_user_id=user_id,
                source_type="ops_test",
                source_event_ref=f"stage10-source-{index}",
                source_ref_json={"proof": "stage10"},
                strategy_signal_id=None,
                idempotency_key="shared-source-key",
            )
        )
        durable_risk_context = risk_context_resolver.resolve(
            query=_risk_context_query(
                organization_id=organization_id,
                user_id=user_id,
                source_event_id=source.event.source_event_id,
                connection_id=connection_id,
            )
        )
        accepted_risk_context = _accepted_probe_risk_context(
            durable_context=durable_risk_context
        )
        intent = ingress.create_intent(
            command=_intent_command(
                organization_id=organization_id,
                user_id=user_id,
                source_event_id=source.event.source_event_id,
                connection_id=connection_id,
                idempotency_key="shared-intent-key",
                risk_context=accepted_risk_context,
            )
        )
        source_results.append(source)
        intent_results.append(intent)
        accepted_risk_contexts.append(accepted_risk_context)

    if source_results[0].event.idempotency_key_hash == source_results[1].event.idempotency_key_hash:
        raise TradingRuntimeProofError("source idempotency namespaces collided")
    if (
        intent_results[0].intent.idempotency_key_hash
        == intent_results[1].intent.idempotency_key_hash
    ):
        raise TradingRuntimeProofError("intent idempotency namespaces collided")
    replay = ingress.create_intent(
        command=_intent_command(
            organization_id=fixture.organizations[0],
            user_id=fixture.users[0],
            source_event_id=source_results[0].event.source_event_id,
            connection_id=fixture.connections[0],
            idempotency_key="shared-intent-key",
            risk_context=accepted_risk_contexts[0],
        )
    )
    if not replay.duplicate or replay.intent.intent_id != intent_results[0].intent.intent_id:
        raise TradingRuntimeProofError("same-organization intent replay was not deduplicated")
    if intent_repository.get_intent_by_id(
        organization_id=fixture.organizations[1],
        owner_user_id=fixture.users[1],
        intent_id=intent_results[0].intent.intent_id,
    ) is not None:
        raise TradingRuntimeProofError("cross-organization intent query leaked a row")

    account_ownership_mismatch = _expect_cross_organization_resolution_rejection(
        resolver=risk_context_resolver,
        organization_id=fixture.organizations[0],
        user_id=fixture.users[0],
        source_event_id=source_results[0].event.source_event_id,
        connection_id=fixture.connections[1],
    )
    database_constraints["intent_connection"] = _expect_intent_connection_rejection(
        ingress=ingress,
        organization_id=fixture.organizations[0],
        user_id=fixture.users[0],
        source_event_id=source_results[0].event.source_event_id,
        connection_id=fixture.connections[1],
        risk_context=accepted_risk_contexts[0],
    )
    http_risk_spoof = _prove_http_risk_spoof_rejected(
        organization_id=fixture.organizations[0],
        user_id=fixture.users[0],
        source_event_id=source_results[0].event.source_event_id,
        connection_id=fixture.connections[0],
        gateway=gateway,
    )

    denied_source = ingress.record_source_event(
        command=RecordExecutionSourceEventCommand(
            organization_id=fixture.organizations[0],
            owner_user_id=fixture.users[0],
            source_type="ops_test",
            source_event_ref="stage10-account-mismatch",
            source_ref_json={"proof": "stage10-account-mismatch"},
            strategy_signal_id=None,
            idempotency_key="account-mismatch-source-key",
        )
    )
    denied_intent = ingress.create_intent(
        command=_intent_command(
            organization_id=fixture.organizations[0],
            user_id=fixture.users[0],
            source_event_id=denied_source.event.source_event_id,
            connection_id=fixture.connections[0],
            idempotency_key="account-mismatch-intent-key",
            risk_context=risk_context_resolver.resolve(
                query=_risk_context_query(
                    organization_id=fixture.organizations[0],
                    user_id=fixture.users[0],
                    source_event_id=denied_source.event.source_event_id,
                    connection_id=fixture.connections[0],
                )
            ),
        )
    )
    if (
        denied_intent.intent.status != "rejected"
        or denied_intent.intent.risk_reason != "kill_switch_closed"
    ):
        raise TradingRuntimeProofError("server-derived risk denial was not enforced")

    paper_service = CapitalReservationPaperAccountingService(
        repository=paper_repository,
        account_projection_repository=None,
        clock=_Clock(now=now),
    )
    paper_snapshots = []
    for index, (organization_id, user_id, strategy, run) in enumerate(
        zip(
            fixture.organizations,
            fixture.users,
            strategies,
            runs,
            strict=True,
        )
    ):
        snapshot = paper_service.record_manual_paper_execution(
            organization_id=organization_id,
            owner_user_id=user_id,
            strategy_id=strategy.strategy_id,
            live_profile_id=None,
            strategy_run_id=run.run_id,
            source_event_id=source_results[index].event.source_event_id,
            instrument_key="bybit:spot:BTCUSDT",
            market_type="spot",
            side="buy",
            quote_notional=Decimal("25"),
            reference_price=Decimal("50000"),
            now=now,
        )
        paper_snapshots.append(snapshot)
    paper_replay = paper_service.record_manual_paper_execution(
        organization_id=fixture.organizations[0],
        owner_user_id=fixture.users[0],
        strategy_id=strategies[0].strategy_id,
        live_profile_id=None,
        strategy_run_id=runs[0].run_id,
        source_event_id=source_results[0].event.source_event_id,
        instrument_key="bybit:spot:BTCUSDT",
        market_type="spot",
        side="buy",
        quote_notional=Decimal("25"),
        reference_price=Decimal("50000"),
        now=now,
    )
    if paper_replay.accounting_id != paper_snapshots[0].accounting_id:
        raise TradingRuntimeProofError("paper replay created duplicate accounting")
    if paper_snapshots[0].accounting_id == paper_snapshots[1].accounting_id:
        raise TradingRuntimeProofError("paper accounting identities collided across organizations")
    if paper_repository.get_latest_accounting_for_strategy(
        organization_id=fixture.organizations[1],
        owner_user_id=fixture.users[1],
        strategy_id=strategies[0].strategy_id,
    ) is not None:
        raise TradingRuntimeProofError("cross-organization paper query leaked a row")

    ownership_service = StrategyPositionOwnershipService(repository=ownership_repository)
    ownerships = []
    for organization_id, user_id, connection_id, strategy, run in zip(
        fixture.organizations,
        fixture.users,
        fixture.connections,
        strategies,
        runs,
        strict=True,
    ):
        ownerships.append(
            ownership_service.reserve_for_strategy_run(
                organization_id=organization_id,
                owner_user_id=user_id,
                exchange_connection_id=connection_id,
                strategy_id=strategy.strategy_id,
                live_profile_id=None,
                strategy_run_id=run.run_id,
                market_type="spot",
                instrument_key="bybit:spot:BTCUSDT",
                position_mode="net",
                now=now,
            )
        )
    if ownership_repository.get_for_run(
        organization_id=fixture.organizations[1],
        owner_user_id=fixture.users[1],
        strategy_run_id=runs[0].run_id,
    ) is not None:
        raise TradingRuntimeProofError("cross-organization position query leaked a row")

    unknown_reconciliation = _prove_unknown_reconciliation(
        gateway=gateway,
        intent_repository=intent_repository,
        intent=intent_results[0].intent,
        now=now,
    )
    testnet_boundary = _prove_controlled_testnet_boundary(
        gateway=gateway,
        ingress=ingress,
        intent_repository=intent_repository,
        organization_id=fixture.organizations[0],
        user_id=fixture.users[0],
        connection_id=fixture.connections[0],
        risk_context=accepted_risk_contexts[0],
        now=now,
    )
    mainnet_guard = _prove_mainnet_hard_block(
        gateway=gateway,
        ingress=ingress,
        intent_repository=intent_repository,
        organization_id=fixture.organizations[0],
        user_id=fixture.users[0],
        connection_id=fixture.connections[0],
        risk_context=accepted_risk_contexts[0],
        now=now,
    )
    return {
        "schema": "io.roehub.trading-tenancy-runtime-proof/v1alpha1",
        "two_organization_paper": "passed",
        "cross_organization_repository_read": "rejected",
        "negative_authorization": http_risk_spoof,
        "client_risk_spoof": http_risk_spoof,
        "account_ownership_mismatch": account_ownership_mismatch,
        "risk_denial": denied_intent.intent.risk_reason,
        "duplicate_intent": "deduplicated",
        "position_ownership": "passed",
        "unknown_state_reconciliation": unknown_reconciliation,
        "private_stream_session": testnet_boundary["private_stream_session"],
        "request_observation": testnet_boundary["request_observation"],
        "controlled_testnet_submits": testnet_boundary["controlled_testnet_submits"],
        "mainnet_attempt": mainnet_guard["mainnet_attempt"],
        "mainnet_submits": mainnet_guard["mainnet_submits"],
        "database_constraints": database_constraints,
        "production_repositories": (
            "strategy,run,intent,paper,position,order,execution_process"
        ),
    }


def _intent_command(
    *,
    organization_id: OrganizationId,
    user_id: UserId,
    source_event_id: UUID,
    connection_id: UUID,
    idempotency_key: str,
    risk_context: ExecutionRiskContext,
) -> CreateExecutionIntentCommand:
    return CreateExecutionIntentCommand(
        organization_id=organization_id,
        owner_user_id=user_id,
        source_event_id=source_event_id,
        idempotency_key=idempotency_key,
        exchange_connection_id=connection_id,
        market_type="spot",
        instrument_key="bybit:spot:BTCUSDT",
        order_type="market",
        side="buy",
        quantity=None,
        quote_notional=Decimal("10"),
        limit_price=None,
        advanced_order_flags={},
        risk_context=risk_context,
    )


def _risk_context_query(
    *,
    organization_id: OrganizationId,
    user_id: UserId,
    source_event_id: UUID,
    connection_id: UUID,
) -> ExecutionRiskContextQuery:
    return ExecutionRiskContextQuery(
        organization_id=organization_id,
        owner_user_id=user_id,
        source_event_id=source_event_id,
        exchange_connection_id=connection_id,
        market_type="spot",
        instrument_key="bybit:spot:BTCUSDT",
    )


def _accepted_probe_risk_context(
    *, durable_context: ExecutionRiskContext
) -> ExecutionRiskContext:
    durable_facts = (
        durable_context.organization_ownership_verified,
        durable_context.account_ownership_verified,
        durable_context.exchange_connection_active,
        durable_context.secret_custody_ready,
        durable_context.source_authorized,
        durable_context.exchange_config_verified,
        durable_context.environment_policy_allows,
    )
    if not all(durable_facts):
        raise TradingRuntimeProofError("trusted durable risk facts are incomplete")
    # Limit and account-state services are outside the Stage 10 resolver. This
    # controlled policy fixture unlocks only the disposable ops-test flow after
    # durable organization/account/secret/config facts were resolved from DB.
    return ExecutionRiskContext(
        organization_ownership_verified=durable_context.organization_ownership_verified,
        account_ownership_verified=durable_context.account_ownership_verified,
        exchange_connection_active=durable_context.exchange_connection_active,
        secret_custody_ready=durable_context.secret_custody_ready,
        source_authorized=durable_context.source_authorized,
        exchange_config_verified=durable_context.exchange_config_verified,
        account_state_fresh=True,
        kill_switch_open=True,
        environment_policy_allows=durable_context.environment_policy_allows,
        max_order_size_ok=True,
        daily_limit_ok=True,
    )


def _expect_cross_organization_resolution_rejection(
    *,
    resolver: PostgresExecutionRiskContextResolver,
    organization_id: OrganizationId,
    user_id: UserId,
    source_event_id: UUID,
    connection_id: UUID,
) -> str:
    try:
        resolver.resolve(
            query=_risk_context_query(
                organization_id=organization_id,
                user_id=user_id,
                source_event_id=source_event_id,
                connection_id=connection_id,
            )
        )
    except ExecutionRiskContextResolutionError as error:
        if error.reason == "account_ownership_mismatch":
            return "rejected_by_server_resolver"
        raise TradingRuntimeProofError(
            f"unexpected cross-organization resolver reason: {error.reason}"
        ) from error
    raise TradingRuntimeProofError("cross-organization connection resolved successfully")


def _expect_intent_connection_rejection(
    *,
    ingress: ExecutionIngressService,
    organization_id: OrganizationId,
    user_id: UserId,
    source_event_id: UUID,
    connection_id: UUID,
    risk_context: ExecutionRiskContext,
) -> str:
    try:
        ingress.create_intent(
            command=_intent_command(
                organization_id=organization_id,
                user_id=user_id,
                source_event_id=source_event_id,
                connection_id=connection_id,
                idempotency_key="cross-org-connection-db-constraint",
                risk_context=risk_context,
            )
        )
    except psycopg.errors.ForeignKeyViolation as error:
        constraint_name = error.diag.constraint_name or "unknown"
        if constraint_name != "execution_intents_org_connection_fk":
            raise TradingRuntimeProofError(
                f"unexpected intent connection constraint: {constraint_name}"
            ) from error
        return constraint_name
    raise TradingRuntimeProofError("cross-organization intent connection write was accepted")


def _prove_http_risk_spoof_rejected(
    *,
    organization_id: OrganizationId,
    user_id: UserId,
    source_event_id: UUID,
    connection_id: UUID,
    gateway: PsycopgStrategyPostgresGateway,
) -> str:
    before = _table_count(gateway=gateway, table_name="execution_intents")
    request_model = _load_execution_intent_request_model()
    app = FastAPI()

    @app.post("/ui/execution/intents")
    async def post_intent(request: Request) -> dict[str, bool]:
        try:
            request_model.model_validate(await request.json())
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from error
        return {"accepted": True}

    response = TestClient(app).post(
        "/ui/execution/intents",
        headers={"x-user-id": str(user_id), "x-organization-id": str(organization_id)},
        json={
            "source_event_id": str(source_event_id),
            "idempotency_key": "client-risk-spoof",
            "exchange_connection_id": str(connection_id),
            "market_type": "spot",
            "instrument_key": "bybit:spot:BTCUSDT",
            "order": {
                "order_type": "market",
                "side": "buy",
                "quantity": "0.001",
            },
            "risk_context": {
                "organization_ownership_verified": True,
                "account_ownership_verified": True,
                "kill_switch_open": True,
                "environment_policy_allows": True,
            },
        },
    )
    after = _table_count(gateway=gateway, table_name="execution_intents")
    if response.status_code != 422 or after != before:
        raise TradingRuntimeProofError("HTTP client risk spoof was not rejected before write")
    return "rejected_422_no_write"


def _load_execution_intent_request_model() -> type[BaseModel]:
    module_path = Path(__file__).resolve().parents[1] / "api" / "dto" / "ui_execution.py"
    module_name = "_roehub_runtime_probe_ui_execution_dto"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise TradingRuntimeProofError("execution intent DTO module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model = getattr(module, "ExecutionIntentRequest", None)
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TradingRuntimeProofError("execution intent DTO model is unavailable")
    return model


def _table_count(
    *, gateway: PsycopgStrategyPostgresGateway, table_name: str
) -> int:
    if not table_name.replace("_", "").isalnum():
        raise TradingRuntimeProofError("invalid proof table name")
    row = gateway.fetch_one(
        query=f"SELECT COUNT(*) AS count FROM {table_name}",
        parameters={},
    )
    if row is None:
        raise TradingRuntimeProofError(f"proof table is unavailable: {table_name}")
    return int(row["count"])


def _record_dispatched_probe_intent(
    *,
    ingress: ExecutionIngressService,
    intent_repository: PostgresExecutionIntentRepository,
    organization_id: OrganizationId,
    user_id: UserId,
    connection_id: UUID,
    risk_context: ExecutionRiskContext,
    proof_key: str,
    now: datetime,
) -> ExecutionIntent:
    source = ingress.record_source_event(
        command=RecordExecutionSourceEventCommand(
            organization_id=organization_id,
            owner_user_id=user_id,
            source_type="ops_test",
            source_event_ref=f"stage10-{proof_key}",
            source_ref_json={"proof": proof_key},
            strategy_signal_id=None,
            idempotency_key=f"{proof_key}-source",
        )
    )
    created = ingress.create_intent(
        command=_intent_command(
            organization_id=organization_id,
            user_id=user_id,
            source_event_id=source.event.source_event_id,
            connection_id=connection_id,
            idempotency_key=f"{proof_key}-intent",
            risk_context=risk_context,
        )
    )
    if created.intent.risk_status != "accepted":
        raise TradingRuntimeProofError(
            f"controlled {proof_key} intent did not pass the risk gate"
        )
    claimed = intent_repository.claim_intent_for_dispatch(
        organization_id=organization_id,
        intent_id=created.intent.intent_id,
        now=now,
        retry_budget=3,
    )
    if claimed is None:
        raise TradingRuntimeProofError(f"controlled {proof_key} intent was not claimed")
    dispatched = intent_repository.mark_intent_dispatched(
        organization_id=organization_id,
        intent_id=created.intent.intent_id,
        stream_name="execution.requests.v1",
        redis_message_id=f"{proof_key}-1",
        now=now,
    )
    if dispatched is None:
        raise TradingRuntimeProofError(f"controlled {proof_key} intent was not dispatched")
    return dispatched


def _proof_message(
    *, intent: ExecutionIntent, message_id: str
) -> ExchangeExecutionRedisMessage:
    return ExchangeExecutionRedisMessage(
        stream_name="execution.requests.v1",
        message_id=message_id,
        payload={
            "organization_id": str(intent.organization_id),
            "owner_user_id": str(intent.owner_user_id),
            "intent_id": str(intent.intent_id),
        },
    )


def _prove_controlled_testnet_boundary(
    *,
    gateway: PsycopgStrategyPostgresGateway,
    ingress: ExecutionIngressService,
    intent_repository: PostgresExecutionIntentRepository,
    organization_id: OrganizationId,
    user_id: UserId,
    connection_id: UUID,
    risk_context: ExecutionRiskContext,
    now: datetime,
) -> dict[str, object]:
    intent = _record_dispatched_probe_intent(
        ingress=ingress,
        intent_repository=intent_repository,
        organization_id=organization_id,
        user_id=user_id,
        connection_id=connection_id,
        risk_context=risk_context,
        proof_key="controlled-testnet",
        now=now,
    )
    adapter = _ControlledSubmitAdapter(now=now)
    private_stream_events: list[tuple[str, str]] = []
    observations: list[tuple[str, str]] = []
    service_id = "stage10-controlled-testnet"
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            service_id=service_id,
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
        ),
        repository=PostgresExchangeExecutionProcessRepository(gateway=gateway),
        intent_repository=intent_repository,
        order_repository=PostgresExchangeExecutionOrderRepository(gateway=gateway),
        credential_resolver=_TestnetConnectionResolver(),
        order_adapters=(adapter,),
        gateway_policy_repository=_testnet_gateway_policy(
            gateway=gateway,
            organization_id=organization_id,
            user_id=user_id,
            connection_id=connection_id,
            now=now,
        ),
        consumer=_SingleMessageConsumer(
            message=_proof_message(intent=intent, message_id="stage10-testnet-1")
        ),
        clock=_Clock(now=now),
        on_observation=lambda status, reason: observations.append((status, reason)),
        on_private_stream=lambda exchange, reason: private_stream_events.append(
            (exchange, reason)
        ),
    )
    result = service.run_once()
    session_row = gateway.fetch_one(
        query="""
        SELECT status, status_reason
         FROM exchange_private_stream_sessions
         WHERE organization_id = %(organization_id)s
           AND exchange_connection_id = %(connection_id)s
         ORDER BY updated_at DESC
         LIMIT 1
        """,
        parameters={
            "organization_id": str(organization_id),
            "connection_id": str(connection_id),
        },
    )
    observation_row = gateway.fetch_one(
        query="""
        SELECT status, status_reason
          FROM exchange_execution_request_observations
         WHERE service_id = %(service_id)s
           AND intent_id = %(intent_id)s
         ORDER BY observed_at DESC
         LIMIT 1
        """,
        parameters={"service_id": service_id, "intent_id": str(intent.intent_id)},
    )
    if (
        result.submitted_count != 1
        or result.acked_count != 1
        or adapter.submitted != 1
        or private_stream_events != [("bybit", "disposable_probe")]
        or observations != [("testnet_submitted", "testnet_submit_status_recorded")]
        or session_row is None
        or session_row["status"] != "ready"
        or observation_row is None
        or observation_row["status"] != "testnet_submitted"
    ):
        raise TradingRuntimeProofError(
            "controlled testnet private-stream/observation boundary was incomplete"
        )
    return {
        "private_stream_session": "persisted_ready",
        "request_observation": "persisted_testnet_submitted",
        "controlled_testnet_submits": adapter.submitted,
    }


def _prove_mainnet_hard_block(
    *,
    gateway: PsycopgStrategyPostgresGateway,
    ingress: ExecutionIngressService,
    intent_repository: PostgresExecutionIntentRepository,
    organization_id: OrganizationId,
    user_id: UserId,
    connection_id: UUID,
    risk_context: ExecutionRiskContext,
    now: datetime,
) -> dict[str, object]:
    intent = _record_dispatched_probe_intent(
        ingress=ingress,
        intent_repository=intent_repository,
        organization_id=organization_id,
        user_id=user_id,
        connection_id=connection_id,
        risk_context=risk_context,
        proof_key="mainnet-hard-block",
        now=now,
    )
    adapter = _ControlledSubmitAdapter(now=now)
    observations: list[tuple[str, str]] = []
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            service_id="stage10-mainnet-hard-block",
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
        ),
        repository=PostgresExchangeExecutionProcessRepository(gateway=gateway),
        intent_repository=intent_repository,
        order_repository=PostgresExchangeExecutionOrderRepository(gateway=gateway),
        credential_resolver=_TestnetConnectionResolver(environment="mainnet"),
        order_adapters=(adapter,),
        gateway_policy_repository=_testnet_gateway_policy(
            gateway=gateway,
            organization_id=organization_id,
            user_id=user_id,
            connection_id=connection_id,
            now=now,
        ),
        consumer=_SingleMessageConsumer(
            message=_proof_message(intent=intent, message_id="stage10-mainnet-1")
        ),
        clock=_Clock(now=now),
        on_observation=lambda status, reason: observations.append((status, reason)),
    )
    result = service.run_once()
    if (
        result.guard_rejected_count != 1
        or result.acked_count != 1
        or adapter.submitted != 0
        or observations != [("guard_rejected", "mainnet_hard_block")]
    ):
        raise TradingRuntimeProofError("mainnet attempt was not stopped before submit")
    return {"mainnet_attempt": "guard_rejected_before_submit", "mainnet_submits": 0}


def _prove_unknown_reconciliation(
    *,
    gateway: PsycopgStrategyPostgresGateway,
    intent_repository: PostgresExecutionIntentRepository,
    intent: object,
    now: datetime,
) -> str:
    if not isinstance(intent, ExecutionIntent):
        raise TradingRuntimeProofError("execution intent proof fixture is invalid")
    claimed = intent_repository.claim_intent_for_dispatch(
        organization_id=intent.organization_id,
        intent_id=intent.intent_id,
        now=now,
        retry_budget=3,
    )
    if claimed is None:
        raise TradingRuntimeProofError("accepted intent could not be claimed for dispatch")
    dispatched = intent_repository.mark_intent_dispatched(
        organization_id=intent.organization_id,
        intent_id=intent.intent_id,
        stream_name="execution.requests.v1",
        redis_message_id="stage10-1",
        now=now,
    )
    if dispatched is None:
        raise TradingRuntimeProofError("accepted intent could not be marked dispatched")

    order_repository = PostgresExchangeExecutionOrderRepository(gateway=gateway)
    command = ExchangeOrderCommand.from_intent(
        intent=dispatched,
        exchange_name="bybit",
        environment="testnet",
        client_order_id=f"rh1_{dispatched.idempotency_key_hash[:28]}",
    )
    gateway_policy = _testnet_gateway_policy(
        gateway=gateway,
        organization_id=dispatched.organization_id,
        user_id=dispatched.owner_user_id,
        connection_id=dispatched.exchange_connection_id,
        now=now,
    )
    adapter = _ReconciliationOnlyAdapter(now=now)
    seed_audit_event_id = uuid4()
    seed_audit = gateway.fetch_one(
        query="""
        INSERT INTO execution_gateway_audit_events (
            event_id, organization_id, owner_user_id, exchange_connection_id,
            intent_id, approval_id, event_type, decision, reason,
            actor_user_id, created_at, metadata_json
        ) VALUES (
            %(event_id)s, %(organization_id)s, %(owner_user_id)s,
            %(exchange_connection_id)s, %(intent_id)s, NULL,
            'execution_unknown_state_fixture', 'accepted',
            'disposable_runtime_seed', NULL, %(created_at)s, '{}'::jsonb
        ) RETURNING event_id
        """,
        parameters={
            "event_id": str(seed_audit_event_id),
            "organization_id": str(dispatched.organization_id),
            "owner_user_id": str(dispatched.owner_user_id),
            "exchange_connection_id": str(dispatched.exchange_connection_id),
            "intent_id": str(dispatched.intent_id),
            "created_at": now,
        },
    )
    if seed_audit is None:
        raise TradingRuntimeProofError("unknown-state seed audit was not persisted")
    claim_id = uuid4()
    order_repository.claim_submit(
        command=command,
        claim_id=claim_id,
        claimed_at=now,
        expires_at=now + timedelta(seconds=30),
        submit_guard_audit_event_id=seed_audit_event_id,
        mainnet_approval_id=None,
    )
    order_repository.record_adapter_error(
        organization_id=dispatched.organization_id,
        intent_id=dispatched.intent_id,
        claim_id=claim_id,
        occurred_at=now + timedelta(seconds=1),
        reason="adapter_unknown_state_reconciliation_required",
    )
    message = ExchangeExecutionRedisMessage(
        stream_name="execution.requests.v1",
        message_id="stage10-1",
        payload={
            "organization_id": str(dispatched.organization_id),
            "owner_user_id": str(dispatched.owner_user_id),
            "intent_id": str(dispatched.intent_id),
        },
    )
    consumer = _SingleMessageConsumer(message=message)
    observations: list[tuple[str, str]] = []
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
        ),
        repository=PostgresExchangeExecutionProcessRepository(gateway=gateway),
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_TestnetConnectionResolver(),
        order_adapters=(adapter,),
        gateway_policy_repository=gateway_policy,
        consumer=consumer,
        clock=_Clock(now=now),
        on_observation=lambda status, reason: observations.append((status, reason)),
    )
    result = service.run_once()
    order = order_repository.get_by_intent(
        organization_id=dispatched.organization_id,
        intent_id=dispatched.intent_id,
    )
    if (
        result.acked_count != 1
        or adapter.submitted != 0
        or order is None
        or order.status != "status_checked"
        or observations != [
            ("reconciled", "unknown_state_reconciled_present_without_resubmit")
        ]
    ):
        raise TradingRuntimeProofError("unknown order state reconciliation was incomplete")
    return "matched_without_resubmit"


def _strategy_spec() -> StrategySpecV1:
    return StrategySpecV1(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        instrument_key="bybit:spot:BTCUSDT",
        market_type="spot",
        timeframe=Timeframe("1m"),
        indicators=({"name": "MA", "params": {"period": 20}},),
        signal_template="MA(20)",
    )


def _seed_fixture(
    *, dsn: str, now: datetime
) -> _Fixture:
    organization_values = (uuid4(), uuid4())
    user_values = (uuid4(), uuid4())
    connection_values = (uuid4(), uuid4())
    credential_values = (uuid4(), uuid4())
    session_values = (uuid4(), uuid4())
    snapshot_values = (uuid4(), uuid4())
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT installation_id FROM identity_installations WHERE singleton_key = TRUE"
        )
        row = cursor.fetchone()
        if row is None:
            raise TradingRuntimeProofError("disposable installation is unavailable")
        installation_id = row[0]
        cursor.executemany(
            """
            INSERT INTO identity_users (
                user_id, telegram_user_id, paid_level, created_at,
                last_login_at, is_deleted, keycloak_subject
            ) VALUES (%s, NULL, 'free', %s, %s, FALSE, NULL)
            """,
            [(user_id, now, now) for user_id in user_values],
        )
        cursor.executemany(
            """
            INSERT INTO identity_sessions (
                session_id, user_id, created_at, last_seen_at,
                idle_expires_at, absolute_expires_at, revoked_at
            ) VALUES (%s, %s, %s, %s, %s, %s, NULL)
            """,
            tuple(
                (
                    session_id,
                    user_id,
                    now,
                    now,
                    now + timedelta(hours=1),
                    now + timedelta(hours=2),
                )
                for session_id, user_id in zip(session_values, user_values, strict=True)
            ),
        )
        cursor.executemany(
            """
            INSERT INTO identity_installation_owners (
                installation_id, user_id, granted_by_user_id, granted_at
            ) VALUES (%s, %s, %s, %s)
            """,
            tuple(
                (installation_id, user_id, user_id, now)
                for user_id in user_values
            ),
        )
        cursor.executemany(
            """
            INSERT INTO identity_organizations (
                organization_id, installation_id, slug, display_name, status, created_at
            ) VALUES (%s, %s, %s, %s, 'active', %s)
            """,
            tuple(
                (
                    organization_id,
                    installation_id,
                    f"stage10-{index}-{organization_id.hex[:8]}",
                    f"Stage 10 {index}",
                    now,
                )
                for index, organization_id in enumerate(organization_values)
            ),
        )
        cursor.executemany(
            """
            INSERT INTO identity_memberships (
                organization_id, user_id, role, status, created_at, updated_at
            ) VALUES (%s, %s, 'owner', 'active', %s, %s)
            """,
            tuple(
                (organization_id, user_id, now, now)
                for organization_id, user_id in zip(
                    organization_values, user_values, strict=True
                )
            ),
        )
        cursor.executemany(
            """
            INSERT INTO exchange_connections (
                connection_id, organization_id, owner_user_id, exchange_name,
                market_type, environment, label, active_credential_version_id,
                status, status_reason, permission_summary_json,
                ip_restriction_status, created_at, updated_at,
                disabled_at, archived_at
            ) VALUES (
                %s, %s, %s, 'bybit', 'spot', 'testnet', 'Stage 10 proof', NULL,
                'active', 'disposable_runtime_proof', '{"permissions": ["trade"]}'::jsonb,
                'unknown', %s, %s, NULL, NULL
            )
            """,
            tuple(
                (connection_id, organization_id, user_id, now, now)
                for connection_id, organization_id, user_id in zip(
                    connection_values,
                    organization_values,
                    user_values,
                    strict=True,
                )
            ),
        )
        cursor.executemany(
            """
            INSERT INTO exchange_account_snapshots (
                account_snapshot_id, organization_id, owner_user_id,
                exchange_connection_id, exchange_name, market_type,
                environment, account_mode, source_hash, sync_status,
                sync_reason, observed_at, synced_at, balance_count,
                position_count, open_order_count, filter_count, metadata_json
            ) VALUES (
                %s, %s, %s, %s, 'bybit', 'spot', 'testnet', 'one_way',
                %s, 'fresh', 'disposable_runtime_proof', %s, %s,
                0, 0, 0, 0, '{}'::jsonb
            )
            """,
            tuple(
                (
                    snapshot_id,
                    organization_id,
                    user_id,
                    connection_id,
                    "6" * 64,
                    now,
                    now,
                )
                for snapshot_id, organization_id, user_id, connection_id in zip(
                    snapshot_values,
                    organization_values,
                    user_values,
                    connection_values,
                    strict=True,
                )
            ),
        )
        cursor.executemany(
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
            tuple(
                (uuid4(), organization_id, snapshot_id, user_id, connection_id, now)
                for organization_id, snapshot_id, user_id, connection_id in zip(
                    organization_values,
                    snapshot_values,
                    user_values,
                    connection_values,
                    strict=True,
                )
            ),
        )
        cursor.executemany(
            """
            INSERT INTO exchange_credential_versions (
                credential_version_id, organization_id, connection_id,
                api_key_ciphertext, api_secret_ciphertext, passphrase_ciphertext,
                api_key_last4, api_key_fingerprint_hmac, secret_cipher,
                transit_key_id, credential_scheme, status, created_by_user_id,
                created_by_session_id, created_at, rotated_at, disabled_at
            ) VALUES (
                %s, %s, %s, 'proof:key', 'proof:secret', NULL,
                'test', %s, 'disposable_proof', 'disposable_proof',
                'disposable_proof_v1', 'active', %s, NULL, %s, NULL, NULL
            )
            """,
            tuple(
                (
                    credential_id,
                    organization_id,
                    connection_id,
                    b"stage10-disposable-fingerprint",
                    user_id,
                    now,
                )
                for credential_id, organization_id, connection_id, user_id in zip(
                    credential_values,
                    organization_values,
                    connection_values,
                    user_values,
                    strict=True,
                )
            ),
        )
        cursor.executemany(
            """
            UPDATE exchange_connections
               SET active_credential_version_id = %s,
                   updated_at = %s
             WHERE organization_id = %s
               AND connection_id = %s
            """,
            tuple(
                (credential_id, now, organization_id, connection_id)
                for credential_id, organization_id, connection_id in zip(
                    credential_values,
                    organization_values,
                    connection_values,
                    strict=True,
                )
            ),
        )
    return _Fixture(
        organizations=(
            OrganizationId(organization_values[0]),
            OrganizationId(organization_values[1]),
        ),
        users=(UserId(user_values[0]), UserId(user_values[1])),
        connections=connection_values,
        sessions=session_values,
    )


def _expect_strategy_membership_rejection(
    *,
    repository: PostgresStrategyRepository,
    organization_id: OrganizationId,
    user_id: UserId,
    now: datetime,
) -> str:
    try:
        repository.create(
            strategy=Strategy.create(
                organization_id=organization_id,
                user_id=user_id,
                spec=_strategy_spec(),
                created_at=now,
            )
        )
    except psycopg.errors.ForeignKeyViolation:
        return "rejected"
    raise TradingRuntimeProofError("cross-organization strategy write was accepted")


def _bounded_error_message(error: Exception) -> str:
    message = " ".join(str(error).split())
    message = re.sub(r"password=[^\s]+", "password=<redacted>", message)
    message = re.sub(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
        r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
        "[uuid]",
        message,
    )
    return message[:240] or type(error).__name__


def main() -> int:
    if os.environ.get("ROEHUB_DISPOSABLE_STORAGE_PROOF") != "1":
        print("trading runtime proof failed: disposable proof guard is not enabled")
        return 1
    postgres_dsn = os.environ.get("ROEHUB_STORAGE_POSTGRES_DSN", "").strip()
    if not postgres_dsn:
        print("trading runtime proof failed: PostgreSQL DSN is unavailable")
        return 1
    try:
        result = run_probe(postgres_dsn=postgres_dsn)
    except Exception as error:  # noqa: BLE001
        print(f"trading runtime proof failed: {_bounded_error_message(error)}")
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
