from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from apps.exchange_execution.adapters import ExchangeExecutionEmulatorAdapter
from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExchangeExecutionOrderRepository,
    InMemoryExchangeExecutionProcessRepository,
    InMemoryExecutionGatewayPolicyRepository,
    InMemoryExecutionIntentRepository,
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
    ExecutionAccountSafetyState,
    ExecutionAdapterIdentity,
    ExecutionGatewayPolicyError,
    ExecutionKillSwitchState,
    ExecutionMainnetApproval,
    ExecutionPolicyPrincipal,
    ExecutionProviderRegistration,
    ExecutionRiskContext,
    ExecutionSourceValidationError,
    ExecutionSubmitGuardQuery,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-4000-8000-000000016000")
_OWNER_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000016001")
_CONNECTION_ID = UUID("00000000-0000-4000-8000-000000016002")
_SESSION_ID = UUID("00000000-0000-4000-8000-000000016003")
_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def _principal() -> ExecutionPolicyPrincipal:
    return ExecutionPolicyPrincipal(
        user_id=_OWNER_USER_ID,
        session_id=_SESSION_ID,
        authenticated_at=_NOW - timedelta(minutes=1),
    )


class _Clock:
    def __init__(self) -> None:
        self.value = _NOW

    def now(self) -> datetime:
        current = self.value
        self.value += timedelta(milliseconds=1)
        return current


class _Consumer:
    def __init__(self, *, intent_id: UUID) -> None:
        self.message = ExchangeExecutionRedisMessage(
            stream_name="execution.requests.v1",
            message_id=f"1-{uuid4().int}",
            payload={
                "organization_id": str(_ORGANIZATION_ID),
                "owner_user_id": str(_OWNER_USER_ID),
                "intent_id": str(intent_id),
            },
        )
        self.acked = 0

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
        self.acked += 1


class _Resolver:
    def __init__(self, *, environment: str) -> None:
        credential_fields = {
            "api_" + "key": "emulator-public",
            "api_" + "secret": "emulator-private-placeholder",
        }
        self.connection = ExchangeExecutionConnection(
            connection_id=_CONNECTION_ID,
            organization_id=_ORGANIZATION_ID,
            owner_user_id=_OWNER_USER_ID,
            exchange_name="bybit",
            market_type="spot",
            environment=environment,
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
        assert organization_id == _ORGANIZATION_ID
        assert owner_user_id == _OWNER_USER_ID
        assert exchange_connection_id == _CONNECTION_ID
        return self.connection


class _KillOnPreSubmitPolicy:
    def __init__(self, *, repository: InMemoryExecutionGatewayPolicyRepository) -> None:
        self.repository = repository

    def evaluate_and_record(self, *, query: ExecutionSubmitGuardQuery):
        if query.phase == "pre_submit":
            state = ExecutionKillSwitchState(
                scope_type="account",
                organization_id=query.intent.organization_id,
                exchange_connection_id=query.intent.exchange_connection_id,
                active=True,
                reason="incident_during_flight",
                updated_by_user_id=query.intent.owner_user_id,
                updated_at=query.evaluated_at,
            )
            self.repository.kill_switches[
                (state.scope_type, state.organization_id, state.exchange_connection_id)
            ] = state
        return self.repository.evaluate_and_record(query=query)


def test_mainnet_policy_requires_owner_recent_auth_and_invalidates_material_changes() -> None:
    repository, service, provider, state = _policy(mode="mainnet")
    intent = _intent()
    connection = _Resolver(environment="mainnet").connection
    adapter = ExecutionAdapterIdentity(
        provider_id=provider.provider_id,
        provider_version=provider.provider_version,
        provider_kind=provider.provider_kind,
        exchange_name=provider.exchange_name,
        revision_hash=provider.revision_hash,
    )
    approval = replace(
        _approval(provider=provider, state=state),
        expires_at=_NOW + timedelta(minutes=3),
    )

    with pytest.raises(ExecutionGatewayPolicyError, match="mainnet_owner_approval_required"):
        service.approve_mainnet(
            approval=replace(approval, owner_user_id=UserId(uuid4())),
            principal=_principal(),
            now=_NOW,
        )
    accepted_approval = service.approve_mainnet(
        approval=approval,
        principal=_principal(),
        now=_NOW,
    )
    decision = repository.evaluate_and_record(
        query=_guard_query(intent=intent, connection=connection, adapter=adapter)
    )
    assert decision.accepted
    assert decision.approval_id == accepted_approval.approval_id
    assert decision.audit_event_id is not None

    expired = repository.evaluate_and_record(
        query=_guard_query(
            intent=intent,
            connection=connection,
            adapter=adapter,
            evaluated_at=_NOW + timedelta(minutes=4),
        )
    )
    assert expired.reason == "mainnet_approval_expired"

    repository.safety_states[(_ORGANIZATION_ID, _CONNECTION_ID)] = replace(
        state,
        risk_revision_hash="9" * 64,
    )
    invalidated = repository.evaluate_and_record(
        query=_guard_query(intent=intent, connection=connection, adapter=adapter)
    )
    assert invalidated.reason == "mainnet_approval_invalidated_by_risk_change"


def test_general_plugin_and_kill_switch_fail_closed() -> None:
    repository, service, provider, state = _policy(mode="testnet")
    plugin = replace(provider, provider_id="plugin:arbitrary", provider_kind="plugin")
    with pytest.raises(ExecutionGatewayPolicyError, match="execution_plugin_submit_forbidden"):
        service.register_provider(provider=plugin, principal=_principal())

    service.set_kill_switch(
        state=ExecutionKillSwitchState(
            scope_type="account",
            organization_id=_ORGANIZATION_ID,
            exchange_connection_id=_CONNECTION_ID,
            active=True,
            reason="operator_incident_stop",
            updated_by_user_id=_OWNER_USER_ID,
            updated_at=_NOW,
        ),
        principal=_principal(),
    )
    decision = repository.evaluate_and_record(
        query=_guard_query(
            intent=_intent(),
            connection=_Resolver(environment="testnet").connection,
            adapter=ExecutionAdapterIdentity(
                provider_id=provider.provider_id,
                provider_version=provider.provider_version,
                provider_kind=provider.provider_kind,
                exchange_name=provider.exchange_name,
                revision_hash=provider.revision_hash,
            ),
        )
    )
    assert decision.reason == "execution_kill_switch_active"
    assert state.risk_allows_submit


@pytest.mark.parametrize(
    ("connection_change", "expected_reason"),
    (
        ({"secret_reference_hash": "8" * 64}, "execution_secret_reference_unbound"),
        ({"account_revision_hash": "7" * 64}, "execution_account_revision_changed"),
    ),
)
def test_submit_rechecks_current_credential_and_account_revision(
    connection_change: dict[str, str], expected_reason: str
) -> None:
    repository, _service, provider, _state = _policy(mode="testnet")
    connection = replace(
        _Resolver(environment="testnet").connection,
        **connection_change,
    )
    decision = repository.evaluate_and_record(
        query=_guard_query(
            intent=_intent(),
            connection=connection,
            adapter=ExecutionAdapterIdentity(
                provider_id=provider.provider_id,
                provider_version=provider.provider_version,
                provider_kind=provider.provider_kind,
                exchange_name=provider.exchange_name,
                revision_hash=provider.revision_hash,
            ),
        )
    )

    assert decision.reason == expected_reason


def test_idempotency_key_is_bound_to_canonical_intent_payload() -> None:
    repository = InMemoryExecutionIntentRepository()
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=_OWNER_USER_ID,
            source_type="ops_test",
            source_event_ref="stage16-idempotency",
            source_ref_json={"scenario": "canonical-intent"},
            strategy_signal_id=None,
            idempotency_key="stage16-source",
        )
    )
    base = _intent_command(source_event_id=source.event.source_event_id)
    first = service.create_intent(command=base)
    duplicate = service.create_intent(command=base)
    assert duplicate.duplicate
    assert duplicate.intent.canonical_intent_hash == first.intent.canonical_intent_hash

    with pytest.raises(ExecutionSourceValidationError, match="idempotency_payload_mismatch"):
        service.create_intent(command=replace(base, quote_notional=Decimal("2")))


def test_submit_claim_allows_only_one_concurrent_submitter() -> None:
    repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent()
    command = ExchangeOrderCommand.from_intent(
        intent=intent,
        exchange_name="bybit",
        environment="testnet",
        client_order_id=f"rh1_{intent.idempotency_key_hash[:28]}",
    )
    first = repository.claim_submit(
        command=command,
        claim_id=uuid4(),
        claimed_at=_NOW,
        expires_at=_NOW + timedelta(seconds=30),
        submit_guard_audit_event_id=uuid4(),
        mainnet_approval_id=None,
    )
    second = repository.claim_submit(
        command=command,
        claim_id=uuid4(),
        claimed_at=_NOW + timedelta(seconds=1),
        expires_at=_NOW + timedelta(seconds=31),
        submit_guard_audit_event_id=uuid4(),
        mainnet_approval_id=None,
    )
    assert first.acquired
    assert not second.acquired
    assert second.reason == "submission_in_flight"


def test_kill_switch_activated_during_flight_blocks_submit() -> None:
    policy, _service, provider, state = _policy(mode="mainnet")
    approval = _approval(provider=provider, state=state)
    policy.approvals[approval.approval_id] = approval
    intent_repository = InMemoryExecutionIntentRepository()
    intent = _intent()
    intent_repository.record_intent(intent=intent)
    order_repository = InMemoryExchangeExecutionOrderRepository()
    adapter = ExchangeExecutionEmulatorAdapter(exchange_name="bybit")

    result = _process(
        intent=intent,
        intent_repository=intent_repository,
        order_repository=order_repository,
        policy=_KillOnPreSubmitPolicy(repository=policy),  # type: ignore[arg-type]
        adapter=adapter,
    ).run_once()

    assert result.guard_rejected_count == 1
    assert order_repository.orders[intent.intent_id].status_reason == (
        "execution_kill_switch_active"
    )
    assert adapter._orders == {}  # noqa: SLF001


def test_fresh_submit_claim_survives_worker_restart_without_ack_or_retry() -> None:
    policy, _service, provider, state = _policy(mode="mainnet")
    approval = _approval(provider=provider, state=state)
    policy.approvals[approval.approval_id] = approval
    intent_repository = InMemoryExecutionIntentRepository()
    intent = _intent()
    intent_repository.record_intent(intent=intent)
    order_repository = InMemoryExchangeExecutionOrderRepository()
    command = ExchangeOrderCommand.from_intent(
        intent=intent,
        exchange_name="bybit",
        environment="mainnet",
        client_order_id=f"rh1_{intent.idempotency_key_hash[:28]}",
    )
    claim = order_repository.claim_submit(
        command=command,
        claim_id=uuid4(),
        claimed_at=_NOW,
        expires_at=_NOW + timedelta(seconds=30),
        submit_guard_audit_event_id=uuid4(),
        mainnet_approval_id=approval.approval_id,
    )
    assert claim.acquired

    result = _process(
        intent=intent,
        intent_repository=intent_repository,
        order_repository=order_repository,
        policy=policy,
        adapter=ExchangeExecutionEmulatorAdapter(exchange_name="bybit"),
    ).run_once()

    assert result.acked_count == 0
    assert result.adapter_error_count == 0
    assert order_repository.orders[intent.intent_id].status == "submit_pending"


@pytest.mark.parametrize(
    ("outcome", "provider_present"),
    (("timeout_before_accept", False), ("timeout_after_accept", True)),
)
def test_emulator_unknown_state_reconciles_before_any_retry(
    outcome: str, provider_present: bool
) -> None:
    policy, _service, provider, state = _policy(mode="mainnet")
    approval = _approval(provider=provider, state=state)
    policy.approvals[approval.approval_id] = approval
    intent_repository = InMemoryExecutionIntentRepository()
    intent = _intent()
    intent_repository.record_intent(intent=intent)
    order_repository = InMemoryExchangeExecutionOrderRepository()
    client_order_id = f"rh1_{intent.idempotency_key_hash[:28]}"
    adapter = ExchangeExecutionEmulatorAdapter(
        exchange_name="bybit",
        scripted_outcomes={client_order_id: outcome},
    )

    first = _process(
        intent=intent,
        intent_repository=intent_repository,
        order_repository=order_repository,
        policy=policy,
        adapter=adapter,
    ).run_once()
    assert first.adapter_error_count == 1
    assert order_repository.orders[intent.intent_id].status == "unknown"

    second = _process(
        intent=intent,
        intent_repository=intent_repository,
        order_repository=order_repository,
        policy=policy,
        adapter=adapter,
    ).run_once()
    assert second.submitted_count == 0
    assert second.acked_count == 1
    assert second.reason == "emulator_adapter_processed"
    assert len(adapter._orders) == (1 if provider_present else 0)  # noqa: SLF001
    assert len(order_repository.reconciliation_runs) >= 1
    assert order_repository.reconciliation_runs[-1].reason == (
        "unknown_provider_status_matched"
        if provider_present
        else "unknown_provider_confirmed_absent"
    )

    if not provider_present:
        adapter.scripted_outcomes[client_order_id] = "accepted"
        third = _process(
            intent=intent,
            intent_repository=intent_repository,
            order_repository=order_repository,
            policy=policy,
            adapter=adapter,
        ).run_once()
        assert third.submitted_count == 1
        assert len(adapter._orders) == 1  # noqa: SLF001


def _policy(
    *, mode: str
) -> tuple[
    InMemoryExecutionGatewayPolicyRepository,
    ExecutionGatewayPolicyService,
    ExecutionProviderRegistration,
    ExecutionAccountSafetyState,
]:
    repository = InMemoryExecutionGatewayPolicyRepository()
    service = ExecutionGatewayPolicyService(repository=repository)
    provider = ExecutionProviderRegistration(
        provider_id="core:exchange-emulator",
        provider_version="v1",
        provider_kind="core",
        exchange_name="bybit",
        revision_hash=ExchangeExecutionEmulatorAdapter(
            exchange_name="bybit"
        ).revision_hash,
        order_submit_capability=True,
        enabled=True,
        approved_by_user_id=_OWNER_USER_ID,
        updated_at=_NOW,
    )
    service.register_provider(provider=provider, principal=_principal())
    state = ExecutionAccountSafetyState(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_OWNER_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        mode=mode,  # type: ignore[arg-type]
        risk_revision_hash="2" * 64,
        account_revision_hash="3" * 64,
        secret_reference_hash="4" * 64,
        risk_allows_submit=True,
        max_order_notional=Decimal("1000"),
        daily_notional_limit=Decimal("10000"),
        max_account_exposure_notional=Decimal("25000"),
        risk_valid_until=_NOW + timedelta(hours=1),
        updated_by_user_id=_OWNER_USER_ID,
        updated_at=_NOW,
    )
    service.set_account_safety_state(state=state, principal=_principal())
    return repository, service, provider, state


def _approval(
    *, provider: ExecutionProviderRegistration, state: ExecutionAccountSafetyState
) -> ExecutionMainnetApproval:
    return ExecutionMainnetApproval(
        approval_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_OWNER_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        exchange_name="bybit",
        market_type="spot",
        provider_id=provider.provider_id,
        risk_revision_hash=state.risk_revision_hash,
        account_revision_hash=state.account_revision_hash,
        provider_revision_hash=provider.revision_hash,
        recent_auth_session_id=_SESSION_ID,
        recent_auth_at=_NOW - timedelta(minutes=1),
        approved_at=_NOW,
        expires_at=_NOW + timedelta(minutes=10),
        audit_event_id=uuid4(),
    )


def _guard_query(
    *,
    intent: object,
    connection: ExchangeExecutionConnection,
    adapter: ExecutionAdapterIdentity,
    evaluated_at: datetime | None = None,
) -> ExecutionSubmitGuardQuery:
    return ExecutionSubmitGuardQuery(
        intent=intent,  # type: ignore[arg-type]
        connection=connection,
        adapter=adapter,
        phase="pre_submit",
        submission_attempt_id=uuid4(),
        evaluated_at=evaluated_at or _NOW + timedelta(minutes=1),
    )


def _intent():
    repository = InMemoryExecutionIntentRepository()
    service = ExecutionIngressService(repository=repository, clock=_Clock())
    source = service.record_source_event(
        command=RecordExecutionSourceEventCommand(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=_OWNER_USER_ID,
            source_type="ops_test",
            source_event_ref=f"stage16-{uuid4()}",
            source_ref_json={"scenario": "emulator"},
            strategy_signal_id=None,
            idempotency_key=f"stage16-source-{uuid4()}",
        )
    )
    created = service.create_intent(
        command=_intent_command(source_event_id=source.event.source_event_id)
    ).intent
    return replace(created, status="dispatched", status_reason="dispatch_recorded")


def _intent_command(*, source_event_id: UUID) -> CreateExecutionIntentCommand:
    return CreateExecutionIntentCommand(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_OWNER_USER_ID,
        source_event_id=source_event_id,
        idempotency_key="stage16-intent",
        exchange_connection_id=_CONNECTION_ID,
        market_type="spot",
        instrument_key="bybit:spot:BTCUSDT",
        order_type="market",
        side="buy",
        quantity=None,
        quote_notional=Decimal("1"),
        limit_price=None,
        advanced_order_flags={},
        constraints={"expires_at": (_NOW + timedelta(minutes=5)).isoformat()},
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


def _process(
    *,
    intent: object,
    intent_repository: InMemoryExecutionIntentRepository,
    order_repository: InMemoryExchangeExecutionOrderRepository,
    policy: InMemoryExecutionGatewayPolicyRepository,
    adapter: ExchangeExecutionEmulatorAdapter,
) -> ExchangeExecutionProcessService:
    return ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="emulator",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000_000_000,
            submit_claim_ttl_seconds=1,
        ),
        repository=InMemoryExchangeExecutionProcessRepository(),
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="mainnet"),
        order_adapters=(adapter,),
        gateway_policy_repository=policy,
        consumer=_Consumer(intent_id=intent.intent_id),  # type: ignore[attr-defined]
        clock=_Clock(),
    )
