from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExchangeExecutionOrderRepository,
    InMemoryExchangeExecutionProcessRepository,
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import (
    ExchangeExecutionProcessConfig,
    ExchangeExecutionProcessService,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionRedisHealth,
    ExchangeExecutionRedisMessage,
    ExchangeOrderAdapterError,
    ExecutionDispatchPublishResult,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
    ExchangeOrderCancelResult,
    ExchangeOrderCommand,
    ExchangeOrderStatusResult,
    ExchangeOrderSubmitResult,
    ExchangePrivateStreamSession,
    ExecutionFillFact,
    ExecutionIntent,
)
from trading.shared_kernel.primitives import UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000013001")
_NOW = datetime(2026, 5, 31, 18, 0, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self._index = 0

    def now(self) -> datetime:
        value = _NOW + timedelta(seconds=self._index)
        self._index += 1
        return value


class _LimiterClock:
    def __init__(self) -> None:
        self._now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += seconds


class _Consumer:
    def __init__(
        self,
        messages: tuple[ExchangeExecutionRedisMessage, ...],
        pending_messages: tuple[ExchangeExecutionRedisMessage, ...] = (),
    ) -> None:
        self.messages = messages
        self.pending_messages = pending_messages
        self.groups_ensured = 0
        self.dlq: list[tuple[str, str]] = []
        self.acked: list[tuple[str, str]] = []

    def ensure_request_group(self) -> None:
        self.groups_ensured += 1

    def health_snapshot(self) -> ExchangeExecutionRedisHealth:
        return ExchangeExecutionRedisHealth(
            request_stream_length=len(self.messages),
            retry_stream_length=0,
            dlq_stream_length=len(self.dlq),
            pending_count=0,
            clock_drift_ms=8.0,
        )

    def read_new_requests(
        self, *, count: int, block_ms: int
    ) -> tuple[ExchangeExecutionRedisMessage, ...]:
        _ = block_ms
        return self.messages[:count]

    def read_pending_requests(self, *, count: int) -> tuple[ExchangeExecutionRedisMessage, ...]:
        return self.pending_messages[:count]

    def publish_dlq(
        self, *, message: ExchangeExecutionRedisMessage, reason: str
    ) -> ExecutionDispatchPublishResult:
        self.dlq.append((message.message_id, reason))
        return ExecutionDispatchPublishResult(
            stream_name="execution.requests.dlq.v1",
            message_id=f"dlq-{len(self.dlq)}",
        )

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None:
        self.acked.append((stream_name, message_id))


def test_readiness_records_heartbeat_and_reports_adapter_disabled_degraded() -> None:
    repository = InMemoryExchangeExecutionProcessRepository()
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(consumer_enabled=True),
        repository=repository,
        intent_repository=InMemoryExecutionIntentRepository(),
        consumer=_Consumer(messages=()),
        clock=_Clock(),
    )

    snapshot = service.readiness()

    assert snapshot.status == "degraded"
    assert snapshot.status_reason == "adapter_disabled_stage13"
    assert repository.heartbeats["exchange-execution"].status == "degraded"
    assert {item.name for item in snapshot.dependencies} >= {
        "postgres",
        "redis",
        "backpressure",
        "dlq",
        "clock_drift",
        "adapter",
        "rate_limit",
    }


def test_valid_dispatched_intent_is_observed_but_not_acked_with_adapter_disabled() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    message = _message(
        payload={
            "intent_id": str(intent.intent_id),
            "owner_user_id": str(intent.owner_user_id),
        }
    )
    consumer = _Consumer(messages=(message,))
    observations: list[tuple[str, str]] = []
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(consumer_enabled=True),
        repository=process_repository,
        intent_repository=intent_repository,
        consumer=consumer,
        clock=_Clock(),
        on_observation=lambda status, reason: observations.append((status, reason)),
    )

    result = service.run_once()

    assert result.read_count == 1
    assert result.observed_count == 1
    assert result.quarantined_count == 0
    assert result.acked_count == 0
    assert consumer.acked == []
    assert process_repository.observations[0].status == "adapter_disabled"
    assert process_repository.observations[0].status_reason == "adapter_disabled_stage13"
    assert observations == [("adapter_disabled", "adapter_disabled_stage13")]


def test_invalid_or_non_dispatchable_message_is_quarantined_to_dlq_and_acked() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(uuid4()),
                    "owner_user_id": str(_USER_ID),
                }
            ),
        )
    )
    dlq_reasons: list[str] = []
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(consumer_enabled=True),
        repository=process_repository,
        intent_repository=InMemoryExecutionIntentRepository(),
        consumer=consumer,
        clock=_Clock(),
        on_dlq=dlq_reasons.append,
    )

    result = service.run_once()

    assert result.quarantined_count == 1
    assert result.acked_count == 1
    assert process_repository.observations[0].status == "quarantined"
    assert process_repository.observations[0].status_reason == "intent_not_found"
    assert consumer.dlq == [("1-0", "intent_not_found")]
    assert consumer.acked == [("execution.requests.v1", "1-0")]
    assert dlq_reasons == ["intent_not_found"]


def test_run_once_recovers_pending_messages_before_reading_new_requests() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    first_intent = _intent(status="dispatched", risk_status="accepted")
    second_intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=first_intent)
    intent_repository.record_intent(intent=second_intent)
    pending_message = _message(
        payload={
            "intent_id": str(first_intent.intent_id),
            "owner_user_id": str(first_intent.owner_user_id),
        }
    )
    new_message = ExchangeExecutionRedisMessage(
        stream_name="execution.requests.v1",
        message_id="2-0",
        payload={
            "intent_id": str(second_intent.intent_id),
            "owner_user_id": str(second_intent.owner_user_id),
        },
    )
    consumer = _Consumer(messages=(new_message,), pending_messages=(pending_message,))
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(consumer_enabled=True),
        repository=process_repository,
        intent_repository=intent_repository,
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.read_count == 2
    assert result.observed_count == 2
    assert [item.redis_message_id for item in process_repository.observations] == [
        "1-0",
        "2-0",
    ]


def test_testnet_adapter_records_submit_status_cancel_before_ack() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        )
    )
    adapter = _Adapter(exchange_name="bybit", include_fills=False)
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            max_clock_drift_ms=10_000,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="testnet"),
        order_adapters=(adapter,),
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.submitted_count == 1
    assert result.acked_count == 1
    assert consumer.acked == [("execution.requests.v1", "1-0")]
    assert order_repository.orders[intent.intent_id].status == "cancelled"
    assert order_repository.orders[intent.intent_id].exchange_order_id == "order-1"
    assert process_repository.observations[0].status == "testnet_submitted"
    assert order_repository.private_stream_sessions[intent.exchange_connection_id].status == "ready"
    assert len(order_repository.order_events) == 5
    assert len(order_repository.fills) == 0
    assert len(order_repository.reconciliation_runs) == 1
    assert order_repository.reconciliation_runs[0].status == "matched"
    assert order_repository.reconciliation_runs[0].reason == "spot_order_status_matched"
    assert adapter.submitted == 1


def test_testnet_adapter_respects_rate_limit_before_adapter_calls() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        )
    )
    limiter_clock = _LimiterClock()
    waits: list[tuple[str, str, float]] = []
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
            rate_limit_per_second=1.0,
            rate_limit_burst=1,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="testnet"),
        order_adapters=(_Adapter(exchange_name="bybit"),),
        consumer=consumer,
        clock=_Clock(),
        on_rate_limit_wait=lambda exchange, operation, wait: waits.append(
            (exchange, operation, wait)
        ),
        rate_limit_monotonic=limiter_clock.monotonic,
        rate_limit_sleep=limiter_clock.sleep,
    )

    result = service.run_once()

    assert result.submitted_count == 1
    assert [operation for _exchange, operation, _wait in waits] == [
        "private_stream",
        "submit",
        "status",
    ]
    assert [round(wait, 3) for _exchange, _operation, wait in waits] == [1.0, 1.0, 1.0]
    assert limiter_clock.sleeps == [1.0, 1.0, 1.0]


def test_testnet_adapter_canary_can_record_fill_without_cancel() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        )
    )
    adapter = _Adapter(exchange_name="bybit")
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="testnet"),
        order_adapters=(adapter,),
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.submitted_count == 1
    assert result.acked_count == 1
    assert process_repository.observations[0].status == "testnet_submitted"
    assert process_repository.observations[0].status_reason == "testnet_submit_status_recorded"
    assert order_repository.orders[intent.intent_id].status == "status_checked"
    assert len(order_repository.order_events) == 4
    assert {item.event_type for item in order_repository.order_events} == {
        "submit_pending",
        "private_stream_backfill",
        "submitted",
        "status_checked",
    }
    assert len(order_repository.fills) == 1
    assert len(order_repository.reconciliation_runs) == 1
    assert order_repository.reconciliation_runs[0].status == "matched"
    assert adapter.submitted == 1


def test_testnet_adapter_skips_cancel_when_market_order_already_filled() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        )
    )
    adapter = _Adapter(exchange_name="bybit", cancel_reason="filled_order_cannot_cancel")
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=True,
            max_clock_drift_ms=10_000,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="testnet"),
        order_adapters=(adapter,),
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.submitted_count == 1
    assert result.adapter_error_count == 0
    assert process_repository.observations[0].status == "testnet_submitted"
    assert process_repository.observations[0].status_reason == "testnet_submit_status_recorded"
    assert order_repository.orders[intent.intent_id].status == "status_checked"
    assert len(order_repository.fills) == 1
    assert order_repository.reconciliation_runs[0].status == "matched"
    assert adapter.cancelled == 0


def test_testnet_futures_fill_without_funding_is_matched_order_reconciliation() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(
        status="dispatched",
        risk_status="accepted",
        instrument_key="bybit:futures:BTCUSDT",
        market_type="futures",
        side="sell",
    )
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        )
    )
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="testnet", market_type="futures"),
        order_adapters=(_Adapter(exchange_name="bybit"),),
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.submitted_count == 1
    assert result.adapter_error_count == 0
    assert process_repository.observations[0].status == "testnet_submitted"
    assert order_repository.orders[intent.intent_id].status == "status_checked"
    assert len(order_repository.fills) == 1
    assert order_repository.reconciliation_runs[0].status == "matched"
    assert (
        order_repository.reconciliation_runs[0].reason
        == "futures_order_status_and_fills_matched"
    )
    assert order_repository.reconciliation_runs[0].funding_event_count == 0


def test_testnet_adapter_accepts_dispatching_intent_after_redis_publish_race() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(status="dispatching", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        )
    )
    adapter = _Adapter(exchange_name="bybit")
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            cancel_after_submit=False,
            max_clock_drift_ms=10_000,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="testnet"),
        order_adapters=(adapter,),
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.submitted_count == 1
    assert result.quarantined_count == 0
    assert result.acked_count == 1
    assert consumer.dlq == []
    assert process_repository.observations[0].status == "testnet_submitted"
    assert order_repository.orders[intent.intent_id].status == "status_checked"
    assert adapter.submitted == 1


def test_testnet_adapter_hard_blocks_mainnet_connection_before_ack() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(status="dispatched", risk_status="accepted")
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        )
    )
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            max_clock_drift_ms=10_000,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="mainnet"),
        order_adapters=(_Adapter(exchange_name="bybit"),),
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.guard_rejected_count == 1
    assert result.acked_count == 1
    assert order_repository.orders[intent.intent_id].status == "guard_rejected"
    assert order_repository.orders[intent.intent_id].status_reason == "mainnet_hard_block"
    assert order_repository.order_events[0].event_type == "guard_rejected"
    assert intent_repository.notifications[0].event_type == "producer_order_rejected"
    assert intent_repository.notifications[0].reason == "mainnet_hard_block"
    assert process_repository.observations[0].status == "guard_rejected"


def test_testnet_adapter_rejects_unsupported_exchange_without_order_row() -> None:
    process_repository = InMemoryExchangeExecutionProcessRepository()
    intent_repository = InMemoryExecutionIntentRepository()
    order_repository = InMemoryExchangeExecutionOrderRepository()
    intent = _intent(
        status="dispatched",
        risk_status="accepted",
        instrument_key="codexstage16:spot:BTCUSDT",
    )
    intent_repository.record_intent(intent=intent)
    consumer = _Consumer(
        pending_messages=(
            _message(
                payload={
                    "intent_id": str(intent.intent_id),
                    "owner_user_id": str(intent.owner_user_id),
                }
            ),
        ),
        messages=(),
    )
    service = ExchangeExecutionProcessService(
        config=ExchangeExecutionProcessConfig(
            adapter_mode="testnet",
            consumer_enabled=True,
            max_clock_drift_ms=10_000,
        ),
        repository=process_repository,
        intent_repository=intent_repository,
        order_repository=order_repository,
        credential_resolver=_Resolver(environment="testnet"),
        order_adapters=(_Adapter(exchange_name="bybit"),),
        consumer=consumer,
        clock=_Clock(),
    )

    result = service.run_once()

    assert result.guard_rejected_count == 1
    assert result.acked_count == 1
    assert consumer.acked == [("execution.requests.v1", "1-0")]
    assert process_repository.observations[0].status == "guard_rejected"
    assert process_repository.observations[0].status_reason == "exchange_adapter_not_enabled"
    assert order_repository.orders == {}
    assert order_repository.order_events == []


def _message(*, payload: dict[str, str]) -> ExchangeExecutionRedisMessage:
    return ExchangeExecutionRedisMessage(
        stream_name="execution.requests.v1",
        message_id="1-0",
        payload=payload,
    )


def _intent(
    *,
    status: str,
    risk_status: str,
    instrument_key: str = "bybit:spot:BTCUSDT",
    market_type: str = "spot",
    side: str = "buy",
) -> ExecutionIntent:
    return ExecutionIntent(
        intent_id=uuid4(),
        source_event_id=uuid4(),
        owner_user_id=_USER_ID,
        source_type="ops_test",
        strategy_signal_id=None,
        exchange_connection_id=UUID("00000000-0000-0000-0000-000000013101"),
        market_type=market_type,  # type: ignore[arg-type]
        instrument_key=instrument_key,
        side=side,  # type: ignore[arg-type]
        order_type="market",
        quantity=Decimal("0.01"),
        quote_notional=None,
        limit_price=None,
        status=status,  # type: ignore[arg-type]
        status_reason="redis_xadd_ok",
        risk_status=risk_status,
        risk_reason="risk_gate_accepted",
        idempotency_key_hash="b" * 64,
        created_at=_NOW,
        dispatch_attempt_count=1,
        dispatch_stream_name="execution.requests.v1",
        dispatch_redis_message_id="1-0",
        dispatch_updated_at=_NOW,
    )


class _Resolver:
    def __init__(self, *, environment: str, market_type: str = "spot") -> None:
        self._environment = environment
        self._market_type = market_type

    def resolve(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeExecutionConnection:
        return ExchangeExecutionConnection(
            connection_id=exchange_connection_id,
            owner_user_id=owner_user_id,
            exchange_name="bybit",
            market_type=self._market_type,  # type: ignore[arg-type]
            environment=self._environment,
            connection_readiness="ready_for_trading",
            effective_capability="trading",
            credential=ExchangeExecutionCredential(
                api_key="test-key",
                api_secret="test-secret",
            ),
        )


class _Adapter:
    def __init__(
        self,
        *,
        exchange_name: str,
        cancel_reason: str | None = None,
        include_fills: bool = True,
    ) -> None:
        self.exchange_name = exchange_name
        self.submitted = 0
        self.cancelled = 0
        self._cancel_reason = cancel_reason
        self._include_fills = include_fills

    def server_time_ms(self) -> int:
        return int(_NOW.timestamp() * 1000)

    def ensure_private_stream_session(
        self,
        *,
        connection: ExchangeExecutionConnection,
    ) -> ExchangePrivateStreamSession:
        return ExchangePrivateStreamSession(
            session_id=UUID("00000000-0000-0000-0000-000000013901"),
            exchange_name=connection.exchange_name,
            environment=connection.environment,
            market_type=connection.market_type,
            status="ready",
            status_reason="listen_key_keepalive_ok",
            opened_at=_NOW,
            keepalive_at=_NOW,
            expires_at=None,
            metadata={"provider": connection.exchange_name},
        )

    def submit_order(
        self, *, command: ExchangeOrderCommand, credential: object
    ) -> ExchangeOrderSubmitResult:
        _ = command, credential
        self.submitted += 1
        return ExchangeOrderSubmitResult(
            exchange_order_id="order-1",
            exchange_status="new",
            submitted_at=_NOW,
            latency_ms=1.5,
            metadata={"provider": self.exchange_name},
        )

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
            checked_at=_NOW,
            latency_ms=1.0,
            metadata={"provider": self.exchange_name},
            fills=(
                (
                    ExecutionFillFact(
                        provider_trade_id="trade-1",
                        price=Decimal("20"),
                        quantity=Decimal("0.01"),
                        fee_amount=Decimal("0.001"),
                        fee_asset="USDT",
                        filled_at=_NOW,
                        liquidity="taker",
                        metadata={"provider": self.exchange_name},
                    ),
                )
                if self._include_fills
                else ()
            ),
        )

    def cancel_order(
        self,
        *,
        command: ExchangeOrderCommand,
        exchange_order_id: str,
        credential: object,
    ) -> ExchangeOrderCancelResult:
        _ = command, credential
        self.cancelled += 1
        if self._cancel_reason is not None:
            raise ExchangeOrderAdapterError(reason=self._cancel_reason)
        return ExchangeOrderCancelResult(
            exchange_order_id=exchange_order_id,
            exchange_status="cancelled",
            cancelled_at=_NOW,
            latency_ms=1.0,
            metadata={"provider": self.exchange_name},
        )
