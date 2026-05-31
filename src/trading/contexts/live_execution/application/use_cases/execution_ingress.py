from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Mapping
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import (
    ExecutionIntentRepository,
    LiveExecutionClock,
)
from trading.contexts.live_execution.domain import (
    ExecutionIntent,
    ExecutionOrderModelRejectedError,
    ExecutionRequest,
    ExecutionSourceEvent,
    ExecutionSourceValidationError,
    hash_idempotency_key,
    validate_order_model,
    validate_source_event_fields,
)
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class RecordExecutionSourceEventCommand:
    owner_user_id: UserId
    source_type: str
    source_event_ref: str
    source_ref_json: Mapping[str, str]
    strategy_signal_id: UUID | None
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class CreateExecutionIntentCommand:
    owner_user_id: UserId
    source_event_id: UUID
    idempotency_key: str
    exchange_connection_id: UUID
    market_type: str
    instrument_key: str
    order_type: str
    side: str
    quantity: Decimal | None
    quote_notional: Decimal | None
    limit_price: Decimal | None
    advanced_order_flags: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ExecutionSourceEventResult:
    event: ExecutionSourceEvent
    duplicate: bool


@dataclass(frozen=True, slots=True)
class ExecutionIntentResult:
    event: ExecutionSourceEvent
    intent: ExecutionIntent
    duplicate: bool


class ExecutionIngressService:
    def __init__(
        self,
        *,
        repository: ExecutionIntentRepository,
        clock: LiveExecutionClock,
        on_source_event: Callable[[str, str], None] | None = None,
        on_intent: Callable[[str, str, str], None] | None = None,
        on_order_model_rejected: Callable[[str, str], None] | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionIngressService requires repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionIngressService requires clock")
        self._repository = repository
        self._clock = clock
        self._on_source_event = on_source_event
        self._on_intent = on_intent
        self._on_order_model_rejected = on_order_model_rejected

    def record_source_event(
        self, *, command: RecordExecutionSourceEventCommand
    ) -> ExecutionSourceEventResult:
        source_type = validate_source_event_fields(
            source_type=command.source_type,
            source_event_ref=command.source_event_ref,
            source_ref_json=command.source_ref_json,
            strategy_signal_id=command.strategy_signal_id,
        )
        idempotency_key_hash = hash_idempotency_key(command.idempotency_key)
        existing = self._repository.get_source_event_by_idempotency(
            owner_user_id=command.owner_user_id,
            source_type=source_type,
            idempotency_key_hash=idempotency_key_hash,
        )
        if existing is not None:
            return ExecutionSourceEventResult(event=existing, duplicate=True)
        event = ExecutionSourceEvent(
            source_event_id=uuid4(),
            owner_user_id=command.owner_user_id,
            source_type=source_type,
            source_event_ref=command.source_event_ref.strip(),
            source_ref_json=dict(command.source_ref_json),
            strategy_signal_id=command.strategy_signal_id,
            idempotency_key_hash=idempotency_key_hash,
            outcome="recorded",
            outcome_reason="source_event_recorded",
            intent_id=None,
            received_at=self._clock.now(),
        )
        recorded = self._repository.record_source_event(event=event)
        self._record_source_event(source_type=recorded.source_type, result="recorded")
        return ExecutionSourceEventResult(event=recorded, duplicate=False)

    def create_intent(self, *, command: CreateExecutionIntentCommand) -> ExecutionIntentResult:
        event = self._repository.get_source_event_by_id(
            owner_user_id=command.owner_user_id,
            source_event_id=command.source_event_id,
        )
        if event is None:
            raise ExecutionSourceValidationError(reason="source_event_not_found")
        idempotency_key_hash = hash_idempotency_key(command.idempotency_key)
        existing = self._repository.get_intent_by_idempotency(
            owner_user_id=command.owner_user_id,
            idempotency_key_hash=idempotency_key_hash,
        )
        if existing is not None:
            linked = self._repository.update_source_event_outcome(
                owner_user_id=command.owner_user_id,
                source_event_id=event.source_event_id,
                outcome="intent_created",
                outcome_reason="idempotent_replay",
                intent_id=existing.intent_id,
            )
            return ExecutionIntentResult(event=linked or event, intent=existing, duplicate=True)
        try:
            request = self._build_request(command=command, event=event)
        except ExecutionOrderModelRejectedError as error:
            self._repository.update_source_event_outcome(
                owner_user_id=command.owner_user_id,
                source_event_id=event.source_event_id,
                outcome="order_model_rejected",
                outcome_reason=error.reason,
                intent_id=None,
            )
            self._record_order_model_rejected(source_type=event.source_type, reason=error.reason)
            raise
        intent = ExecutionIntent(
            intent_id=uuid4(),
            source_event_id=request.source_event_id,
            owner_user_id=command.owner_user_id,
            source_type=request.source_type,
            strategy_signal_id=event.strategy_signal_id,
            exchange_connection_id=request.exchange_connection_id,
            market_type=request.market_type,
            instrument_key=request.instrument_key,
            side=request.order.side,
            order_type=request.order.order_type,
            quantity=request.order.quantity,
            quote_notional=request.order.quote_notional,
            limit_price=request.order.limit_price,
            status="recorded",
            status_reason="stage10_recorded_no_dispatch",
            risk_status="not_evaluated",
            risk_reason="stage11_not_implemented",
            idempotency_key_hash=idempotency_key_hash,
            created_at=self._clock.now(),
        )
        recorded = self._repository.record_intent(intent=intent)
        linked = self._repository.update_source_event_outcome(
            owner_user_id=command.owner_user_id,
            source_event_id=event.source_event_id,
            outcome="intent_created",
            outcome_reason="stage10_recorded_no_dispatch",
            intent_id=recorded.intent_id,
        )
        self._record_intent(
            source_type=recorded.source_type,
            result="recorded",
            reason=recorded.status_reason,
        )
        return ExecutionIntentResult(event=linked or event, intent=recorded, duplicate=False)

    def _build_request(
        self, *, command: CreateExecutionIntentCommand, event: ExecutionSourceEvent
    ) -> ExecutionRequest:
        if not command.market_type.strip():
            raise ExecutionSourceValidationError(reason="market_type_required")
        if not command.instrument_key.strip():
            raise ExecutionSourceValidationError(reason="instrument_key_required")
        order = validate_order_model(
            order_type=command.order_type,
            side=command.side,
            quantity=command.quantity,
            quote_notional=command.quote_notional,
            limit_price=command.limit_price,
            advanced_order_flags=command.advanced_order_flags,
        )
        return ExecutionRequest(
            source_event_id=event.source_event_id,
            source_type=event.source_type,
            idempotency_key_hash=hash_idempotency_key(command.idempotency_key),
            exchange_connection_id=command.exchange_connection_id,
            market_type=command.market_type.strip(),
            instrument_key=command.instrument_key.strip(),
            order=order,
        )

    def _record_source_event(self, *, source_type: str, result: str) -> None:
        if self._on_source_event is not None:
            self._on_source_event(source_type, result)

    def _record_intent(self, *, source_type: str, result: str, reason: str) -> None:
        if self._on_intent is not None:
            self._on_intent(source_type, result, reason)

    def _record_order_model_rejected(self, *, source_type: str, reason: str) -> None:
        if self._on_order_model_rejected is not None:
            self._on_order_model_rejected(source_type, reason)
