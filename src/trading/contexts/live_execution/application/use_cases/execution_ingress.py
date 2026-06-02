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
    ExecutionNotificationOutboxEvent,
    ExecutionOrderModelRejectedError,
    ExecutionProducerOutcomeLink,
    ExecutionRequest,
    ExecutionRiskAuditEvent,
    ExecutionRiskContext,
    ExecutionSourceEvent,
    ExecutionSourceValidationError,
    evaluate_execution_risk,
    hash_idempotency_key,
    sanitize_notification_labels,
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
    risk_context: ExecutionRiskContext | None = None


@dataclass(frozen=True, slots=True)
class EmitExecutionNotificationCommand:
    owner_user_id: UserId
    source_type: str
    event_type: str
    severity: str
    reason: str
    source_event_id: UUID | None = None
    intent_id: UUID | None = None
    order_id: UUID | None = None
    strategy_signal_id: UUID | None = None
    labels: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSourceEventResult:
    event: ExecutionSourceEvent
    duplicate: bool


@dataclass(frozen=True, slots=True)
class ExecutionIntentResult:
    event: ExecutionSourceEvent
    intent: ExecutionIntent
    duplicate: bool


@dataclass(frozen=True, slots=True)
class ExecutionNotificationResult:
    notification: ExecutionNotificationOutboxEvent
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
        on_risk_decision: Callable[[str, str, str, float], None] | None = None,
        on_notification: Callable[[str, str, str], None] | None = None,
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
        self._on_risk_decision = on_risk_decision
        self._on_notification = on_notification

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
        now = self._clock.now()
        draft_intent = ExecutionIntent(
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
            status_reason="risk_gate_pending",
            risk_status="not_evaluated",
            risk_reason="risk_gate_pending",
            idempotency_key_hash=idempotency_key_hash,
            created_at=now,
        )
        risk_started_at = self._clock.now()
        decision = evaluate_execution_risk(intent=draft_intent, context=command.risk_context)
        risk_finished_at = self._clock.now()
        intent = ExecutionIntent(
            intent_id=draft_intent.intent_id,
            source_event_id=draft_intent.source_event_id,
            owner_user_id=draft_intent.owner_user_id,
            source_type=draft_intent.source_type,
            strategy_signal_id=draft_intent.strategy_signal_id,
            exchange_connection_id=draft_intent.exchange_connection_id,
            market_type=draft_intent.market_type,
            instrument_key=draft_intent.instrument_key,
            side=draft_intent.side,
            order_type=draft_intent.order_type,
            quantity=draft_intent.quantity,
            quote_notional=draft_intent.quote_notional,
            limit_price=draft_intent.limit_price,
            status=decision.status,
            status_reason=decision.reason,
            risk_status=decision.status,
            risk_reason=decision.reason,
            idempotency_key_hash=draft_intent.idempotency_key_hash,
            created_at=draft_intent.created_at,
        )
        recorded = self._repository.record_intent(intent=intent)
        self._repository.record_risk_audit_event(
            event=ExecutionRiskAuditEvent(
                event_id=uuid4(),
                intent_id=recorded.intent_id,
                source_event_id=recorded.source_event_id,
                owner_user_id=recorded.owner_user_id,
                source_type=recorded.source_type,
                event_type=f"risk_gate_{recorded.risk_status}",
                risk_status=recorded.risk_status,  # type: ignore[arg-type]
                risk_reason=recorded.risk_reason,
                check_name=decision.check_name,
                metadata_json={"dispatch": "no-dispatch"},
                created_at=self._clock.now(),
            )
        )
        source_outcome = "risk_rejected" if recorded.risk_status == "rejected" else "intent_created"
        linked = self._repository.update_source_event_outcome(
            owner_user_id=command.owner_user_id,
            source_event_id=event.source_event_id,
            outcome=source_outcome,
            outcome_reason=recorded.risk_reason,
            intent_id=recorded.intent_id,
        )
        if recorded.risk_status == "rejected":
            event_type = (
                "producer_kill_switch"
                if recorded.risk_reason == "kill_switch_closed"
                else "producer_rejected"
            )
            self.emit_notification(
                command=EmitExecutionNotificationCommand(
                    owner_user_id=recorded.owner_user_id,
                    source_type=recorded.source_type,
                    event_type=event_type,
                    severity="critical" if event_type == "producer_kill_switch" else "warning",
                    reason=recorded.risk_reason,
                    source_event_id=recorded.source_event_id,
                    intent_id=recorded.intent_id,
                    strategy_signal_id=recorded.strategy_signal_id,
                    labels={
                        "risk_status": recorded.risk_status,
                        "intent_status": recorded.status,
                        "instrument_key": recorded.instrument_key,
                    },
                )
            )
        self._record_intent(
            source_type=recorded.source_type,
            result=recorded.status,
            reason=recorded.status_reason,
        )
        self._record_risk_decision(
            source_type=recorded.source_type,
            result=recorded.risk_status,
            reason=recorded.risk_reason,
            latency_seconds=max(0.0, (risk_finished_at - risk_started_at).total_seconds()),
        )
        return ExecutionIntentResult(event=linked or event, intent=recorded, duplicate=False)

    def emit_notification(
        self, *, command: EmitExecutionNotificationCommand
    ) -> ExecutionNotificationResult:
        source_type = validate_source_event_fields(
            source_type=command.source_type,
            source_event_ref=str(command.source_event_id or command.intent_id or command.order_id),
            source_ref_json={"notification": command.event_type},
            strategy_signal_id=(
                command.strategy_signal_id if command.source_type == "strategy_signal" else None
            ),
        )
        labels = sanitize_notification_labels(command.labels or {})
        notification = ExecutionNotificationOutboxEvent(
            notification_id=uuid4(),
            owner_user_id=command.owner_user_id,
            source_type=source_type,
            event_type=command.event_type,  # type: ignore[arg-type]
            severity=command.severity,  # type: ignore[arg-type]
            reason=command.reason.strip(),
            source_event_id=command.source_event_id,
            intent_id=command.intent_id,
            order_id=command.order_id,
            strategy_signal_id=command.strategy_signal_id,
            labels_json=labels,
            status="pending",
            created_at=self._clock.now(),
        )
        recorded = self._repository.record_notification_outbox(event=notification)
        self._record_notification(
            event_type=recorded.event_type,
            source_type=recorded.source_type,
            severity=recorded.severity,
        )
        return ExecutionNotificationResult(
            notification=recorded,
            duplicate=recorded.notification_id != notification.notification_id,
        )

    def list_recent_notifications(
        self,
        *,
        owner_user_id: UserId,
        limit: int,
        strategy_id: UUID | None = None,
    ) -> tuple[ExecutionNotificationOutboxEvent, ...]:
        return self._repository.list_recent_notifications(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            limit=limit,
        )

    def list_producer_outcome_links_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID, limit: int
    ) -> tuple[ExecutionProducerOutcomeLink, ...]:
        return self._repository.list_producer_outcome_links_for_strategy(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
            limit=limit,
        )

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

    def _record_risk_decision(
        self, *, source_type: str, result: str, reason: str, latency_seconds: float
    ) -> None:
        if self._on_risk_decision is not None:
            self._on_risk_decision(source_type, result, reason, latency_seconds)

    def _record_notification(self, *, event_type: str, source_type: str, severity: str) -> None:
        if self._on_notification is not None:
            self._on_notification(event_type, source_type, severity)
