from __future__ import annotations

from trading.contexts.live_execution.application import (
    ExecutionIngressService,
    ExecutionIntentRepository,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.domain import ExecutionSourceEvent
from trading.contexts.rl_trading.domain import (
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_V1,
    STAGE13_SOURCE_TYPE_V1,
    Stage13DecisionContext,
    Stage13InferenceDecision,
    build_stage13_source_event_payload_v1,
)
from trading.shared_kernel.primitives import UserId


class LiveExecutionRlInferenceProducer:
    def __init__(
        self,
        *,
        ingress_service: ExecutionIngressService,
        repository: ExecutionIntentRepository,
    ) -> None:
        if ingress_service is None:  # type: ignore[truthy-bool]
            raise ValueError("LiveExecutionRlInferenceProducer requires ingress_service")
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("LiveExecutionRlInferenceProducer requires repository")
        self._ingress_service = ingress_service
        self._repository = repository

    def record_monitor_only_decision(
        self,
        *,
        context: Stage13DecisionContext,
        decision: Stage13InferenceDecision,
    ) -> ExecutionSourceEvent:
        owner_user_id = UserId.from_string(context.owner_user_id)
        payload = build_stage13_source_event_payload_v1(context=context, decision=decision)
        result = self._ingress_service.record_source_event(
            command=RecordExecutionSourceEventCommand(
                owner_user_id=owner_user_id,
                source_type=STAGE13_SOURCE_TYPE_V1,
                source_event_ref=payload.source_event_ref,
                source_ref_json=payload.source_ref_json,
                strategy_signal_id=None,
                idempotency_key=payload.idempotency_key,
            )
        )
        updated = self._repository.update_source_event_outcome(
            owner_user_id=owner_user_id,
            source_event_id=result.event.source_event_id,
            outcome=STAGE13_SOURCE_EVENT_OUTCOME_V1,
            outcome_reason=STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
            intent_id=None,
        )
        return updated or result.event
