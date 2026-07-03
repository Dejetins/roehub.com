from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping
from uuid import UUID

from trading.contexts.live_execution.application import (
    CapitalReservationPaperAccountingService,
    CreateExecutionIntentCommand,
    ExecutionIngressService,
    ExecutionIntentRepository,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.domain import (
    PAPER_VIRTUAL_EXCHANGE_CONNECTION_ID,
    ExecutionIntent,
    ExecutionRiskContext,
    ExecutionSourceEvent,
    StrategyPaperAccountingSnapshot,
)
from trading.contexts.rl_trading.domain import (
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_V1,
    STAGE13_SOURCE_TYPE_V1,
    Stage13DecisionContext,
    Stage13InferenceDecision,
    build_stage13_source_event_payload_v1,
)
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class RlPaperExecutionResult:
    event: ExecutionSourceEvent
    intent: ExecutionIntent | None
    accounting: StrategyPaperAccountingSnapshot | None
    duplicate: bool


@dataclass(frozen=True, slots=True)
class RlTestnetExecutionResult:
    event: ExecutionSourceEvent
    intent: ExecutionIntent | None
    duplicate: bool


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

    def record_paper_decision(
        self,
        *,
        context: Stage13DecisionContext,
        decision: Stage13InferenceDecision,
        risk_context: ExecutionRiskContext,
        paper_accounting_service: CapitalReservationPaperAccountingService,
        quote_notional: Decimal,
        reference_price: Decimal,
        exchange_connection_id: UUID = PAPER_VIRTUAL_EXCHANGE_CONNECTION_ID,
        live_profile_id: UUID | None = None,
    ) -> RlPaperExecutionResult:
        owner_user_id = UserId.from_string(context.owner_user_id)
        source = self._ingress_service.record_source_event(
            command=RecordExecutionSourceEventCommand(
                owner_user_id=owner_user_id,
                source_type=STAGE13_SOURCE_TYPE_V1,
                source_event_ref=f"rl:{decision.decision_id}",
                source_ref_json=_paper_source_ref_json(context=context, decision=decision),
                strategy_signal_id=None,
                idempotency_key=_paper_source_idempotency_key(context=context, decision=decision),
            )
        )
        side = _paper_side_for_action(decision.action_name)
        if side is None:
            updated = self._repository.update_source_event_outcome(
                owner_user_id=owner_user_id,
                source_event_id=source.event.source_event_id,
                outcome="no_intent",
                outcome_reason=_paper_no_intent_reason(decision.action_name),
                intent_id=None,
            )
            return RlPaperExecutionResult(
                event=updated or source.event,
                intent=None,
                accounting=None,
                duplicate=source.duplicate,
            )

        intent = self._ingress_service.create_intent(
            command=CreateExecutionIntentCommand(
                owner_user_id=owner_user_id,
                source_event_id=source.event.source_event_id,
                idempotency_key=_paper_intent_idempotency_key(context=context, decision=decision),
                exchange_connection_id=exchange_connection_id,
                market_type=context.market_type,
                instrument_key=context.instrument_key,
                order_type="market",
                side=side,
                quantity=None,
                quote_notional=quote_notional,
                limit_price=None,
                advanced_order_flags={},
                risk_context=risk_context,
            )
        )
        accounting = None
        if intent.intent.risk_reason == "paper_no_exchange_submit":
            accounting = paper_accounting_service.record_rl_paper_execution(
                owner_user_id=owner_user_id,
                strategy_id=UUID(context.strategy_id),
                live_profile_id=live_profile_id,
                strategy_run_id=UUID(context.strategy_run_id),
                source_event_id=source.event.source_event_id,
                instrument_key=context.instrument_key,
                market_type=context.market_type,
                side=side,
                quote_notional=quote_notional,
                reference_price=reference_price,
                now=intent.intent.created_at,
            )
        return RlPaperExecutionResult(
            event=intent.event,
            intent=intent.intent,
            accounting=accounting,
            duplicate=source.duplicate or intent.duplicate,
        )

    def record_testnet_decision(
        self,
        *,
        context: Stage13DecisionContext,
        decision: Stage13InferenceDecision,
        risk_context: ExecutionRiskContext,
        exchange_connection_id: UUID,
        quote_notional: Decimal,
        quantity: Decimal | None = None,
    ) -> RlTestnetExecutionResult:
        owner_user_id = UserId.from_string(context.owner_user_id)
        source = self._ingress_service.record_source_event(
            command=RecordExecutionSourceEventCommand(
                owner_user_id=owner_user_id,
                source_type=STAGE13_SOURCE_TYPE_V1,
                source_event_ref=f"rl:{decision.decision_id}",
                source_ref_json=_testnet_source_ref_json(context=context, decision=decision),
                strategy_signal_id=None,
                idempotency_key=_testnet_source_idempotency_key(
                    context=context,
                    decision=decision,
                ),
            )
        )
        side = _testnet_side_for_action(context=context, action_name=decision.action_name)
        if side is None:
            updated = self._repository.update_source_event_outcome(
                owner_user_id=owner_user_id,
                source_event_id=source.event.source_event_id,
                outcome="no_intent",
                outcome_reason=_testnet_no_intent_reason(
                    context=context,
                    action_name=decision.action_name,
                ),
                intent_id=None,
            )
            return RlTestnetExecutionResult(
                event=updated or source.event,
                intent=None,
                duplicate=source.duplicate,
            )

        intent = self._ingress_service.create_intent(
            command=CreateExecutionIntentCommand(
                owner_user_id=owner_user_id,
                source_event_id=source.event.source_event_id,
                idempotency_key=_testnet_intent_idempotency_key(
                    context=context,
                    decision=decision,
                ),
                exchange_connection_id=exchange_connection_id,
                market_type=context.market_type,
                instrument_key=context.instrument_key,
                order_type="market",
                side=side,
                quantity=quantity if context.market_type == "futures" else None,
                quote_notional=quote_notional if context.market_type == "spot" else None,
                limit_price=None,
                advanced_order_flags={},
                risk_context=risk_context,
            )
        )
        return RlTestnetExecutionResult(
            event=intent.event,
            intent=intent.intent,
            duplicate=source.duplicate or intent.duplicate,
        )


def _paper_source_ref_json(
    *,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> Mapping[str, str]:
    return {
        "action": decision.action_name,
        "action_id": str(decision.action_id),
        "exchange": context.exchange,
        "feature_hash": decision.feature_hash,
        "instrument_key": context.instrument_key,
        "market_type": context.market_type,
        "mode": "paper",
        "model_version_id": decision.model_version_id,
        "strategy_id": context.strategy_id,
        "strategy_run_id": context.strategy_run_id,
        "symbol": context.symbol,
    }


def _paper_source_idempotency_key(
    *,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> str:
    return "|".join(
        (
            STAGE13_SOURCE_TYPE_V1,
            context.strategy_id,
            context.strategy_run_id,
            context.instrument_key,
            decision.feature_hash,
            decision.model_version_id,
            "paper",
        )
    )


def _paper_intent_idempotency_key(
    *,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> str:
    return "|".join((_paper_source_idempotency_key(context=context, decision=decision), "intent"))


def _paper_side_for_action(action_name: str) -> str | None:
    if action_name == "open_long":
        return "buy"
    if action_name == "open_short":
        return "sell"
    return None


def _paper_no_intent_reason(action_name: str) -> str:
    if action_name == "hold":
        return "paper_hold_no_intent"
    if action_name == "close":
        return "paper_close_position_snapshot_required"
    return "paper_unsupported_rl_action"


def _testnet_source_ref_json(
    *,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> Mapping[str, str]:
    return {
        "action": decision.action_name,
        "action_id": str(decision.action_id),
        "exchange": context.exchange,
        "feature_hash": decision.feature_hash,
        "instrument_key": context.instrument_key,
        "market_type": context.market_type,
        "mode": "testnet",
        "model_version_id": decision.model_version_id,
        "strategy_id": context.strategy_id,
        "strategy_run_id": context.strategy_run_id,
        "symbol": context.symbol,
    }


def _testnet_source_idempotency_key(
    *,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> str:
    return "|".join(
        (
            STAGE13_SOURCE_TYPE_V1,
            context.strategy_id,
            context.strategy_run_id,
            context.instrument_key,
            decision.feature_hash,
            decision.model_version_id,
            "testnet",
        )
    )


def _testnet_intent_idempotency_key(
    *,
    context: Stage13DecisionContext,
    decision: Stage13InferenceDecision,
) -> str:
    return "|".join((_testnet_source_idempotency_key(context=context, decision=decision), "intent"))


def _testnet_side_for_action(
    *,
    context: Stage13DecisionContext,
    action_name: str,
) -> str | None:
    if action_name == "open_long" and _is_supported_testnet_market(context=context):
        return "buy"
    if (
        action_name == "open_short"
        and context.market_type == "futures"
        and _is_supported_testnet_market(context=context)
    ):
        return "sell"
    return None


def _testnet_no_intent_reason(
    *,
    context: Stage13DecisionContext,
    action_name: str,
) -> str:
    if action_name == "hold":
        return "testnet_hold_no_intent"
    if action_name == "close":
        return "testnet_close_position_snapshot_required"
    if context.exchange not in {"binance", "bybit"}:
        return "testnet_unsupported_exchange"
    if context.market_type not in {"spot", "futures"}:
        return "testnet_unsupported_market_type"
    if action_name == "open_short" and context.market_type == "spot":
        return "testnet_spot_short_not_supported"
    return "testnet_unsupported_rl_action"


def _is_supported_testnet_market(*, context: Stage13DecisionContext) -> bool:
    return context.exchange in {"binance", "bybit"} and context.market_type in {"spot", "futures"}
