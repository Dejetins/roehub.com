from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, Response

from apps.api.dto.ui_execution import (
    ExecutionIntentRequest,
    ExecutionIntentResponse,
    ExecutionNotificationRequest,
    ExecutionNotificationResponse,
    ExecutionNotificationsResponse,
    ExecutionSourceEventRequest,
    ExecutionSourceEventResponse,
)
from trading.contexts.backtest.application.ports import ResearchOrganizationScopeResolver
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.application import (
    CreateExecutionIntentCommand,
    EmitExecutionNotificationCommand,
    ExecutionDispatchService,
    ExecutionIngressService,
    RecordExecutionSourceEventCommand,
)
from trading.contexts.live_execution.application.ports import (
    ExecutionRiskContextQuery,
    ExecutionRiskContextResolutionError,
    ExecutionRiskContextResolver,
)
from trading.contexts.live_execution.domain import (
    ExecutionIntent,
    ExecutionNotificationOutboxEvent,
    ExecutionNotificationValidationError,
    ExecutionOrderModelRejectedError,
    ExecutionSourceEvent,
    ExecutionSourceValidationError,
)
from trading.platform.errors import RoehubError

CurrentUserPrincipalDependency = Callable[..., CurrentUserPrincipal]


def build_ui_execution_router(
    *,
    ingress_service: ExecutionIngressService,
    dispatch_service: ExecutionDispatchService | None = None,
    current_user_dependency: CurrentUserPrincipalDependency,
    organization_scope_resolver: ResearchOrganizationScopeResolver,
    risk_context_resolver: ExecutionRiskContextResolver,
) -> APIRouter:
    if ingress_service is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_execution_router requires ingress_service")
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_execution_router requires current_user_dependency")
    if organization_scope_resolver is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_execution_router requires organization_scope_resolver")
    if risk_context_resolver is None:  # type: ignore[truthy-bool]
        raise ValueError("build_ui_execution_router requires risk_context_resolver")

    router = APIRouter(tags=["ui-execution"])

    @router.post(
        "/ui/execution/source-events",
        response_model=ExecutionSourceEventResponse,
        status_code=201,
    )
    def post_source_event(
        payload: ExecutionSourceEventRequest,
        response: Response,
        current_user: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> ExecutionSourceEventResponse:
        try:
            result = ingress_service.record_source_event(
                command=RecordExecutionSourceEventCommand(
                    organization_id=organization_scope_resolver.resolve(
                        user_id=current_user.user_id
                    ).organization_id,
                    owner_user_id=current_user.user_id,
                    source_type=payload.source_type,
                    source_event_ref=payload.source_event_ref,
                    source_ref_json=payload.source_ref,
                    strategy_signal_id=payload.strategy_signal_id,
                    idempotency_key=payload.idempotency_key,
                )
            )
        except ExecutionSourceValidationError as error:
            raise _source_event_error(reason=error.reason) from error
        if result.duplicate:
            response.status_code = 200
        return _to_source_event_response(event=result.event, duplicate=result.duplicate)

    @router.post(
        "/ui/execution/intents",
        response_model=ExecutionIntentResponse,
        status_code=201,
    )
    def post_intent(
        payload: ExecutionIntentRequest,
        response: Response,
        current_user: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> ExecutionIntentResponse:
        try:
            organization_id = organization_scope_resolver.resolve(
                user_id=current_user.user_id
            ).organization_id
            risk_context = risk_context_resolver.resolve(
                query=ExecutionRiskContextQuery(
                    organization_id=organization_id,
                    owner_user_id=current_user.user_id,
                    source_event_id=payload.source_event_id,
                    exchange_connection_id=payload.exchange_connection_id,
                    market_type=payload.market_type,
                    instrument_key=payload.instrument_key,
                )
            )
            result = ingress_service.create_intent(
                command=CreateExecutionIntentCommand(
                    organization_id=organization_id,
                    owner_user_id=current_user.user_id,
                    source_event_id=payload.source_event_id,
                    idempotency_key=payload.idempotency_key,
                    exchange_connection_id=payload.exchange_connection_id,
                    market_type=payload.market_type,
                    instrument_key=payload.instrument_key,
                    order_type=payload.order.order_type,
                    side=payload.order.side,
                    quantity=payload.order.quantity,
                    quote_notional=payload.order.quote_notional,
                    limit_price=payload.order.limit_price,
                    advanced_order_flags={
                        "oco": payload.order.oco,
                        "trailing": payload.order.trailing,
                        "take_profit": payload.order.take_profit,
                        "stop_loss": payload.order.stop_loss,
                        "amend_replace": payload.order.amend_replace,
                        "legs": payload.order.legs,
                    },
                    risk_context=risk_context,
                )
            )
        except ExecutionOrderModelRejectedError as error:
            raise _unsupported_order_model_error(reason=error.reason) from error
        except ExecutionSourceValidationError as error:
            raise _execution_request_error(reason=error.reason) from error
        except ExecutionRiskContextResolutionError as error:
            raise _execution_request_error(reason=error.reason) from error
        if result.duplicate:
            response.status_code = 200
        response_intent = result.intent
        if dispatch_service is not None:
            response_intent = dispatch_service.dispatch_intent(intent=result.intent).intent
        return _to_intent_response(
            intent=response_intent,
            source_event=result.event,
            duplicate=result.duplicate,
        )

    @router.post(
        "/ui/execution/notifications",
        response_model=ExecutionNotificationResponse,
        status_code=201,
    )
    def post_notification(
        payload: ExecutionNotificationRequest,
        response: Response,
        current_user: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> ExecutionNotificationResponse:
        try:
            result = ingress_service.emit_notification(
                command=EmitExecutionNotificationCommand(
                    organization_id=organization_scope_resolver.resolve(
                        user_id=current_user.user_id
                    ).organization_id,
                    owner_user_id=current_user.user_id,
                    source_type=payload.source_type,
                    event_type=payload.event_type,
                    severity=payload.severity,
                    reason=payload.reason,
                    source_event_id=payload.source_event_id,
                    intent_id=payload.intent_id,
                    order_id=payload.order_id,
                    strategy_signal_id=payload.strategy_signal_id,
                    labels=payload.labels,
                )
            )
        except (ExecutionSourceValidationError, ExecutionNotificationValidationError) as error:
            reason = getattr(error, "reason", "invalid_notification")
            raise _notification_error(reason=reason) from error
        if result.duplicate:
            response.status_code = 200
        return _to_notification_response(
            notification=result.notification,
            duplicate=result.duplicate,
        )

    @router.get(
        "/ui/execution/notifications",
        response_model=ExecutionNotificationsResponse,
    )
    def get_notifications(
        current_user: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> ExecutionNotificationsResponse:
        return ExecutionNotificationsResponse(
            items=[
                _to_notification_response(notification=notification, duplicate=False)
                for notification in ingress_service.list_recent_notifications(
                    organization_id=organization_scope_resolver.resolve(
                        user_id=current_user.user_id
                    ).organization_id,
                    owner_user_id=current_user.user_id,
                    limit=50,
                )
            ]
        )

    return router


def _to_source_event_response(
    *, event: ExecutionSourceEvent, duplicate: bool
) -> ExecutionSourceEventResponse:
    return ExecutionSourceEventResponse(
        source_event_id=event.source_event_id,
        source_type=event.source_type,
        source_event_ref=event.source_event_ref,
        source_ref=dict(event.source_ref_json),
        strategy_signal_id=event.strategy_signal_id,
        outcome=event.outcome,
        outcome_reason=event.outcome_reason,
        intent_id=event.intent_id,
        received_at=event.received_at,
        duplicate=duplicate,
    )


def _to_intent_response(
    *,
    intent: ExecutionIntent,
    source_event: ExecutionSourceEvent,
    duplicate: bool,
) -> ExecutionIntentResponse:
    return ExecutionIntentResponse(
        intent_id=intent.intent_id,
        source_event_id=intent.source_event_id,
        source_type=intent.source_type,
        strategy_signal_id=intent.strategy_signal_id,
        exchange_connection_id=intent.exchange_connection_id,
        market_type=intent.market_type,
        instrument_key=intent.instrument_key,
        side=intent.side,
        order_type=intent.order_type,
        quantity=intent.quantity,
        quote_notional=intent.quote_notional,
        limit_price=intent.limit_price,
        status=intent.status,
        status_reason=intent.status_reason,
        risk_status=intent.risk_status,
        risk_reason=intent.risk_reason,
        dispatch_attempt_count=intent.dispatch_attempt_count,
        dispatch_stream_name=intent.dispatch_stream_name,
        dispatch_redis_message_id=intent.dispatch_redis_message_id,
        dispatch_last_error=intent.dispatch_last_error,
        dispatch_updated_at=intent.dispatch_updated_at,
        created_at=intent.created_at,
        duplicate=duplicate,
        source_event=_to_source_event_response(event=source_event, duplicate=False),
    )


def _to_notification_response(
    *,
    notification: ExecutionNotificationOutboxEvent,
    duplicate: bool,
) -> ExecutionNotificationResponse:
    return ExecutionNotificationResponse(
        notification_id=notification.notification_id,
        source_type=notification.source_type,
        event_type=notification.event_type,
        severity=notification.severity,
        reason=notification.reason,
        source_event_id=notification.source_event_id,
        intent_id=notification.intent_id,
        order_id=notification.order_id,
        strategy_signal_id=notification.strategy_signal_id,
        labels=dict(notification.labels_json),
        status=notification.status,
        created_at=notification.created_at,
        sent_at=notification.sent_at,
        duplicate=duplicate,
    )


def _source_event_error(*, reason: str) -> RoehubError:
    return RoehubError(
        code="execution.invalid_source_event",
        message="Execution source event is invalid",
        details={"reason": reason},
    )


def _execution_request_error(*, reason: str) -> RoehubError:
    code = (
        "execution.source_event_not_found"
        if reason == "source_event_not_found"
        else "execution.invalid_execution_request"
    )
    return RoehubError(
        code=code,
        message="Execution request is invalid",
        details={"reason": reason},
    )


def _unsupported_order_model_error(*, reason: str) -> RoehubError:
    return RoehubError(
        code="execution.unsupported_order_model",
        message="Execution order model is not supported in v1",
        details={"reason": reason},
    )


def _notification_error(*, reason: str) -> RoehubError:
    return RoehubError(
        code="execution.invalid_notification",
        message="Execution notification is invalid",
        details={"reason": reason},
    )


__all__ = ["build_ui_execution_router"]
