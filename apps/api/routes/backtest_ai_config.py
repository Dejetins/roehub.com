from __future__ import annotations

from typing import Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response

from apps.api.dto import (
    BacktestAiConversationCreateRequest,
    BacktestAiConversationListResponse,
    BacktestAiConversationResponse,
    BacktestAiConversationSendMessageRequest,
    BacktestAiConversationSendMessageResponse,
    BacktestAiConversationStatusResponse,
    build_backtest_ai_conversation_list_response,
    build_backtest_ai_conversation_response,
    build_backtest_ai_conversation_send_message_response,
    build_backtest_ai_conversation_status_response,
    build_backtest_ai_load_action_response,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConversationUseCase,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]


def build_backtest_ai_config_router(
    *,
    current_user_dependency: CurrentUserDependency,
    conversation_use_case: BacktestAiConversationUseCase | None = None,
    jobs_use_case: object | None = None,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_ai_config_router requires current_user_dependency")
    _ = jobs_use_case
    router = APIRouter(tags=["backtest-ai-config"])

    def require_backtest_user(request: Request) -> CurrentUserPrincipal:
        return current_user_dependency(request)

    def require_conversation_use_case() -> BacktestAiConversationUseCase:
        if conversation_use_case is None:
            raise RoehubError(
                code="backtest.ai_config.unavailable",
                message="Backtest AI conversation service is not configured",
                details={"reason": "conversation_repository_unavailable"},
            )
        return conversation_use_case

    @router.get(
        "/backtests/ai-config/conversations",
        response_model=BacktestAiConversationListResponse,
    )
    def list_conversations(
        limit: int = Query(default=50, ge=1, le=50),
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestAiConversationUseCase = Depends(
            require_conversation_use_case
        ),
    ) -> BacktestAiConversationListResponse:
        conversations = use_case.list_conversations(
            user_id=principal.user_id,
            limit=limit,
        )
        return build_backtest_ai_conversation_list_response(
            conversations=conversations,
            use_case=use_case,
        )

    @router.post(
        "/backtests/ai-config/conversations",
        response_model=BacktestAiConversationResponse,
        status_code=201,
    )
    def create_conversation(
        _response: Response,
        request: BacktestAiConversationCreateRequest,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestAiConversationUseCase = Depends(
            require_conversation_use_case
        ),
    ) -> BacktestAiConversationResponse:
        read = use_case.create_conversation(
            user_id=principal.user_id,
            locale=request.locale,
        )
        return build_backtest_ai_conversation_response(read=read, use_case=use_case)

    @router.get(
        "/backtests/ai-config/conversations/{conversation_id}/messages",
        response_model=BacktestAiConversationResponse,
    )
    def get_conversation_messages(
        conversation_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestAiConversationUseCase = Depends(
            require_conversation_use_case
        ),
    ) -> BacktestAiConversationResponse:
        read = use_case.read_conversation(
            user_id=principal.user_id,
            conversation_id=conversation_id,
        )
        return build_backtest_ai_conversation_response(read=read, use_case=use_case)

    @router.post(
        "/backtests/ai-config/conversations/{conversation_id}/messages",
        response_model=BacktestAiConversationSendMessageResponse,
        status_code=201,
    )
    def send_conversation_message(
        conversation_id: UUID,
        request: BacktestAiConversationSendMessageRequest,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestAiConversationUseCase = Depends(
            require_conversation_use_case
        ),
    ) -> BacktestAiConversationSendMessageResponse:
        result = use_case.send_message(
            user_id=principal.user_id,
            conversation_id=conversation_id,
            message=request.message,
            current_config=request.current_config,
            ui_context=request.ui_context,
        )
        return build_backtest_ai_conversation_send_message_response(result=result)

    @router.get(
        "/backtests/ai-config/conversations/{conversation_id}/status",
        response_model=BacktestAiConversationStatusResponse,
    )
    def get_conversation_status(
        conversation_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestAiConversationUseCase = Depends(
            require_conversation_use_case
        ),
    ) -> BacktestAiConversationStatusResponse:
        run = use_case.get_status(
            user_id=principal.user_id,
            conversation_id=conversation_id,
        )
        return build_backtest_ai_conversation_status_response(
            conversation_id=str(conversation_id),
            run=run,
        )

    @router.get(
        "/backtests/ai-config/conversations/{conversation_id}/load-action",
        response_model=BacktestAiConversationStatusResponse,
    )
    def get_conversation_load_action(
        conversation_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_backtest_user),
        use_case: BacktestAiConversationUseCase = Depends(
            require_conversation_use_case
        ),
    ) -> BacktestAiConversationStatusResponse:
        run = use_case.get_status(
            user_id=principal.user_id,
            conversation_id=conversation_id,
        )
        return build_backtest_ai_load_action_response(
            conversation_id=str(conversation_id),
            run=run,
        )

    return router


__all__ = ["build_backtest_ai_config_router"]
