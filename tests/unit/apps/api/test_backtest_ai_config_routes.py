from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.backtest_ai_config import build_backtest_ai_config_router
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConversation,
    BacktestAiConversationMessage,
    BacktestAiConversationRun,
    BacktestAiConversationUseCase,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_ai_config_router_registers_conversation_api_without_old_job_paths() -> None:
    router = build_backtest_ai_config_router(
        current_user_dependency=_CurrentUserDependency(),
        conversation_use_case=BacktestAiConversationUseCase(
            repository=_InMemoryConversationRepository()
        ),
        jobs_use_case=object(),
    )

    paths = {getattr(route, "path", "") for route in router.routes}
    old_job_endpoint = "/backtests" + "/ai-config" + "/jobs"

    assert "/backtests/ai-config/conversations" in paths
    assert not any(path.startswith(old_job_endpoint) for path in paths)


def test_conversation_routes_create_send_status_and_load_action() -> None:
    repository = _InMemoryConversationRepository()
    client = _build_client(repository=repository)

    create_response = client.post(
        "/backtests/ai-config/conversations",
        json={"locale": "en"},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    conversation_id = created["conversation"]["conversation_id"]
    assert created["conversation"]["conversation_title"] == "New backtest chat"
    assert created["limits"]["retention_days"] == 30
    assert created["limits"]["max_conversations_per_user"] == 50
    assert created["limits"]["max_messages_per_conversation"] == 100
    assert created["messages"][0]["role"] == "assistant"

    send_response = client.post(
        f"/backtests/ai-config/conversations/{conversation_id}/messages",
        json={
            "message": "Create RSI for BTCUSDT",
            "current_config": {"coordinates": {"symbol": "BTCUSDT"}},
        },
    )
    assert send_response.status_code == 201
    sent = send_response.json()
    assert sent["message_id"]
    assert sent["assistant_message"]["role"] == "assistant"
    assert sent["status"]["status"] == "awaiting_model"
    assert sent["status"]["load_action"] == {
        "enabled": False,
        "state": "unavailable",
        "reason": "backend_not_ready",
        "config": None,
    }

    messages_response = client.get(
        f"/backtests/ai-config/conversations/{conversation_id}/messages"
    )
    assert messages_response.status_code == 200
    assert [item["role"] for item in messages_response.json()["messages"]] == [
        "assistant",
        "user",
        "assistant",
    ]

    status_response = client.get(
        f"/backtests/ai-config/conversations/{conversation_id}/status"
    )
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "awaiting_model"

    load_action_response = client.get(
        f"/backtests/ai-config/conversations/{conversation_id}/load-action"
    )
    assert load_action_response.status_code == 200
    assert load_action_response.json()["load_action"]["enabled"] is False


def test_conversation_routes_are_owner_isolated() -> None:
    repository = _InMemoryConversationRepository()
    current_user = _CurrentUserDependency()
    client = _build_client(repository=repository, current_user=current_user)

    created = client.post(
        "/backtests/ai-config/conversations",
        json={"locale": "en"},
    ).json()
    conversation_id = created["conversation"]["conversation_id"]
    current_user.user_id = UserId.from_string("00000000-0000-0000-0000-000000000802")

    response = client.get(
        f"/backtests/ai-config/conversations/{conversation_id}/messages"
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "backtest.ai_config.not_found"
    assert client.get("/backtests/ai-config/conversations").json()["conversations"] == []


def test_retired_ai_config_job_endpoints_are_not_active() -> None:
    client = _build_client(repository=_InMemoryConversationRepository())
    retired_endpoint = "/backtests" + "/ai-config" + "/jobs"

    response = client.post(
        retired_endpoint,
        json={"mode": "assistant_v1", "locale": "en", "message": "Create RSI config"},
    )

    assert response.status_code == 404


def _build_client(
    *,
    repository: _InMemoryConversationRepository,
    current_user: _CurrentUserDependency | None = None,
) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtest_ai_config_router(
            current_user_dependency=current_user or _CurrentUserDependency(),
            conversation_use_case=BacktestAiConversationUseCase(repository=repository),
        )
    )
    return TestClient(app)


class _CurrentUserDependency:
    def __init__(self) -> None:
        self.user_id = UserId.from_string("00000000-0000-0000-0000-000000000801")

    def __call__(self, request: Request) -> CurrentUserPrincipal:
        _ = request
        return CurrentUserPrincipal(
            user_id=self.user_id,
            paid_level=PaidLevel.free(),
        )


class _InMemoryConversationRepository:
    def __init__(self) -> None:
        self.conversations: dict[UUID, BacktestAiConversation] = {}
        self.messages: dict[UUID, list[BacktestAiConversationMessage]] = {}
        self.runs: dict[UUID, list[BacktestAiConversationRun]] = {}

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        return sum(
            1
            for conversation in self.conversations.values()
            if conversation.owner_user_id == owner_user_id
        )

    def create_with_startup_message(
        self,
        *,
        conversation: BacktestAiConversation,
        startup_message: BacktestAiConversationMessage,
    ) -> BacktestAiConversation:
        self.conversations[conversation.conversation_id] = conversation
        self.messages[conversation.conversation_id] = [startup_message]
        self.runs[conversation.conversation_id] = []
        return conversation

    def list_for_user(
        self,
        *,
        owner_user_id: UserId,
        limit: int,
    ) -> tuple[BacktestAiConversation, ...]:
        return tuple(
            item
            for item in self.conversations.values()
            if item.owner_user_id == owner_user_id
        )[:limit]

    def get(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> BacktestAiConversation | None:
        conversation = self.conversations.get(conversation_id)
        if conversation is None or conversation.owner_user_id != owner_user_id:
            return None
        return conversation

    def count_messages(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> int:
        if self.get(conversation_id=conversation_id, owner_user_id=owner_user_id) is None:
            return 0
        return len(self.messages.get(conversation_id, ()))

    def append_user_exchange(
        self,
        *,
        conversation: BacktestAiConversation,
        user_message: BacktestAiConversationMessage,
        assistant_message: BacktestAiConversationMessage,
        run: BacktestAiConversationRun,
    ) -> BacktestAiConversation:
        self.conversations[conversation.conversation_id] = conversation
        self.messages.setdefault(conversation.conversation_id, []).extend(
            [user_message, assistant_message]
        )
        self.runs.setdefault(conversation.conversation_id, []).append(run)
        return conversation

    def list_messages(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> tuple[BacktestAiConversationMessage, ...]:
        if self.get(conversation_id=conversation_id, owner_user_id=owner_user_id) is None:
            return ()
        return tuple(self.messages.get(conversation_id, ()))

    def latest_run(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> BacktestAiConversationRun | None:
        if self.get(conversation_id=conversation_id, owner_user_id=owner_user_id) is None:
            return None
        runs = self.runs.get(conversation_id, ())
        return None if not runs else runs[-1]
