from __future__ import annotations

from uuid import UUID

import pytest

from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConversation,
    BacktestAiConversationMessage,
    BacktestAiConversationModelResponse,
    BacktestAiConversationRun,
    BacktestAiConversationUseCase,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId


def test_send_message_persists_first_valid_model_conversation_title() -> None:
    repository = _InMemoryConversationRepository()
    gateway = _Gateway(
        responses=[
            BacktestAiConversationModelResponse(
                assistant_message="Prepared a draft.",
                conversation_title="BTCUSDT RSI setup",
            ),
            BacktestAiConversationModelResponse(
                assistant_message="Updated a draft.",
                conversation_title="Ignored later title",
            ),
        ]
    )
    use_case = BacktestAiConversationUseCase(
        repository=repository,
        gateway=gateway,
    )
    user_id = _user("00000000-0000-0000-0000-000000000701")
    created = use_case.create_conversation(user_id=user_id, locale="en")

    first = use_case.send_message(
        user_id=user_id,
        conversation_id=created.conversation.conversation_id,
        message="Create RSI for BTCUSDT",
    )
    second = use_case.send_message(
        user_id=user_id,
        conversation_id=created.conversation.conversation_id,
        message="Add EMA too",
    )

    assert first.conversation.title == "BTCUSDT RSI setup"
    assert first.conversation.title_source == "model"
    assert second.conversation.title == "BTCUSDT RSI setup"
    assert repository.messages[created.conversation.conversation_id][-1].content == (
        "Updated a draft."
    )


def test_missing_or_unsafe_model_title_uses_fallback() -> None:
    repository = _InMemoryConversationRepository()
    use_case = BacktestAiConversationUseCase(
        repository=repository,
        gateway=_Gateway(
            responses=[
                BacktestAiConversationModelResponse(
                    assistant_message="No safe title.",
                    conversation_title="<script>alert(1)</script>",
                )
            ]
        ),
    )
    user_id = _user("00000000-0000-0000-0000-000000000702")
    created = use_case.create_conversation(user_id=user_id, locale="en")

    result = use_case.send_message(
        user_id=user_id,
        conversation_id=created.conversation.conversation_id,
        message="Create RSI for BTCUSDT",
    )

    assert result.conversation.title == "New backtest chat"
    assert result.conversation.title_source == "fallback"


def test_owner_isolation_hides_foreign_conversation() -> None:
    repository = _InMemoryConversationRepository()
    use_case = BacktestAiConversationUseCase(repository=repository)
    owner = _user("00000000-0000-0000-0000-000000000703")
    other = _user("00000000-0000-0000-0000-000000000704")
    created = use_case.create_conversation(user_id=owner, locale="en")

    with pytest.raises(RoehubError) as exc_info:
        use_case.read_conversation(
            user_id=other,
            conversation_id=created.conversation.conversation_id,
        )

    assert exc_info.value.code == "backtest.ai_config.not_found"
    assert use_case.list_conversations(user_id=other) == ()


def test_load_action_remains_disabled_until_backend_ready_state() -> None:
    repository = _InMemoryConversationRepository()
    use_case = BacktestAiConversationUseCase(repository=repository)
    user_id = _user("00000000-0000-0000-0000-000000000705")
    created = use_case.create_conversation(user_id=user_id, locale="en")

    result = use_case.send_message(
        user_id=user_id,
        conversation_id=created.conversation.conversation_id,
        message="Create RSI for BTCUSDT",
        current_config={"coordinates": {"symbol": "BTCUSDT"}},
    )

    assert result.run.load_action.as_mapping() == {
        "enabled": False,
        "state": "unavailable",
        "reason": "backend_not_ready",
        "config": None,
    }


class _Gateway:
    def __init__(self, *, responses: list[BacktestAiConversationModelResponse]) -> None:
        self._responses = list(responses)

    def generate_reply(
        self,
        *,
        conversation: BacktestAiConversation,
        user_message: str,
        current_config: object | None,
        ui_context: object | None,
    ) -> BacktestAiConversationModelResponse:
        _ = conversation, user_message, current_config, ui_context
        return self._responses.pop(0)


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
            sorted(
                (
                    conversation
                    for conversation in self.conversations.values()
                    if conversation.owner_user_id == owner_user_id
                ),
                key=lambda item: item.last_message_at,
                reverse=True,
            )[:limit]
        )

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
        conversation = self.get(
            conversation_id=conversation_id,
            owner_user_id=owner_user_id,
        )
        if conversation is None:
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
        conversation = self.get(
            conversation_id=conversation_id,
            owner_user_id=owner_user_id,
        )
        if conversation is None:
            return ()
        return tuple(self.messages.get(conversation_id, ()))

    def latest_run(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> BacktestAiConversationRun | None:
        conversation = self.get(
            conversation_id=conversation_id,
            owner_user_id=owner_user_id,
        )
        if conversation is None:
            return None
        runs = self.runs.get(conversation_id, ())
        return None if not runs else runs[-1]


def _user(value: str) -> UserId:
    return UserId.from_string(value)
