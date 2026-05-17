from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId

if TYPE_CHECKING:
    from trading.contexts.backtest.application.ai_configurator.dto import (
        BacktestAiConversation,
        BacktestAiConversationMessage,
        BacktestAiConversationRun,
    )


class BacktestAiConversationRepository(Protocol):
    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        """Count non-expired active conversations for one owner."""
        ...

    def create_with_startup_message(
        self,
        *,
        conversation: BacktestAiConversation,
        startup_message: BacktestAiConversationMessage,
    ) -> BacktestAiConversation:
        """Persist one conversation and its startup assistant message atomically."""
        ...

    def list_for_user(
        self,
        *,
        owner_user_id: UserId,
        limit: int,
    ) -> tuple[BacktestAiConversation, ...]:
        """Return owner-scoped active conversations."""
        ...

    def get(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> BacktestAiConversation | None:
        """Load one owner-scoped active conversation."""
        ...

    def count_messages(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> int:
        """Count owner-scoped messages in one conversation."""
        ...

    def append_user_exchange(
        self,
        *,
        conversation: BacktestAiConversation,
        user_message: BacktestAiConversationMessage,
        assistant_message: BacktestAiConversationMessage,
        run: BacktestAiConversationRun,
    ) -> BacktestAiConversation:
        """Persist a user message, assistant response, run row, and conversation update."""
        ...

    def list_messages(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> tuple[BacktestAiConversationMessage, ...]:
        """Return owner-scoped messages in chronological order."""
        ...

    def latest_run(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> BacktestAiConversationRun | None:
        """Return latest owner-scoped run for one conversation."""
        ...


__all__ = ["BacktestAiConversationRepository"]
