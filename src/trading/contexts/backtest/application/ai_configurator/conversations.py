from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Protocol
from uuid import UUID, uuid4

from trading.contexts.backtest.application.ports.backtest_ai_conversations import (
    BacktestAiConversationRepository,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import UserId

from .dto import (
    BacktestAiConversation,
    BacktestAiConversationLocale,
    BacktestAiConversationMessage,
    BacktestAiConversationModelResponse,
    BacktestAiConversationRead,
    BacktestAiConversationRun,
    BacktestAiConversationSendResult,
    BacktestAiConversationTitleSource,
    BacktestAiLoadAction,
)

BACKTEST_AI_CONVERSATION_ERROR_INVALID_REQUEST = "backtest.ai_config.invalid_request"
BACKTEST_AI_CONVERSATION_ERROR_NOT_FOUND = "backtest.ai_config.not_found"
BACKTEST_AI_CONVERSATION_ERROR_UNAVAILABLE = "backtest.ai_config.unavailable"

DEFAULT_BACKTEST_AI_CONVERSATION_TITLE = "New backtest chat"
DEFAULT_BACKTEST_AI_RETENTION_DAYS = 30
DEFAULT_BACKTEST_AI_MAX_CONVERSATIONS_PER_USER = 50
DEFAULT_BACKTEST_AI_MAX_MESSAGES_PER_CONVERSATION = 100

_VALID_LOCALES = {"ru", "en"}
_MAX_MESSAGE_CHARS = 16_000
_MAX_TITLE_CHARS = 80
_UNSAFE_TITLE_RE = re.compile(r"[\x00-\x1f\x7f<>]")


class BacktestAiConversationGateway(Protocol):
    def generate_reply(
        self,
        *,
        conversation: BacktestAiConversation,
        user_message: str,
        current_config: Mapping[str, Any] | None,
        ui_context: Mapping[str, Any] | None,
    ) -> BacktestAiConversationModelResponse:
        """Return one assistant response. Iteration 03 does not call LM Studio."""
        ...


@dataclass(frozen=True, slots=True)
class BacktestAiConversationLimits:
    retention_days: int = DEFAULT_BACKTEST_AI_RETENTION_DAYS
    max_conversations_per_user: int = DEFAULT_BACKTEST_AI_MAX_CONVERSATIONS_PER_USER
    max_messages_per_conversation: int = (
        DEFAULT_BACKTEST_AI_MAX_MESSAGES_PER_CONVERSATION
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("retention_days", self.retention_days),
            ("max_conversations_per_user", self.max_conversations_per_user),
            ("max_messages_per_conversation", self.max_messages_per_conversation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def as_mapping(self) -> dict[str, int]:
        return {
            "retention_days": self.retention_days,
            "max_conversations_per_user": self.max_conversations_per_user,
            "max_messages_per_conversation": self.max_messages_per_conversation,
        }


@dataclass(frozen=True, slots=True)
class DisabledBacktestAiConversationGateway:
    def generate_reply(
        self,
        *,
        conversation: BacktestAiConversation,
        user_message: str,
        current_config: Mapping[str, Any] | None,
        ui_context: Mapping[str, Any] | None,
    ) -> BacktestAiConversationModelResponse:
        _ = conversation, user_message, current_config, ui_context
        return BacktestAiConversationModelResponse(
            assistant_message=(
                "AI assistant generation is not connected in this iteration."
            ),
            conversation_title=None,
            status="awaiting_model",
            load_action=BacktestAiLoadAction(
                enabled=False,
                state="unavailable",
                reason="backend_not_ready",
            ),
        )


@dataclass(frozen=True, slots=True)
class BacktestAiConversationUseCase:
    repository: BacktestAiConversationRepository
    gateway: BacktestAiConversationGateway = DisabledBacktestAiConversationGateway()
    limits: BacktestAiConversationLimits = BacktestAiConversationLimits()

    def create_conversation(
        self,
        *,
        user_id: UserId,
        locale: str,
        now: datetime | None = None,
    ) -> BacktestAiConversationRead:
        normalized_locale = _normalize_locale(locale=locale)
        effective_now = _utc_now(now=now)
        active_count = self.repository.count_active_for_user(owner_user_id=user_id)
        if active_count >= self.limits.max_conversations_per_user:
            raise RoehubError(
                code=BACKTEST_AI_CONVERSATION_ERROR_INVALID_REQUEST,
                message="Backtest AI conversation limit exceeded",
                details={
                    "limit": self.limits.max_conversations_per_user,
                    "used": active_count,
                },
            )

        conversation_id = uuid4()
        conversation = BacktestAiConversation(
            conversation_id=conversation_id,
            owner_user_id=user_id,
            locale=normalized_locale,
            status="active",
            title=DEFAULT_BACKTEST_AI_CONVERSATION_TITLE,
            title_source="fallback",
            created_at=effective_now,
            updated_at=effective_now,
            last_message_at=effective_now,
            expires_at=effective_now + timedelta(days=self.limits.retention_days),
        )
        startup_message = BacktestAiConversationMessage(
            message_id=uuid4(),
            conversation_id=conversation_id,
            owner_user_id=user_id,
            role="assistant",
            content=_startup_message(locale=normalized_locale),
            created_at=effective_now,
            metadata_json={
                "kind": "startup",
                "retention_days": self.limits.retention_days,
            },
        )
        stored = self.repository.create_with_startup_message(
            conversation=conversation,
            startup_message=startup_message,
        )
        return BacktestAiConversationRead(
            conversation=stored,
            messages=(startup_message,),
            latest_run=None,
        )

    def list_conversations(
        self,
        *,
        user_id: UserId,
        limit: int = 50,
    ) -> tuple[BacktestAiConversation, ...]:
        normalized_limit = max(1, min(limit, self.limits.max_conversations_per_user))
        return self.repository.list_for_user(
            owner_user_id=user_id,
            limit=normalized_limit,
        )

    def read_conversation(
        self,
        *,
        user_id: UserId,
        conversation_id: UUID,
    ) -> BacktestAiConversationRead:
        conversation = self._get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return BacktestAiConversationRead(
            conversation=conversation,
            messages=self.repository.list_messages(
                conversation_id=conversation_id,
                owner_user_id=user_id,
            ),
            latest_run=self.repository.latest_run(
                conversation_id=conversation_id,
                owner_user_id=user_id,
            ),
        )

    def send_message(
        self,
        *,
        user_id: UserId,
        conversation_id: UUID,
        message: str,
        current_config: Mapping[str, Any] | None = None,
        ui_context: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> BacktestAiConversationSendResult:
        conversation = self._get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        normalized_message = _normalize_message(message=message)
        message_count = self.repository.count_messages(
            conversation_id=conversation_id,
            owner_user_id=user_id,
        )
        if message_count + 2 > self.limits.max_messages_per_conversation:
            raise RoehubError(
                code=BACKTEST_AI_CONVERSATION_ERROR_INVALID_REQUEST,
                message="Backtest AI conversation message limit exceeded",
                details={
                    "limit": self.limits.max_messages_per_conversation,
                    "used": message_count,
                },
            )

        model_response = self.gateway.generate_reply(
            conversation=conversation,
            user_message=normalized_message,
            current_config=current_config,
            ui_context=ui_context,
        )
        effective_now = _utc_now(now=now)
        next_title = _next_title(
            conversation=conversation,
            model_title=model_response.conversation_title,
        )
        updated_conversation = BacktestAiConversation(
            conversation_id=conversation.conversation_id,
            owner_user_id=conversation.owner_user_id,
            locale=conversation.locale,
            status=conversation.status,
            title=next_title[0],
            title_source=next_title[1],
            created_at=conversation.created_at,
            updated_at=effective_now,
            last_message_at=effective_now,
            expires_at=effective_now + timedelta(days=self.limits.retention_days),
        )
        user_message = BacktestAiConversationMessage(
            message_id=uuid4(),
            conversation_id=conversation_id,
            owner_user_id=user_id,
            role="user",
            content=normalized_message,
            created_at=effective_now,
            metadata_json={
                "current_config_present": current_config is not None,
                "ui_context_present": ui_context is not None,
            },
        )
        assistant_message = BacktestAiConversationMessage(
            message_id=uuid4(),
            conversation_id=conversation_id,
            owner_user_id=user_id,
            role="assistant",
            content=model_response.assistant_message,
            created_at=effective_now,
            metadata_json={
                "run_status": model_response.status,
                "load_action": model_response.load_action.as_mapping(),
            },
        )
        run = BacktestAiConversationRun(
            run_id=uuid4(),
            conversation_id=conversation_id,
            owner_user_id=user_id,
            user_message_id=user_message.message_id,
            assistant_message_id=assistant_message.message_id,
            status=model_response.status,
            intent=model_response.intent,
            load_action=model_response.load_action,
            current_config_json=current_config,
            validated_config_json=model_response.validated_config_json,
            model_id=model_response.model_id,
            failure_reason=model_response.load_action.reason,
            created_at=effective_now,
            updated_at=effective_now,
        )
        stored_conversation = self.repository.append_user_exchange(
            conversation=updated_conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            run=run,
        )
        return BacktestAiConversationSendResult(
            conversation=stored_conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            run=run,
        )

    def get_status(
        self,
        *,
        user_id: UserId,
        conversation_id: UUID,
    ) -> BacktestAiConversationRun | None:
        self._get_conversation(user_id=user_id, conversation_id=conversation_id)
        return self.repository.latest_run(
            conversation_id=conversation_id,
            owner_user_id=user_id,
        )

    def _get_conversation(
        self,
        *,
        user_id: UserId,
        conversation_id: UUID,
    ) -> BacktestAiConversation:
        conversation = self.repository.get(
            conversation_id=conversation_id,
            owner_user_id=user_id,
        )
        if conversation is None:
            raise RoehubError(
                code=BACKTEST_AI_CONVERSATION_ERROR_NOT_FOUND,
                message="Backtest AI conversation was not found",
                details={"conversation_id": str(conversation_id)},
            )
        return conversation


def _normalize_locale(*, locale: str) -> BacktestAiConversationLocale:
    normalized = locale.strip().lower()
    if normalized not in _VALID_LOCALES:
        raise RoehubError(
            code=BACKTEST_AI_CONVERSATION_ERROR_INVALID_REQUEST,
            message="Backtest AI conversation locale is invalid",
            details={"locale": locale},
        )
    return normalized  # type: ignore[return-value]


def _normalize_message(*, message: str) -> str:
    normalized = message.strip()
    if not normalized:
        raise RoehubError(
            code=BACKTEST_AI_CONVERSATION_ERROR_INVALID_REQUEST,
            message="Backtest AI message must be non-empty",
            details={"field": "message"},
        )
    if len(normalized) > _MAX_MESSAGE_CHARS:
        raise RoehubError(
            code=BACKTEST_AI_CONVERSATION_ERROR_INVALID_REQUEST,
            message="Backtest AI message is too large",
            details={"field": "message", "max_chars": _MAX_MESSAGE_CHARS},
        )
    return normalized


def _next_title(
    *,
    conversation: BacktestAiConversation,
    model_title: str | None,
) -> tuple[str, BacktestAiConversationTitleSource]:
    if conversation.title_source == "model":
        return conversation.title, conversation.title_source
    safe_title = _safe_model_title(model_title=model_title)
    if safe_title is None:
        return DEFAULT_BACKTEST_AI_CONVERSATION_TITLE, "fallback"
    return safe_title, "model"


def _safe_model_title(*, model_title: str | None) -> str | None:
    if model_title is None:
        return None
    normalized = " ".join(model_title.strip().split())
    if (
        not normalized
        or len(normalized) > _MAX_TITLE_CHARS
        or _UNSAFE_TITLE_RE.search(normalized)
    ):
        return None
    return normalized


def _startup_message(*, locale: BacktestAiConversationLocale) -> str:
    if locale == "ru":
        return "Ассистент готов помочь с конфигурацией бектеста."
    return "The assistant is ready to help with backtest configuration."


def _utc_now(*, now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(UTC)


__all__ = [
    "BACKTEST_AI_CONVERSATION_ERROR_INVALID_REQUEST",
    "BACKTEST_AI_CONVERSATION_ERROR_NOT_FOUND",
    "BACKTEST_AI_CONVERSATION_ERROR_UNAVAILABLE",
    "DEFAULT_BACKTEST_AI_CONVERSATION_TITLE",
    "DEFAULT_BACKTEST_AI_MAX_CONVERSATIONS_PER_USER",
    "DEFAULT_BACKTEST_AI_MAX_MESSAGES_PER_CONVERSATION",
    "DEFAULT_BACKTEST_AI_RETENTION_DAYS",
    "BacktestAiConversationGateway",
    "BacktestAiConversationLimits",
    "BacktestAiConversationUseCase",
    "DisabledBacktestAiConversationGateway",
]
