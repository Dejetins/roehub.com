from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from trading.contexts.backtest.adapters.outbound.persistence.postgres.gateway import (
    BacktestPostgresGateway,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConversation,
    BacktestAiConversationMessage,
    BacktestAiConversationRun,
    BacktestAiConversationRunStatus,
    BacktestAiLoadAction,
)
from trading.contexts.backtest.application.ports import BacktestAiConversationRepository
from trading.contexts.backtest.domain.errors import BacktestStorageError
from trading.shared_kernel.primitives import UserId

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_CONVERSATION_COLUMNS = """
    conversation_id,
    owner_user_id,
    locale,
    status,
    title,
    title_source,
    created_at,
    updated_at,
    last_message_at,
    expires_at
"""

_MESSAGE_COLUMNS = """
    message_id,
    conversation_id,
    owner_user_id,
    role,
    content,
    metadata_json,
    created_at
"""

_RUN_COLUMNS = """
    run_id,
    conversation_id,
    owner_user_id,
    user_message_id,
    assistant_message_id,
    status,
    intent,
    load_action_json,
    current_config_json,
    validated_config_json,
    model_id,
    failure_reason,
    created_at,
    updated_at
"""


class PostgresBacktestAiConversationRepository(BacktestAiConversationRepository):
    def __init__(
        self,
        *,
        gateway: BacktestPostgresGateway,
        conversations_table: str = "backtest_ai_conversations",
        messages_table: str = "backtest_ai_conversation_messages",
        runs_table: str = "backtest_ai_conversation_runs",
    ) -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("PostgresBacktestAiConversationRepository requires gateway")
        self._gateway = gateway
        self._conversations_table = _table_name(
            conversations_table,
            field_name="conversations_table",
        )
        self._messages_table = _table_name(messages_table, field_name="messages_table")
        self._runs_table = _table_name(runs_table, field_name="runs_table")

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT count(*) AS count
            FROM {self._conversations_table}
            WHERE owner_user_id = %(owner_user_id)s
              AND status = 'active'
              AND expires_at > now()
            """,
            parameters={"owner_user_id": str(owner_user_id)},
        )
        return 0 if row is None else int(row["count"])

    def create_with_startup_message(
        self,
        *,
        conversation: BacktestAiConversation,
        startup_message: BacktestAiConversationMessage,
    ) -> BacktestAiConversation:
        query = f"""
        WITH inserted_conversation AS (
            INSERT INTO {self._conversations_table}
            (
                conversation_id,
                owner_user_id,
                locale,
                status,
                title,
                title_source,
                created_at,
                updated_at,
                last_message_at,
                expires_at
            )
            VALUES
            (
                %(conversation_id)s,
                %(owner_user_id)s,
                %(locale)s,
                %(status)s,
                %(title)s,
                %(title_source)s,
                %(created_at)s,
                %(updated_at)s,
                %(last_message_at)s,
                %(expires_at)s
            )
            RETURNING
                {_CONVERSATION_COLUMNS}
        ),
        inserted_startup_message AS (
            INSERT INTO {self._messages_table}
            (
                message_id,
                conversation_id,
                owner_user_id,
                role,
                content,
                metadata_json,
                created_at
            )
            VALUES
            (
                %(startup_message_id)s,
                %(conversation_id)s,
                %(owner_user_id)s,
                %(startup_role)s,
                %(startup_content)s,
                %(startup_metadata_json)s::jsonb,
                %(startup_created_at)s
            )
        )
        SELECT
            {_CONVERSATION_COLUMNS}
        FROM inserted_conversation
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                **_conversation_parameters(conversation=conversation),
                "startup_message_id": str(startup_message.message_id),
                "startup_role": startup_message.role,
                "startup_content": startup_message.content,
                "startup_metadata_json": _json_dumps(startup_message.metadata_json),
                "startup_created_at": startup_message.created_at,
            },
        )
        if row is None:
            raise BacktestStorageError("AI conversation insert returned no row")
        return _map_conversation_row(row=row)

    def list_for_user(
        self,
        *,
        owner_user_id: UserId,
        limit: int,
    ) -> tuple[BacktestAiConversation, ...]:
        rows = self._gateway.fetch_all(
            query=f"""
            SELECT
                {_CONVERSATION_COLUMNS}
            FROM {self._conversations_table}
            WHERE owner_user_id = %(owner_user_id)s
              AND status = 'active'
              AND expires_at > now()
            ORDER BY last_message_at DESC, conversation_id DESC
            LIMIT %(limit)s
            """,
            parameters={"owner_user_id": str(owner_user_id), "limit": limit},
        )
        return tuple(_map_conversation_row(row=row) for row in rows)

    def get(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> BacktestAiConversation | None:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT
                {_CONVERSATION_COLUMNS}
            FROM {self._conversations_table}
            WHERE conversation_id = %(conversation_id)s
              AND owner_user_id = %(owner_user_id)s
              AND status = 'active'
              AND expires_at > now()
            LIMIT 1
            """,
            parameters={
                "conversation_id": str(conversation_id),
                "owner_user_id": str(owner_user_id),
            },
        )
        if row is None:
            return None
        return _map_conversation_row(row=row)

    def count_messages(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> int:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT count(*) AS count
            FROM {self._messages_table}
            WHERE conversation_id = %(conversation_id)s
              AND owner_user_id = %(owner_user_id)s
            """,
            parameters={
                "conversation_id": str(conversation_id),
                "owner_user_id": str(owner_user_id),
            },
        )
        return 0 if row is None else int(row["count"])

    def append_user_exchange(
        self,
        *,
        conversation: BacktestAiConversation,
        user_message: BacktestAiConversationMessage,
        assistant_message: BacktestAiConversationMessage,
        run: BacktestAiConversationRun,
    ) -> BacktestAiConversation:
        query = f"""
        WITH updated_conversation AS (
            UPDATE {self._conversations_table}
            SET
                title = %(title)s,
                title_source = %(title_source)s,
                updated_at = %(updated_at)s,
                last_message_at = %(last_message_at)s,
                expires_at = %(expires_at)s
            WHERE conversation_id = %(conversation_id)s
              AND owner_user_id = %(owner_user_id)s
              AND status = 'active'
              AND expires_at > now()
            RETURNING
                {_CONVERSATION_COLUMNS}
        ),
        inserted_user_message AS (
            INSERT INTO {self._messages_table}
            (
                message_id,
                conversation_id,
                owner_user_id,
                role,
                content,
                metadata_json,
                created_at
            )
            SELECT
                %(user_message_id)s,
                %(conversation_id)s,
                %(owner_user_id)s,
                %(user_role)s,
                %(user_content)s,
                %(user_metadata_json)s::jsonb,
                %(user_created_at)s
            FROM updated_conversation
        ),
        inserted_assistant_message AS (
            INSERT INTO {self._messages_table}
            (
                message_id,
                conversation_id,
                owner_user_id,
                role,
                content,
                metadata_json,
                created_at
            )
            SELECT
                %(assistant_message_id)s,
                %(conversation_id)s,
                %(owner_user_id)s,
                %(assistant_role)s,
                %(assistant_content)s,
                %(assistant_metadata_json)s::jsonb,
                %(assistant_created_at)s
            FROM updated_conversation
        ),
        inserted_run AS (
            INSERT INTO {self._runs_table}
            (
                run_id,
                conversation_id,
                owner_user_id,
                user_message_id,
                assistant_message_id,
                status,
                intent,
                load_action_json,
                current_config_json,
                validated_config_json,
                model_id,
                failure_reason,
                created_at,
                updated_at
            )
            SELECT
                %(run_id)s,
                %(conversation_id)s,
                %(owner_user_id)s,
                %(run_user_message_id)s,
                %(run_assistant_message_id)s,
                %(run_status)s,
                %(intent)s,
                %(load_action_json)s::jsonb,
                %(current_config_json)s::jsonb,
                %(validated_config_json)s::jsonb,
                %(model_id)s,
                %(failure_reason)s,
                %(run_created_at)s,
                %(run_updated_at)s
            FROM updated_conversation
        )
        SELECT
            {_CONVERSATION_COLUMNS}
        FROM updated_conversation
        """
        row = self._gateway.fetch_one(
            query=query,
            parameters={
                **_conversation_parameters(conversation=conversation),
                "user_message_id": str(user_message.message_id),
                "user_role": user_message.role,
                "user_content": user_message.content,
                "user_metadata_json": _json_dumps(user_message.metadata_json),
                "user_created_at": user_message.created_at,
                "assistant_message_id": str(assistant_message.message_id),
                "assistant_role": assistant_message.role,
                "assistant_content": assistant_message.content,
                "assistant_metadata_json": _json_dumps(
                    assistant_message.metadata_json
                ),
                "assistant_created_at": assistant_message.created_at,
                "run_id": str(run.run_id),
                "run_user_message_id": str(run.user_message_id),
                "run_assistant_message_id": str(run.assistant_message_id),
                "run_status": run.status,
                "intent": run.intent,
                "load_action_json": _json_dumps(run.load_action.as_mapping()),
                "current_config_json": _json_dumps(run.current_config_json),
                "validated_config_json": _json_dumps(run.validated_config_json),
                "model_id": run.model_id,
                "failure_reason": run.failure_reason,
                "run_created_at": run.created_at,
                "run_updated_at": run.updated_at,
            },
        )
        if row is None:
            raise BacktestStorageError("AI conversation exchange insert returned no row")
        return _map_conversation_row(row=row)

    def list_messages(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> tuple[BacktestAiConversationMessage, ...]:
        rows = self._gateway.fetch_all(
            query=f"""
            SELECT
                {_MESSAGE_COLUMNS}
            FROM {self._messages_table}
            WHERE conversation_id = %(conversation_id)s
              AND owner_user_id = %(owner_user_id)s
            ORDER BY created_at ASC, message_seq ASC, message_id ASC
            """,
            parameters={
                "conversation_id": str(conversation_id),
                "owner_user_id": str(owner_user_id),
            },
        )
        return tuple(_map_message_row(row=row) for row in rows)

    def latest_run(
        self,
        *,
        conversation_id: UUID,
        owner_user_id: UserId,
    ) -> BacktestAiConversationRun | None:
        row = self._gateway.fetch_one(
            query=f"""
            SELECT
                {_RUN_COLUMNS}
            FROM {self._runs_table}
            WHERE conversation_id = %(conversation_id)s
              AND owner_user_id = %(owner_user_id)s
            ORDER BY created_at DESC, run_id DESC
            LIMIT 1
            """,
            parameters={
                "conversation_id": str(conversation_id),
                "owner_user_id": str(owner_user_id),
            },
        )
        if row is None:
            return None
        return _map_run_row(row=row)


def _conversation_parameters(
    *,
    conversation: BacktestAiConversation,
) -> dict[str, Any]:
    return {
        "conversation_id": str(conversation.conversation_id),
        "owner_user_id": str(conversation.owner_user_id),
        "locale": conversation.locale,
        "status": conversation.status,
        "title": conversation.title,
        "title_source": conversation.title_source,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "last_message_at": conversation.last_message_at,
        "expires_at": conversation.expires_at,
    }


def _map_conversation_row(*, row: Mapping[str, Any]) -> BacktestAiConversation:
    return BacktestAiConversation(
        conversation_id=_uuid(row["conversation_id"]),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        locale=str(row["locale"]),  # type: ignore[arg-type]
        status=str(row["status"]),  # type: ignore[arg-type]
        title=str(row["title"]),
        title_source=str(row["title_source"]),  # type: ignore[arg-type]
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
        last_message_at=cast(datetime, row["last_message_at"]),
        expires_at=cast(datetime, row["expires_at"]),
    )


def _map_message_row(*, row: Mapping[str, Any]) -> BacktestAiConversationMessage:
    return BacktestAiConversationMessage(
        message_id=_uuid(row["message_id"]),
        conversation_id=_uuid(row["conversation_id"]),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        role=str(row["role"]),  # type: ignore[arg-type]
        content=str(row["content"]),
        metadata_json=_json_loads(row["metadata_json"]),
        created_at=cast(datetime, row["created_at"]),
    )


def _map_run_row(*, row: Mapping[str, Any]) -> BacktestAiConversationRun:
    load_action_payload = _json_loads(row["load_action_json"])
    return BacktestAiConversationRun(
        run_id=_uuid(row["run_id"]),
        conversation_id=_uuid(row["conversation_id"]),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        user_message_id=_uuid(row["user_message_id"]),
        assistant_message_id=_uuid(row["assistant_message_id"]),
        status=cast(BacktestAiConversationRunStatus, str(row["status"])),
        intent=_optional_str(row.get("intent")),
        load_action=BacktestAiLoadAction(
            enabled=bool(load_action_payload.get("enabled")),
            state=str(load_action_payload.get("state") or "unavailable"),
            reason=_optional_str(load_action_payload.get("reason")),
            config=cast(Mapping[str, Any] | None, load_action_payload.get("config")),
        ),
        current_config_json=_optional_mapping(row.get("current_config_json")),
        validated_config_json=_optional_mapping(row.get("validated_config_json")),
        model_id=_optional_str(row.get("model_id")),
        failure_reason=_optional_str(row.get("failure_reason")),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
    )


def _json_dumps(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _json_loads(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, Mapping):
        return None
    return dict(parsed)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _table_name(value: str, *, field_name: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a safe SQL identifier")
    return value


__all__ = ["PostgresBacktestAiConversationRepository"]
