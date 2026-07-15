from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any, Callable, Mapping, cast
from uuid import UUID

import httpx

from trading.contexts.notifications.adapters.outbound.persistence.postgres.gateway import (
    NotificationPostgresGateway,
)
from trading.shared_kernel.primitives import OrganizationId


class TelegramBotApiUpdateSource:
    def __init__(
        self,
        *,
        api_base_url: str,
        credential_source: Callable[[], str],
        client: httpx.Client | None = None,
        connect_timeout_seconds: float = 3.0,
    ) -> None:
        self._api_base_url = api_base_url.strip().rstrip("/")
        self._credential_source = credential_source
        self._client = client or httpx.Client()
        self._connect_timeout_seconds = connect_timeout_seconds

    def fetch_updates(
        self, *, offset: int, long_poll_timeout_seconds: int
    ) -> tuple[Mapping[str, Any], ...]:
        credential = self._credential_source().strip()
        if not credential:
            raise ValueError("Telegram credential is unavailable")
        response = self._client.get(
            f"{self._api_base_url}/bot{credential}/getUpdates",
            params={"offset": offset, "timeout": long_poll_timeout_seconds},
            timeout=httpx.Timeout(
                long_poll_timeout_seconds + 5.0,
                connect=self._connect_timeout_seconds,
            ),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Telegram update response is invalid")
        result = payload.get("result")
        if not isinstance(result, list):
            raise ValueError("Telegram update result is invalid")
        updates: list[Mapping[str, Any]] = []
        for item in result:
            if not isinstance(item, Mapping):
                raise ValueError("Telegram update item is invalid")
            updates.append(cast(Mapping[str, Any], item))
        return tuple(updates)


class PostgresTelegramRecipientScopeResolver:
    def __init__(self, *, gateway: NotificationPostgresGateway) -> None:
        self._gateway = gateway

    def resolve_organization(
        self,
        *,
        provider_instance_id: UUID,
        chat_id_ref: str,
        command_text: str | None,
        now: datetime,
    ) -> OrganizationId | None:
        rows = self._gateway.fetch_all(
            query="""
            SELECT organization_id
            FROM notification_telegram_recipient_bindings
            WHERE provider_instance_id = %(provider_instance_id)s
              AND chat_id_ref = %(chat_id_ref)s
              AND status = 'confirmed'
            ORDER BY organization_id
            LIMIT 2
            """,
            parameters={
                "provider_instance_id": str(provider_instance_id),
                "chat_id_ref": chat_id_ref,
            },
        )
        if not rows:
            binding_code_hash = _start_binding_code_hash(command_text=command_text)
            if binding_code_hash is None:
                return None
            rows = self._gateway.fetch_all(
                query="""
                SELECT organization_id
                FROM notification_telegram_binding_codes
                WHERE provider_instance_id = %(provider_instance_id)s
                  AND code_hash = %(code_hash)s
                  AND consumed_at IS NULL
                  AND expires_at >= %(now)s
                ORDER BY organization_id
                LIMIT 2
                """,
                parameters={
                    "provider_instance_id": str(provider_instance_id),
                    "code_hash": binding_code_hash,
                    "now": now,
                },
            )
            if not rows:
                return None
        if len(rows) != 1:
            raise ValueError("Telegram recipient binding is ambiguous")
        return OrganizationId.from_string(str(rows[0]["organization_id"]))


def _start_binding_code_hash(*, command_text: str | None) -> str | None:
    if command_text is None:
        return None
    tokens = tuple(part for part in command_text.strip().split() if part)
    if len(tokens) != 2:
        return None
    command_name = tokens[0].removeprefix("/").split("@", maxsplit=1)[0].casefold()
    if command_name != "start":
        return None
    normalized = tokens[1].strip().upper()
    if not normalized:
        return None
    return sha256(normalized.encode()).hexdigest()
