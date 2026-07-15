from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID

from trading.contexts.notifications.application.telegram_binding import (
    NotificationTelegramBindingCode,
    NotificationTelegramBindingStatus,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

from .gateway import NotificationPostgresGateway


class PostgresNotificationTelegramBindingStore:
    def __init__(self, *, gateway: NotificationPostgresGateway) -> None:
        self._gateway = gateway

    def save_binding_code(
        self, *, binding_code: NotificationTelegramBindingCode
    ) -> NotificationTelegramBindingCode:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_telegram_binding_codes
              (binding_code_id, organization_id, provider_instance_id, owner_user_id,
               code_hash, expires_at, created_at, consumed_at)
            VALUES
              (%(binding_code_id)s, %(organization_id)s, %(provider_instance_id)s,
               %(owner_user_id)s, %(code_hash)s, %(expires_at)s, %(created_at)s,
               %(consumed_at)s)
            RETURNING binding_code_id, organization_id, provider_instance_id,
                      owner_user_id, code_hash, expires_at, created_at, consumed_at
            """,
            parameters=_code_parameters(binding_code),
        )
        return _map_code(_require_row(row, "Telegram binding code insert"))

    def get_active_binding_code_by_hash(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        code_hash: str,
        now: Any,
    ) -> NotificationTelegramBindingCode | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT binding_code_id, organization_id, provider_instance_id,
                   owner_user_id, code_hash, expires_at, created_at, consumed_at
            FROM notification_telegram_binding_codes
            WHERE organization_id = %(organization_id)s
              AND provider_instance_id = %(provider_instance_id)s
              AND code_hash = %(code_hash)s
              AND consumed_at IS NULL
              AND expires_at >= %(now)s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            parameters={
                "organization_id": str(organization_id),
                "provider_instance_id": str(provider_instance_id),
                "code_hash": code_hash,
                "now": now,
            },
        )
        return None if row is None else _map_code(row)

    def consume_binding_code(
        self, *, binding_code_id: UUID, consumed_at: Any
    ) -> NotificationTelegramBindingCode | None:
        row = self._gateway.fetch_one(
            query="""
            UPDATE notification_telegram_binding_codes SET consumed_at = %(consumed_at)s
            WHERE binding_code_id = %(binding_code_id)s
              AND consumed_at IS NULL
              AND expires_at >= %(consumed_at)s
            RETURNING binding_code_id, organization_id, provider_instance_id,
                      owner_user_id, code_hash, expires_at, created_at, consumed_at
            """,
            parameters={
                "binding_code_id": str(binding_code_id),
                "consumed_at": consumed_at,
            },
        )
        return None if row is None else _map_code(row)

    def confirm_chat(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        owner_user_id: UserId,
        chat_id_ref: str,
        recipient_secret_ref: str,
        confirmed_at: Any,
    ) -> NotificationTelegramBindingStatus:
        row = self._gateway.fetch_one(
            query="""
            INSERT INTO notification_telegram_recipient_bindings
              (organization_id, provider_instance_id, owner_user_id, chat_id_ref,
               recipient_secret_ref, status, confirmed_at, updated_at)
            VALUES
              (%(organization_id)s, %(provider_instance_id)s, %(owner_user_id)s,
               %(chat_id_ref)s, %(recipient_secret_ref)s, 'confirmed', %(confirmed_at)s,
               %(confirmed_at)s)
            ON CONFLICT (organization_id, provider_instance_id, owner_user_id) DO UPDATE SET
              chat_id_ref = EXCLUDED.chat_id_ref,
              status = 'confirmed',
              confirmed_at = EXCLUDED.confirmed_at,
              updated_at = EXCLUDED.updated_at
            RETURNING organization_id, provider_instance_id, owner_user_id,
                      chat_id_ref, status, confirmed_at
            """,
            parameters={
                "organization_id": str(organization_id),
                "provider_instance_id": str(provider_instance_id),
                "owner_user_id": str(owner_user_id),
                "chat_id_ref": chat_id_ref,
                "recipient_secret_ref": recipient_secret_ref,
                "confirmed_at": confirmed_at,
            },
        )
        return _map_status(_require_row(row, "Telegram binding confirmation"))

    def consume_binding_code_and_confirm_chat(
        self,
        *,
        binding_code_id: UUID,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        owner_user_id: UserId,
        chat_id_ref: str,
        recipient_secret_ref: str,
        confirmed_at: Any,
    ) -> NotificationTelegramBindingStatus | None:
        row = self._gateway.fetch_one(
            query="""
            WITH consumed AS (
              UPDATE notification_telegram_binding_codes SET
                consumed_at = %(confirmed_at)s
              WHERE binding_code_id = %(binding_code_id)s
                AND organization_id = %(organization_id)s
                AND provider_instance_id = %(provider_instance_id)s
                AND owner_user_id = %(owner_user_id)s
                AND consumed_at IS NULL
                AND expires_at >= %(confirmed_at)s
              RETURNING organization_id, provider_instance_id, owner_user_id
            )
            INSERT INTO notification_telegram_recipient_bindings
              (organization_id, provider_instance_id, owner_user_id, chat_id_ref,
               recipient_secret_ref, status, confirmed_at, updated_at)
            SELECT organization_id, provider_instance_id, owner_user_id,
                   %(chat_id_ref)s, %(recipient_secret_ref)s, 'confirmed',
                   %(confirmed_at)s, %(confirmed_at)s
            FROM consumed
            ON CONFLICT (organization_id, provider_instance_id, owner_user_id) DO UPDATE SET
              chat_id_ref = EXCLUDED.chat_id_ref,
              recipient_secret_ref = EXCLUDED.recipient_secret_ref,
              status = 'confirmed',
              confirmed_at = EXCLUDED.confirmed_at,
              updated_at = EXCLUDED.updated_at
            RETURNING organization_id, provider_instance_id, owner_user_id,
                      chat_id_ref, status, confirmed_at
            """,
            parameters={
                "binding_code_id": str(binding_code_id),
                "organization_id": str(organization_id),
                "provider_instance_id": str(provider_instance_id),
                "owner_user_id": str(owner_user_id),
                "chat_id_ref": chat_id_ref,
                "recipient_secret_ref": recipient_secret_ref,
                "confirmed_at": confirmed_at,
            },
        )
        return None if row is None else _map_status(row)

    def get_binding_status(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        owner_user_id: UserId,
    ) -> NotificationTelegramBindingStatus:
        row = self._gateway.fetch_one(
            query="""
            SELECT organization_id, provider_instance_id, owner_user_id,
                   chat_id_ref, status, confirmed_at
            FROM notification_telegram_recipient_bindings
            WHERE organization_id = %(organization_id)s
              AND provider_instance_id = %(provider_instance_id)s
              AND owner_user_id = %(owner_user_id)s
            """,
            parameters={
                "organization_id": str(organization_id),
                "provider_instance_id": str(provider_instance_id),
                "owner_user_id": str(owner_user_id),
            },
        )
        if row is None:
            return NotificationTelegramBindingStatus(
                organization_id=organization_id,
                provider_instance_id=provider_instance_id,
                owner_user_id=owner_user_id,
                is_confirmed=False,
                chat_id_ref=None,
                confirmed_at=None,
            )
        return _map_status(row)

    def find_owner_by_chat_ref(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        chat_id_ref: str,
    ) -> UserId | None:
        row = self._gateway.fetch_one(
            query="""
            SELECT owner_user_id
            FROM notification_telegram_recipient_bindings
            WHERE organization_id = %(organization_id)s
              AND provider_instance_id = %(provider_instance_id)s
              AND chat_id_ref = %(chat_id_ref)s
              AND status = 'confirmed'
            """,
            parameters={
                "organization_id": str(organization_id),
                "provider_instance_id": str(provider_instance_id),
                "chat_id_ref": chat_id_ref,
            },
        )
        return None if row is None else UserId.from_string(str(row["owner_user_id"]))


def _code_parameters(code: NotificationTelegramBindingCode) -> dict[str, object]:
    return {
        "binding_code_id": str(code.binding_code_id),
        "organization_id": str(code.organization_id),
        "provider_instance_id": str(code.provider_instance_id),
        "owner_user_id": str(code.owner_user_id),
        "code_hash": code.code_hash,
        "expires_at": code.expires_at,
        "created_at": code.created_at,
        "consumed_at": code.consumed_at,
    }


def _map_code(row: Mapping[str, Any]) -> NotificationTelegramBindingCode:
    return NotificationTelegramBindingCode(
        binding_code_id=UUID(str(row["binding_code_id"])),
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        provider_instance_id=UUID(str(row["provider_instance_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        code_hash=str(row["code_hash"]),
        expires_at=row["expires_at"],
        created_at=row["created_at"],
        consumed_at=row["consumed_at"],
    )


def _map_status(row: Mapping[str, Any]) -> NotificationTelegramBindingStatus:
    return NotificationTelegramBindingStatus(
        organization_id=OrganizationId.from_string(str(row["organization_id"])),
        provider_instance_id=UUID(str(row["provider_instance_id"])),
        owner_user_id=UserId.from_string(str(row["owner_user_id"])),
        is_confirmed=row["status"] == "confirmed",
        chat_id_ref=None if row["chat_id_ref"] is None else str(row["chat_id_ref"]),
        confirmed_at=row["confirmed_at"],
    )


def _require_row(row: Mapping[str, Any] | None, operation: str) -> Mapping[str, Any]:
    if row is None:
        raise ValueError(f"{operation} returned no row")
    return row
