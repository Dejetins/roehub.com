from __future__ import annotations

from uuid import UUID

from trading.contexts.notifications.adapters.outbound.persistence.postgres.gateway import (
    NotificationPostgresGateway,
)
from trading.platform.secrets import OpenBaoSecretResolver, SecretKind, SecretValue
from trading.shared_kernel.primitives import OrganizationId, UserId


class OpenBaoTelegramRecipientSecretStore:
    """Persist raw Telegram recipient ids only in the scoped OpenBao boundary."""

    def __init__(self, *, secret_resolver: OpenBaoSecretResolver) -> None:
        self._secret_resolver = secret_resolver

    def store_chat_id(
        self,
        *,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
        owner_user_id: UserId,
        binding_code_id: UUID,
        chat_id: SecretValue,
    ) -> str:
        secret_ref = (
            f"openbao://kv/roehub/telegram/recipients/{organization_id}/"
            f"{provider_instance_id}/{owner_user_id}/{binding_code_id}#chat_id"
        )
        self._secret_resolver.store(
            secret_ref,
            value=chat_id,
            expected_kind=SecretKind.TELEGRAM,
        )
        return secret_ref


class PostgresOpenBaoTelegramRecipientResolver:
    """Resolve a scoped opaque recipient ref only inside the provider boundary."""

    def __init__(
        self,
        *,
        gateway: NotificationPostgresGateway,
        secret_resolver: OpenBaoSecretResolver,
    ) -> None:
        self._gateway = gateway
        self._secret_resolver = secret_resolver

    def resolve(
        self,
        recipient_address_ref: str,
        organization_id: OrganizationId,
        provider_instance_id: UUID,
    ) -> str:
        row = self._gateway.fetch_one(
            query="""
            SELECT recipient_secret_ref
            FROM notification_telegram_recipient_bindings
            WHERE organization_id = %(organization_id)s
              AND provider_instance_id = %(provider_instance_id)s
              AND chat_id_ref = %(chat_id_ref)s
              AND status = 'confirmed'
            """,
            parameters={
                "organization_id": str(organization_id),
                "provider_instance_id": str(provider_instance_id),
                "chat_id_ref": recipient_address_ref,
            },
        )
        if row is None:
            raise ValueError("Telegram recipient binding is unavailable")
        return self._secret_resolver.resolve(
            str(row["recipient_secret_ref"]),
            expected_kind=SecretKind.TELEGRAM,
        ).reveal_text()
