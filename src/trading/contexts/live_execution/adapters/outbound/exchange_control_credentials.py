from __future__ import annotations

from uuid import UUID

from trading.contexts.exchange_control.application.connections import (
    ExchangeConnectionRepository,
    trading_capability_summary,
)
from trading.contexts.exchange_control.application.secret_cipher import (
    ExchangeCredentialCiphertext,
    ExchangeSecretCipher,
    ExchangeSecretCipherError,
)
from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionCredentialResolver,
    ExchangeExecutionCredentialUnavailable,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionConnection,
    ExchangeExecutionCredential,
)
from trading.shared_kernel.primitives import UserId


class ExchangeControlCredentialResolver(ExchangeExecutionCredentialResolver):
    def __init__(
        self,
        *,
        connection_repository: ExchangeConnectionRepository,
        secret_cipher: ExchangeSecretCipher,
    ) -> None:
        self._connection_repository = connection_repository
        self._secret_cipher = secret_cipher

    def resolve(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeExecutionConnection:
        connection = self._connection_repository.get(connection_id=exchange_connection_id)
        if connection is None or connection.owner_user_id != owner_user_id:
            raise ExchangeExecutionCredentialUnavailable(reason="exchange_connection_not_found")
        if connection.status != "active":
            raise ExchangeExecutionCredentialUnavailable(reason="exchange_connection_not_active")
        credential_version = self._connection_repository.get_active_credential(
            connection_id=exchange_connection_id
        )
        if credential_version is None or credential_version.status != "active":
            raise ExchangeExecutionCredentialUnavailable(reason="exchange_credential_not_active")
        try:
            api_key = self._secret_cipher.decrypt(
                ExchangeCredentialCiphertext(value=credential_version.api_key_ciphertext)
            ).value
            api_secret = self._secret_cipher.decrypt(
                ExchangeCredentialCiphertext(value=credential_version.api_secret_ciphertext)
            ).value
            passphrase = (
                self._secret_cipher.decrypt(
                    ExchangeCredentialCiphertext(value=credential_version.passphrase_ciphertext)
                ).value
                if credential_version.passphrase_ciphertext is not None
                else None
            )
        except (ValueError, ExchangeSecretCipherError) as exc:
            raise ExchangeExecutionCredentialUnavailable(
                reason="exchange_credential_decrypt_failed"
            ) from exc
        permission_summary = connection.permission_summary or {}
        capability_summary = trading_capability_summary(
            status=connection.status,
            validation_status=connection.validation_status,
            validation_reason=connection.validation_reason,
            ip_restriction_status=connection.ip_restriction_status,
            exchange_permissions=str(permission_summary.get("exchange_permissions") or "unknown"),
            status_reason=connection.status_reason,
        )
        return ExchangeExecutionConnection(
            connection_id=connection.connection_id,
            owner_user_id=connection.owner_user_id,
            exchange_name=connection.exchange_name,
            market_type=connection.market_type,
            environment=connection.environment,
            connection_readiness=str(capability_summary["connection_readiness"]),
            effective_capability=str(capability_summary["effective_capability"]),
            credential=ExchangeExecutionCredential(
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
            ),
        )
