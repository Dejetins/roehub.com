from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol
from uuid import UUID, uuid4

from trading.contexts.exchange_control.application.account_state import (
    ExchangeAccountStateReader,
    ExchangeAccountStateReadRequest,
    ExchangeAccountStateReadResult,
)
from trading.contexts.exchange_control.application.secret_cipher import (
    ExchangeCredentialCiphertext,
    ExchangeCredentialSecret,
    ExchangeSecretCipher,
    ExchangeSecretCipherError,
)
from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialPlaintext,
    ExchangeCredentialValidationRequest,
    ExchangeCredentialValidationResult,
    ExchangeCredentialValidator,
)
from trading.shared_kernel.primitives import UserId

ALLOWED_EXCHANGES = {"binance", "bybit"}
ALLOWED_MARKET_TYPES = {"spot", "futures"}
ALLOWED_ENVIRONMENTS = {"mainnet", "testnet"}
ALLOWED_PERMISSIONS = {"read", "trade"}
ALLOWED_EFFECTIVE_CAPABILITIES = {"none", "trading"}
ALLOWED_CONNECTION_READINESS = {
    "ready_for_trading",
    "needs_action",
    "rejected",
    "disconnected",
    "archived",
}
AUTO_VALIDATION_FAILED_STATUS_REASON = "auto_validation_failed"
RECLASSIFIED_NON_TRADING_STATUS_REASON = "reclassified_non_trading_ready"
VALIDATION_UNAVAILABLE_REASON = "validation_unavailable"


class ExchangeConnectionError(RuntimeError):
    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def payload(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class ExchangeConnectionView:
    connection_id: UUID
    credential_version_id: UUID
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    label: str | None
    permissions: str
    requested_permissions: str
    exchange_permissions: str
    effective_permissions: str
    requested_capability: str
    effective_capability: str
    connection_readiness: str
    connection_readiness_reason: str
    permissions_deprecated: bool
    permission_warnings: tuple[str, ...]
    api_key: str
    status: str
    status_reason: str | None
    validation_status: str
    validation_reason: str | None
    ip_restriction_status: str
    last_validated_at: datetime | None
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None = None
    archived_at: datetime | None = None
    used_by_strategies_count: int = 0
    active_strategy_bindings_count: int = 0


@dataclass(frozen=True, slots=True)
class ExchangeCredentialVersionRecord:
    credential_version_id: UUID
    connection_id: UUID
    api_key_ciphertext: str
    api_secret_ciphertext: str
    passphrase_ciphertext: str | None
    api_key_last4: str
    api_key_fingerprint_hmac: str
    secret_cipher: str
    transit_key_id: str
    credential_scheme: str
    status: str
    created_by_user_id: UserId
    created_at: datetime
    rotated_at: datetime | None = None
    disabled_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExchangeConnectionRecord:
    connection_id: UUID
    owner_user_id: UserId
    exchange_name: str
    market_type: str
    environment: str
    label: str | None
    permissions: str
    active_credential_version_id: UUID
    status: str
    status_reason: str | None
    validation_status: str
    validation_reason: str | None
    ip_restriction_status: str
    last_validated_at: datetime | None
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None = None
    archived_at: datetime | None = None
    permission_summary: dict[str, object] | None = None


class ExchangeConnectionRepository(Protocol):
    def create(
        self,
        *,
        connection: ExchangeConnectionRecord,
        credential_version: ExchangeCredentialVersionRecord,
    ) -> ExchangeConnectionRecord | None: ...

    def get(self, *, connection_id: UUID) -> ExchangeConnectionRecord | None: ...

    def list_for_user(self, *, owner_user_id: UserId) -> tuple[ExchangeConnectionRecord, ...]: ...

    def get_active_credential(
        self, *, connection_id: UUID
    ) -> ExchangeCredentialVersionRecord | None: ...

    def replace_active_credential(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        credential_version: ExchangeCredentialVersionRecord,
        updated_at: datetime,
    ) -> ExchangeConnectionRecord | None: ...

    def disable(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        disabled_at: datetime,
        status_reason: str = "user_disabled",
    ) -> ExchangeConnectionRecord | None: ...

    def archive(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        archived_at: datetime,
    ) -> ExchangeConnectionRecord | None: ...

    def record_validation(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        result: ExchangeCredentialValidationResult,
        updated_at: datetime,
    ) -> ExchangeConnectionRecord | None: ...


class ExchangeConnectionUsageGuard(Protocol):
    def active_trading_bindings_count(
        self, *, owner_user_id: UserId, connection_id: UUID
    ) -> int: ...


class AllowAllExchangeConnectionUsageGuard(ExchangeConnectionUsageGuard):
    def active_trading_bindings_count(
        self, *, owner_user_id: UserId, connection_id: UUID
    ) -> int:
        _ = owner_user_id, connection_id
        return 0


class InMemoryExchangeConnectionRepository(ExchangeConnectionRepository):
    def __init__(self) -> None:
        self._connections: dict[UUID, ExchangeConnectionRecord] = {}
        self._credential_versions: dict[UUID, ExchangeCredentialVersionRecord] = {}

    def create(
        self,
        *,
        connection: ExchangeConnectionRecord,
        credential_version: ExchangeCredentialVersionRecord,
    ) -> ExchangeConnectionRecord | None:
        for existing in self._connections.values():
            if existing.status != "active":
                continue
            if existing.owner_user_id != connection.owner_user_id:
                continue
            if existing.exchange_name != connection.exchange_name:
                continue
            if existing.market_type != connection.market_type:
                continue
            if existing.environment != connection.environment:
                continue
            existing_credential = self._credential_versions[
                existing.active_credential_version_id
            ]
            if (
                existing_credential.api_key_fingerprint_hmac
                == credential_version.api_key_fingerprint_hmac
            ):
                return None
        self._connections[connection.connection_id] = connection
        self._credential_versions[credential_version.credential_version_id] = (
            credential_version
        )
        return connection

    def get(self, *, connection_id: UUID) -> ExchangeConnectionRecord | None:
        return self._connections.get(connection_id)

    def list_for_user(self, *, owner_user_id: UserId) -> tuple[ExchangeConnectionRecord, ...]:
        rows = [
            connection
            for connection in self._connections.values()
            if connection.owner_user_id == owner_user_id
        ]
        rows.sort(key=lambda item: (item.created_at, str(item.connection_id)))
        return tuple(rows)

    def get_active_credential(
        self, *, connection_id: UUID
    ) -> ExchangeCredentialVersionRecord | None:
        connection = self._connections.get(connection_id)
        if connection is None:
            return None
        return self._credential_versions.get(connection.active_credential_version_id)

    def replace_active_credential(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        credential_version: ExchangeCredentialVersionRecord,
        updated_at: datetime,
    ) -> ExchangeConnectionRecord | None:
        connection = self._connections.get(connection_id)
        if connection is None or connection.owner_user_id != owner_user_id:
            return None
        if connection.status != "active":
            return None
        previous = self._credential_versions[connection.active_credential_version_id]
        self._credential_versions[previous.credential_version_id] = replace(
            previous,
            status="rotated",
            rotated_at=updated_at,
        )
        self._credential_versions[credential_version.credential_version_id] = (
            credential_version
        )
        updated = replace(
            connection,
            active_credential_version_id=credential_version.credential_version_id,
            validation_status="skipped_external_validation",
            validation_reason="credential_rotated",
            ip_restriction_status="unknown",
            last_validated_at=None,
            updated_at=updated_at,
            permission_summary=_initial_permission_summary(
                requested_permissions=connection.permissions,
                validation_status="skipped_external_validation",
                validation_reason="credential_rotated",
            ),
        )
        self._connections[connection_id] = updated
        return updated

    def disable(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        disabled_at: datetime,
        status_reason: str = "user_disabled",
    ) -> ExchangeConnectionRecord | None:
        connection = self._connections.get(connection_id)
        if connection is None or connection.owner_user_id != owner_user_id:
            return None
        if connection.status != "active":
            return None
        credential = self._credential_versions[connection.active_credential_version_id]
        self._credential_versions[credential.credential_version_id] = replace(
            credential,
            status="disabled",
            disabled_at=disabled_at,
        )
        disabled = replace(
            connection,
            status="disabled",
            status_reason=status_reason,
            updated_at=disabled_at,
            disabled_at=disabled_at,
            permission_summary={
                **(connection.permission_summary or {}),
                **trading_capability_summary(
                    status="disabled",
                    status_reason=status_reason,
                    validation_status=connection.validation_status,
                    validation_reason=connection.validation_reason,
                    ip_restriction_status=connection.ip_restriction_status,
                    exchange_permissions=_summary_string(
                        summary=connection.permission_summary or {},
                        key="exchange_permissions",
                        default="unknown",
                        allowed={"unknown", "read", "trade", "withdraw_or_transfer"},
                    ),
                ),
            },
        )
        self._connections[connection_id] = disabled
        return disabled

    def archive(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        archived_at: datetime,
    ) -> ExchangeConnectionRecord | None:
        connection = self._connections.get(connection_id)
        if connection is None or connection.owner_user_id != owner_user_id:
            return None
        if connection.status == "archived":
            return connection
        if connection.status != "disabled" or connection.disabled_at is None:
            return None
        archived = replace(
            connection,
            status="archived",
            status_reason="user_archived",
            updated_at=archived_at,
            archived_at=archived_at,
            permission_summary={
                **(connection.permission_summary or {}),
                **trading_capability_summary(
                    status="archived",
                    validation_status=connection.validation_status,
                    validation_reason=connection.validation_reason,
                    ip_restriction_status=connection.ip_restriction_status,
                    exchange_permissions=_summary_string(
                        summary=connection.permission_summary or {},
                        key="exchange_permissions",
                        default="unknown",
                        allowed={"unknown", "read", "trade", "withdraw_or_transfer"},
                    ),
                ),
            },
        )
        self._connections[connection_id] = archived
        return archived

    def record_validation(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        result: ExchangeCredentialValidationResult,
        updated_at: datetime,
    ) -> ExchangeConnectionRecord | None:
        connection = self._connections.get(connection_id)
        if connection is None or connection.owner_user_id != owner_user_id:
            return None
        if connection.status != "active":
            return None
        merged_summary = {
            **(connection.permission_summary or {}),
            **(result.permission_summary or {}),
            "validation_status": result.status,
            "validation_reason": result.reason,
        }
        updated = replace(
            connection,
            validation_status=result.status,
            validation_reason=result.reason,
            ip_restriction_status=result.ip_restriction_status,
            last_validated_at=result.observed_at or updated_at,
            updated_at=updated_at,
            permission_summary={
                **merged_summary,
                **trading_capability_summary(
                    status=connection.status,
                    validation_status=result.status,
                    validation_reason=result.reason,
                    ip_restriction_status=result.ip_restriction_status,
                    exchange_permissions=_summary_string(
                        summary=merged_summary,
                        key="exchange_permissions",
                        default="unknown",
                        allowed={"unknown", "read", "trade", "withdraw_or_transfer"},
                    ),
                ),
            },
        )
        self._connections[connection_id] = updated
        return updated


class ExchangeConnectionService:
    def __init__(
        self,
        *,
        repository: ExchangeConnectionRepository,
        secret_cipher: ExchangeSecretCipher,
        usage_guard: ExchangeConnectionUsageGuard | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeConnectionService requires repository")
        if secret_cipher is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeConnectionService requires secret_cipher")
        self._repository = repository
        self._secret_cipher = secret_cipher
        self._usage_guard = usage_guard or AllowAllExchangeConnectionUsageGuard()

    def create_connection(
        self,
        *,
        owner_user_id: UserId,
        exchange_name: str,
        market_type: str,
        environment: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
        now: datetime,
    ) -> ExchangeConnectionView:
        normalized = _NormalizedConnectionInput.from_raw(
            exchange_name=exchange_name,
            market_type=market_type,
            environment=environment,
            label=label,
            permissions=permissions,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
        )
        connection_id = uuid4()
        credential_version = self._build_credential_version(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            normalized=normalized,
            now=now,
        )
        connection = ExchangeConnectionRecord(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            exchange_name=normalized.exchange_name,
            market_type=normalized.market_type,
            environment=normalized.environment,
            label=normalized.label,
            permissions=normalized.permissions,
            active_credential_version_id=credential_version.credential_version_id,
            status="active",
            status_reason=None,
            validation_status="skipped_external_validation",
            validation_reason="not_validated",
            ip_restriction_status="unknown",
            last_validated_at=None,
            created_at=now,
            updated_at=now,
            permission_summary=_initial_permission_summary(
                requested_permissions=normalized.permissions,
                validation_status="skipped_external_validation",
                validation_reason="not_validated",
            ),
        )
        created = self._repository.create(
            connection=connection,
            credential_version=credential_version,
        )
        if created is None:
            raise ExchangeConnectionError(
                code="exchange_connection_already_exists",
                message="Exchange connection already exists.",
                status_code=409,
            )
        return self._to_view(connection=created)

    def create_connection_with_validation(
        self,
        *,
        owner_user_id: UserId,
        exchange_name: str,
        market_type: str,
        environment: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
        validator: ExchangeCredentialValidator,
        now: datetime,
    ) -> ExchangeConnectionView:
        normalized = _NormalizedConnectionInput.from_raw(
            exchange_name=exchange_name,
            market_type=market_type,
            environment=environment,
            label=label,
            permissions=permissions,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
        )
        result = self._validate_plaintext(
            exchange_name=normalized.exchange_name,
            market_type=normalized.market_type,
            environment=normalized.environment,
            requested_permissions=normalized.permissions,
            normalized=_NormalizedCredentialInput(
                api_key=normalized.api_key,
                api_secret=normalized.api_secret,
                passphrase=normalized.passphrase,
            ),
            validator=validator,
            now=now,
        )
        permission_summary = _validated_permission_summary(
            requested_permissions=normalized.permissions,
            result=result,
            status="active",
            status_reason=None,
            auto_validation=True,
        )
        is_ready = _is_trading_ready_summary(summary=permission_summary)
        status = "active" if is_ready else "disabled"
        status_reason = None if is_ready else AUTO_VALIDATION_FAILED_STATUS_REASON
        connection_id = uuid4()
        credential_version = self._build_credential_version(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            normalized=normalized,
            now=now,
            status=status,
            disabled_at=now if status == "disabled" else None,
        )
        if not is_ready:
            permission_summary = _validated_permission_summary(
                requested_permissions=normalized.permissions,
                result=result,
                status=status,
                status_reason=status_reason,
                auto_validation=True,
            )
        connection = ExchangeConnectionRecord(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            exchange_name=normalized.exchange_name,
            market_type=normalized.market_type,
            environment=normalized.environment,
            label=normalized.label,
            permissions=normalized.permissions,
            active_credential_version_id=credential_version.credential_version_id,
            status=status,
            status_reason=status_reason,
            validation_status=result.status,
            validation_reason=result.reason,
            ip_restriction_status=result.ip_restriction_status,
            last_validated_at=result.observed_at or now,
            created_at=now,
            updated_at=now,
            disabled_at=now if status == "disabled" else None,
            permission_summary=permission_summary,
        )
        created = self._repository.create(
            connection=connection,
            credential_version=credential_version,
        )
        if created is None:
            raise ExchangeConnectionError(
                code="exchange_connection_already_exists",
                message="Exchange connection already exists.",
                status_code=409,
            )
        return self._to_view(connection=created)

    def list_connections(self, *, owner_user_id: UserId) -> tuple[ExchangeConnectionView, ...]:
        return tuple(
            self._to_view(connection=connection)
            for connection in self._repository.list_for_user(owner_user_id=owner_user_id)
        )

    def rotate_connection(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
        now: datetime,
    ) -> ExchangeConnectionView:
        connection = self._require_active_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        normalized = _NormalizedCredentialInput.from_raw(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
        )
        credential_version = self._build_credential_version(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            normalized=_NormalizedConnectionInput(
                exchange_name=connection.exchange_name,
                market_type=connection.market_type,
                environment=connection.environment,
                label=connection.label,
                permissions=connection.permissions,
                api_key=normalized.api_key,
                api_secret=normalized.api_secret,
                passphrase=normalized.passphrase,
            ),
            now=now,
        )
        updated = self._repository.replace_active_credential(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            credential_version=credential_version,
            updated_at=now,
        )
        if updated is None:
            raise _not_found()
        return self._to_view(connection=updated)

    def rotate_connection_with_validation(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
        validator: ExchangeCredentialValidator,
        now: datetime,
    ) -> ExchangeConnectionView:
        connection = self._require_active_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        normalized = _NormalizedCredentialInput.from_raw(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
        )
        result = self._validate_plaintext(
            exchange_name=connection.exchange_name,
            market_type=connection.market_type,
            environment=connection.environment,
            requested_permissions=connection.permissions,
            normalized=normalized,
            validator=validator,
            now=now,
        )
        readiness = trading_capability_summary(
            status="active",
            validation_status=result.status,
            validation_reason=result.reason,
            ip_restriction_status=result.ip_restriction_status,
            exchange_permissions=_summary_string(
                summary=result.permission_summary or {},
                key="exchange_permissions",
                default="unknown",
                allowed={"unknown", "read", "trade", "withdraw_or_transfer"},
            ),
            auto_validation=True,
        )
        if (
            readiness["effective_capability"] != "trading"
            or readiness["connection_readiness"] != "ready_for_trading"
        ):
            reason = str(readiness["connection_readiness_reason"])
            raise ExchangeConnectionError(
                code=reason,
                message="Exchange credential rotation failed validation.",
                status_code=422 if reason != VALIDATION_UNAVAILABLE_REASON else 503,
            )
        credential_version = self._build_credential_version(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            normalized=_NormalizedConnectionInput(
                exchange_name=connection.exchange_name,
                market_type=connection.market_type,
                environment=connection.environment,
                label=connection.label,
                permissions=connection.permissions,
                api_key=normalized.api_key,
                api_secret=normalized.api_secret,
                passphrase=normalized.passphrase,
            ),
            now=now,
        )
        updated = self._repository.replace_active_credential(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            credential_version=credential_version,
            updated_at=now,
        )
        if updated is None:
            raise _not_found()
        recorded = self._repository.record_validation(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            result=result,
            updated_at=now,
        )
        if recorded is None:
            raise _not_found()
        return self._to_view(connection=recorded)

    def disable_connection(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        now: datetime,
        status_reason: str = "user_disabled",
    ) -> ExchangeConnectionView:
        self._require_active_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        self._assert_not_in_use(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            action="disconnect",
        )
        disabled = self._repository.disable(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            disabled_at=now,
            status_reason=status_reason,
        )
        if disabled is None:
            raise _not_found()
        return self._to_view(connection=disabled)

    def reclassify_non_trading_active_connection(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        now: datetime,
    ) -> ExchangeConnectionView:
        connection = self._require_existing_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        if (
            connection.status == "disabled"
            and connection.status_reason == RECLASSIFIED_NON_TRADING_STATUS_REASON
        ):
            return self._to_view(connection=connection)
        if connection.status != "active":
            raise _not_found()
        view = self._to_view(connection=connection)
        if (
            view.effective_capability == "trading"
            and view.connection_readiness == "ready_for_trading"
        ):
            raise ExchangeConnectionError(
                code="exchange_connection_trading_ready",
                message="Trading-ready active exchange connection cannot be reclassified.",
                status_code=409,
            )
        return self.disable_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
            now=now,
            status_reason=RECLASSIFIED_NON_TRADING_STATUS_REASON,
        )

    def archive_connection(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        now: datetime,
    ) -> ExchangeConnectionView:
        connection = self._require_existing_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        if connection.status != "archived":
            self._assert_not_in_use(
                owner_user_id=owner_user_id,
                connection_id=connection_id,
                action="archive",
            )
        if connection.status == "active":
            raise ExchangeConnectionError(
                code="exchange_connection_not_disabled",
                message="Exchange connection must be disabled before archive.",
                status_code=409,
            )
        if connection.status not in {"disabled", "archived"}:
            raise _not_found()
        archived = self._repository.archive(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            archived_at=now,
        )
        if archived is None:
            raise _not_found()
        return self._to_view(connection=archived)

    def validate_connection(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        validator: ExchangeCredentialValidator,
        now: datetime,
    ) -> ExchangeConnectionView:
        connection = self._require_active_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        if bool(getattr(validator, "requires_plaintext", True)):
            credential = self._repository.get_active_credential(connection_id=connection_id)
            if credential is None:
                raise _not_found()
            try:
                plaintext = ExchangeCredentialPlaintext(
                    api_key=self._secret_cipher.decrypt(
                        ExchangeCredentialCiphertext(value=credential.api_key_ciphertext)
                    ).value,
                    api_secret=self._secret_cipher.decrypt(
                        ExchangeCredentialCiphertext(value=credential.api_secret_ciphertext)
                    ).value,
                    passphrase=(
                        self._secret_cipher.decrypt(
                            ExchangeCredentialCiphertext(
                                value=credential.passphrase_ciphertext
                            )
                        ).value
                        if credential.passphrase_ciphertext is not None
                        else None
                    ),
                )
            except ExchangeSecretCipherError as exc:
                raise ExchangeConnectionError(
                    code="exchange_connection_validation_unavailable",
                    message="Exchange connection validation is unavailable.",
                    status_code=503,
                ) from exc
        else:
            plaintext = ExchangeCredentialPlaintext(
                api_key="skipped_external_validation",
                api_secret="skipped_external_validation",
            )
        result = validator.validate(
            request=ExchangeCredentialValidationRequest(
                exchange_name=connection.exchange_name,
                market_type=connection.market_type,
                environment=connection.environment,
                requested_permissions=connection.permissions,
                credential=plaintext,
            ),
            now=now,
        )
        updated = self._repository.record_validation(
            connection_id=connection_id,
            owner_user_id=owner_user_id,
            result=result,
            updated_at=now,
        )
        if updated is None:
            raise _not_found()
        view = self._to_view(connection=updated)
        if (
            view.effective_capability != "trading"
            or view.connection_readiness != "ready_for_trading"
        ):
            return self.disable_connection(
                owner_user_id=owner_user_id,
                connection_id=connection_id,
                now=now,
                status_reason=RECLASSIFIED_NON_TRADING_STATUS_REASON,
            )
        return view

    def read_account_state(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        reader: ExchangeAccountStateReader,
        instrument_keys: tuple[str, ...],
        now: datetime,
    ) -> ExchangeAccountStateReadResult:
        connection = self._require_active_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        view = self._to_view(connection=connection)
        if (
            view.effective_capability != "trading"
            or view.connection_readiness != "ready_for_trading"
        ):
            raise ExchangeConnectionError(
                code=view.connection_readiness_reason or "exchange_connection_not_ready",
                message="Exchange connection is not ready for account-state reads.",
                status_code=422,
            )
        if bool(getattr(reader, "requires_plaintext", True)):
            plaintext = self._decrypt_active_credential(connection_id=connection_id)
        else:
            plaintext = ExchangeCredentialPlaintext(
                api_key="account_state_sync_disabled",
                api_secret="account_state_sync_disabled",
            )
        return reader.read_account_state(
            request=ExchangeAccountStateReadRequest(
                exchange_name=connection.exchange_name,
                market_type=connection.market_type,
                environment=connection.environment,
                credential=plaintext,
                instrument_keys=instrument_keys,
            ),
            now=now,
        )

    def _require_active_owned_connection(
        self, *, owner_user_id: UserId, connection_id: UUID
    ) -> ExchangeConnectionRecord:
        connection = self._require_existing_owned_connection(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        if connection.status != "active":
            raise _not_found()
        return connection

    def _require_existing_owned_connection(
        self, *, owner_user_id: UserId, connection_id: UUID
    ) -> ExchangeConnectionRecord:
        connection = self._repository.get(connection_id=connection_id)
        if connection is None:
            raise _not_found()
        if connection.owner_user_id != owner_user_id:
            raise ExchangeConnectionError(
                code="exchange_connection_not_owned",
                message="Exchange connection is not owned by current user.",
                status_code=404,
            )
        return connection

    def _assert_not_in_use(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
        action: str,
    ) -> None:
        active_bindings_count = self._active_strategy_bindings_count(
            owner_user_id=owner_user_id,
            connection_id=connection_id,
        )
        if active_bindings_count <= 0:
            return
        raise ExchangeConnectionError(
            code="exchange_connection_in_use",
            message=(
                f"Cannot {action}. This exchange account is used by "
                f"{active_bindings_count} active strategies. Pause or reassign "
                "strategies first."
            ),
            status_code=409,
        )

    def _active_strategy_bindings_count(
        self,
        *,
        owner_user_id: UserId,
        connection_id: UUID,
    ) -> int:
        try:
            count = self._usage_guard.active_trading_bindings_count(
                owner_user_id=owner_user_id,
                connection_id=connection_id,
            )
        except Exception as exc:
            raise ExchangeConnectionError(
                code="exchange_connection_usage_guard_unavailable",
                message="Exchange connection usage guard is unavailable.",
                status_code=503,
            ) from exc
        return max(0, int(count))

    def _decrypt_active_credential(
        self, *, connection_id: UUID
    ) -> ExchangeCredentialPlaintext:
        credential = self._repository.get_active_credential(connection_id=connection_id)
        if credential is None:
            raise _not_found()
        try:
            return ExchangeCredentialPlaintext(
                api_key=self._secret_cipher.decrypt(
                    ExchangeCredentialCiphertext(value=credential.api_key_ciphertext)
                ).value,
                api_secret=self._secret_cipher.decrypt(
                    ExchangeCredentialCiphertext(value=credential.api_secret_ciphertext)
                ).value,
                passphrase=(
                    self._secret_cipher.decrypt(
                        ExchangeCredentialCiphertext(
                            value=credential.passphrase_ciphertext
                        )
                    ).value
                    if credential.passphrase_ciphertext is not None
                    else None
                ),
            )
        except ExchangeSecretCipherError as exc:
            raise ExchangeConnectionError(
                code="exchange_connection_account_state_unavailable",
                message="Exchange account-state read is unavailable.",
                status_code=503,
            ) from exc

    def _build_credential_version(
        self,
        *,
        connection_id: UUID,
        owner_user_id: UserId,
        normalized: "_NormalizedConnectionInput",
        now: datetime,
        status: str = "active",
        disabled_at: datetime | None = None,
    ) -> ExchangeCredentialVersionRecord:
        api_key_secret = ExchangeCredentialSecret(value=normalized.api_key)
        api_secret_secret = ExchangeCredentialSecret(value=normalized.api_secret)
        passphrase_secret = (
            ExchangeCredentialSecret(value=normalized.passphrase)
            if normalized.passphrase is not None
            else None
        )
        return ExchangeCredentialVersionRecord(
            credential_version_id=uuid4(),
            connection_id=connection_id,
            api_key_ciphertext=self._secret_cipher.encrypt(api_key_secret).value,
            api_secret_ciphertext=self._secret_cipher.encrypt(api_secret_secret).value,
            passphrase_ciphertext=(
                self._secret_cipher.encrypt(passphrase_secret).value
                if passphrase_secret is not None
                else None
            ),
            api_key_last4=_last4(normalized.api_key),
            api_key_fingerprint_hmac=self._secret_cipher.fingerprint(api_key_secret).value,
            secret_cipher="exchange_control_transit_v1",
            transit_key_id="roehub-exchange-credentials",
            credential_scheme="api_key_secret_v1",
            status=status,
            created_by_user_id=owner_user_id,
            created_at=now,
            disabled_at=disabled_at,
        )

    def _validate_plaintext(
        self,
        *,
        exchange_name: str,
        market_type: str,
        environment: str,
        requested_permissions: str,
        normalized: "_NormalizedCredentialInput",
        validator: ExchangeCredentialValidator,
        now: datetime,
    ) -> ExchangeCredentialValidationResult:
        return validator.validate(
            request=ExchangeCredentialValidationRequest(
                exchange_name=exchange_name,
                market_type=market_type,
                environment=environment,
                requested_permissions=requested_permissions,
                credential=ExchangeCredentialPlaintext(
                    api_key=normalized.api_key,
                    api_secret=normalized.api_secret,
                    passphrase=normalized.passphrase,
                ),
            ),
            now=now,
        )

    def _to_view(self, *, connection: ExchangeConnectionRecord) -> ExchangeConnectionView:
        credential = self._repository.get_active_credential(
            connection_id=connection.connection_id
        )
        if credential is None:
            raise RuntimeError("active exchange credential version is missing")
        permission_summary = connection.permission_summary or {}
        requested_permissions = _summary_string(
            summary=permission_summary,
            key="requested_permissions",
            default=connection.permissions,
            allowed=ALLOWED_PERMISSIONS,
        )
        exchange_permissions = _summary_string(
            summary=permission_summary,
            key="exchange_permissions",
            default="unknown",
            allowed={"unknown", "read", "trade", "withdraw_or_transfer"},
        )
        effective_permissions = _summary_string(
            summary=permission_summary,
            key="effective_permissions",
            default="none",
            allowed={"none", "read", "trade"},
        )
        capability_summary = trading_capability_summary(
            status=connection.status,
            validation_status=connection.validation_status,
            validation_reason=connection.validation_reason,
            ip_restriction_status=connection.ip_restriction_status,
            exchange_permissions=exchange_permissions,
            status_reason=connection.status_reason,
            auto_validation=(
                connection.status_reason == AUTO_VALIDATION_FAILED_STATUS_REASON
            ),
        )
        active_bindings_count = self._active_strategy_bindings_count(
            owner_user_id=connection.owner_user_id,
            connection_id=connection.connection_id,
        )
        return ExchangeConnectionView(
            connection_id=connection.connection_id,
            credential_version_id=connection.active_credential_version_id,
            owner_user_id=connection.owner_user_id,
            exchange_name=connection.exchange_name,
            market_type=connection.market_type,
            environment=connection.environment,
            label=connection.label,
            permissions=connection.permissions,
            requested_permissions=requested_permissions,
            exchange_permissions=exchange_permissions,
            effective_permissions=effective_permissions,
            requested_capability=_summary_string(
                summary={**permission_summary, **capability_summary},
                key="requested_capability",
                default="trading",
                allowed={"trading"},
            ),
            effective_capability=_summary_string(
                summary={**permission_summary, **capability_summary},
                key="effective_capability",
                default=str(capability_summary["effective_capability"]),
                allowed=ALLOWED_EFFECTIVE_CAPABILITIES,
            ),
            connection_readiness=_summary_string(
                summary={**permission_summary, **capability_summary},
                key="connection_readiness",
                default=str(capability_summary["connection_readiness"]),
                allowed=ALLOWED_CONNECTION_READINESS,
            ),
            connection_readiness_reason=_summary_string_unbounded(
                summary={**permission_summary, **capability_summary},
                key="connection_readiness_reason",
                default=str(capability_summary["connection_readiness_reason"]),
            ),
            permissions_deprecated=True,
            permission_warnings=_summary_warnings(summary=permission_summary),
            api_key=f"****{credential.api_key_last4}",
            status=connection.status,
            status_reason=connection.status_reason,
            validation_status=connection.validation_status,
            validation_reason=connection.validation_reason,
            ip_restriction_status=connection.ip_restriction_status,
            last_validated_at=connection.last_validated_at,
            created_at=connection.created_at,
            updated_at=connection.updated_at,
            disabled_at=connection.disabled_at,
            archived_at=connection.archived_at,
            used_by_strategies_count=active_bindings_count,
            active_strategy_bindings_count=active_bindings_count,
        )


@dataclass(frozen=True, slots=True)
class _NormalizedCredentialInput:
    api_key: str
    api_secret: str
    passphrase: str | None

    @classmethod
    def from_raw(
        cls,
        *,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
    ) -> "_NormalizedCredentialInput":
        return cls(
            api_key=_required_secret(value=api_key, field_name="api_key"),
            api_secret=_required_secret(value=api_secret, field_name="api_secret"),
            passphrase=_optional_secret(value=passphrase),
        )


@dataclass(frozen=True, slots=True)
class _NormalizedConnectionInput:
    exchange_name: str
    market_type: str
    environment: str
    label: str | None
    permissions: str
    api_key: str
    api_secret: str
    passphrase: str | None

    @classmethod
    def from_raw(
        cls,
        *,
        exchange_name: str,
        market_type: str,
        environment: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
    ) -> "_NormalizedConnectionInput":
        credential = _NormalizedCredentialInput.from_raw(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
        )
        return cls(
            exchange_name=_enum(
                value=exchange_name,
                allowed=ALLOWED_EXCHANGES,
                field_name="exchange_name",
            ),
            market_type=_enum(
                value=market_type,
                allowed=ALLOWED_MARKET_TYPES,
                field_name="market_type",
            ),
            environment=_enum(
                value=environment,
                allowed=ALLOWED_ENVIRONMENTS,
                field_name="environment",
            ),
            label=_optional_label(value=label),
            permissions=_enum(
                value=permissions,
                allowed=ALLOWED_PERMISSIONS,
                field_name="permissions",
            ),
            api_key=credential.api_key,
            api_secret=credential.api_secret,
            passphrase=credential.passphrase,
        )


def _enum(*, value: str, allowed: set[str], field_name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ExchangeConnectionError(
            code="exchange_connection_invalid",
            message=f"{field_name} is unsupported.",
            status_code=422,
        )
    return normalized


def _optional_label(*, value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > 80:
        raise ExchangeConnectionError(
            code="exchange_connection_invalid",
            message="label is too long.",
            status_code=422,
        )
    return stripped


def _required_secret(*, value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ExchangeConnectionError(
            code="exchange_connection_invalid",
            message=f"{field_name} must be non-empty.",
            status_code=422,
        )
    return stripped


def _initial_permission_summary(
    *,
    requested_permissions: str,
    validation_status: str,
    validation_reason: str,
) -> dict[str, object]:
    return {
        "permissions": requested_permissions,
        "requested_permissions": requested_permissions,
        "exchange_permissions": "unknown",
        "effective_permissions": "none",
        "permission_warnings": [],
        "validation_status": validation_status,
        "validation_reason": validation_reason,
        **trading_capability_summary(
            status="active",
            validation_status=validation_status,
            validation_reason=validation_reason,
            ip_restriction_status="unknown",
            exchange_permissions="unknown",
        ),
    }


def _validated_permission_summary(
    *,
    requested_permissions: str,
    result: ExchangeCredentialValidationResult,
    status: str,
    status_reason: str | None,
    auto_validation: bool,
) -> dict[str, object]:
    summary = {
        "permissions": requested_permissions,
        "requested_permissions": requested_permissions,
        "exchange_permissions": "unknown",
        "effective_permissions": "none",
        "permission_warnings": [],
        **(result.permission_summary or {}),
        "validation_status": result.status,
        "validation_reason": result.reason,
    }
    return {
        **summary,
        **trading_capability_summary(
            status=status,
            validation_status=result.status,
            validation_reason=result.reason,
            ip_restriction_status=result.ip_restriction_status,
            exchange_permissions=_summary_string(
                summary=summary,
                key="exchange_permissions",
                default="unknown",
                allowed={"unknown", "read", "trade", "withdraw_or_transfer"},
            ),
            status_reason=status_reason,
            auto_validation=auto_validation,
        ),
    }


def _is_trading_ready_summary(*, summary: dict[str, object]) -> bool:
    return (
        summary.get("effective_capability") == "trading"
        and summary.get("connection_readiness") == "ready_for_trading"
    )


def trading_capability_summary(
    *,
    status: str,
    validation_status: str,
    validation_reason: str | None,
    ip_restriction_status: str,
    exchange_permissions: str,
    status_reason: str | None = None,
    auto_validation: bool = False,
) -> dict[str, object]:
    effective_capability, readiness, readiness_reason = _resolve_trading_readiness(
        status=status,
        status_reason=status_reason,
        validation_status=validation_status,
        validation_reason=validation_reason,
        ip_restriction_status=ip_restriction_status,
        exchange_permissions=exchange_permissions,
        auto_validation=auto_validation,
    )
    return {
        "requested_capability": "trading",
        "effective_capability": effective_capability,
        "connection_readiness": readiness,
        "connection_readiness_reason": readiness_reason,
        "permissions_deprecated": True,
    }


def _resolve_trading_readiness(
    *,
    status: str,
    status_reason: str | None,
    validation_status: str,
    validation_reason: str | None,
    ip_restriction_status: str,
    exchange_permissions: str,
    auto_validation: bool,
) -> tuple[str, str, str]:
    if status == "archived":
        return "none", "archived", "archived"
    if status == "disabled" and status_reason not in {
        AUTO_VALIDATION_FAILED_STATUS_REASON,
        RECLASSIFIED_NON_TRADING_STATUS_REASON,
    }:
        return "none", "disconnected", "user_disconnected"
    if (
        validation_status == "valid_trade_enabled"
        and exchange_permissions == "trade"
        and ip_restriction_status != "missing_mainnet_restriction"
    ):
        return "trading", "ready_for_trading", "trading_policy_ok"
    if validation_status in {"valid_readonly", "permission_mismatch"}:
        return "none", "rejected", "read_only_not_supported"
    if exchange_permissions == "withdraw_or_transfer" or validation_reason in {
        "withdraw_or_transfer_enabled",
        "transfer_permission_enabled",
    }:
        return "none", "rejected", "unsafe_permissions"
    if (
        validation_status == "invalid_ip_restriction"
        or ip_restriction_status == "missing_mainnet_restriction"
    ):
        return "none", "needs_action", "ip_restriction_required"
    if validation_status == "invalid_credentials":
        return "none", "rejected", "invalid_credentials"
    if validation_status == "invalid_permissions":
        return "none", "rejected", "invalid_permissions"
    if validation_status == "unsupported_account_mode":
        return "none", "rejected", "unsupported_account_mode"
    if auto_validation and validation_status == "skipped_external_validation":
        return "none", "needs_action", VALIDATION_UNAVAILABLE_REASON
    return "none", "needs_action", "validation_required"


def _summary_string(
    *,
    summary: dict[str, object],
    key: str,
    default: str,
    allowed: set[str],
) -> str:
    value = summary.get(key)
    if isinstance(value, str) and value in allowed:
        return value
    if key == "requested_permissions":
        alias = summary.get("permissions")
        if isinstance(alias, str) and alias in allowed:
            return alias
    return default


def _summary_string_unbounded(
    *,
    summary: dict[str, object],
    key: str,
    default: str,
) -> str:
    value = summary.get(key)
    if isinstance(value, str) and value:
        return value
    return default


def _summary_warnings(*, summary: dict[str, object]) -> tuple[str, ...]:
    warnings = summary.get("permission_warnings")
    if not isinstance(warnings, list):
        return ()
    return tuple(
        warning
        for warning in warnings
        if isinstance(warning, str)
        and warning in {"exchange_permissions_exceed_requested"}
    )


def _optional_secret(*, value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _last4(value: str) -> str:
    return value[-4:] if len(value) >= 4 else value


def _not_found() -> ExchangeConnectionError:
    return ExchangeConnectionError(
        code="exchange_connection_not_found",
        message="Exchange connection was not found.",
        status_code=404,
    )


def stable_legacy_connection_id(*, key_id: UUID) -> UUID:
    return key_id


def stable_legacy_credential_version_id(*, key_id: UUID) -> UUID:
    digest = hashlib.md5(f"credential:{key_id}".encode("ascii"), usedforsecurity=False)
    return UUID(digest.hexdigest())


__all__ = [
    "ExchangeConnectionError",
    "ExchangeConnectionRepository",
    "ExchangeConnectionService",
    "ExchangeConnectionView",
    "InMemoryExchangeConnectionRepository",
    "stable_legacy_connection_id",
    "stable_legacy_credential_version_id",
]
