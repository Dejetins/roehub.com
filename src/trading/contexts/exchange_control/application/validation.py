from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

VALIDATION_STATUSES = {
    "valid_readonly",
    "valid_trade_enabled",
    "permission_mismatch",
    "invalid_credentials",
    "invalid_permissions",
    "invalid_ip_restriction",
    "unsupported_account_mode",
    "skipped_external_validation",
}

ExchangeValidationStatus = Literal[
    "valid_readonly",
    "valid_trade_enabled",
    "permission_mismatch",
    "invalid_credentials",
    "invalid_permissions",
    "invalid_ip_restriction",
    "unsupported_account_mode",
    "skipped_external_validation",
]

RequestedPermissions = Literal["read", "trade"]
ExchangePermissions = Literal["unknown", "read", "trade", "withdraw_or_transfer"]
EffectivePermissions = Literal["none", "read", "trade"]
PermissionWarning = Literal["exchange_permissions_exceed_requested"]


@dataclass(frozen=True, slots=True, repr=False)
class ExchangeCredentialPlaintext:
    api_key: str
    api_secret: str
    passphrase: str | None = None

    def __repr__(self) -> str:
        return "ExchangeCredentialPlaintext(<redacted>)"


@dataclass(frozen=True, slots=True)
class ExchangeCredentialValidationRequest:
    exchange_name: str
    market_type: str
    environment: str
    requested_permissions: str
    credential: ExchangeCredentialPlaintext


@dataclass(frozen=True, slots=True)
class ExchangeCredentialValidationResult:
    status: ExchangeValidationStatus
    reason: str
    ip_restriction_status: str
    account_mode: str | None = None
    permission_summary: dict[str, object] | None = None
    observed_at: datetime | None = None


@runtime_checkable
class ExchangeCredentialValidator(Protocol):
    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
        now: datetime,
    ) -> ExchangeCredentialValidationResult: ...


@dataclass(frozen=True)
class SkippedExchangeCredentialValidator:
    reason: str = "live_validation_disabled"
    requires_plaintext: bool = False

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
        now: datetime,
    ) -> ExchangeCredentialValidationResult:
        return ExchangeCredentialValidationResult(
            status="skipped_external_validation",
            reason=self.reason,
            ip_restriction_status="not_checked",
            permission_summary={
                "validation_live": False,
                "requested_permissions": request.requested_permissions,
                "permissions": request.requested_permissions,
                "exchange_permissions": "unknown",
                "effective_permissions": "none",
                "permission_warnings": [],
            },
            observed_at=now,
        )


__all__ = [
    "VALIDATION_STATUSES",
    "ExchangeCredentialPlaintext",
    "ExchangeCredentialValidationRequest",
    "ExchangeCredentialValidationResult",
    "ExchangeCredentialValidator",
    "EffectivePermissions",
    "ExchangePermissions",
    "ExchangeValidationStatus",
    "PermissionWarning",
    "RequestedPermissions",
    "SkippedExchangeCredentialValidator",
]
