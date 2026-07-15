"""Stable provider package and provider-instance contract for notifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping
from uuid import UUID

from trading.platform.secrets import SecretKind, SecretReference, SecretReferenceError
from trading.shared_kernel.primitives import OrganizationId

NOTIFICATION_PROVIDER_CONTRACT = "NotificationProvider/v1"
TELEGRAM_BOT_PROVIDER_KEY = "telegram_bot_api"

NotificationProviderScope = Literal["installation", "organization"]
NotificationProviderInstanceStatus = Literal["active", "disabled", "degraded"]
NotificationProviderHealthStatus = Literal["ready", "degraded", "disabled"]

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9._-]{2,127}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][a-zA-Z0-9.-]+)?$")
_SUPPORTED_CHANNELS = frozenset({"telegram", "email", "webhook", "push", "in_app"})
_SENSITIVE_CONFIG_KEY_RE = re.compile(
    r"(?:^|_)(?:password|token|secret|credential|cookie|authorization|api_key|chat_id)(?:$|_)",
    re.IGNORECASE,
)
NOTIFICATION_PROVIDER_ERROR_CODES = frozenset(
    {
        "provider_disabled",
        "provider_scope_mismatch",
        "provider_secret_unavailable",
        "provider_connect_timeout",
        "provider_transport_error",
        "provider_timeout_after_acceptance_possible",
        "provider_rate_limited",
        "provider_http_error",
        "provider_response_invalid",
        "provider_cancelled",
        "provider_shutdown",
    }
)


class NotificationProviderValidationError(ValueError):
    """Raised when a provider package or instance violates the v1 contract."""


@dataclass(frozen=True, slots=True)
class NotificationProviderDescriptor:
    provider_key: str
    display_name: str
    package_version: str
    config_schema: Mapping[str, object]
    channels: tuple[str, ...]
    templates: tuple[str, ...]
    error_codes: tuple[str, ...]
    contract_version: str = NOTIFICATION_PROVIDER_CONTRACT

    def __post_init__(self) -> None:
        if self.contract_version != NOTIFICATION_PROVIDER_CONTRACT:
            raise NotificationProviderValidationError("unsupported provider contract")
        if not _IDENTIFIER_RE.fullmatch(self.provider_key):
            raise NotificationProviderValidationError("provider_key is invalid")
        if not 2 <= len(self.display_name.strip()) <= 120:
            raise NotificationProviderValidationError("display_name is invalid")
        if not _VERSION_RE.fullmatch(self.package_version):
            raise NotificationProviderValidationError("package_version must be semver")
        if self.config_schema.get("type") != "object":
            raise NotificationProviderValidationError("config_schema must describe an object")
        if not self.channels or any(item not in _SUPPORTED_CHANNELS for item in self.channels):
            raise NotificationProviderValidationError("provider channels are invalid")
        if not self.templates or any(not item.strip() for item in self.templates):
            raise NotificationProviderValidationError("provider templates are required")
        if (
            not self.error_codes
            or not set(self.error_codes) <= NOTIFICATION_PROVIDER_ERROR_CODES
        ):
            raise NotificationProviderValidationError("provider error codes are not bounded")


@dataclass(frozen=True, slots=True)
class NotificationProviderPackage:
    package_id: UUID
    descriptor: NotificationProviderDescriptor
    built_in: bool
    installed_at: datetime


@dataclass(frozen=True, slots=True)
class NotificationProviderInstance:
    instance_id: UUID
    package_id: UUID
    provider_key: str
    scope: NotificationProviderScope
    organization_id: OrganizationId | None
    display_name: str
    config_json: Mapping[str, object]
    secret_ref: str | None
    status: NotificationProviderInstanceStatus
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not _IDENTIFIER_RE.fullmatch(self.provider_key):
            raise NotificationProviderValidationError("provider_key is invalid")
        if self.scope == "installation" and self.organization_id is not None:
            raise NotificationProviderValidationError(
                "installation provider instance must not have organization_id"
            )
        if self.scope == "organization" and self.organization_id is None:
            raise NotificationProviderValidationError(
                "organization provider instance requires organization_id"
            )
        if not 2 <= len(self.display_name.strip()) <= 120:
            raise NotificationProviderValidationError("provider instance display_name is invalid")
        if not isinstance(self.config_json, Mapping):
            raise NotificationProviderValidationError("provider instance config must be an object")
        _validate_config_mapping(self.config_json)
        if self.provider_key == TELEGRAM_BOT_PROVIDER_KEY:
            if self.secret_ref is None:
                raise NotificationProviderValidationError(
                    "Telegram provider instance requires an OpenBao secret reference"
                )
            reference = _validate_secret_reference(
                self.secret_ref, expected_kind=SecretKind.TELEGRAM
            )
            expected_scope = (
                str(self.organization_id)
                if self.organization_id is not None
                else "installation"
            )
            if (
                reference.resource
                != ("providers", expected_scope, str(self.instance_id))
                or reference.field != "bot_token"
            ):
                raise NotificationProviderValidationError(
                    "Telegram secret reference must match provider instance scope"
                )
        elif self.secret_ref is not None:
            reference = _validate_secret_reference(
                self.secret_ref, expected_kind=SecretKind.PLUGIN
            )
            expected_scope = (
                str(self.organization_id)
                if self.organization_id is not None
                else "installation"
            )
            if reference.resource != (expected_scope, str(self.instance_id)):
                raise NotificationProviderValidationError(
                    "Plugin secret reference must match provider instance scope"
                )

    def permits(self, *, organization_id: OrganizationId) -> bool:
        return self.organization_id is None or self.organization_id == organization_id


@dataclass(frozen=True, slots=True)
class NotificationProviderHealth:
    instance_id: UUID
    status: NotificationProviderHealthStatus
    checked_at: datetime
    error_code: str | None = None

    def __post_init__(self) -> None:
        if (
            self.error_code is not None
            and self.error_code not in NOTIFICATION_PROVIDER_ERROR_CODES
        ):
            raise NotificationProviderValidationError("provider health error code is not bounded")


@dataclass(frozen=True, slots=True)
class TelegramUpdateCursor:
    provider_instance_id: UUID
    organization_id: OrganizationId | None
    last_update_id: int
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.last_update_id < -1:
            raise NotificationProviderValidationError("Telegram cursor must be >= -1")


@dataclass(frozen=True, slots=True)
class TelegramCommandDescriptor:
    provider_instance_id: UUID
    command_name: str
    description: str
    enabled: bool
    updated_at: datetime

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", self.command_name):
            raise NotificationProviderValidationError("Telegram command name is invalid")
        if not 2 <= len(self.description.strip()) <= 160:
            raise NotificationProviderValidationError("Telegram command description is invalid")


def telegram_bot_provider_descriptor() -> NotificationProviderDescriptor:
    return NotificationProviderDescriptor(
        provider_key=TELEGRAM_BOT_PROVIDER_KEY,
        display_name="Telegram Bot",
        package_version="1.0.0",
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "api_base_url": {"type": "string", "format": "uri"},
                "connect_timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 3},
                "overall_timeout_seconds": {"type": "number", "minimum": 1, "maximum": 10},
            },
        },
        channels=("telegram",),
        templates=("plain_text.v1", "telegram_command_response.v1"),
        error_codes=tuple(sorted(NOTIFICATION_PROVIDER_ERROR_CODES)),
    )


def _validate_secret_reference(
    raw: str, *, expected_kind: SecretKind
) -> SecretReference:
    try:
        return SecretReference.parse(raw, expected_kind=expected_kind)
    except SecretReferenceError as error:
        raise NotificationProviderValidationError(
            "provider secret reference is invalid"
        ) from error


def _validate_config_mapping(value: Mapping[str, object]) -> None:
    for raw_key, child in value.items():
        key = str(raw_key)
        if _SENSITIVE_CONFIG_KEY_RE.search(key):
            raise NotificationProviderValidationError(
                "provider config contains a raw secret-shaped key"
            )
        if isinstance(child, Mapping):
            _validate_config_mapping(child)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, Mapping):
                    _validate_config_mapping(item)
        elif isinstance(child, str) and re.search(r"://[^/@\s]+:[^/@\s]+@", child):
            raise NotificationProviderValidationError(
                "provider config contains credentials in a URL"
            )


__all__ = [
    "NOTIFICATION_PROVIDER_CONTRACT",
    "NOTIFICATION_PROVIDER_ERROR_CODES",
    "TELEGRAM_BOT_PROVIDER_KEY",
    "NotificationProviderDescriptor",
    "NotificationProviderHealth",
    "NotificationProviderInstance",
    "NotificationProviderPackage",
    "NotificationProviderValidationError",
    "TelegramCommandDescriptor",
    "TelegramUpdateCursor",
    "telegram_bot_provider_descriptor",
]
