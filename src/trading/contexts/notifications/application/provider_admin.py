"""Application commands used by roehubctl and the future admin API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

import jsonschema

from trading.contexts.notifications.application.ports.provider_repository import (
    NotificationProviderRepository,
)
from trading.contexts.notifications.domain import (
    NotificationProviderDescriptor,
    NotificationProviderInstance,
    NotificationProviderPackage,
)
from trading.shared_kernel.primitives import OrganizationId


@dataclass(frozen=True, slots=True)
class AddNotificationProviderCommand:
    instance_id: UUID
    package_id: UUID
    provider_key: str
    scope: str
    organization_id: OrganizationId | None
    display_name: str
    config_json: Mapping[str, object]
    secret_ref: str | None


@dataclass(frozen=True, slots=True)
class InstallNotificationProviderPackageCommand:
    provider_key: str
    display_name: str
    package_version: str
    config_schema: Mapping[str, object]
    channels: tuple[str, ...]
    templates: tuple[str, ...]
    error_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConnectTelegramProviderCommand:
    provider_instance_id: UUID
    organization_id: OrganizationId


class NotificationProviderAdministrationService:
    def __init__(self, *, repository: NotificationProviderRepository) -> None:
        self._repository = repository

    def add_provider(
        self, *, command: AddNotificationProviderCommand, now: datetime
    ) -> NotificationProviderInstance:
        package = self._repository.get_package(package_id=command.package_id)
        if package is None:
            raise ValueError("Notification provider package is unavailable")
        if package.descriptor.provider_key != command.provider_key:
            raise ValueError("Notification provider package key mismatch")
        validator = jsonschema.Draft202012Validator(
            package.descriptor.config_schema,
            format_checker=jsonschema.FormatChecker(),
        )
        if next(
            validator.iter_errors(cast(Any, dict(command.config_json))),
            None,
        ) is not None:
            raise ValueError("Notification provider config does not match package schema")
        instance = NotificationProviderInstance(
            instance_id=command.instance_id,
            package_id=command.package_id,
            provider_key=command.provider_key,
            scope=command.scope,  # type: ignore[arg-type]
            organization_id=command.organization_id,
            display_name=command.display_name,
            config_json=dict(command.config_json),
            secret_ref=command.secret_ref,
            status="active",
            created_at=now,
            updated_at=now,
        )
        return self._repository.add_instance(instance=instance)

    def install_provider_package(
        self,
        *,
        command: InstallNotificationProviderPackageCommand,
        now: datetime,
    ) -> NotificationProviderPackage:
        try:
            jsonschema.Draft202012Validator.check_schema(command.config_schema)
        except jsonschema.SchemaError as error:
            raise ValueError("Notification provider config schema is invalid") from error
        package = NotificationProviderPackage(
            package_id=uuid4(),
            descriptor=NotificationProviderDescriptor(
                provider_key=command.provider_key,
                display_name=command.display_name,
                package_version=command.package_version,
                config_schema=dict(command.config_schema),
                channels=command.channels,
                templates=command.templates,
                error_codes=command.error_codes,
            ),
            built_in=False,
            installed_at=now,
        )
        return self._repository.install_package(package=package)

    def connect_telegram(
        self, *, command: ConnectTelegramProviderCommand
    ) -> NotificationProviderInstance:
        instance = self._repository.get_instance(instance_id=command.provider_instance_id)
        if instance is None or instance.provider_key != "telegram_bot_api":
            raise ValueError("Telegram provider instance is unavailable")
        if not instance.permits(organization_id=command.organization_id):
            raise ValueError("Telegram provider instance belongs to another organization")
        return instance
