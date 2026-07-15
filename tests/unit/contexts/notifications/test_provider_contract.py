from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest

from trading.contexts.notifications.application import (
    AddNotificationProviderCommand,
    InstallNotificationProviderPackageCommand,
    NotificationProviderAdministrationService,
)
from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import (
    NOTIFICATION_PROVIDER_CONTRACT,
    NotificationProviderDescriptor,
    NotificationProviderInstance,
    NotificationProviderPackage,
    NotificationProviderValidationError,
)
from trading.shared_kernel.primitives import OrganizationId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))


class _ProviderRepository:
    def __init__(self) -> None:
        self.packages: dict[UUID, object] = {}
        self.instances: dict[UUID, NotificationProviderInstance] = {}

    def install_package(self, *, package: object) -> object:
        package_id = cast(Any, package).package_id
        self.packages[package_id] = package
        return package

    def add_instance(
        self, *, instance: NotificationProviderInstance
    ) -> NotificationProviderInstance:
        if instance.package_id not in self.packages:
            raise ValueError("package unavailable")
        self.instances[instance.instance_id] = instance
        return instance

    def get_package(self, *, package_id: UUID) -> NotificationProviderPackage | None:
        package = self.packages.get(package_id)
        return cast(NotificationProviderPackage | None, package)


def test_provider_descriptor_and_instance_reject_unbounded_or_raw_secret_config() -> None:
    with pytest.raises(ValueError, match="not bounded"):
        NotificationProviderResult(status="retry", error_code="arbitrary_provider_text")

    with pytest.raises(NotificationProviderValidationError, match="contract"):
        NotificationProviderDescriptor(
            provider_key="custom_http",
            display_name="Custom HTTP",
            package_version="1.0.0",
            config_schema={"type": "object"},
            channels=("webhook",),
            templates=("plain_text.v1",),
            error_codes=("provider_disabled",),
            contract_version="NotificationProvider/v2",
        )

    with pytest.raises(NotificationProviderValidationError, match="raw secret"):
        NotificationProviderInstance(
            instance_id=UUID("00000000-0000-4000-8000-000000000201"),
            package_id=UUID("00000000-0000-4000-8000-000000000301"),
            provider_key="custom_http",
            scope="organization",
            organization_id=_ORGANIZATION_ID,
            display_name="Custom HTTP",
            config_json={"api_token": "must-not-live-here"},
            secret_ref=None,
            status="active",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )


def test_package_installation_and_instance_creation_are_separate_operations() -> None:
    repository = _ProviderRepository()
    service = NotificationProviderAdministrationService(
        repository=cast(Any, repository)
    )
    now = datetime(2026, 7, 13, tzinfo=UTC)

    package = service.install_provider_package(
        command=InstallNotificationProviderPackageCommand(
            provider_key="custom_http",
            display_name="Custom HTTP",
            package_version="1.0.0",
            config_schema={"type": "object"},
            channels=("webhook",),
            templates=("plain_text.v1",),
            error_codes=(
                "provider_disabled",
                "provider_scope_mismatch",
                "provider_connect_timeout",
                "provider_transport_error",
                "provider_timeout_after_acceptance_possible",
                "provider_rate_limited",
                "provider_http_error",
            ),
        ),
        now=now,
    )
    instance = service.add_provider(
        command=AddNotificationProviderCommand(
            instance_id=UUID("00000000-0000-4000-8000-000000000201"),
            package_id=package.package_id,
            provider_key="custom_http",
            scope="organization",
            organization_id=_ORGANIZATION_ID,
            display_name="Organization HTTP",
            config_json={"endpoint_url": "https://provider.test/v1/deliveries"},
            secret_ref=(
                "openbao://kv/roehub/plugins/"
                f"{_ORGANIZATION_ID}/00000000-0000-4000-8000-000000000201"
                "#bearer_token"
            ),
        ),
        now=now,
    )

    assert package.descriptor.contract_version == NOTIFICATION_PROVIDER_CONTRACT
    assert package.built_in is False
    assert instance.package_id == package.package_id
    assert instance.organization_id == _ORGANIZATION_ID


def test_instance_config_is_validated_against_installed_package_schema() -> None:
    repository = _ProviderRepository()
    service = NotificationProviderAdministrationService(
        repository=cast(Any, repository)
    )
    now = datetime(2026, 7, 13, tzinfo=UTC)
    package = service.install_provider_package(
        command=InstallNotificationProviderPackageCommand(
            provider_key="custom_http",
            display_name="Custom HTTP",
            package_version="1.0.0",
            config_schema={
                "type": "object",
                "required": ["endpoint_url"],
                "additionalProperties": False,
                "properties": {"endpoint_url": {"type": "string", "format": "uri"}},
            },
            channels=("webhook",),
            templates=("plain_text.v1",),
            error_codes=("provider_disabled",),
        ),
        now=now,
    )

    with pytest.raises(ValueError, match="does not match package schema"):
        service.add_provider(
            command=AddNotificationProviderCommand(
                instance_id=UUID("00000000-0000-4000-8000-000000000202"),
                package_id=package.package_id,
                provider_key="custom_http",
                scope="organization",
                organization_id=_ORGANIZATION_ID,
                display_name="Invalid HTTP",
                config_json={"unexpected": 123},
                secret_ref=(
                    "openbao://kv/roehub/plugins/"
                    f"{_ORGANIZATION_ID}/00000000-0000-4000-8000-000000000202"
                    "#bearer_token"
                ),
            ),
            now=now,
        )
