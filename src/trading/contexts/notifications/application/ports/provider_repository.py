from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.contexts.notifications.domain import (
    NotificationProviderHealth,
    NotificationProviderInstance,
    NotificationProviderPackage,
    TelegramCommandDescriptor,
    TelegramUpdateCursor,
)
from trading.shared_kernel.primitives import OrganizationId


class NotificationProviderRepository(Protocol):
    def install_package(
        self, *, package: NotificationProviderPackage
    ) -> NotificationProviderPackage: ...

    def get_package(self, *, package_id: UUID) -> NotificationProviderPackage | None: ...

    def add_instance(
        self, *, instance: NotificationProviderInstance
    ) -> NotificationProviderInstance: ...

    def get_instance(self, *, instance_id: UUID) -> NotificationProviderInstance | None: ...

    def list_active_instances(self) -> tuple[NotificationProviderInstance, ...]: ...

    def list_instances_for_organization(
        self, *, organization_id: OrganizationId
    ) -> tuple[NotificationProviderInstance, ...]: ...

    def record_health(self, *, health: NotificationProviderHealth) -> None: ...

    def get_cursor(self, *, provider_instance_id: UUID) -> TelegramUpdateCursor | None: ...

    def advance_cursor(
        self,
        *,
        provider_instance_id: UUID,
        expected_last_update_id: int,
        next_update_id: int,
        updated_at: datetime,
    ) -> TelegramUpdateCursor: ...

    def replace_command_registry(
        self,
        *,
        provider_instance_id: UUID,
        commands: tuple[TelegramCommandDescriptor, ...],
    ) -> tuple[TelegramCommandDescriptor, ...]: ...
