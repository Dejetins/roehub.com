from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.contexts.notifications.application.delivery_counters import (
    NotificationDeliveryCounters,
)
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationDeliveryAttempt,
    NotificationEvent,
    NotificationReportRun,
    NotificationRoute,
    TelegramUpdate,
)
from trading.shared_kernel.primitives import UserId


class NotificationRepository(Protocol):
    def record_event(self, *, event: NotificationEvent) -> NotificationEvent: ...

    def get_event_by_dedupe_key(self, *, dedupe_key: str) -> NotificationEvent | None: ...

    def upsert_route(self, *, route: NotificationRoute) -> NotificationRoute: ...

    def get_route(self, *, route_id: UUID) -> NotificationRoute | None: ...

    def list_active_routes(
        self, *, owner_user_id: UserId | None, recipient_kind: str, category: str
    ) -> tuple[NotificationRoute, ...]: ...

    def list_active_report_routes(self) -> tuple[NotificationRoute, ...]: ...

    def record_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery: ...

    def list_due_deliveries(
        self, *, now: datetime, limit: int
    ) -> tuple[NotificationDelivery, ...]: ...

    def update_delivery(self, *, delivery: NotificationDelivery) -> NotificationDelivery: ...

    def count_deliveries_by_status(self, *, status: str) -> int: ...

    def get_delivery_counters(
        self, *, owner_user_id: UserId, now: datetime
    ) -> NotificationDeliveryCounters: ...

    def claim_delivery(
        self, *, delivery_id: UUID, lease_until: datetime, now: datetime
    ) -> NotificationDelivery | None: ...

    def record_delivery_attempt(
        self, *, attempt: NotificationDeliveryAttempt
    ) -> NotificationDeliveryAttempt: ...

    def record_telegram_update(self, *, update: TelegramUpdate) -> TelegramUpdate: ...

    def get_telegram_update(
        self, *, telegram_update_id: int
    ) -> TelegramUpdate | None: ...

    def record_report_run(self, *, report_run: NotificationReportRun) -> NotificationReportRun: ...

    def get_report_run_by_dedupe_key(
        self, *, dedupe_key: str
    ) -> NotificationReportRun | None: ...
