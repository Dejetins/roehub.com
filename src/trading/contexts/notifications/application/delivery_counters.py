from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class NotificationDeliveryCounters:
    telegram_sent_total: int
    telegram_sent_last_24h: int
    last_telegram_sent_at: datetime | None


class NotificationDeliveryCounterReader(Protocol):
    def get_delivery_counters(
        self, *, owner_user_id: UserId, now: datetime
    ) -> NotificationDeliveryCounters: ...


@dataclass(frozen=True, slots=True)
class NotificationDeliveryCounterService:
    reader: NotificationDeliveryCounterReader

    def get_counters(
        self, *, owner_user_id: UserId, now: datetime
    ) -> NotificationDeliveryCounters:
        return self.reader.get_delivery_counters(owner_user_id=owner_user_id, now=now)
