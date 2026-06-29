from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading.contexts.notifications.application import (
    NotificationReportScheduler,
    NotificationReportSchedulerConfig,
    NotificationStatsQueryService,
)
from trading.contexts.notifications.application.ports import NotificationRepository
from trading.contexts.notifications.application.report_scheduler import (
    NotificationReportSchedulerClock,
    NotificationReportSchedulerMetrics,
)
from trading.contexts.notifications.application.stats_query import (
    NotificationStatsSourceReader,
)


@dataclass(frozen=True, slots=True)
class FixedNotificationReportSchedulerClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


def build_notification_report_scheduler(
    *,
    repository: NotificationRepository,
    stats_source_reader: NotificationStatsSourceReader,
    clock: NotificationReportSchedulerClock,
    config: NotificationReportSchedulerConfig | None = None,
    metrics: NotificationReportSchedulerMetrics | None = None,
) -> NotificationReportScheduler:
    return NotificationReportScheduler(
        repository=repository,
        stats_query_service=NotificationStatsQueryService(
            source_reader=stats_source_reader
        ),
        clock=clock,
        config=config,
        metrics=metrics,
    )


__all__ = [
    "FixedNotificationReportSchedulerClock",
    "build_notification_report_scheduler",
]
