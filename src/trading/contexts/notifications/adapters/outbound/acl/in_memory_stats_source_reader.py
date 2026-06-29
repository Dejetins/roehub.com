from __future__ import annotations

from datetime import datetime

from trading.contexts.notifications.application.stats_query import (
    NotificationStatsSourceReader,
    NotificationStatsSourceResult,
    NotificationStatsSourceRow,
)
from trading.shared_kernel.primitives import UserId


class InMemoryNotificationStatsSourceReader(NotificationStatsSourceReader):
    def __init__(
        self,
        *,
        rows: tuple[NotificationStatsSourceRow, ...] = (),
        unavailable_sources: tuple[str, ...] = (),
    ) -> None:
        self._rows = rows
        self._unavailable_sources = unavailable_sources

    def read_stats_rows(
        self,
        *,
        owner_user_id: UserId,
        period_start: datetime,
        period_end: datetime,
        strategy_ref: str | None = None,
        exchange_ref: str | None = None,
    ) -> NotificationStatsSourceResult:
        rows = tuple(
            row
            for row in self._rows
            if row.owner_user_id == owner_user_id
            and period_start <= row.observed_at <= period_end
            and (strategy_ref is None or row.strategy_ref == strategy_ref)
            and (exchange_ref is None or row.exchange_ref == exchange_ref)
        )
        return NotificationStatsSourceResult(
            rows=rows,
            unavailable_sources=self._unavailable_sources,
        )
