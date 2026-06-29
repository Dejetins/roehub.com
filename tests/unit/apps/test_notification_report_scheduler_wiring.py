from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from apps.worker.notification_report_scheduler.wiring import (
    FixedNotificationReportSchedulerClock,
    build_notification_report_scheduler,
)
from trading.contexts.notifications.adapters import (
    InMemoryNotificationRepository,
    InMemoryNotificationStatsSourceReader,
)
from trading.contexts.notifications.application.stats_query import (
    NotificationStatsSourceRow,
)
from trading.contexts.notifications.domain import NotificationRoute
from trading.shared_kernel.primitives import UserId


def test_notification_report_scheduler_wiring_builds_scheduler() -> None:
    repository = InMemoryNotificationRepository()
    owner_user_id = UserId(UUID("11111111-1111-4111-8111-111111111111"))
    now = datetime(2026, 6, 29, 10, 30, tzinfo=UTC)
    repository.upsert_route(
        route=NotificationRoute(
            route_id=uuid4(),
            recipient_kind="user",
            owner_user_id=owner_user_id,
            channel_key="telegram",
            provider_key="fake",
            mode="reports",
            category_filter=("portfolio_report",),
            scope_filter_json={},
            schedule_json={"timezone": "UTC"},
            recipient_address_ref="telegram_ref:user:stage06",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    scheduler = build_notification_report_scheduler(
        repository=repository,
        stats_source_reader=InMemoryNotificationStatsSourceReader(
            rows=(
                NotificationStatsSourceRow(
                    owner_user_id=owner_user_id,
                    source="strategy_signals",
                    observed_at=datetime(2026, 6, 25, tzinfo=UTC),
                    signal_count=1,
                ),
                NotificationStatsSourceRow(
                    owner_user_id=owner_user_id,
                    source="strategy_paper_accounting",
                    observed_at=datetime(2026, 6, 26, tzinfo=UTC),
                    realized_pnl=Decimal("1.00"),
                    pnl_complete=True,
                ),
            )
        ),
        clock=FixedNotificationReportSchedulerClock(now),
    )

    result = scheduler.run_once()

    assert result.created_runs == 2
    assert result.deliveries_created == 2
    assert all(delivery.provider_key == "fake" for delivery in result.deliveries)
