from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from trading.contexts.notifications.adapters import (
    InMemoryNotificationRepository,
    InMemoryNotificationStatsSourceReader,
    LogOnlyNotificationProvider,
)
from trading.contexts.notifications.application import (
    NotificationDispatcher,
    NotificationReportScheduler,
    NotificationReportSchedulerConfig,
    NotificationStatsQueryService,
)
from trading.contexts.notifications.application.report_scheduler import (
    render_portfolio_report,
)
from trading.contexts.notifications.application.stats_query import (
    NotificationStatsSourceRow,
)
from trading.contexts.notifications.domain import NotificationRoute
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000001")


def _now() -> datetime:
    return datetime(2026, 6, 29, 10, 30, tzinfo=UTC)


def _user_id() -> UserId:
    return UserId(UUID("11111111-1111-4111-8111-111111111111"))


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


@dataclass(slots=True)
class CapturingReportMetrics:
    created: list[tuple[str, str]]
    deduped: list[str]
    missed: list[tuple[str, int, str]]

    def __init__(self) -> None:
        self.created = []
        self.deduped = []
        self.missed = []

    def on_report_run_created(self, *, report_type: str, quality_status: str) -> None:
        self.created.append((report_type, quality_status))

    def on_report_run_deduped(self, *, report_type: str) -> None:
        self.deduped.append(report_type)

    def on_missed_schedule(
        self, *, report_type: str, delay_seconds: int, timezone: str
    ) -> None:
        self.missed.append((report_type, delay_seconds, timezone))


def test_report_scheduler_creates_weekly_monthly_runs_and_suppresses_duplicates() -> None:
    repository = InMemoryNotificationRepository()
    repository.upsert_route(route=_route(schedule_json={"timezone": "Europe/Moscow"}))
    scheduler = NotificationReportScheduler(
        repository=repository,
        stats_query_service=NotificationStatsQueryService(
            source_reader=InMemoryNotificationStatsSourceReader(rows=_stats_rows())
        ),
        clock=FixedClock(_now()),
    )

    first = scheduler.run_once()
    second = scheduler.run_once()

    assert first.scanned_routes == 1
    assert first.created_runs == 2
    assert first.deliveries_created == 2
    assert second.created_runs == 0
    assert second.deduped_runs == 2
    assert len(repository.report_runs) == 2
    assert len(repository.deliveries) == 2
    assert {run.report_type for run in repository.report_runs.values()} == {
        "portfolio_weekly",
        "portfolio_monthly",
    }
    assert all(run.quality_status == "complete" for run in repository.report_runs.values())
    assert all(run.status == "rendered" for run in repository.report_runs.values())
    weekly = next(
        run
        for run in repository.report_runs.values()
        if run.report_type == "portfolio_weekly"
    )
    assert weekly.period_start == datetime(2026, 6, 21, 21, 0, tzinfo=UTC)
    assert weekly.period_end == datetime(2026, 6, 28, 21, 0, tzinfo=UTC)
    assert weekly.scope_json["timezone"] == "Europe/Moscow"


def test_report_scheduler_uses_default_timezone_and_missed_schedule_metric() -> None:
    repository = InMemoryNotificationRepository()
    repository.upsert_route(route=_route(schedule_json={"monthly": {"enabled": True}}))
    metrics = CapturingReportMetrics()
    scheduler = NotificationReportScheduler(
        repository=repository,
        stats_query_service=NotificationStatsQueryService(
            source_reader=InMemoryNotificationStatsSourceReader(rows=_stats_rows())
        ),
        clock=FixedClock(_now()),
        config=NotificationReportSchedulerConfig(
            default_timezone="UTC",
            missed_schedule_grace_seconds=0,
        ),
        metrics=metrics,
    )

    result = scheduler.run_once()

    assert result.created_runs == 2
    assert result.missed_schedules == 2
    assert all(run.scope_json["timezone_source"] == "default" for run in result.report_runs)
    assert metrics.created == [
        ("portfolio_weekly", "complete"),
        ("portfolio_monthly", "complete"),
    ]
    assert [item[0] for item in metrics.missed] == [
        "portfolio_weekly",
        "portfolio_monthly",
    ]


def test_report_scheduler_rendering_includes_period_id_and_quality() -> None:
    service = NotificationStatsQueryService(
        source_reader=InMemoryNotificationStatsSourceReader(rows=_stats_rows())
    )
    snapshot = service.get_portfolio_stats_for_window(
        owner_user_id=_user_id(),
        period="week",
        period_start=datetime(2026, 6, 22, tzinfo=UTC),
        period_end=datetime(2026, 6, 29, tzinfo=UTC),
        generated_at=_now(),
        timezone="UTC",
    )

    text = render_portfolio_report(
        snapshot=snapshot,
        report_type="portfolio_weekly",
        period_id="2026-W26",
    )

    assert "Weekly portfolio report 2026-W26: complete" in text
    assert "period=2026-06-22T00:00:00+00:00..2026-06-29T00:00:00+00:00" in text


def test_report_scheduler_smoke_creates_log_delivery_attempts_through_dispatcher() -> None:
    repository = InMemoryNotificationRepository()
    repository.upsert_route(route=_route(schedule_json={"weekly": {"enabled": True}}))
    scheduler = NotificationReportScheduler(
        repository=repository,
        stats_query_service=NotificationStatsQueryService(
            source_reader=InMemoryNotificationStatsSourceReader(rows=_stats_rows())
        ),
        clock=FixedClock(_now()),
    )

    schedule_result = scheduler.run_once()
    dispatch_result = NotificationDispatcher(
        repository=repository,
        providers=(LogOnlyNotificationProvider(),),
        clock=FixedClock(_now()),
    ).drain_once()

    assert schedule_result.created_runs == 2
    assert schedule_result.deliveries_created == 2
    assert dispatch_result.sent == 2
    assert len(repository.attempts) == 2
    assert all(delivery.status == "sent" for delivery in repository.deliveries.values())


def _route(*, schedule_json: dict[str, object]) -> NotificationRoute:
    return NotificationRoute(
        route_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        recipient_kind="user",
        owner_user_id=_user_id(),
        channel_key="telegram",
        provider_key="log_only",
        mode="reports",
        category_filter=("portfolio_report",),
        scope_filter_json={},
        schedule_json=schedule_json,
        recipient_address_ref="telegram_ref:user:stage06",
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )


def _stats_rows() -> tuple[NotificationStatsSourceRow, ...]:
    return (
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="strategy_signals",
            observed_at=datetime(2026, 6, 25, tzinfo=UTC),
            signal_count=4,
        ),
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="strategy_paper_accounting",
            observed_at=datetime(2026, 6, 26, tzinfo=UTC),
            realized_pnl=Decimal("12.50"),
            unrealized_pnl=Decimal("2.25"),
            equity=Decimal("1002.25"),
            pnl_complete=True,
        ),
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="execution_fills",
            observed_at=datetime(2026, 6, 27, tzinfo=UTC),
            fill_count=2,
            order_count=2,
        ),
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="exchange_account_projection",
            observed_at=datetime(2026, 6, 28, tzinfo=UTC),
            balance_count=1,
            position_count=1,
            open_order_count=0,
        ),
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="strategy_signals",
            observed_at=datetime(2026, 5, 20, tzinfo=UTC),
            signal_count=7,
        ),
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="strategy_paper_accounting",
            observed_at=datetime(2026, 5, 21, tzinfo=UTC),
            realized_pnl=Decimal("22.00"),
            unrealized_pnl=Decimal("4.00"),
            equity=Decimal("1022.00"),
            pnl_complete=True,
        ),
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="execution_fills",
            observed_at=datetime(2026, 5, 22, tzinfo=UTC),
            fill_count=5,
            order_count=5,
        ),
        NotificationStatsSourceRow(
            owner_user_id=_user_id(),
            source="exchange_account_projection",
            observed_at=datetime(2026, 5, 23, tzinfo=UTC),
            balance_count=1,
            position_count=1,
            open_order_count=0,
        ),
    )
