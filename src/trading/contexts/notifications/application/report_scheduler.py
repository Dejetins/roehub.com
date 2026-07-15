from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from hashlib import sha256
from typing import Literal, Mapping, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from trading.contexts.notifications.application.ports import NotificationRepository
from trading.contexts.notifications.application.stats_query import (
    NotificationStatsQueryService,
    NotificationStatsSnapshot,
)
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationReportRun,
    NotificationRoute,
    build_notification_dedupe_key,
    sanitize_notification_mapping,
)
from trading.contexts.notifications.domain.notification import NotificationReportType

NotificationReportSchedulerPeriod = Literal["week", "month"]


class NotificationReportSchedulerClock(Protocol):
    def now(self) -> datetime: ...


class NotificationReportSchedulerMetrics(Protocol):
    def on_report_run_created(self, *, report_type: str, quality_status: str) -> None: ...

    def on_report_run_deduped(self, *, report_type: str) -> None: ...

    def on_missed_schedule(
        self, *, report_type: str, delay_seconds: int, timezone: str
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NotificationReportSchedulerConfig:
    default_timezone: str = "UTC"
    missed_schedule_grace_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.missed_schedule_grace_seconds < 0:
            raise ValueError("missed_schedule_grace_seconds must be >= 0")
        _zone_or_default(self.default_timezone, fallback="UTC")


@dataclass(frozen=True, slots=True)
class NotificationReportSchedulerResult:
    scanned_routes: int
    created_runs: int
    deduped_runs: int
    deliveries_created: int
    missed_schedules: int
    report_runs: tuple[NotificationReportRun, ...]
    deliveries: tuple[NotificationDelivery, ...]


class NotificationReportScheduler:
    def __init__(
        self,
        *,
        repository: NotificationRepository,
        stats_query_service: NotificationStatsQueryService,
        clock: NotificationReportSchedulerClock,
        config: NotificationReportSchedulerConfig | None = None,
        metrics: NotificationReportSchedulerMetrics | None = None,
    ) -> None:
        self._repository = repository
        self._stats_query_service = stats_query_service
        self._clock = clock
        self._config = config or NotificationReportSchedulerConfig()
        self._metrics = metrics

    def run_once(self) -> NotificationReportSchedulerResult:
        now = _as_utc(self._clock.now())
        routes = self._repository.list_active_report_routes()
        created_report_runs: list[NotificationReportRun] = []
        created_deliveries: list[NotificationDelivery] = []
        deduped_runs = 0
        missed_schedules = 0

        for route in routes:
            owner_user_id = route.owner_user_id
            if owner_user_id is None:
                continue
            for report_type in ("portfolio_weekly", "portfolio_monthly"):
                if not _schedule_enabled(route=route, report_type=report_type):
                    continue
                timezone_name, used_default_timezone = _route_timezone(
                    route=route,
                    report_type=report_type,
                    default_timezone=self._config.default_timezone,
                )
                window = _previous_period_window(
                    report_type=report_type,
                    now=now,
                    timezone_name=timezone_name,
                )
                delay_seconds = int((now - window.period_end_utc).total_seconds())
                if delay_seconds < 0:
                    continue
                if delay_seconds > self._config.missed_schedule_grace_seconds:
                    missed_schedules += 1
                    if self._metrics is not None:
                        self._metrics.on_missed_schedule(
                            report_type=report_type,
                            delay_seconds=delay_seconds,
                            timezone=timezone_name,
                        )

                dedupe_key = _report_dedupe_key(
                    route=route,
                    report_type=report_type,
                    window=window,
                )
                existing = self._repository.get_report_run_by_dedupe_key(
                    organization_id=route.organization_id,
                    dedupe_key=dedupe_key
                )
                if existing is not None:
                    deduped_runs += 1
                    if self._metrics is not None:
                        self._metrics.on_report_run_deduped(report_type=report_type)
                    continue

                snapshot = self._stats_query_service.get_portfolio_stats_for_window(
                    owner_user_id=owner_user_id,
                    period=window.stats_period,
                    period_start=window.period_start_utc,
                    period_end=window.period_end_utc,
                    generated_at=now,
                    timezone=timezone_name,
                )
                period_id = _period_id(report_type=report_type, window=window)
                report_run = NotificationReportRun(
                    report_run_id=uuid4(),
                    organization_id=route.organization_id,
                    owner_user_id=owner_user_id,
                    report_type=report_type,
                    period_start=window.period_start_utc,
                    period_end=window.period_end_utc,
                    scope_json=sanitize_notification_mapping(
                        {
                            "scope": "portfolio",
                            "period_id": period_id,
                            "timezone": timezone_name,
                            "timezone_source": "default"
                            if used_default_timezone
                            else "route",
                        }
                    ),
                    quality_status=snapshot.quality_status,
                    status="rendered",
                    dedupe_key=dedupe_key,
                    created_at=now,
                    rendered_at=now,
                )
                stored_report_run = self._repository.record_report_run(
                    report_run=report_run
                )
                delivery = NotificationDelivery(
                    delivery_id=uuid4(),
                    organization_id=route.organization_id,
                    provider_instance_id=route.provider_instance_id,
                    event_id=None,
                    report_run_id=stored_report_run.report_run_id,
                    command_id=None,
                    route_id=route.route_id,
                    provider_key=route.provider_key,
                    channel_key=route.channel_key,
                    recipient_address_ref=route.recipient_address_ref,
                    template_key=f"{report_type}.v1",
                    rendered_payload_json=sanitize_notification_mapping(
                        {
                            "report_type": report_type,
                            "period_id": period_id,
                            "quality_status": snapshot.quality_status,
                            "text": render_portfolio_report(
                                snapshot=snapshot,
                                report_type=report_type,
                                period_id=period_id,
                            ),
                        }
                    ),
                    status="pending",
                    attempt_count=0,
                    created_at=now,
                )
                stored_delivery = self._repository.record_delivery(delivery=delivery)
                created_report_runs.append(stored_report_run)
                created_deliveries.append(stored_delivery)
                if self._metrics is not None:
                    self._metrics.on_report_run_created(
                        report_type=report_type,
                        quality_status=snapshot.quality_status,
                    )

        return NotificationReportSchedulerResult(
            scanned_routes=len(routes),
            created_runs=len(created_report_runs),
            deduped_runs=deduped_runs,
            deliveries_created=len(created_deliveries),
            missed_schedules=missed_schedules,
            report_runs=tuple(created_report_runs),
            deliveries=tuple(created_deliveries),
        )


def render_portfolio_report(
    *,
    snapshot: NotificationStatsSnapshot,
    report_type: NotificationReportType,
    period_id: str,
) -> str:
    label = "Weekly" if report_type == "portfolio_weekly" else "Monthly"
    parts = [
        f"{label} portfolio report {period_id}: {snapshot.quality_status}",
        f"period={snapshot.period_start.isoformat()}..{snapshot.period_end.isoformat()}",
        f"signals={snapshot.signal_count}",
        f"fills={snapshot.fill_count}",
        f"orders={snapshot.order_count}",
    ]
    if snapshot.realized_pnl is not None:
        parts.append(f"realized_pnl={snapshot.realized_pnl}")
    if snapshot.unrealized_pnl is not None:
        parts.append(f"unrealized_pnl={snapshot.unrealized_pnl}")
    if snapshot.missing_sources:
        parts.append("missing_sources=" + ",".join(snapshot.missing_sources))
    return "; ".join(parts) + "."


@dataclass(frozen=True, slots=True)
class _ReportPeriodWindow:
    report_type: NotificationReportType
    stats_period: NotificationReportSchedulerPeriod
    timezone: str
    period_start_utc: datetime
    period_end_utc: datetime
    period_start_local: datetime
    period_end_local: datetime


def _previous_period_window(
    *,
    report_type: NotificationReportType,
    now: datetime,
    timezone_name: str,
) -> _ReportPeriodWindow:
    zone = ZoneInfo(timezone_name)
    local_now = now.astimezone(zone)
    if report_type == "portfolio_weekly":
        week_start_date = local_now.date() - timedelta(days=local_now.weekday())
        current_start = datetime.combine(week_start_date, time.min, tzinfo=zone)
        period_start = current_start - timedelta(days=7)
        period_end = current_start
        stats_period: NotificationReportSchedulerPeriod = "week"
    elif report_type == "portfolio_monthly":
        current_start = datetime.combine(
            local_now.date().replace(day=1), time.min, tzinfo=zone
        )
        previous_month_last_day = current_start.date() - timedelta(days=1)
        period_start = datetime.combine(
            previous_month_last_day.replace(day=1), time.min, tzinfo=zone
        )
        period_end = current_start
        stats_period = "month"
    else:
        raise ValueError(f"unsupported scheduled report type: {report_type}")
    return _ReportPeriodWindow(
        report_type=report_type,
        stats_period=stats_period,
        timezone=timezone_name,
        period_start_utc=period_start.astimezone(UTC),
        period_end_utc=period_end.astimezone(UTC),
        period_start_local=period_start,
        period_end_local=period_end,
    )


def _period_id(*, report_type: NotificationReportType, window: _ReportPeriodWindow) -> str:
    if report_type == "portfolio_weekly":
        iso = window.period_start_local.date().isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if report_type == "portfolio_monthly":
        return window.period_start_local.strftime("%Y-%m")
    raise ValueError(f"unsupported scheduled report type: {report_type}")


def _report_dedupe_key(
    *,
    route: NotificationRoute,
    report_type: NotificationReportType,
    window: _ReportPeriodWindow,
) -> str:
    scope_digest = sha256(
        json.dumps(route.scope_filter_json, sort_keys=True, default=str).encode()
    ).hexdigest()
    source_id = (
        f"user={route.owner_user_id}:route={route.route_id}:type={report_type}:"
        f"start={window.period_start_utc.isoformat()}:"
        f"end={window.period_end_utc.isoformat()}:scope={scope_digest}"
    )
    return build_notification_dedupe_key(
        organization_id=route.organization_id,
        source_context="notifications",
        source_event_type=report_type,
        source_id=source_id,
    )


def _schedule_enabled(*, route: NotificationRoute, report_type: NotificationReportType) -> bool:
    period_key = "weekly" if report_type == "portfolio_weekly" else "monthly"
    schedule = route.schedule_json
    value = schedule.get(period_key, schedule.get(report_type, True))
    if isinstance(value, Mapping):
        enabled = value.get("enabled", True)
        return bool(enabled)
    return bool(value)


def _route_timezone(
    *,
    route: NotificationRoute,
    report_type: NotificationReportType,
    default_timezone: str,
) -> tuple[str, bool]:
    period_key = "weekly" if report_type == "portfolio_weekly" else "monthly"
    candidates: list[object] = []
    period_value = route.schedule_json.get(period_key, route.schedule_json.get(report_type))
    if isinstance(period_value, Mapping):
        candidates.append(period_value.get("timezone"))
    candidates.append(route.schedule_json.get("timezone"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            timezone_name = candidate.strip()
            return _zone_or_default(timezone_name, fallback=default_timezone), False
    return _zone_or_default(default_timezone, fallback="UTC"), True


def _zone_or_default(timezone_name: str, *, fallback: str) -> str:
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        ZoneInfo(fallback)
        return fallback
    return timezone_name


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
