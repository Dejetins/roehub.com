from __future__ import annotations

from apps.api.monitoring import (
    build_metrics_response,
    record_admin_notification,
    record_notification_delivery_unknown,
    record_notifications_delivery_retry,
    record_notifications_report_schedule_missed,
    set_notification_worker_up,
    set_notifications_pending_oldest_age_seconds,
)


def test_notification_admin_metrics_are_bounded_and_exposed() -> None:
    record_admin_notification(
        category="admin_critical",
        severity="critical",
        status="pending",
    )
    record_notification_delivery_unknown(
        provider="log_only",
        channel="telegram",
        category="admin_critical",
    )
    set_notifications_pending_oldest_age_seconds(
        provider="log_only",
        channel="telegram",
        severity="critical",
        seconds=301.0,
    )
    record_notifications_delivery_retry(
        provider="telegram_bot_api",
        channel="telegram",
        reason="rate_limited",
    )
    set_notification_worker_up(worker="notification_dispatcher", up=False)
    record_notifications_report_schedule_missed(
        report_type="portfolio_weekly",
        timezone="UTC",
    )

    payload = bytes(build_metrics_response().body).decode()

    assert (
        'admin_notifications_total{category="admin_critical",'
        'severity="critical",status="pending"}'
    ) in payload
    assert (
        'notifications_delivery_unknown_total{category="admin_critical",'
        'channel="telegram",provider="log_only"}'
    ) in payload
    assert (
        'notifications_pending_oldest_age_seconds{channel="telegram",'
        'provider="log_only",severity="critical"} 301.0'
    ) in payload
    assert (
        'notifications_deliveries_retry_total{channel="telegram",'
        'provider="telegram_bot_api",reason="rate_limited"}'
    ) in payload
    assert 'notifications_worker_up{worker="notification_dispatcher"} 0.0' in payload
    assert (
        'notifications_report_schedule_missed_total{report_type="portfolio_weekly",'
        'timezone="UTC"}'
    ) in payload
