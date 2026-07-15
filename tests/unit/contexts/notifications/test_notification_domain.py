from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationDeliveryAttempt,
    NotificationEvent,
    NotificationReportRun,
    NotificationRoute,
    NotificationValidationError,
    TelegramUpdate,
    build_notification_dedupe_key,
    sanitize_notification_mapping,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


def _now() -> datetime:
    return datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)


def _user_id() -> UserId:
    return UserId(UUID("11111111-1111-4111-8111-111111111111"))


def _organization_id() -> OrganizationId:
    return OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))


def _provider_instance_id() -> UUID:
    return UUID("00000000-0000-4000-8000-000000000001")


def test_notification_event_validates_dedupe_and_rejects_secret_like_payload() -> None:
    event = NotificationEvent(
        event_id=uuid4(),
        organization_id=_organization_id(),
        owner_user_id=_user_id(),
        recipient_kind="user",
        source_context="strategy",
        source_event_type="strategy_run_failed",
        category="strategy_run_failed",
        severity="warning",
        scope_json={"strategy_id": str(uuid4())},
        payload_json={"reason": "run failed"},
        dedupe_key=build_notification_dedupe_key(
            organization_id=_organization_id(),
            source_context="strategy",
            source_event_type="strategy_run_failed",
            source_id=str(uuid4()),
        ),
        occurred_at=_now(),
        created_at=_now(),
    )

    assert event.dedupe_key.startswith("strategy:strategy_run_failed:")

    with pytest.raises(NotificationValidationError, match="payload_json"):
        NotificationEvent(
            event_id=uuid4(),
            organization_id=_organization_id(),
            owner_user_id=_user_id(),
            recipient_kind="user",
            source_context="strategy",
            source_event_type="strategy_run_failed",
            category="strategy_run_failed",
            severity="warning",
            scope_json={},
            payload_json={"api_token": "redacted"},
            dedupe_key=event.dedupe_key,
            occurred_at=_now(),
            created_at=_now(),
        )


def test_notification_route_separates_user_and_admin_recipients() -> None:
    NotificationRoute(
        route_id=uuid4(),
        organization_id=_organization_id(),
        provider_instance_id=_provider_instance_id(),
        recipient_kind="user",
        owner_user_id=_user_id(),
        channel_key="telegram",
        provider_key="log_only",
        mode="critical_only",
        category_filter=("strategy_run_failed", "execution_unknown"),
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref="telegram_ref:user:abc123",
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )
    NotificationRoute(
        route_id=uuid4(),
        organization_id=_organization_id(),
        provider_instance_id=_provider_instance_id(),
        recipient_kind="admin",
        owner_user_id=None,
        channel_key="telegram",
        provider_key="log_only",
        mode="all",
        category_filter=("admin_critical", "admin_alert"),
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref="telegram_ref:admin:abc123",
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )

    with pytest.raises(NotificationValidationError, match="user_route_requires_owner_user_id"):
        NotificationRoute(
            route_id=uuid4(),
            organization_id=_organization_id(),
            provider_instance_id=_provider_instance_id(),
            recipient_kind="user",
            owner_user_id=None,
            channel_key="telegram",
            provider_key="log_only",
            mode="critical_only",
            category_filter=("strategy_run_failed",),
            scope_filter_json={},
            schedule_json={},
            recipient_address_ref="telegram_ref:user:abc123",
            status="active",
            created_at=_now(),
            updated_at=_now(),
        )
    with pytest.raises(
        NotificationValidationError, match="admin_route_must_not_have_owner_user_id"
    ):
        NotificationRoute(
            route_id=uuid4(),
            organization_id=_organization_id(),
            provider_instance_id=_provider_instance_id(),
            recipient_kind="admin",
            owner_user_id=_user_id(),
            channel_key="telegram",
            provider_key="log_only",
            mode="all",
            category_filter=("admin_alert",),
            scope_filter_json={},
            schedule_json={},
            recipient_address_ref="telegram_ref:admin:abc123",
            status="active",
            created_at=_now(),
            updated_at=_now(),
        )


def test_delivery_attempt_statuses_and_hashes_are_validated() -> None:
    NotificationDelivery(
        delivery_id=uuid4(),
        organization_id=_organization_id(),
        provider_instance_id=_provider_instance_id(),
        event_id=uuid4(),
        report_run_id=None,
        command_id=None,
        route_id=uuid4(),
        provider_key="log_only",
        channel_key="telegram",
        recipient_address_ref="telegram_ref:user:abc123",
        template_key="strategy.failure.v1",
        rendered_payload_json={"template": "strategy.failure.v1"},
        status="unknown",
        attempt_count=1,
        created_at=_now(),
    )
    NotificationDeliveryAttempt(
        attempt_id=uuid4(),
        organization_id=_organization_id(),
        provider_instance_id=_provider_instance_id(),
        delivery_id=uuid4(),
        provider_key="log_only",
        started_at=_now(),
        status="unknown",
        http_status=504,
        retry_after_seconds=0,
        redacted_request_hash="a" * 64,
        redacted_response_hash="b" * 64,
    )

    with pytest.raises(NotificationValidationError, match="redacted_request_hash"):
        NotificationDeliveryAttempt(
            attempt_id=uuid4(),
            organization_id=_organization_id(),
            provider_instance_id=_provider_instance_id(),
            delivery_id=uuid4(),
            provider_key="log_only",
            started_at=_now(),
            status="failed",
            redacted_request_hash="not-a-sha256",
        )


def test_report_and_telegram_update_scaffolds_validate_status_and_periods() -> None:
    NotificationReportRun(
        report_run_id=uuid4(),
        organization_id=_organization_id(),
        owner_user_id=_user_id(),
        report_type="portfolio_weekly",
        period_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
        period_end=datetime(2026, 6, 8, tzinfo=timezone.utc),
        scope_json={"timezone": "Europe/Moscow"},
        quality_status="partial",
        status="pending",
        dedupe_key=build_notification_dedupe_key(
            organization_id=_organization_id(),
            source_context="notifications",
            source_event_type="portfolio_weekly",
            source_id="2026-W23",
        ),
        created_at=_now(),
    )
    TelegramUpdate(
        organization_id=_organization_id(),
        provider_instance_id=_provider_instance_id(),
        telegram_update_id=123,
        received_at=_now(),
        chat_id_ref="telegram_ref:user:abc123",
        owner_user_id=_user_id(),
        command_name="stats",
        command_args_json={"period": "week"},
        status="pending",
        idempotency_key=build_notification_dedupe_key(
            organization_id=_organization_id(),
            source_context="notifications",
            source_event_type="telegram_update",
            source_id="123",
        ),
        created_at=_now(),
    )

    with pytest.raises(NotificationValidationError, match="report_period_must_be_non_empty"):
        NotificationReportRun(
            report_run_id=uuid4(),
            organization_id=_organization_id(),
            owner_user_id=_user_id(),
            report_type="portfolio_weekly",
            period_start=_now(),
            period_end=_now(),
            scope_json={},
            quality_status="complete",
            status="pending",
            dedupe_key="notifications:portfolio_weekly:123456",
            created_at=_now(),
        )


def test_sanitize_notification_mapping_rejects_secret_like_keys_and_values() -> None:
    assert sanitize_notification_mapping({"symbol": "BTCUSDT"}) == {"symbol": "BTCUSDT"}

    with pytest.raises(NotificationValidationError, match="sensitive_notification_key_rejected"):
        sanitize_notification_mapping({"chat_id": "redacted"})
    with pytest.raises(NotificationValidationError, match="sensitive_notification_value_rejected"):
        sanitize_notification_mapping({"label": "contains token marker"})
