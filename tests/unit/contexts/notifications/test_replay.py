from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from trading.contexts.notifications.adapters import InMemoryNotificationRepository
from trading.contexts.notifications.application import (
    NotificationDeliveryReplayService,
    ReplayNotificationDeliveryCommand,
)
from trading.contexts.notifications.domain import NotificationDelivery, NotificationRoute
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"))
_OTHER_ORGANIZATION_ID = OrganizationId(UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"))
_PROVIDER_INSTANCE_ID = UUID("00000000-0000-4000-8000-000000000003")
_ORIGINAL_DELIVERY_ID = UUID("00000000-0000-4000-8000-000000000401")
_REPLAY_DELIVERY_ID = UUID("00000000-0000-4000-8000-000000000402")


def test_explicit_replay_creates_linked_delivery_and_preserves_unknown_source() -> None:
    repository = _repository_with_delivery(status="unknown")
    service = NotificationDeliveryReplayService(repository=repository)
    command = ReplayNotificationDeliveryCommand(
        organization_id=_ORGANIZATION_ID,
        original_delivery_id=_ORIGINAL_DELIVERY_ID,
        replay_delivery_id=_REPLAY_DELIVERY_ID,
    )

    replay = service.replay(command=command, now=_now() + timedelta(minutes=1))
    repeated = service.replay(command=command, now=_now() + timedelta(minutes=2))

    assert replay == repeated
    assert replay.status == "pending"
    assert replay.attempt_count == 0
    assert replay.replayed_from_delivery_id == _ORIGINAL_DELIVERY_ID
    assert repository.deliveries[_ORIGINAL_DELIVERY_ID].status == "unknown"
    assert len(repository.deliveries) == 2


def test_explicit_replay_rejects_cross_organization_and_non_terminal_source() -> None:
    pending_repository = _repository_with_delivery(status="pending")
    pending_service = NotificationDeliveryReplayService(
        repository=pending_repository
    )
    with pytest.raises(ValueError, match="unknown or dead_letter"):
        pending_service.replay(
            command=ReplayNotificationDeliveryCommand(
                organization_id=_ORGANIZATION_ID,
                original_delivery_id=_ORIGINAL_DELIVERY_ID,
                replay_delivery_id=_REPLAY_DELIVERY_ID,
            ),
            now=_now(),
        )
    with pytest.raises(ValueError, match="source is unavailable"):
        pending_service.replay(
            command=ReplayNotificationDeliveryCommand(
                organization_id=_OTHER_ORGANIZATION_ID,
                original_delivery_id=_ORIGINAL_DELIVERY_ID,
                replay_delivery_id=_REPLAY_DELIVERY_ID,
            ),
            now=_now(),
        )


def _repository_with_delivery(*, status: str) -> InMemoryNotificationRepository:
    repository = InMemoryNotificationRepository()
    route = NotificationRoute(
        route_id=UUID("00000000-0000-4000-8000-000000000301"),
        organization_id=_ORGANIZATION_ID,
        provider_instance_id=_PROVIDER_INSTANCE_ID,
        recipient_kind="user",
        owner_user_id=UserId(UUID("11111111-1111-4111-8111-111111111111")),
        channel_key="telegram",
        provider_key="telegram_bot_api",
        mode="all",
        category_filter=(),
        scope_filter_json={},
        schedule_json={},
        recipient_address_ref="telegram_ref:replay:masked",
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )
    repository.upsert_route(route=route)
    repository.record_delivery(
        delivery=NotificationDelivery(
            delivery_id=_ORIGINAL_DELIVERY_ID,
            organization_id=_ORGANIZATION_ID,
            provider_instance_id=_PROVIDER_INSTANCE_ID,
            event_id=None,
            report_run_id=None,
            command_id=UUID("00000000-0000-4000-8000-000000000201"),
            route_id=route.route_id,
            provider_key=route.provider_key,
            channel_key=route.channel_key,
            recipient_address_ref=route.recipient_address_ref,
            template_key="plain_text.v1",
            rendered_payload_json={"text": "sanitized"},
            status=status,  # type: ignore[arg-type]
            attempt_count=1,
            created_at=_now(),
            last_error_code="provider_timeout_after_acceptance_possible",
        )
    )
    return repository


def _now() -> datetime:
    return datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
