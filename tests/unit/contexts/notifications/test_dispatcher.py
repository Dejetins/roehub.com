from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

from trading.contexts.notifications.adapters import InMemoryNotificationRepository
from trading.contexts.notifications.application import (
    NotificationDispatcher,
    NotificationDispatcherConfig,
)
from trading.contexts.notifications.application.ports import NotificationProviderResult
from trading.contexts.notifications.domain import NotificationDelivery


def _now() -> datetime:
    return datetime(2026, 6, 29, 14, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class FixedClock:
    value: datetime

    def now(self) -> datetime:
        return self.value


@dataclass(slots=True)
class CapturingMetrics:
    claimed: int = 0
    results: list[tuple[str, str]] | None = None
    latencies: list[float] | None = None
    pending_age_seconds: float = 0.0
    unknown_count: int = 0

    def __post_init__(self) -> None:
        self.results = []
        self.latencies = []

    def on_delivery_claimed(self, *, provider_key: str) -> None:
        _ = provider_key
        self.claimed += 1

    def on_delivery_result(self, *, provider_key: str, status: str) -> None:
        assert self.results is not None
        self.results.append((provider_key, status))

    def observe_delivery_latency_seconds(
        self, *, provider_key: str, seconds: float
    ) -> None:
        _ = provider_key
        assert self.latencies is not None
        self.latencies.append(seconds)

    def set_pending_age_seconds(self, *, seconds: float) -> None:
        self.pending_age_seconds = seconds

    def set_unknown_count(self, *, count: int) -> None:
        self.unknown_count = count


@dataclass(frozen=True, slots=True)
class StaticProvider:
    status: Literal["sent", "retry", "unknown", "dead_letter", "suppressed"]
    error_code: str | None = None
    retry_after_seconds: int | None = None
    provider_key: str = "log_only"

    def send(self, *, delivery: NotificationDelivery) -> NotificationProviderResult:
        return NotificationProviderResult(
            status=self.status,
            error_code=self.error_code,
            provider_message_id=f"message:{delivery.delivery_id}"
            if self.status == "sent"
            else None,
            retry_after_seconds=self.retry_after_seconds,
            redacted_request_hash="0" * 64,
            redacted_response_hash="1" * 64,
        )


def test_dispatcher_claims_pending_delivery_and_marks_sent() -> None:
    repository = InMemoryNotificationRepository()
    delivery = repository.record_delivery(
        delivery=_delivery(status="pending", created_at=_now() - timedelta(seconds=12))
    )
    metrics = CapturingMetrics()
    dispatcher = NotificationDispatcher(
        repository=repository,
        providers=(StaticProvider(status="sent"),),
        clock=FixedClock(_now()),
        metrics=metrics,
    )

    result = dispatcher.drain_once()

    updated = repository.deliveries[delivery.delivery_id]
    assert result.sent == 1
    assert result.claimed == 1
    assert updated.status == "sent"
    assert updated.attempt_count == 1
    assert updated.lease_until is None
    assert updated.sent_at == _now()
    assert len(repository.attempts) == 1
    assert metrics.claimed == 1
    assert metrics.results == [("log_only", "sent")]
    assert metrics.pending_age_seconds == 12.0


def test_dispatcher_schedules_retry_until_attempt_budget_is_exhausted() -> None:
    repository = InMemoryNotificationRepository()
    delivery = repository.record_delivery(delivery=_delivery(status="pending"))
    dispatcher = NotificationDispatcher(
        repository=repository,
        providers=(
            StaticProvider(
                status="retry", error_code="rate_limited", retry_after_seconds=9
            ),
        ),
        clock=FixedClock(_now()),
        config=NotificationDispatcherConfig(max_attempts=2),
    )

    result = dispatcher.drain_once()

    updated = repository.deliveries[delivery.delivery_id]
    assert result.retry == 1
    assert updated.status == "retry"
    assert updated.attempt_count == 1
    assert updated.next_attempt_at == _now() + timedelta(seconds=9)
    assert updated.last_error_code == "rate_limited"

    second_dispatcher = NotificationDispatcher(
        repository=repository,
        providers=(StaticProvider(status="retry", error_code="rate_limited"),),
        clock=FixedClock(_now() + timedelta(seconds=10)),
        config=NotificationDispatcherConfig(max_attempts=2),
    )

    second_result = second_dispatcher.drain_once()

    exhausted = repository.deliveries[delivery.delivery_id]
    assert second_result.dead_letter == 1
    assert exhausted.status == "dead_letter"
    assert exhausted.attempt_count == 2


def test_dispatcher_reclaims_expired_claim_without_double_sending_active_claim() -> None:
    repository = InMemoryNotificationRepository()
    active = repository.record_delivery(
        delivery=_delivery(
            status="claimed", lease_until=_now() + timedelta(seconds=30), created_at=_now()
        )
    )
    expired = repository.record_delivery(
        delivery=_delivery(
            status="claimed",
            lease_until=_now() - timedelta(seconds=1),
            created_at=_now() - timedelta(seconds=1),
        )
    )
    dispatcher = NotificationDispatcher(
        repository=repository,
        providers=(StaticProvider(status="sent"),),
        clock=FixedClock(_now()),
    )

    result = dispatcher.drain_once()

    assert result.scanned == 1
    assert result.sent == 1
    assert repository.deliveries[active.delivery_id].status == "claimed"
    assert repository.deliveries[expired.delivery_id].status == "sent"
    assert repository.deliveries[expired.delivery_id].attempt_count == 1


def test_dispatcher_marks_unknown_without_blind_retry() -> None:
    repository = InMemoryNotificationRepository()
    delivery = repository.record_delivery(delivery=_delivery(status="pending"))
    metrics = CapturingMetrics()
    dispatcher = NotificationDispatcher(
        repository=repository,
        providers=(StaticProvider(status="unknown", error_code="telegram_timeout"),),
        clock=FixedClock(_now()),
        metrics=metrics,
    )

    result = dispatcher.drain_once()
    repeated = dispatcher.drain_once()

    updated = repository.deliveries[delivery.delivery_id]
    assert result.unknown == 1
    assert repeated.scanned == 0
    assert updated.status == "unknown"
    assert updated.last_error_code == "telegram_timeout"
    assert metrics.unknown_count == 1


def test_dispatcher_dead_letters_missing_provider() -> None:
    repository = InMemoryNotificationRepository()
    delivery = repository.record_delivery(delivery=_delivery(status="pending", provider_key="fake"))
    dispatcher = NotificationDispatcher(
        repository=repository,
        providers=(StaticProvider(status="sent", provider_key="log_only"),),
        clock=FixedClock(_now()),
    )

    result = dispatcher.drain_once()

    updated = repository.deliveries[delivery.delivery_id]
    assert result.provider_missing == 1
    assert result.dead_letter == 1
    assert updated.status == "dead_letter"
    assert updated.last_error_code == "provider_missing"
    assert len(repository.attempts) == 1


def _delivery(
    *,
    status: str,
    provider_key: str = "log_only",
    created_at: datetime | None = None,
    lease_until: datetime | None = None,
) -> NotificationDelivery:
    return NotificationDelivery(
        delivery_id=uuid4(),
        event_id=UUID("22222222-2222-4222-8222-222222222222"),
        report_run_id=None,
        command_id=None,
        route_id=uuid4(),
        provider_key=provider_key,  # type: ignore[arg-type]
        channel_key="telegram",
        recipient_address_ref="telegram_ref:user:stage03",
        template_key="strategy_signal",
        rendered_payload_json={"text": "Stage 03 dispatcher smoke"},
        status=status,  # type: ignore[arg-type]
        attempt_count=0,
        created_at=created_at or _now(),
        lease_until=lease_until,
    )
