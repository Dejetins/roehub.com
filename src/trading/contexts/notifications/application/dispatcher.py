from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Protocol
from uuid import uuid4

from trading.contexts.notifications.application.ports import (
    NotificationProvider,
    NotificationProviderResult,
    NotificationRepository,
)
from trading.contexts.notifications.domain import (
    NotificationDelivery,
    NotificationDeliveryAttempt,
)

log = logging.getLogger(__name__)


class NotificationDispatcherClock(Protocol):
    def now(self) -> datetime: ...


class NotificationDispatcherMetrics(Protocol):
    def on_delivery_claimed(self, *, provider_key: str) -> None: ...

    def on_delivery_result(self, *, provider_key: str, status: str) -> None: ...

    def observe_delivery_latency_seconds(
        self, *, provider_key: str, seconds: float
    ) -> None: ...

    def set_pending_age_seconds(self, *, seconds: float) -> None: ...

    def set_unknown_count(self, *, count: int) -> None: ...


@dataclass(frozen=True, slots=True)
class NotificationDispatcherConfig:
    batch_size: int = 100
    lease_seconds: int = 30
    retry_backoff_seconds: int = 60
    max_attempts: int = 3
    allowed_provider_keys: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("NotificationDispatcherConfig.batch_size must be > 0")
        if self.lease_seconds <= 0:
            raise ValueError("NotificationDispatcherConfig.lease_seconds must be > 0")
        if self.retry_backoff_seconds <= 0:
            raise ValueError(
                "NotificationDispatcherConfig.retry_backoff_seconds must be > 0"
            )
        if self.max_attempts <= 0:
            raise ValueError("NotificationDispatcherConfig.max_attempts must be > 0")


@dataclass(frozen=True, slots=True)
class NotificationDispatchBatchResult:
    scanned: int
    claimed: int
    sent: int
    retry: int
    unknown: int
    dead_letter: int
    suppressed: int
    provider_missing: int


class NotificationDispatcher:
    def __init__(
        self,
        *,
        repository: NotificationRepository,
        providers: tuple[NotificationProvider, ...],
        clock: NotificationDispatcherClock,
        config: NotificationDispatcherConfig | None = None,
        metrics: NotificationDispatcherMetrics | None = None,
    ) -> None:
        self._repository = repository
        self._providers = {provider.provider_key: provider for provider in providers}
        self._clock = clock
        self._config = config or NotificationDispatcherConfig()
        self._metrics = metrics

    def drain_once(self) -> NotificationDispatchBatchResult:
        now = self._clock.now()
        due = self._repository.list_due_deliveries(
            now=now, limit=self._config.batch_size
        )
        counts = _MutableBatchCounts(scanned=len(due))
        self._update_pending_age_metric(deliveries=due, now=now)

        for delivery in due:
            if (
                self._config.allowed_provider_keys is not None
                and delivery.provider_key not in self._config.allowed_provider_keys
            ):
                continue
            claimed = self._repository.claim_delivery(
                delivery_id=delivery.delivery_id,
                lease_until=now + timedelta(seconds=self._config.lease_seconds),
                now=now,
            )
            if claimed is None:
                continue
            counts.claimed += 1
            if self._metrics is not None:
                self._metrics.on_delivery_claimed(provider_key=claimed.provider_key)

            provider = self._providers.get(claimed.provider_key)
            if provider is None:
                counts.provider_missing += 1
                self._apply_result(
                    delivery=claimed,
                    result=NotificationProviderResult(
                        status="dead_letter", error_code="provider_missing"
                    ),
                    now=now,
                    counts=counts,
                )
                continue

            try:
                result = provider.send(delivery=claimed)
            except Exception:  # noqa: BLE001
                result = NotificationProviderResult(
                    status="unknown", error_code="provider_exception"
                )
            self._apply_result(delivery=claimed, result=result, now=now, counts=counts)

        self._update_unknown_metric()
        return counts.freeze()

    def _apply_result(
        self,
        *,
        delivery: NotificationDelivery,
        result: NotificationProviderResult,
        now: datetime,
        counts: _MutableBatchCounts,
    ) -> None:
        attempt = NotificationDeliveryAttempt(
            attempt_id=uuid4(),
            delivery_id=delivery.delivery_id,
            provider_key=delivery.provider_key,
            started_at=now,
            finished_at=now,
            status=result.status,
            error_code=result.error_code,
            retry_after_seconds=result.retry_after_seconds,
            redacted_request_hash=result.redacted_request_hash,
            redacted_response_hash=result.redacted_response_hash,
        )
        self._repository.record_delivery_attempt(attempt=attempt)

        if result.status == "sent":
            updated = replace(
                delivery,
                status="sent",
                lease_until=None,
                last_error_code=None,
                provider_message_id=result.provider_message_id,
                sent_at=now,
            )
            counts.sent += 1
        elif result.status == "suppressed":
            updated = replace(
                delivery,
                status="suppressed",
                lease_until=None,
                last_error_code=result.error_code,
            )
            counts.suppressed += 1
        elif result.status == "retry":
            if delivery.attempt_count >= self._config.max_attempts:
                updated = replace(
                    delivery,
                    status="dead_letter",
                    lease_until=None,
                    last_error_code=result.error_code or "max_attempts_exhausted",
                )
                counts.dead_letter += 1
            else:
                retry_after = result.retry_after_seconds or self._config.retry_backoff_seconds
                updated = replace(
                    delivery,
                    status="retry",
                    lease_until=None,
                    next_attempt_at=now + timedelta(seconds=retry_after),
                    last_error_code=result.error_code,
                )
                counts.retry += 1
        elif result.status == "unknown":
            updated = replace(
                delivery,
                status="unknown",
                lease_until=None,
                last_error_code=result.error_code or "unknown_provider_state",
            )
            counts.unknown += 1
        else:
            updated = replace(
                delivery,
                status="dead_letter",
                lease_until=None,
                last_error_code=result.error_code,
            )
            counts.dead_letter += 1

        self._repository.update_delivery(delivery=updated)
        log.info(
            "notification delivery result delivery_id=%s route_id=%s provider=%s "
            "status=%s attempt_count=%s error_code=%s",
            updated.delivery_id,
            updated.route_id,
            updated.provider_key,
            updated.status,
            updated.attempt_count,
            updated.last_error_code or "none",
        )
        if self._metrics is not None:
            self._metrics.on_delivery_result(
                provider_key=delivery.provider_key, status=updated.status
            )
            self._metrics.observe_delivery_latency_seconds(
                provider_key=delivery.provider_key,
                seconds=max((now - delivery.created_at).total_seconds(), 0.0),
            )

    def _update_pending_age_metric(
        self, *, deliveries: tuple[NotificationDelivery, ...], now: datetime
    ) -> None:
        if self._metrics is None:
            return
        pending = [
            max((now - delivery.created_at).total_seconds(), 0.0)
            for delivery in deliveries
            if delivery.status in {"pending", "retry", "claimed"}
        ]
        self._metrics.set_pending_age_seconds(seconds=max(pending, default=0.0))

    def _update_unknown_metric(self) -> None:
        if self._metrics is None:
            return
        self._metrics.set_unknown_count(
            count=self._repository.count_deliveries_by_status(status="unknown")
        )


@dataclass(slots=True)
class _MutableBatchCounts:
    scanned: int = 0
    claimed: int = 0
    sent: int = 0
    retry: int = 0
    unknown: int = 0
    dead_letter: int = 0
    suppressed: int = 0
    provider_missing: int = 0

    def freeze(self) -> NotificationDispatchBatchResult:
        return NotificationDispatchBatchResult(
            scanned=self.scanned,
            claimed=self.claimed,
            sent=self.sent,
            retry=self.retry,
            unknown=self.unknown,
            dead_letter=self.dead_letter,
            suppressed=self.suppressed,
            provider_missing=self.provider_missing,
        )
