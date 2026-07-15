from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
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
    def on_delivery_claimed(
        self, *, provider_key: str, provider_instance_id: str
    ) -> None: ...

    def on_delivery_result(
        self,
        *,
        provider_key: str,
        provider_instance_id: str,
        category: str,
        status: str,
    ) -> None: ...

    def observe_delivery_latency_seconds(
        self, *, provider_key: str, provider_instance_id: str, seconds: float
    ) -> None: ...

    def set_pending_age_seconds(self, *, seconds: float) -> None: ...

    def set_unknown_count(self, *, count: int) -> None: ...


@dataclass(frozen=True, slots=True)
class NotificationDispatcherConfig:
    batch_size: int = 100
    lease_seconds: int = 30
    retry_backoff_seconds: int = 60
    max_retry_backoff_seconds: int = 900
    retry_jitter_ratio: float = 0.2
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
        if self.max_retry_backoff_seconds < self.retry_backoff_seconds:
            raise ValueError(
                "NotificationDispatcherConfig.max_retry_backoff_seconds must be >= base backoff"
            )
        if not 0 <= self.retry_jitter_ratio <= 0.5:
            raise ValueError("NotificationDispatcherConfig.retry_jitter_ratio must be 0..0.5")
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
        self._providers = {
            provider.provider_instance_id: provider for provider in providers
        }
        if len(self._providers) != len(providers):
            raise ValueError("provider instance identifiers must be unique")
        self._clock = clock
        self._config = config or NotificationDispatcherConfig()
        self._metrics = metrics

    def drain_once(self) -> NotificationDispatchBatchResult:
        now = self._clock.now()
        recovered_unknown = self._repository.recover_expired_claims(now=now)
        due = self._repository.list_due_deliveries(
            now=now, limit=self._config.batch_size
        )
        counts = _MutableBatchCounts(scanned=len(due), unknown=recovered_unknown)
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
                self._metrics.on_delivery_claimed(
                    provider_key=claimed.provider_key,
                    provider_instance_id=str(claimed.provider_instance_id),
                )

            provider = self._providers.get(claimed.provider_instance_id)
            if provider is None:
                counts.provider_missing += 1
                self._apply_result(
                    delivery=claimed,
                    result=NotificationProviderResult(
                        status="dead_letter", error_code="provider_disabled"
                    ),
                    now=now,
                    counts=counts,
                )
                continue

            if (
                provider.provider_key != claimed.provider_key
                or (
                    provider.organization_id is not None
                    and provider.organization_id != claimed.organization_id
                )
            ):
                self._apply_result(
                    delivery=claimed,
                    result=NotificationProviderResult(
                        status="dead_letter", error_code="provider_scope_mismatch"
                    ),
                    now=now,
                    counts=counts,
                )
                continue

            try:
                result = provider.send(delivery=claimed)
            except (asyncio.CancelledError, KeyboardInterrupt):
                self._apply_result(
                    delivery=claimed,
                    result=NotificationProviderResult(
                        status="unknown", error_code="provider_cancelled"
                    ),
                    now=now,
                    counts=counts,
                )
                self._update_unknown_metric()
                raise
            except Exception:  # noqa: BLE001
                result = NotificationProviderResult(
                    status="unknown", error_code="provider_transport_error"
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
            organization_id=delivery.organization_id,
            provider_instance_id=delivery.provider_instance_id,
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
                retry_after = _bounded_retry_delay_seconds(
                    delivery=delivery,
                    requested=result.retry_after_seconds,
                    config=self._config,
                )
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
                provider_key=delivery.provider_key,
                provider_instance_id=str(delivery.provider_instance_id),
                category=str(delivery.rendered_payload_json.get("category", "unknown")),
                status=updated.status,
            )
            self._metrics.observe_delivery_latency_seconds(
                provider_key=delivery.provider_key,
                provider_instance_id=str(delivery.provider_instance_id),
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


def _bounded_retry_delay_seconds(
    *,
    delivery: NotificationDelivery,
    requested: int | None,
    config: NotificationDispatcherConfig,
) -> int:
    if requested is not None:
        return max(1, min(requested, config.max_retry_backoff_seconds))
    exponent = max(delivery.attempt_count - 1, 0)
    base = min(
        config.retry_backoff_seconds * (2**exponent),
        config.max_retry_backoff_seconds,
    )
    if config.retry_jitter_ratio == 0:
        return base
    digest = sha256(
        f"{delivery.delivery_id}:{delivery.attempt_count}".encode()
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
    factor = 1 - config.retry_jitter_ratio + (2 * config.retry_jitter_ratio * unit)
    return max(1, min(round(base * factor), config.max_retry_backoff_seconds))
