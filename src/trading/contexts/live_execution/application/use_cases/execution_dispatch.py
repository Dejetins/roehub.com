from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from trading.contexts.live_execution.application.ports import (
    ExecutionDispatchPoisonMessageError,
    ExecutionDispatchTransport,
    ExecutionDispatchUnavailableError,
    ExecutionIntentRepository,
    LiveExecutionClock,
)
from trading.contexts.live_execution.domain import ExecutionIntent


@dataclass(frozen=True, slots=True)
class ExecutionDispatchConfig:
    retry_budget: int = 3
    backpressure_max_stream_length: int = 10_000

    def __post_init__(self) -> None:
        if self.retry_budget <= 0:
            raise ValueError("ExecutionDispatchConfig.retry_budget must be > 0")
        if self.backpressure_max_stream_length <= 0:
            raise ValueError(
                "ExecutionDispatchConfig.backpressure_max_stream_length must be > 0"
            )


@dataclass(frozen=True, slots=True)
class ExecutionDispatchResult:
    intent: ExecutionIntent
    result: str
    reason: str


class ExecutionDispatchService:
    def __init__(
        self,
        *,
        repository: ExecutionIntentRepository,
        transport: ExecutionDispatchTransport,
        clock: LiveExecutionClock,
        config: ExecutionDispatchConfig | None = None,
        on_dispatch: Callable[[str, str], None] | None = None,
        on_retry: Callable[[str], None] | None = None,
        on_dlq: Callable[[str], None] | None = None,
        on_backpressure: Callable[[str], None] | None = None,
        on_redis_error: Callable[[str], None] | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionDispatchService requires repository")
        if transport is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionDispatchService requires transport")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ExecutionDispatchService requires clock")
        self._repository = repository
        self._transport = transport
        self._clock = clock
        self._config = config if config is not None else ExecutionDispatchConfig()
        self._on_dispatch = on_dispatch
        self._on_retry = on_retry
        self._on_dlq = on_dlq
        self._on_backpressure = on_backpressure
        self._on_redis_error = on_redis_error

    def dispatch_intent(self, *, intent: ExecutionIntent) -> ExecutionDispatchResult:
        if intent.risk_status != "accepted":
            return ExecutionDispatchResult(
                intent=intent,
                result="skipped",
                reason="risk_not_accepted",
            )
        if intent.status == "dispatched":
            return ExecutionDispatchResult(
                intent=intent,
                result="duplicate",
                reason="already_dispatched",
            )
        if intent.status in {"rejected", "recorded", "quarantined", "dispatching"}:
            return ExecutionDispatchResult(
                intent=intent,
                result="skipped",
                reason=f"status_{intent.status}_not_dispatchable",
            )
        if intent.status not in {"accepted", "retry"}:
            return ExecutionDispatchResult(
                intent=intent,
                result="skipped",
                reason="unknown_dispatch_state",
            )

        if not self._has_retry_budget(intent=intent):
            quarantined = self._quarantine(
                intent=intent,
                reason="retry_budget_exhausted",
                stream_name=None,
            )
            return ExecutionDispatchResult(
                intent=quarantined,
                result="dlq",
                reason="retry_budget_exhausted",
            )

        try:
            self._transport.ensure_request_group()
            stream_length = self._transport.request_stream_length()
            if stream_length >= self._config.backpressure_max_stream_length:
                retry_intent = self._mark_retry(intent=intent, reason="dispatch_backpressure")
                self._publish_retry_marker(
                    intent=retry_intent,
                    reason="dispatch_backpressure",
                )
                self._record_backpressure(reason="dispatch_backpressure")
                return ExecutionDispatchResult(
                    intent=retry_intent,
                    result="retry",
                    reason="dispatch_backpressure",
                )

            dispatching = self._repository.claim_intent_for_dispatch(
                intent_id=intent.intent_id,
                now=self._clock.now(),
                retry_budget=self._config.retry_budget,
            )
            if dispatching is None:
                current = self._repository.get_intent_by_id(
                    owner_user_id=intent.owner_user_id,
                    intent_id=intent.intent_id,
                )
                return ExecutionDispatchResult(
                    intent=current or intent,
                    result="duplicate",
                    reason="dispatch_claim_unavailable",
                )
            published = self._transport.publish_request(
                intent=dispatching,
                attempt_count=dispatching.dispatch_attempt_count,
            )
            dispatched = self._repository.mark_intent_dispatched(
                intent_id=dispatching.intent_id,
                stream_name=published.stream_name,
                redis_message_id=published.message_id,
                now=self._clock.now(),
            )
            final_intent = dispatched or dispatching
            self._record_dispatch(result="dispatched", reason="redis_xadd_ok")
            return ExecutionDispatchResult(
                intent=final_intent,
                result="dispatched",
                reason="redis_xadd_ok",
            )
        except ExecutionDispatchPoisonMessageError as error:
            quarantined = self._quarantine(
                intent=intent,
                reason=error.reason,
                stream_name=None,
            )
            self._publish_dlq_marker(intent=quarantined, reason=error.reason)
            return ExecutionDispatchResult(
                intent=quarantined,
                result="dlq",
                reason=error.reason,
            )
        except ExecutionDispatchUnavailableError as error:
            retry_intent = self._mark_retry(intent=intent, reason=error.reason)
            self._record_redis_error(reason=error.reason)
            return ExecutionDispatchResult(
                intent=retry_intent,
                result="retry",
                reason=error.reason,
            )

    def _has_retry_budget(self, *, intent: ExecutionIntent) -> bool:
        return intent.dispatch_attempt_count < self._config.retry_budget

    def _mark_retry(self, *, intent: ExecutionIntent, reason: str) -> ExecutionIntent:
        retry_intent = self._repository.mark_intent_dispatch_retry(
            intent_id=intent.intent_id,
            reason=reason,
            now=self._clock.now(),
        )
        final_intent = retry_intent or intent
        self._record_retry(reason=reason)
        return final_intent

    def _quarantine(
        self, *, intent: ExecutionIntent, reason: str, stream_name: str | None
    ) -> ExecutionIntent:
        quarantined = self._repository.mark_intent_quarantined(
            intent_id=intent.intent_id,
            reason=reason,
            stream_name=stream_name,
            now=self._clock.now(),
        )
        final_intent = quarantined or intent
        self._record_dlq(reason=reason)
        return final_intent

    def _publish_retry_marker(self, *, intent: ExecutionIntent, reason: str) -> None:
        try:
            self._transport.publish_retry(
                intent=intent,
                reason=reason,
                attempt_count=intent.dispatch_attempt_count,
            )
        except ExecutionDispatchUnavailableError as error:
            self._record_redis_error(reason=error.reason)

    def _publish_dlq_marker(self, *, intent: ExecutionIntent, reason: str) -> None:
        try:
            self._transport.publish_dlq(
                intent=intent,
                reason=reason,
                attempt_count=intent.dispatch_attempt_count,
            )
        except ExecutionDispatchUnavailableError as error:
            self._record_redis_error(reason=error.reason)

    def _record_dispatch(self, *, result: str, reason: str) -> None:
        if self._on_dispatch is not None:
            self._on_dispatch(result, reason)

    def _record_retry(self, *, reason: str) -> None:
        if self._on_retry is not None:
            self._on_retry(reason)

    def _record_dlq(self, *, reason: str) -> None:
        if self._on_dlq is not None:
            self._on_dlq(reason)

    def _record_backpressure(self, *, reason: str) -> None:
        if self._on_backpressure is not None:
            self._on_backpressure(reason)

    def _record_redis_error(self, *, reason: str) -> None:
        if self._on_redis_error is not None:
            self._on_redis_error(reason)
