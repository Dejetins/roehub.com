from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from uuid import UUID, uuid4

from trading.contexts.live_execution.application.ports import (
    ExchangeExecutionConsumer,
    ExchangeExecutionProcessRepository,
    ExecutionDispatchUnavailableError,
    ExecutionIntentRepository,
    LiveExecutionClock,
)
from trading.contexts.live_execution.domain import (
    ExchangeExecutionDependencyHealth,
    ExchangeExecutionHealthSnapshot,
    ExchangeExecutionObservationStatus,
    ExchangeExecutionProcessHeartbeat,
    ExchangeExecutionProcessStatus,
    ExchangeExecutionRequestObservation,
)
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class ExchangeExecutionProcessConfig:
    service_id: str = "exchange-execution"
    adapter_mode: str = "disabled"
    request_stream: str = "execution.requests.v1"
    retry_stream: str = "execution.requests.retry.v1"
    dlq_stream: str = "execution.requests.dlq.v1"
    consumer_group: str = "exchange-execution.v1"
    consumer_name: str = "exchange-execution-local"
    consumer_enabled: bool = False
    read_count: int = 10
    block_ms: int = 100
    backpressure_max_stream_length: int = 10_000
    max_clock_drift_ms: float = 1_000.0
    rate_limit_per_second: float = 5.0
    rate_limit_burst: int = 10
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.service_id.strip() == "":
            raise ValueError("ExchangeExecutionProcessConfig.service_id must be non-empty")
        if self.adapter_mode != "disabled":
            raise ValueError("Stage 13 exchange-execution adapter_mode must be disabled")
        if self.consumer_name.strip() == "":
            raise ValueError("ExchangeExecutionProcessConfig.consumer_name must be non-empty")
        if self.read_count <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.read_count must be > 0")
        if self.block_ms < 0:
            raise ValueError("ExchangeExecutionProcessConfig.block_ms must be >= 0")
        if self.backpressure_max_stream_length <= 0:
            raise ValueError(
                "ExchangeExecutionProcessConfig.backpressure_max_stream_length must be > 0"
            )
        if self.max_clock_drift_ms <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.max_clock_drift_ms must be > 0")
        if self.rate_limit_per_second <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.rate_limit_per_second must be > 0")
        if self.rate_limit_burst <= 0:
            raise ValueError("ExchangeExecutionProcessConfig.rate_limit_burst must be > 0")


@dataclass(frozen=True, slots=True)
class ExchangeExecutionProcessStepResult:
    read_count: int
    observed_count: int
    quarantined_count: int
    acked_count: int
    reason: str


class ExchangeExecutionProcessService:
    def __init__(
        self,
        *,
        config: ExchangeExecutionProcessConfig,
        repository: ExchangeExecutionProcessRepository,
        intent_repository: ExecutionIntentRepository,
        consumer: ExchangeExecutionConsumer | None,
        clock: LiveExecutionClock,
        started_at: datetime | None = None,
        on_observation: Callable[[str, str], None] | None = None,
        on_dlq: Callable[[str], None] | None = None,
        on_ack: Callable[[str], None] | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeExecutionProcessService requires repository")
        if intent_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeExecutionProcessService requires intent_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("ExchangeExecutionProcessService requires clock")
        self._config = config
        self._repository = repository
        self._intent_repository = intent_repository
        self._consumer = consumer
        self._clock = clock
        self._started_at = started_at or clock.now()
        self._on_observation = on_observation
        self._on_dlq = on_dlq
        self._on_ack = on_ack

    def readiness(self) -> ExchangeExecutionHealthSnapshot:
        now = self._clock.now()
        dependencies: list[ExchangeExecutionDependencyHealth] = [
            ExchangeExecutionDependencyHealth(
                name="config",
                status="ready",
                reason="config_loaded",
                metadata={
                    "consumer_enabled": int(self._config.consumer_enabled),
                    "fail_fast": int(self._config.fail_fast),
                },
            ),
            ExchangeExecutionDependencyHealth(
                name="adapter",
                status="degraded",
                reason="adapter_disabled_stage13",
                metadata={"submit_enabled": 0},
            ),
            ExchangeExecutionDependencyHealth(
                name="rate_limit",
                status="ready",
                reason="rate_limit_guard_configured",
                metadata={
                    "per_second": self._config.rate_limit_per_second,
                    "burst": self._config.rate_limit_burst,
                },
            ),
        ]
        dependencies.extend(self._redis_dependencies())
        status, reason = _rollup_status(dependencies=dependencies)
        heartbeat = ExchangeExecutionProcessHeartbeat(
            service_id=self._config.service_id,
            status=status,
            status_reason=reason,
            adapter_mode="disabled",
            started_at=self._started_at,
            heartbeat_at=now,
            request_stream=self._config.request_stream,
            consumer_group=self._config.consumer_group,
            consumer_name=self._config.consumer_name,
            metadata={"dependency_count": len(dependencies)},
        )
        try:
            self._repository.record_heartbeat(heartbeat=heartbeat)
            dependencies.append(
                ExchangeExecutionDependencyHealth(
                    name="postgres",
                    status="ready",
                    reason="heartbeat_recorded",
                    metadata={},
                )
            )
        except Exception:  # noqa: BLE001
            dependencies.append(
                ExchangeExecutionDependencyHealth(
                    name="postgres",
                    status="not_ready" if self._config.fail_fast else "degraded",
                    reason="heartbeat_record_failed",
                    metadata={},
                )
            )
        status, reason = _rollup_status(dependencies=dependencies)
        return ExchangeExecutionHealthSnapshot(
            service_id=self._config.service_id,
            status=status,
            status_reason=reason,
            adapter_mode="disabled",
            checked_at=now,
            dependencies=tuple(dependencies),
        )

    def run_once(self) -> ExchangeExecutionProcessStepResult:
        if self._consumer is None or not self._config.consumer_enabled:
            return ExchangeExecutionProcessStepResult(
                read_count=0,
                observed_count=0,
                quarantined_count=0,
                acked_count=0,
                reason="consumer_disabled",
            )
        self._consumer.ensure_request_group()
        messages = self._consumer.read_new_requests(
            count=self._config.read_count,
            block_ms=self._config.block_ms,
        )
        observed_count = 0
        quarantined_count = 0
        acked_count = 0
        for message in messages:
            status, reason, intent_id = self._classify_message(payload=dict(message.payload))
            observation = ExchangeExecutionRequestObservation(
                observation_id=uuid4(),
                service_id=self._config.service_id,
                intent_id=intent_id,
                stream_name=message.stream_name,
                redis_message_id=message.message_id,
                status=status,
                status_reason=reason,
                adapter_mode="disabled",
                observed_at=self._clock.now(),
                metadata={"payload_fields": len(message.payload)},
            )
            self._repository.record_request_observation(observation=observation)
            self._record_observation(status=status, reason=reason)
            observed_count += 1
            if status == "quarantined":
                self._consumer.publish_dlq(message=message, reason=reason)
                self._record_dlq(reason=reason)
                self._consumer.ack_after_durable_state_change(
                    stream_name=message.stream_name,
                    message_id=message.message_id,
                )
                self._record_ack(reason=reason)
                quarantined_count += 1
                acked_count += 1
        return ExchangeExecutionProcessStepResult(
            read_count=len(messages),
            observed_count=observed_count,
            quarantined_count=quarantined_count,
            acked_count=acked_count,
            reason="adapter_disabled_no_submit",
        )

    def _redis_dependencies(self) -> tuple[ExchangeExecutionDependencyHealth, ...]:
        if self._consumer is None:
            return (
                ExchangeExecutionDependencyHealth(
                    name="redis",
                    status="degraded",
                    reason="redis_consumer_disabled",
                    metadata={"consumer_enabled": 0},
                ),
                ExchangeExecutionDependencyHealth(
                    name="backpressure",
                    status="degraded",
                    reason="backpressure_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="dlq",
                    status="degraded",
                    reason="dlq_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="clock_drift",
                    status="degraded",
                    reason="clock_drift_state_unavailable",
                    metadata={},
                ),
            )
        try:
            self._consumer.ensure_request_group()
            snapshot = self._consumer.health_snapshot()
        except ExecutionDispatchUnavailableError as error:
            return (
                ExchangeExecutionDependencyHealth(
                    name="redis",
                    status="not_ready" if self._config.fail_fast else "degraded",
                    reason=error.reason,
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="backpressure",
                    status="degraded",
                    reason="backpressure_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="dlq",
                    status="degraded",
                    reason="dlq_state_unavailable",
                    metadata={},
                ),
                ExchangeExecutionDependencyHealth(
                    name="clock_drift",
                    status="degraded",
                    reason="clock_drift_state_unavailable",
                    metadata={},
                ),
            )
        backpressure_status = (
            "degraded"
            if snapshot.request_stream_length >= self._config.backpressure_max_stream_length
            else "ready"
        )
        clock_status = (
            "degraded"
            if abs(snapshot.clock_drift_ms) > self._config.max_clock_drift_ms
            else "ready"
        )
        return (
            ExchangeExecutionDependencyHealth(
                name="redis",
                status="ready",
                reason="redis_streams_observed",
                metadata={
                    "request_stream_length": snapshot.request_stream_length,
                    "pending_count": snapshot.pending_count,
                },
            ),
            ExchangeExecutionDependencyHealth(
                name="backpressure",
                status=backpressure_status,
                reason=(
                    "dispatch_backpressure"
                    if backpressure_status == "degraded"
                    else "backpressure_within_limit"
                ),
                metadata={
                    "request_stream_length": snapshot.request_stream_length,
                    "max_stream_length": self._config.backpressure_max_stream_length,
                },
            ),
            ExchangeExecutionDependencyHealth(
                name="dlq",
                status="ready",
                reason="dlq_stream_observed",
                metadata={"dlq_stream_length": snapshot.dlq_stream_length},
            ),
            ExchangeExecutionDependencyHealth(
                name="clock_drift",
                status=clock_status,
                reason=(
                    "clock_drift_exceeds_limit"
                    if clock_status == "degraded"
                    else "clock_drift_within_limit"
                ),
                metadata={
                    "clock_drift_ms": round(snapshot.clock_drift_ms, 3),
                    "max_clock_drift_ms": self._config.max_clock_drift_ms,
                },
            ),
        )

    def _classify_message(
        self, *, payload: dict[str, str]
    ) -> tuple[ExchangeExecutionObservationStatus, str, UUID | None]:
        try:
            intent_id = UUID(payload["intent_id"])
            owner_user_id = UserId.from_string(payload["owner_user_id"])
        except (KeyError, ValueError):
            return "quarantined", "dispatch_payload_invalid_identity", None
        intent = self._intent_repository.get_intent_by_id(
            owner_user_id=owner_user_id,
            intent_id=intent_id,
        )
        if intent is None:
            return "quarantined", "intent_not_found", intent_id
        if intent.status != "dispatched" or intent.risk_status != "accepted":
            return "quarantined", "intent_not_dispatchable", intent_id
        return "adapter_disabled", "adapter_disabled_stage13", intent_id

    def _record_observation(self, *, status: str, reason: str) -> None:
        if self._on_observation is not None:
            self._on_observation(status, reason)

    def _record_dlq(self, *, reason: str) -> None:
        if self._on_dlq is not None:
            self._on_dlq(reason)

    def _record_ack(self, *, reason: str) -> None:
        if self._on_ack is not None:
            self._on_ack(reason)


def _rollup_status(
    *, dependencies: list[ExchangeExecutionDependencyHealth]
) -> tuple[ExchangeExecutionProcessStatus, str]:
    if any(item.status == "not_ready" for item in dependencies):
        return "not_ready", "dependency_not_ready"
    if any(item.status == "degraded" for item in dependencies):
        reasons = [item.reason for item in dependencies if item.status == "degraded"]
        return "degraded", reasons[0] if reasons else "dependency_degraded"
    return "ready", "all_dependencies_ready"
