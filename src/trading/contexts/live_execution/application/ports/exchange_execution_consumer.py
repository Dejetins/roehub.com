from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from trading.contexts.live_execution.application.ports.execution_dispatch_transport import (
    ExecutionDispatchPublishResult,
)


@dataclass(frozen=True, slots=True)
class ExchangeExecutionRedisMessage:
    stream_name: str
    message_id: str
    payload: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ExchangeExecutionRedisHealth:
    request_stream_length: int
    retry_stream_length: int
    dlq_stream_length: int
    pending_count: int
    clock_drift_ms: float


class ExchangeExecutionConsumer(Protocol):
    def ensure_request_group(self) -> None: ...

    def health_snapshot(self) -> ExchangeExecutionRedisHealth: ...

    def read_new_requests(
        self, *, count: int, block_ms: int
    ) -> tuple[ExchangeExecutionRedisMessage, ...]: ...

    def read_pending_requests(self, *, count: int) -> tuple[ExchangeExecutionRedisMessage, ...]: ...

    def publish_dlq(
        self, *, message: ExchangeExecutionRedisMessage, reason: str
    ) -> ExecutionDispatchPublishResult: ...

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None: ...
