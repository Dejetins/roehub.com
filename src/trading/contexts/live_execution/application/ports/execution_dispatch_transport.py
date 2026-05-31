from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trading.contexts.live_execution.domain import ExecutionIntent


class ExecutionDispatchUnavailableError(RuntimeError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ExecutionDispatchPoisonMessageError(RuntimeError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ExecutionDispatchPublishResult:
    stream_name: str
    message_id: str


class ExecutionDispatchTransport(Protocol):
    def ensure_request_group(self) -> None: ...

    def request_stream_length(self) -> int: ...

    def publish_request(
        self, *, intent: ExecutionIntent, attempt_count: int
    ) -> ExecutionDispatchPublishResult: ...

    def publish_retry(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult: ...

    def publish_dlq(
        self, *, intent: ExecutionIntent, reason: str, attempt_count: int
    ) -> ExecutionDispatchPublishResult: ...

    def ack_after_durable_state_change(self, *, stream_name: str, message_id: str) -> None: ...
