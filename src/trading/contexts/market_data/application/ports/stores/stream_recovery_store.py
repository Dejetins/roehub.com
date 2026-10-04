"""Durable metadata only: accepted ranges survive process or storage failure."""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Protocol
from uuid import UUID

from trading.contexts.market_data.application.dto import RestFillTask


def recovery_key(task: RestFillTask) -> str:
    value = (
        f"{task.instrument_id.market_id.value}:{task.instrument_id.symbol}:"
        f"{task.time_range.start.value.isoformat()}:{task.time_range.end.value.isoformat()}:"
        f"{task.reason}"
    )
    return sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class RecoveryEntry:
    task: RestFillTask
    token: UUID
    attempt: int = 0
    next_retry_at: datetime | None = None
    error_code: str | None = None
    key: str | None = None

    @property
    def task_key(self) -> str:
        return self.key or recovery_key(self.task)


class StreamRecoveryStore(Protocol):
    def remember(self, task: RestFillTask) -> RecoveryEntry: ...
    def pending(self) -> list[RecoveryEntry]: ...
    def advance(self, entry: RecoveryEntry, *, start_at: datetime) -> None: ...
    def complete(self, entry: RecoveryEntry) -> None: ...
    def complete_many(self, entries: list[RecoveryEntry]) -> None: ...
    def retry(self, entry: RecoveryEntry, *, next_retry_at: datetime, code: str) -> None: ...
    def fail_many(self, entries: list[RecoveryEntry], *, code: str) -> None: ...
    def fail(self, entry: RecoveryEntry, *, code: str) -> None: ...
