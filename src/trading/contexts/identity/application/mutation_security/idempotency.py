from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from threading import RLock
from typing import Protocol

from trading.contexts.identity.application.authorization import CapabilityId
from trading.shared_kernel.primitives import OrganizationId, UserId


class IdempotencyRecordState(StrEnum):
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class IdempotencyBeginStatus(StrEnum):
    NEW = "new"
    REPLAY_TERMINAL = "replay_terminal"
    CONFLICT = "conflict"
    IN_PROGRESS = "in_progress"
    RECONCILIATION_REQUIRED = "reconciliation_required"


@dataclass(frozen=True, slots=True)
class IdempotencyIdentity:
    actor_user_id: UserId
    organization_id: OrganizationId | None
    capability: CapabilityId
    action: str
    resource_reference_hash: str | None
    key_hash: str


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    identity: IdempotencyIdentity
    payload_hash: str
    state: IdempotencyRecordState
    terminal_reference: str | None = None


@dataclass(frozen=True, slots=True)
class IdempotencyBeginResult:
    status: IdempotencyBeginStatus
    record: IdempotencyRecord


class MutationIdempotencyStore(Protocol):
    def begin(
        self, *, identity: IdempotencyIdentity, payload_hash: str
    ) -> IdempotencyBeginResult: ...

    def finish(
        self,
        *,
        identity: IdempotencyIdentity,
        payload_hash: str,
        state: IdempotencyRecordState,
        terminal_reference: str | None,
    ) -> IdempotencyRecord: ...


class InMemoryMutationIdempotencyStore:
    """Thread-safe proof adapter; route tickets must select their durable store."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[IdempotencyIdentity, IdempotencyRecord] = {}

    def begin(self, *, identity: IdempotencyIdentity, payload_hash: str) -> IdempotencyBeginResult:
        with self._lock:
            existing = self._records.get(identity)
            if existing is None:
                record = IdempotencyRecord(
                    identity=identity,
                    payload_hash=payload_hash,
                    state=IdempotencyRecordState.PROCESSING,
                )
                self._records[identity] = record
                return IdempotencyBeginResult(status=IdempotencyBeginStatus.NEW, record=record)
            if existing.payload_hash != payload_hash:
                return IdempotencyBeginResult(
                    status=IdempotencyBeginStatus.CONFLICT,
                    record=existing,
                )
            if existing.state in {
                IdempotencyRecordState.SUCCEEDED,
                IdempotencyRecordState.FAILED,
            }:
                status = IdempotencyBeginStatus.REPLAY_TERMINAL
            elif existing.state is IdempotencyRecordState.UNKNOWN:
                status = IdempotencyBeginStatus.RECONCILIATION_REQUIRED
            else:
                status = IdempotencyBeginStatus.IN_PROGRESS
            return IdempotencyBeginResult(status=status, record=existing)

    def finish(
        self,
        *,
        identity: IdempotencyIdentity,
        payload_hash: str,
        state: IdempotencyRecordState,
        terminal_reference: str | None,
    ) -> IdempotencyRecord:
        if state is IdempotencyRecordState.PROCESSING:
            raise ValueError("finish requires a terminal or unknown idempotency state")
        with self._lock:
            current = self._records.get(identity)
            if current is None or current.payload_hash != payload_hash:
                raise ValueError("idempotency reservation is missing or conflicts")
            if current.state is not IdempotencyRecordState.PROCESSING:
                if current.state is state and current.terminal_reference == terminal_reference:
                    return current
                raise ValueError("idempotency reservation is already finalized")
            updated = replace(
                current,
                state=state,
                terminal_reference=terminal_reference,
            )
            self._records[identity] = updated
            return updated
