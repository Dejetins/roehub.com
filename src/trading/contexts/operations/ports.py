"""Ports owned by the operations application boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ContextManager, Protocol
from uuid import UUID

from .contracts import OperationRequest, OperationResult, OperationState


class ControlBackendPort(Protocol):
    def execute(self, request: OperationRequest) -> OperationResult: ...

    def reconcile(self, request: OperationRequest) -> OperationResult: ...


class OperationJournalPort(Protocol):
    def exclusive(self) -> ContextManager[None]: ...

    def append(
        self,
        *,
        request: OperationRequest,
        state: OperationState,
        detail_code: str,
        active_services: Sequence[str] = (),
    ) -> Mapping[str, object]: ...

    def request(self, operation_id: UUID) -> OperationRequest | None: ...

    def latest(self, operation_id: UUID) -> Mapping[str, object] | None: ...

    def entries(self, *, after_sequence: int = 0) -> tuple[Mapping[str, object], ...]: ...


__all__ = ["ControlBackendPort", "OperationJournalPort"]
