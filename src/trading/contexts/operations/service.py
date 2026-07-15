"""Idempotent application service for privileged host operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from uuid import UUID

from .contracts import (
    ControlOperationError,
    OperationAction,
    OperationRequest,
    OperationResult,
    OperationState,
)
from .ports import ControlBackendPort, OperationJournalPort

_TERMINAL = {
    OperationState.SUCCEEDED,
    OperationState.FAILED,
    OperationState.REJECTED,
}


class ControlOperationService:
    """Linearize effects through an append-only journal and explicit reconciliation."""

    def __init__(
        self,
        *,
        backend: ControlBackendPort,
        journal: OperationJournalPort,
        after_effect: Callable[[OperationRequest, OperationResult], None] | None = None,
        before_lock: Callable[[OperationRequest], None] | None = None,
    ) -> None:
        self._backend = backend
        self._journal = journal
        self._after_effect = after_effect
        self._before_lock = before_lock
        self._process_lock = RLock()

    def submit(self, request: OperationRequest) -> OperationResult:
        if self._before_lock is not None:
            self._before_lock(request)
        with self._process_lock, self._journal.exclusive():
            return self._submit_locked(request)

    def _submit_locked(self, request: OperationRequest) -> OperationResult:
        recorded = self._journal.request(request.operation_id)
        if recorded is not None:
            if recorded.request_digest != request.request_digest:
                raise ControlOperationError(code="operation.idempotency_conflict")
            latest = self._journal.latest(request.operation_id)
            if latest is None:
                raise ControlOperationError(code="operation.journal_corrupt")
            return self._result_from_entry(latest)

        self._journal.append(
            request=request,
            state=OperationState.ACCEPTED,
            detail_code="operation.accepted",
        )
        self._journal.append(
            request=request,
            state=OperationState.RUNNING,
            detail_code="operation.running",
        )
        try:
            result = self._backend.execute(request)
            if self._after_effect is not None:
                self._after_effect(request, result)
        except ControlOperationError as error:
            state = (
                OperationState.UNKNOWN
                if error.code == "operation.effect_unknown"
                else OperationState.FAILED
            )
            entry = self._journal.append(
                request=request,
                state=state,
                detail_code=error.code,
            )
            return self._result_from_entry(entry)
        entry = self._journal.append(
            request=request,
            state=result.state,
            detail_code=result.detail_code,
            active_services=result.active_services,
        )
        return self._result_from_entry(entry)

    def get(self, operation_id: UUID) -> OperationResult:
        latest = self._journal.latest(operation_id)
        if latest is None:
            raise ControlOperationError(code="operation.not_found")
        return self._result_from_entry(latest)

    def reconcile(self, operation_id: UUID) -> OperationResult:
        with self._process_lock, self._journal.exclusive():
            return self._reconcile_locked(operation_id)

    def _reconcile_locked(self, operation_id: UUID) -> OperationResult:
        request = self._journal.request(operation_id)
        latest = self._journal.latest(operation_id)
        if request is None or latest is None:
            raise ControlOperationError(code="operation.not_found")
        state = OperationState(str(latest["state"]))
        if state in _TERMINAL:
            return self._result_from_entry(latest)
        unknown = self._journal.append(
            request=request,
            state=OperationState.UNKNOWN,
            detail_code="operation.reconciliation_required",
        )
        try:
            result = self._backend.reconcile(request)
        except ControlOperationError:
            return self._result_from_entry(unknown)
        entry = self._journal.append(
            request=request,
            state=result.state,
            detail_code=result.detail_code,
            active_services=result.active_services,
        )
        return self._result_from_entry(entry)

    @staticmethod
    def _result_from_entry(entry: Mapping[str, object]) -> OperationResult:
        services = entry.get("active_services", [])
        return OperationResult(
            operation_id=UUID(str(entry["operation_id"])),
            action=OperationAction(str(entry["action"])),
            profile=str(entry["profile"]),  # type: ignore[arg-type]
            state=OperationState(str(entry["state"])),
            detail_code=str(entry["detail_code"]),
            active_services=tuple(str(item) for item in services),  # type: ignore[arg-type]
            journal_sequence=int(str(entry["sequence"])),
        )


__all__ = ["ControlOperationService"]
