from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from trading.contexts.operations import (
    ControlAgentRequest,
    ControlOperationError,
    ControlOperationService,
    OperationAction,
    OperationRequest,
    OperationResult,
    OperationState,
)
from trading.contexts.operations.adapters import AppendOnlyOperationJournal


class _Backend:
    def __init__(self) -> None:
        self.effects = 0
        self.running = False

    def execute(self, request: OperationRequest) -> OperationResult:
        self.effects += 1
        self.running = request.action != OperationAction.STOP
        return self._result(request, OperationState.SUCCEEDED, "topology.changed")

    def reconcile(self, request: OperationRequest) -> OperationResult:
        complete = self.running if request.action != OperationAction.STOP else not self.running
        return self._result(
            request,
            OperationState.SUCCEEDED if complete else OperationState.UNKNOWN,
            "operation.reconciled" if complete else "operation.effect_unknown",
        )

    @staticmethod
    def _result(
        request: OperationRequest, state: OperationState, detail: str
    ) -> OperationResult:
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=state,
            detail_code=detail,
            active_services=("api",) if state == OperationState.SUCCEEDED else (),
        )


def _request(*, action: OperationAction = OperationAction.RECOVER) -> OperationRequest:
    return OperationRequest(
        operation_id=uuid4(),
        action=action,
        profile="base",
        subject_id="installation-state"
        if action in {OperationAction.BACKUP, OperationAction.RESTORE}
        else None,
    )


def test_request_forbids_shell_and_runtime_overrides() -> None:
    payload = _request().model_dump(mode="json", by_alias=True)
    for forbidden in ("command", "image", "mounts", "environment"):
        with pytest.raises(ValidationError):
            OperationRequest.model_validate({**payload, forbidden: ["sh", "-c", "id"]})


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema": "io.roehub.control-operation/v1alpha1",
            "operation_id": "00000000-0000-4000-8000-000000000001",
            "action": "rollback",
            "profile": "base",
        },
        {
            "schema": "io.roehub.control-operation/v1alpha1",
            "operation_id": "00000000-0000-4000-8000-000000000002",
            "action": "plugin.install",
            "profile": "base",
            "subject_id": "plugin.example",
        },
        {
            "schema": "io.roehub.control-operation/v1alpha1",
            "operation_id": "00000000-0000-4000-8000-000000000003",
            "action": "backup",
            "profile": "base",
        },
    ],
)
def test_operation_json_schema_matches_pydantic_conditions(
    payload: dict[str, Any],
) -> None:
    root = Path(__file__).resolve().parents[4]
    schema = json.loads(
        (root / "schemas/operations/control-operation.v1alpha1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert list(Draft202012Validator(schema).iter_errors(payload))
    with pytest.raises(ValidationError):
        OperationRequest.model_validate(payload)


def test_transport_json_schema_matches_pydantic_method_conditions() -> None:
    root = Path(__file__).resolve().parents[4]
    operation_schema = json.loads(
        (root / "schemas/operations/control-operation.v1alpha1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (root / "schemas/operations/control-agent-request.v1alpha1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    schema["properties"]["operation"] = operation_schema
    invalid = {
        "schema": "io.roehub.control-agent-request/v1alpha1",
        "identity": "api",
        "credential": "0" * 32,
        "method": "get",
    }

    assert list(Draft202012Validator(schema).iter_errors(invalid))
    with pytest.raises(ValidationError):
        ControlAgentRequest.model_validate(invalid)


def test_operation_id_is_idempotent_and_conflicts_fail_closed(tmp_path: Path) -> None:
    backend = _Backend()
    service = ControlOperationService(
        backend=backend,
        journal=AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl"),
    )
    request = _request()

    first = service.submit(request)
    replay = service.submit(request)

    assert first.state == replay.state == OperationState.SUCCEEDED
    assert backend.effects == 1
    with pytest.raises(ControlOperationError, match="operation.idempotency_conflict"):
        service.submit(
            OperationRequest(
                operation_id=request.operation_id,
                action=OperationAction.STOP,
                profile="base",
            )
        )


def test_crash_after_effect_requires_typed_reconciliation(tmp_path: Path) -> None:
    backend = _Backend()
    journal = AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl")
    request = _request()

    crashing = ControlOperationService(
        backend=backend,
        journal=journal,
        after_effect=lambda _request, _result: (_ for _ in ()).throw(SystemExit(70)),
    )
    with pytest.raises(SystemExit):
        crashing.submit(request)

    assert journal.latest(request.operation_id)["state"] == "running"  # type: ignore[index]
    recovered = ControlOperationService(backend=backend, journal=journal).reconcile(
        request.operation_id
    )
    assert recovered.state == OperationState.SUCCEEDED
    assert recovered.detail_code == "operation.reconciled"
    assert backend.effects == 1


def test_journal_detects_hash_chain_tampering(tmp_path: Path) -> None:
    path = tmp_path / "operations.jsonl"
    journal = AppendOnlyOperationJournal(path=path)
    request = _request()
    journal.append(
        request=request,
        state=OperationState.ACCEPTED,
        detail_code="operation.accepted",
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    row["detail_code"] = "operation.tampered"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ControlOperationError, match="operation.journal_hash_mismatch"):
        journal.entries()


def test_journal_recovers_only_uncommitted_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "operations.jsonl"
    journal = AppendOnlyOperationJournal(path=path)
    request = _request()
    journal.append(
        request=request,
        state=OperationState.ACCEPTED,
        detail_code="operation.accepted",
    )
    committed = path.read_bytes()
    path.write_bytes(committed + b'{"partial":')

    recovered = AppendOnlyOperationJournal(path=path)

    assert recovered.latest(request.operation_id)["state"] == "accepted"  # type: ignore[index]
    assert path.read_bytes() == committed
    evidence = tuple(tmp_path.glob("operations.jsonl.torn-*.bin"))
    assert len(evidence) == 1
    assert evidence[0].read_bytes() == b'{"partial":'


def test_failed_effect_is_terminal_and_not_retried(tmp_path: Path) -> None:
    class _FailingBackend(_Backend):
        def execute(self, request: OperationRequest) -> OperationResult:
            self.effects += 1
            raise ControlOperationError(code="operation.handler_unavailable")

    backend = _FailingBackend()
    service = ControlOperationService(
        backend=backend,
        journal=AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl"),
    )
    request = _request(action=OperationAction.BACKUP)

    assert service.submit(request).state == OperationState.FAILED
    assert service.submit(request).state == OperationState.FAILED
    assert backend.effects == 1


def test_ambiguous_effect_is_unknown_and_replay_does_not_retry(tmp_path: Path) -> None:
    class _UnknownBackend(_Backend):
        def execute(self, request: OperationRequest) -> OperationResult:
            self.effects += 1
            raise ControlOperationError(code="operation.effect_unknown")

    backend = _UnknownBackend()
    service = ControlOperationService(
        backend=backend,
        journal=AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl"),
    )
    request = _request()

    assert service.submit(request).state == OperationState.UNKNOWN
    assert service.submit(request).state == OperationState.UNKNOWN
    assert backend.effects == 1


def test_concurrent_replay_linearizes_one_effect(tmp_path: Path) -> None:
    backend = _Backend()
    service = ControlOperationService(
        backend=backend,
        journal=AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl"),
    )
    request = _request()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _index: service.submit(request), range(16)))

    assert all(result.state == OperationState.SUCCEEDED for result in results)
    assert backend.effects == 1


def test_before_lock_interrupt_can_cancel_a_running_serialized_effect(
    tmp_path: Path,
) -> None:
    started = Event()
    interrupted = Event()

    class _InterruptibleBackend(_Backend):
        def execute(self, request: OperationRequest) -> OperationResult:
            if request.action is OperationAction.BACKUP:
                started.set()
                assert interrupted.wait(timeout=5)
                raise ControlOperationError(code="backup.cancelled")
            return super().execute(request)

    backend = _InterruptibleBackend()

    def before_lock(request: OperationRequest) -> None:
        if request.action is OperationAction.BACKUP_CANCEL:
            interrupted.set()

    service = ControlOperationService(
        backend=backend,
        journal=AppendOnlyOperationJournal(path=tmp_path / "operations.jsonl"),
        before_lock=before_lock,
    )
    backup = OperationRequest(
        operation_id=uuid4(),
        action=OperationAction.BACKUP,
        profile="base",
        subject_id="backup-001",
    )
    cancellation = OperationRequest(
        operation_id=uuid4(),
        action=OperationAction.BACKUP_CANCEL,
        profile="base",
        subject_id=str(backup.operation_id),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        backup_future = executor.submit(service.submit, backup)
        assert started.wait(timeout=5)
        cancellation_future = executor.submit(service.submit, cancellation)
        assert backup_future.result(timeout=5).detail_code == "backup.cancelled"
        assert cancellation_future.result(timeout=5).state is OperationState.SUCCEEDED
