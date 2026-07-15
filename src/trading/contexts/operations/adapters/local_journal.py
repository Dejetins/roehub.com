"""Hash-chained, fsync-backed emergency journal independent of PostgreSQL."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from ..contracts import ControlOperationError, OperationRequest, OperationState

_ZERO_HASH = "0" * 64
_MAX_JOURNAL_BYTES = 64 * 1024 * 1024


class AppendOnlyOperationJournal:
    """Append operation transitions without rewrite, truncation, or database access."""

    def __init__(
        self,
        *,
        path: Path,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        candidate = path.expanduser()
        if candidate.exists() and candidate.is_symlink():
            raise ControlOperationError(code="operation.journal_unsafe")
        self._path = candidate.resolve()
        if self._path == Path("/"):
            raise ControlOperationError(code="operation.journal_unsafe")
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._path.parent.chmod(0o700)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._now = now or (lambda: datetime.now(UTC))
        descriptor = os.open(
            self._path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(mode) or stat.S_IMODE(mode) & 0o077:
                raise ControlOperationError(code="operation.journal_permissions_invalid")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._recover_incomplete_tail(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
        self.entries()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold a cross-thread/process operation lock over lookup, effect, and final append."""

        descriptor = os.open(
            self._lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise ControlOperationError(code="operation.journal_lock_unsafe")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def append(
        self,
        *,
        request: OperationRequest,
        state: OperationState,
        detail_code: str,
        active_services: Sequence[str] = (),
    ) -> Mapping[str, object]:
        descriptor = os.open(
            self._path,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            entries = self._read_descriptor(descriptor)
            previous_hash = str(entries[-1]["entry_hash"]) if entries else _ZERO_HASH
            sequence = len(entries) + 1
            body: dict[str, object] = {
                "schema": "io.roehub.control-operation-journal-entry/v1alpha1",
                "sequence": sequence,
                "recorded_at": self._now().astimezone(UTC).isoformat(),
                "operation_id": str(request.operation_id),
                "request": request.model_dump(mode="json", by_alias=True, exclude_none=True),
                "request_digest": request.request_digest,
                "action": request.action.value,
                "profile": request.profile,
                "state": state.value,
                "detail_code": detail_code,
                "active_services": sorted(set(active_services)),
                "previous_hash": previous_hash,
            }
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            entry = {**body, "entry_hash": hashlib.sha256(canonical).hexdigest()}
            encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            os.lseek(descriptor, 0, os.SEEK_END)
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
            return entry
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def request(self, operation_id: UUID) -> OperationRequest | None:
        for entry in self.entries():
            if entry["operation_id"] == str(operation_id):
                payload = entry.get("request")
                if not isinstance(payload, dict):
                    raise ControlOperationError(code="operation.journal_corrupt")
                return OperationRequest.model_validate(payload)
        return None

    def latest(self, operation_id: UUID) -> Mapping[str, object] | None:
        found = [
            entry for entry in self.entries() if entry["operation_id"] == str(operation_id)
        ]
        return found[-1] if found else None

    def entries(self, *, after_sequence: int = 0) -> tuple[Mapping[str, object], ...]:
        descriptor = os.open(
            self._path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            return tuple(
                entry
                for entry in self._read_descriptor(descriptor)
                if int(entry["sequence"]) > after_sequence
            )
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def _read_descriptor(descriptor: int) -> tuple[dict[str, Any], ...]:
        size = os.fstat(descriptor).st_size
        if size > _MAX_JOURNAL_BYTES:
            raise ControlOperationError(code="operation.journal_too_large")
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = _read_exact(descriptor, size)
        return AppendOnlyOperationJournal._validate_raw(raw)

    @staticmethod
    def _validate_raw(raw: bytes) -> tuple[dict[str, Any], ...]:
        if raw and not raw.endswith(b"\n"):
            raise ControlOperationError(code="operation.journal_incomplete")
        entries: list[dict[str, Any]] = []
        previous_hash = _ZERO_HASH
        for expected_sequence, line in enumerate(raw.splitlines(), start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ControlOperationError(code="operation.journal_corrupt") from error
            if not isinstance(entry, dict):
                raise ControlOperationError(code="operation.journal_corrupt")
            entry_hash = entry.pop("entry_hash", None)
            canonical = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
            calculated = hashlib.sha256(canonical).hexdigest()
            entry["entry_hash"] = entry_hash
            if (
                entry.get("schema")
                != "io.roehub.control-operation-journal-entry/v1alpha1"
                or entry.get("sequence") != expected_sequence
                or entry.get("previous_hash") != previous_hash
                or entry_hash != calculated
            ):
                raise ControlOperationError(code="operation.journal_hash_mismatch")
            previous_hash = str(entry_hash)
            entries.append(entry)
        return tuple(entries)

    def _recover_incomplete_tail(self, descriptor: int) -> None:
        size = os.fstat(descriptor).st_size
        if size == 0:
            return
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = _read_exact(descriptor, size)
        if raw.endswith(b"\n"):
            return
        committed_end = raw.rfind(b"\n") + 1
        committed = raw[:committed_end]
        tail = raw[committed_end:]
        self._validate_committed_prefix(committed)
        tail_digest = hashlib.sha256(tail).hexdigest()
        evidence = self._path.with_name(f"{self._path.name}.torn-{tail_digest}.bin")
        try:
            evidence_descriptor = os.open(
                evidence,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            if evidence.read_bytes() != tail:
                raise ControlOperationError(code="operation.journal_recovery_conflict")
        else:
            try:
                _write_all(evidence_descriptor, tail)
                os.fsync(evidence_descriptor)
            finally:
                os.close(evidence_descriptor)
        os.ftruncate(descriptor, committed_end)
        os.fsync(descriptor)
        directory = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _validate_committed_prefix(raw: bytes) -> None:
        AppendOnlyOperationJournal._validate_raw(raw)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise ControlOperationError(code="operation.journal_write_failed")
        offset += written


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise ControlOperationError(code="operation.journal_read_failed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = ["AppendOnlyOperationJournal"]
