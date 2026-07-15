"""File-backed local service identities for the control-agent socket."""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
import time
from pathlib import Path
from threading import RLock

from trading.contexts.operations import ControlOperationError, OperationAction
from trading.contexts.operations.auth import Identity


def read_private_credential(path: Path) -> str:
    """Read a bounded mode-0600 credential without following symlinks."""

    candidate = path.expanduser()
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ControlOperationError(code="control_agent.identity_unavailable") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise ControlOperationError(code="control_agent.identity_permissions_invalid")
        with os.fdopen(descriptor, encoding="utf-8", closefd=False) as stream:
            value = stream.read(513).strip()
    finally:
        os.close(descriptor)
    if len(value) < 32 or len(value) > 512:
        raise ControlOperationError(code="control_agent.identity_invalid")
    return value


class ServiceIdentityAuthorizer:
    """Authenticate API and installation-owner identities without journaling credentials."""

    def __init__(
        self,
        *,
        api_token_file: Path,
        owner_token_file: Path,
        job_token_file: Path | None = None,
        replay_state_dir: Path,
        assertion_ttl_seconds: int = 60,
    ) -> None:
        if assertion_ttl_seconds < 10 or assertion_ttl_seconds > 300:
            raise ControlOperationError(code="control_agent.identity_configuration_invalid")
        self._identity_keys = {
            "api": read_private_credential(api_token_file).encode("utf-8"),
            "installation_owner": read_private_credential(owner_token_file).encode("utf-8"),
        }
        if job_token_file is not None:
            self._identity_keys["job_runtime"] = read_private_credential(job_token_file).encode(
                "utf-8"
            )
        self._assertion_ttl = assertion_ttl_seconds
        replay_root = replay_state_dir.expanduser()
        if replay_root.exists() and replay_root.is_symlink():
            raise ControlOperationError(code="control_agent.replay_state_unsafe")
        self._replay_root = replay_root.resolve()
        self._replay_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._replay_root.chmod(0o700)
        self._lock = RLock()

    def authorize(
        self,
        *,
        identity: Identity,
        credential: str,
        action: OperationAction | None,
        request_digest: str,
    ) -> None:
        parts = credential.split(".")
        if len(parts) != 5 or parts[0] != identity or not parts[1].isdigit():
            raise ControlOperationError(code="control_agent.identity_rejected")
        asserted_identity, timestamp_text, nonce, asserted_digest, supplied_signature = parts
        if identity not in self._identity_keys:
            raise ControlOperationError(code="control_agent.identity_rejected")
        if asserted_digest != request_digest:
            raise ControlOperationError(code="control_agent.identity_scope_mismatch")
        timestamp = int(timestamp_text)
        current = int(time.time())
        if timestamp > current + 5 or current - timestamp > self._assertion_ttl:
            raise ControlOperationError(code="control_agent.identity_expired")
        body = f"{asserted_identity}.{timestamp}.{nonce}.{asserted_digest}"
        expected_signature = hmac.new(
            self._identity_keys[identity], body.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise ControlOperationError(code="control_agent.identity_rejected")
        with self._lock:
            self._consume_nonce(nonce=nonce, timestamp=timestamp, current=current)
        if action is not None and action not in set(OperationAction):
            raise ControlOperationError(code="control_agent.action_forbidden")

    def _consume_nonce(self, *, nonce: str, timestamp: int, current: int) -> None:
        if len(nonce) != 32 or any(
            character not in "0123456789abcdef" for character in nonce
        ):
            raise ControlOperationError(code="control_agent.identity_rejected")
        for candidate in self._replay_root.iterdir():
            try:
                seen_at = int(candidate.read_text(encoding="ascii"))
            except (OSError, ValueError) as error:
                raise ControlOperationError(code="control_agent.replay_state_corrupt") from error
            if current - seen_at > self._assertion_ttl:
                candidate.unlink(missing_ok=True)
        target = self._replay_root / nonce
        try:
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError as error:
            raise ControlOperationError(code="control_agent.identity_replay") from error
        except OSError as error:
            raise ControlOperationError(code="control_agent.replay_state_unavailable") from error
        try:
            payload = str(timestamp).encode("ascii")
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ControlOperationError(
                        code="control_agent.replay_state_unavailable"
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(self._replay_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


__all__ = ["ServiceIdentityAuthorizer", "read_private_credential"]
