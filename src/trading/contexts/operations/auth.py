"""Short-lived HMAC service assertions for the local control-agent boundary."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Literal

from .contracts import ControlOperationError

Identity = Literal["api", "installation_owner", "job_runtime"]


def mint_service_assertion(
    *,
    identity: Identity,
    identity_key: str,
    request_digest: str,
    issued_at: int | None = None,
) -> str:
    """Mint an assertion bound to one canonical transport request."""

    if len(identity_key) < 32 or len(identity_key) > 512:
        raise ControlOperationError(code="control_agent.identity_invalid")
    if len(request_digest) != 64 or any(
        character not in "0123456789abcdef" for character in request_digest
    ):
        raise ControlOperationError(code="control_agent.request_digest_invalid")
    timestamp = int(time.time()) if issued_at is None else issued_at
    nonce = secrets.token_hex(16)
    body = f"{identity}.{timestamp}.{nonce}.{request_digest}"
    signature = hmac.new(
        identity_key.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{body}.{signature}"


__all__ = ["Identity", "mint_service_assertion"]
