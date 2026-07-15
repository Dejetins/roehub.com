from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from apps.control_agent.auth import ServiceIdentityAuthorizer
from trading.contexts.operations import ControlOperationError, OperationAction
from trading.contexts.operations.auth import mint_service_assertion


def _private_file(path: Path, value: str) -> Path:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
    return path


def test_short_lived_assertion_is_single_use(tmp_path: Path) -> None:
    api_key = "a" * 48
    owner_key = "b" * 48
    authorizer = ServiceIdentityAuthorizer(
        api_token_file=_private_file(tmp_path / "api", api_key),
        owner_token_file=_private_file(tmp_path / "owner", owner_key),
        replay_state_dir=tmp_path / "replay",
    )
    request_digest = "1" * 64
    assertion = mint_service_assertion(
        identity="api",
        identity_key=api_key,
        request_digest=request_digest,
    )

    authorizer.authorize(
        identity="api",
        credential=assertion,
        action=OperationAction.DIAGNOSTICS,
        request_digest=request_digest,
    )
    with pytest.raises(ControlOperationError, match="control_agent.identity_replay"):
        authorizer.authorize(
            identity="api",
            credential=assertion,
            action=OperationAction.DIAGNOSTICS,
            request_digest=request_digest,
        )
    restarted = ServiceIdentityAuthorizer(
        api_token_file=tmp_path / "api",
        owner_token_file=tmp_path / "owner",
        replay_state_dir=tmp_path / "replay",
    )
    with pytest.raises(ControlOperationError, match="control_agent.identity_replay"):
        restarted.authorize(
            identity="api",
            credential=assertion,
            action=OperationAction.DIAGNOSTICS,
            request_digest=request_digest,
        )


def test_expired_or_wrong_identity_assertion_is_rejected(tmp_path: Path) -> None:
    api_key = "a" * 48
    owner_key = "b" * 48
    authorizer = ServiceIdentityAuthorizer(
        api_token_file=_private_file(tmp_path / "api", api_key),
        owner_token_file=_private_file(tmp_path / "owner", owner_key),
        replay_state_dir=tmp_path / "replay",
    )
    request_digest = "2" * 64
    expired = mint_service_assertion(
        identity="api",
        identity_key=api_key,
        request_digest=request_digest,
        issued_at=int(time.time()) - 61,
    )
    wrong_identity = mint_service_assertion(
        identity="installation_owner",
        identity_key=owner_key,
        request_digest=request_digest,
    )

    with pytest.raises(ControlOperationError, match="control_agent.identity_expired"):
        authorizer.authorize(
            identity="api",
            credential=expired,
            action=OperationAction.INSPECT,
            request_digest=request_digest,
        )
    with pytest.raises(ControlOperationError, match="control_agent.identity_rejected"):
        authorizer.authorize(
            identity="api",
            credential=wrong_identity,
            action=OperationAction.INSPECT,
            request_digest=request_digest,
        )


def test_assertion_is_bound_to_canonical_request_digest(tmp_path: Path) -> None:
    api_key = "a" * 48
    authorizer = ServiceIdentityAuthorizer(
        api_token_file=_private_file(tmp_path / "api", api_key),
        owner_token_file=_private_file(tmp_path / "owner", "b" * 48),
        replay_state_dir=tmp_path / "replay",
    )
    assertion = mint_service_assertion(
        identity="api",
        identity_key=api_key,
        request_digest="3" * 64,
    )

    with pytest.raises(ControlOperationError, match="control_agent.identity_scope_mismatch"):
        authorizer.authorize(
            identity="api",
            credential=assertion,
            action=OperationAction.STOP,
            request_digest="4" * 64,
        )
