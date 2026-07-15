from __future__ import annotations

import json
import pickle
import secrets
import stat
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest
from fastapi.encoders import jsonable_encoder

from trading.platform.secrets import (
    OpenBaoPermissionError,
    OpenBaoSecretResolver,
    OpenBaoUnavailableError,
    SecretKind,
    SecretValue,
    SecureTokenFile,
)


class _Response:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _credential_file(tmp_path: Path) -> tuple[Path, str]:
    value = secrets.token_urlsafe(32)
    path = tmp_path / "service-identity"
    path.write_text(value, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path, value


def test_resolver_reads_selected_version_and_redacts_repr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_path, expected_header = _credential_file(tmp_path)
    canary = secrets.token_urlsafe(24)

    def respond(request: urllib.request.Request, *, timeout: float) -> _Response:
        assert request.full_url.endswith("/v1/kv/data/roehub/oidc/provider-a?version=2")
        assert request.headers["X-vault-token"] == expected_header
        assert timeout == 3.0
        return _Response(
            json.dumps(
                {"data": {"data": {"client_secret": canary}, "metadata": {"version": 2}}}
            ).encode()
        )

    monkeypatch.setattr(
        "trading.platform.secrets.openbao.open_without_redirect",
        respond,
    )
    resolver = OpenBaoSecretResolver(
        address="http://127.0.0.1:8200",
        token_source=SecureTokenFile(credential_path.resolve()),
    )

    value = resolver.resolve(
        "openbao://kv/roehub/oidc/provider-a?version=2#client_secret",
        expected_kind=SecretKind.OIDC,
    )

    assert value.reveal_text() == canary
    assert value.version == 2
    assert canary not in repr(value)
    assert canary not in str(value)
    assert expected_header not in repr(resolver)

    with pytest.raises(ValueError):
        jsonable_encoder(value)
    with pytest.raises(TypeError):
        json.dumps(value)
    with pytest.raises(TypeError):
        pickle.dumps(value)


def test_token_source_rejects_group_or_world_access(tmp_path: Path) -> None:
    credential_path, _ = _credential_file(tmp_path)
    credential_path.chmod(0o640)

    with pytest.raises(OpenBaoUnavailableError, match="permissions"):
        SecureTokenFile(credential_path.resolve()).read()


def test_resolver_stores_secret_without_exposing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_path, expected_header = _credential_file(tmp_path)
    canary = secrets.token_urlsafe(24)
    captured_body = b""

    def respond(request: urllib.request.Request, *, timeout: float) -> _Response:
        nonlocal captured_body
        assert request.full_url.endswith(
            "/v1/kv/data/roehub/telegram/provider-a/org-a/user-a"
        )
        assert request.get_method() == "POST"
        assert request.headers["X-vault-token"] == expected_header
        assert request.headers["Content-type"] == "application/json"
        assert timeout == 3.0
        captured_body = request.data or b""
        return _Response(b"{}", status=204)

    monkeypatch.setattr(
        "trading.platform.secrets.openbao.open_without_redirect",
        respond,
    )
    resolver = OpenBaoSecretResolver(
        address="http://127.0.0.1:8200",
        token_source=SecureTokenFile(credential_path.resolve()),
    )
    value = SecretValue.from_text(canary)

    reference = resolver.store(
        "openbao://kv/roehub/telegram/provider-a/org-a/user-a#chat_id",
        value=value,
        expected_kind=SecretKind.TELEGRAM,
    )

    assert json.loads(captured_body) == {"data": {"chat_id": canary}}
    assert reference.field == "chat_id"
    assert canary not in repr(value)
    assert canary not in repr(resolver)


def test_token_source_rejects_symlink(tmp_path: Path) -> None:
    credential_path, _ = _credential_file(tmp_path)
    linked_path = tmp_path / "linked-service-identity"
    linked_path.symlink_to(credential_path)

    with pytest.raises(OpenBaoUnavailableError, match="unavailable"):
        SecureTokenFile(linked_path).read()


def test_resolver_sanitizes_policy_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_path, marker = _credential_file(tmp_path)

    def reject(_request: urllib.request.Request, *, timeout: float) -> object:
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:8200/v1/kv/data/roehub/oidc/provider-a",
            code=403,
            msg=marker,
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(
        "trading.platform.secrets.openbao.open_without_redirect",
        reject,
    )
    resolver = OpenBaoSecretResolver(
        address="http://127.0.0.1:8200",
        token_source=SecureTokenFile(credential_path.resolve()),
    )

    with pytest.raises(OpenBaoPermissionError) as exc_info:
        resolver.resolve("openbao://kv/roehub/oidc/provider-a#client_secret")

    assert marker not in str(exc_info.value)


@pytest.mark.parametrize(
    ("status", "classification", "ready"),
    [(200, "unsealed", True), (501, "uninitialized", False), (503, "sealed", False)],
)
def test_readiness_maps_to_secret_free_ops_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    classification: str,
    ready: bool,
) -> None:
    credential_path, marker = _credential_file(tmp_path)

    def respond(_request: urllib.request.Request, *, timeout: float) -> _Response:
        if status == 200:
            return _Response(b"{}", status=200)
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:8200/v1/sys/health",
            code=status,
            msg=marker,
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(
        "trading.platform.secrets.openbao.open_without_redirect",
        respond,
    )
    resolver = OpenBaoSecretResolver(
        address="http://127.0.0.1:8200",
        token_source=SecureTokenFile(credential_path.resolve()),
    )

    evidence = resolver.readiness().as_ops_status()

    assert evidence["apiVersion"] == "ops.roehub.io/v1"
    assert evidence["status"]["classification"] == classification
    assert evidence["status"]["ready"] is ready
    assert marker not in json.dumps(evidence)
