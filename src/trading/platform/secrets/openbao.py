"""Secret-safe OpenBao KV v2 client used inside trusted service boundaries."""

from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from .reference import SecretKind, SecretReference, SecretReferenceError
from .transport import normalize_openbao_address, open_without_redirect


class SecretResolutionError(RuntimeError):
    """Sanitized base error for secret resolution."""


class OpenBaoUnavailableError(SecretResolutionError):
    """OpenBao cannot currently serve the request."""


class OpenBaoPermissionError(SecretResolutionError):
    """The service identity is not authorized for the requested path."""


class OpenBaoSecretNotFoundError(SecretResolutionError):
    """The reference, version, or selected field does not exist."""


class SecretValue:
    """Opaque in-memory value that fails closed for generic serialization."""

    __slots__ = ("__text", "__version")

    def __init__(self, *, _text: str, version: int) -> None:
        object.__setattr__(self, "_SecretValue__text", _text)
        object.__setattr__(self, "_SecretValue__version", version)

    @classmethod
    def from_text(cls, value: str) -> "SecretValue":
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("secret value must be non-empty text")
        return cls(_text=value, version=0)

    @property
    def version(self) -> int:
        return self.__version

    def reveal_text(self) -> str:
        """Reveal only at the final trusted provider boundary."""

        return self.__text

    def __repr__(self) -> str:
        return f"SecretValue(version={self.version!r}, value=<redacted>)"

    def __str__(self) -> str:
        return "<secret-value:redacted>"

    def __setattr__(self, name: str, value: object) -> Never:
        _ = name, value
        raise AttributeError("SecretValue is immutable")

    def __reduce__(self) -> Never:
        raise TypeError("SecretValue cannot be serialized")


@dataclass(frozen=True, slots=True, repr=False)
class SecureCredentialFile:
    """Reloadable OpenBao token source with strict host-file permissions."""

    path: Path
    max_bytes: int = 16_384

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.path.is_absolute():
            raise ValueError("service credential file path must be absolute")

    def read(self) -> str:
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise OpenBaoUnavailableError("secure no-follow file access is unavailable")
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY | os.O_CLOEXEC | no_follow,
            )
        except OSError as error:
            raise OpenBaoUnavailableError("service credential is unavailable") from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise OpenBaoUnavailableError("service credential file is unsafe")
            if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise OpenBaoUnavailableError("service credential permissions are unsafe")
            if info.st_size <= 0 or info.st_size > self.max_bytes:
                raise OpenBaoUnavailableError("service credential file size is invalid")
            chunks: list[bytes] = []
            remaining = self.max_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > self.max_bytes:
                raise OpenBaoUnavailableError("service credential file size is invalid")
            value = payload.decode("utf-8").strip()
        except (OSError, UnicodeError) as error:
            raise OpenBaoUnavailableError("service credential is unreadable") from error
        finally:
            os.close(descriptor)
        if not value or "\x00" in value or any(character.isspace() for character in value):
            raise OpenBaoUnavailableError("service credential content is invalid")
        return value

    def __repr__(self) -> str:
        return "SecureCredentialFile(path=<redacted>)"


SecureTokenFile = SecureCredentialFile


@dataclass(frozen=True, slots=True)
class OpenBaoReadiness:
    classification: str
    ready: bool
    status_code: int | None

    def as_ops_status(self) -> dict[str, Any]:
        """Return the secret-free `ops.roehub.io/v1` dependency status."""

        return {
            "apiVersion": "ops.roehub.io/v1",
            "kind": "DependencyReadiness",
            "metadata": {"id": "openbao"},
            "status": {
                "classification": self.classification,
                "ready": self.ready,
                "runbook_id": "auth.openbao-unavailable",
            },
        }


@dataclass(frozen=True, slots=True, repr=False)
class OpenBaoSecretResolver:
    address: str
    token_source: SecureCredentialFile
    secret_root: str = "kv/roehub"
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        normalized = normalize_openbao_address(self.address)
        if self.timeout_seconds <= 0 or self.timeout_seconds > 10:
            raise ValueError("OpenBao timeout must be in (0, 10] seconds")
        object.__setattr__(self, "address", normalized)

    def resolve(
        self,
        raw_reference: str | SecretReference,
        *,
        expected_kind: SecretKind | None = None,
    ) -> SecretValue:
        try:
            reference = (
                raw_reference
                if isinstance(raw_reference, SecretReference)
                else SecretReference.parse(
                    raw_reference,
                    expected_root=self.secret_root,
                    expected_kind=expected_kind,
                )
            )
        except SecretReferenceError as error:
            raise SecretResolutionError("OpenBao secret reference is invalid") from error
        if expected_kind is not None and reference.kind is not expected_kind:
            raise SecretResolutionError("OpenBao secret reference kind is invalid")

        query = "" if reference.version is None else f"?version={reference.version}"
        payload = self._request_json(
            method="GET",
            path=f"/v1/{reference.kv_v2_path}{query}",
            authenticated=True,
        )
        data = _mapping_at(payload, "data", "data")
        metadata = _mapping_at(payload, "data", "metadata")
        value = data.get(reference.field)
        version = metadata.get("version")
        if not isinstance(value, str) or not value:
            raise OpenBaoSecretNotFoundError("OpenBao secret field is unavailable")
        if not isinstance(version, int) or version <= 0:
            raise OpenBaoUnavailableError("OpenBao secret metadata is invalid")
        return SecretValue(_text=value, version=version)

    def store(
        self,
        raw_reference: str | SecretReference,
        *,
        value: SecretValue,
        expected_kind: SecretKind | None = None,
    ) -> SecretReference:
        try:
            reference = (
                raw_reference
                if isinstance(raw_reference, SecretReference)
                else SecretReference.parse(
                    raw_reference,
                    expected_root=self.secret_root,
                    expected_kind=expected_kind,
                )
            )
        except SecretReferenceError as error:
            raise SecretResolutionError("OpenBao secret reference is invalid") from error
        if expected_kind is not None and reference.kind is not expected_kind:
            raise SecretResolutionError("OpenBao secret reference kind is invalid")
        payload = json.dumps(
            {"data": {reference.field: value.reveal_text()}},
            separators=(",", ":"),
        ).encode("utf-8")
        self._request(
            method="POST",
            path=f"/v1/{reference.kv_v2_path}",
            authenticated=True,
            request_body=payload,
        )
        return reference

    def readiness(self) -> OpenBaoReadiness:
        try:
            _, status = self._request(
                method="GET",
                path="/v1/sys/health?standbyok=true&perfstandbyok=true",
                authenticated=False,
                accepted_statuses={200, 429, 472, 473, 501, 503},
            )
        except OpenBaoUnavailableError:
            return OpenBaoReadiness("unavailable", False, None)
        if status in {200, 429, 472, 473}:
            return OpenBaoReadiness("unsealed", True, status)
        if status == 501:
            return OpenBaoReadiness("uninitialized", False, status)
        if status == 503:
            return OpenBaoReadiness("sealed", False, status)
        return OpenBaoReadiness("unavailable", False, status)

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        authenticated: bool,
    ) -> dict[str, Any]:
        body, _ = self._request(method=method, path=path, authenticated=authenticated)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise OpenBaoUnavailableError("OpenBao returned an invalid response") from error
        if not isinstance(payload, dict):
            raise OpenBaoUnavailableError("OpenBao returned an invalid response")
        return payload

    def _request(
        self,
        *,
        method: str,
        path: str,
        authenticated: bool,
        accepted_statuses: set[int] | None = None,
        request_body: bytes | None = None,
    ) -> tuple[bytes, int]:
        headers = {"Accept": "application/json"}
        if request_body is not None:
            headers["Content-Type"] = "application/json"
        if authenticated:
            headers["X-Vault-Token"] = self.token_source.read()
        request = urllib.request.Request(
            url=f"{self.address}{path}",
            headers=headers,
            method=method,
            data=request_body,
        )
        try:
            with open_without_redirect(request, timeout=self.timeout_seconds) as response:
                status = int(response.status)
                return response.read(), status
        except urllib.error.HTTPError as error:
            if accepted_statuses and error.code in accepted_statuses:
                return error.read(), int(error.code)
            if error.code in {401, 403}:
                raise OpenBaoPermissionError("OpenBao request is not authorized") from error
            if error.code == 404:
                raise OpenBaoSecretNotFoundError("OpenBao secret is unavailable") from error
            raise OpenBaoUnavailableError(
                f"OpenBao request failed with status {error.code}"
            ) from error
        except (OSError, TimeoutError) as error:
            raise OpenBaoUnavailableError("OpenBao request is unavailable") from error

    def __repr__(self) -> str:
        return (
            "OpenBaoSecretResolver(address=<redacted>, token_source=<redacted>, "
            f"secret_root={self.secret_root!r})"
        )


def _mapping_at(payload: dict[str, Any], *path: str) -> dict[str, Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise OpenBaoUnavailableError("OpenBao response is missing expected fields")
        current = current[key]
    if not isinstance(current, dict):
        raise OpenBaoUnavailableError("OpenBao response field is invalid")
    return current


__all__ = [
    "OpenBaoPermissionError",
    "OpenBaoReadiness",
    "OpenBaoSecretNotFoundError",
    "OpenBaoSecretResolver",
    "OpenBaoUnavailableError",
    "SecretResolutionError",
    "SecretValue",
    "SecureCredentialFile",
    "SecureTokenFile",
]
