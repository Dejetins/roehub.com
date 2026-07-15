from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from trading.contexts.extensions.domain import PLUGIN_RPC_VERSION

_SIGNING_CONTEXT = b"roehub-plugin-service-identity-v1alpha1\0"
_ALLOWED_CAPABILITIES = frozenset(
    {"app.action", "data.read", "notification.send", "panel.describe"}
)
_DEFAULT_RPC_RESPONSE_BYTE_LIMIT = 1_114_112
_RPC_ENVELOPE_BYTE_ALLOWANCE = 65_536


class PluginRpcError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class PluginServiceIdentityClaims:
    contract: str
    organization_id: str
    instance_id: str
    package_digest: str
    package_version: str
    capability: str
    issued_at: int
    expires_at: int
    nonce_id: str

    def __post_init__(self) -> None:
        if self.contract != "PluginServiceIdentity/v1alpha1":
            raise ValueError("unsupported plugin service identity contract")
        UUID(self.organization_id)
        UUID(self.instance_id)
        UUID(self.nonce_id)
        if self.capability not in _ALLOWED_CAPABILITIES:
            raise ValueError("unsupported plugin capability")
        if self.expires_at <= self.issued_at or self.expires_at - self.issued_at > 60:
            raise ValueError("plugin service identity lifetime must be in (0, 60] seconds")
        if len(self.package_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.package_digest
        ):
            raise ValueError("plugin package digest is invalid")


class PluginServiceIdentitySigner:
    def __init__(self, *, private_key: Ed25519PrivateKey, key_id: str) -> None:
        self._signing_key = private_key
        self._key_id = key_id

    def issue(
        self,
        *,
        organization_id: UUID,
        instance_id: UUID,
        package_digest: str,
        package_version: str,
        capability: str,
        now: datetime,
        ttl_seconds: int = 30,
    ) -> str:
        normalized_now = _utc(now)
        claims = PluginServiceIdentityClaims(
            contract="PluginServiceIdentity/v1alpha1",
            organization_id=str(organization_id),
            instance_id=str(instance_id),
            package_digest=package_digest,
            package_version=package_version,
            capability=capability,
            issued_at=int(normalized_now.timestamp()),
            expires_at=int((normalized_now + timedelta(seconds=ttl_seconds)).timestamp()),
            nonce_id=str(uuid4()),
        )
        header = _encode_json({"alg": "Ed25519", "kid": self._key_id, "typ": "RoehubPlugin"})
        payload = _encode_json(asdict(claims))
        signing_input = f"{header}.{payload}".encode("ascii")
        signature = _b64url(self._signing_key.sign(_SIGNING_CONTEXT + signing_input))
        return f"{header}.{payload}.{signature}"


class PluginServiceIdentityVerifier:
    def __init__(self, *, public_keys: Mapping[str, Ed25519PublicKey]) -> None:
        self._public_keys = dict(public_keys)
        self._used_nonces: dict[str, int] = {}
        self._nonce_lock = Lock()

    def verify(
        self,
        *,
        identity: str,
        expected_organization_id: UUID,
        expected_instance_id: UUID,
        expected_package_digest: str,
        expected_package_version: str,
        expected_capability: str,
        now: datetime,
    ) -> PluginServiceIdentityClaims:
        try:
            header_segment, payload_segment, signature_segment = identity.split(".")
            header = _decode_json(header_segment)
            claims_payload = _decode_json(payload_segment)
            if header.get("alg") != "Ed25519" or header.get("typ") != "RoehubPlugin":
                raise ValueError("plugin identity header is invalid")
            key_id = header["kid"]
            public_key = self._public_keys[key_id]
            signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
            public_key.verify(_b64url_decode(signature_segment), _SIGNING_CONTEXT + signing_input)
            claims = PluginServiceIdentityClaims(**claims_payload)
        except (KeyError, TypeError, ValueError, InvalidSignature) as error:
            raise PluginRpcError(
                code="plugin.identity_invalid", message="Plugin service identity is invalid"
            ) from error
        now_epoch = int(_utc(now).timestamp())
        if not claims.issued_at <= now_epoch < claims.expires_at:
            raise PluginRpcError(
                code="plugin.identity_expired", message="Plugin service identity has expired"
            )
        expected = (
            claims.organization_id == str(expected_organization_id)
            and claims.instance_id == str(expected_instance_id)
            and claims.package_digest == expected_package_digest
            and claims.package_version == expected_package_version
            and claims.capability == expected_capability
        )
        if not expected:
            raise PluginRpcError(
                code="plugin.identity_scope_mismatch",
                message="Plugin service identity does not match invocation scope",
            )
        with self._nonce_lock:
            self._used_nonces = {
                nonce_id: expires_at
                for nonce_id, expires_at in self._used_nonces.items()
                if expires_at > now_epoch
            }
            if claims.nonce_id in self._used_nonces:
                raise PluginRpcError(
                    code="plugin.identity_replayed",
                    message="Plugin service identity nonce was already used",
                )
            self._used_nonces[claims.nonce_id] = claims.expires_at
        return claims


class PluginRpcClient:
    """Gateway-owned typed RPC client; there is deliberately no generic execute method."""

    def __init__(
        self,
        *,
        base_url: str,
        signer: PluginServiceIdentitySigner,
        organization_id: UUID,
        instance_id: UUID,
        package_digest: str,
        package_version: str,
        granted_capabilities: frozenset[str],
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("plugin RPC base_url must use http or https")
        if not 0 < timeout_seconds <= 10:
            raise ValueError("plugin RPC timeout must be in (0, 10] seconds")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )
        self._signer = signer
        self._organization_id = organization_id
        self._instance_id = instance_id
        self._package_digest = package_digest
        self._package_version = package_version
        self._granted_capabilities = granted_capabilities

    def close(self) -> None:
        self._client.close()

    def health(self, *, now: datetime) -> Mapping[str, Any]:
        return self._request(capability="data.read", path="/v1alpha1/health", payload=None, now=now)

    def metrics(self, *, now: datetime) -> Mapping[str, Any]:
        return self._request(
            capability="data.read", path="/v1alpha1/metrics", payload=None, now=now
        )

    def query_data(
        self, *, request: Mapping[str, object], now: datetime
    ) -> Mapping[str, Any]:
        limits = request.get("limits")
        response_byte_limit = _DEFAULT_RPC_RESPONSE_BYTE_LIMIT
        if isinstance(limits, Mapping):
            requested_limit = limits.get("bytes")
            if isinstance(requested_limit, int) and not isinstance(
                requested_limit, bool
            ):
                response_byte_limit = min(
                    requested_limit + _RPC_ENVELOPE_BYTE_ALLOWANCE,
                    _DEFAULT_RPC_RESPONSE_BYTE_LIMIT,
                )
        return self._request(
            capability="data.read",
            path="/v1alpha1/data-source/query",
            payload=request,
            now=now,
            response_byte_limit=response_byte_limit,
        )

    def describe_panel(self, *, now: datetime) -> Mapping[str, Any]:
        return self._request(
            capability="panel.describe", path="/v1alpha1/panel/describe", payload={}, now=now
        )

    def invoke_app_action(
        self,
        *,
        request: Mapping[str, object],
        idempotency_key: str,
        now: datetime,
    ) -> Mapping[str, Any]:
        return self._request(
            capability="app.action",
            path="/v1alpha1/app/action",
            payload=request,
            now=now,
            idempotency_key=idempotency_key,
        )

    def send_notification(
        self,
        *,
        request: Mapping[str, object],
        idempotency_key: str,
        now: datetime,
    ) -> Mapping[str, Any]:
        return self._request(
            capability="notification.send",
            path="/v1alpha1/notification-provider/send",
            payload=request,
            now=now,
            idempotency_key=idempotency_key,
        )

    def _request(
        self,
        *,
        capability: str,
        path: str,
        payload: Mapping[str, object] | None,
        now: datetime,
        idempotency_key: str | None = None,
        response_byte_limit: int = _DEFAULT_RPC_RESPONSE_BYTE_LIMIT,
    ) -> Mapping[str, Any]:
        if capability not in self._granted_capabilities:
            raise PluginRpcError(
                code="plugin.capability_not_granted",
                message="Plugin capability is not granted to this instance",
            )
        identity_value = self._signer.issue(
            organization_id=self._organization_id,
            instance_id=self._instance_id,
            package_digest=self._package_digest,
            package_version=self._package_version,
            capability=capability,
            now=now,
        )
        headers = {
            "Authorization": f"RoehubPluginIdentity {identity_value}",
            "X-Roehub-Plugin-Protocol": PLUGIN_RPC_VERSION,
        }
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            request_context = (
                self._client.stream("GET", path, headers=headers)
                if payload is None
                else self._client.stream(
                    "POST", path, headers=headers, json=dict(payload)
                )
            )
            with request_context as response:
                if response.headers.get("X-Roehub-Plugin-Protocol") != PLUGIN_RPC_VERSION:
                    raise PluginRpcError(
                        code="plugin.protocol_negotiation_failed",
                        message="Plugin RPC protocol negotiation failed",
                    )
                if response.status_code >= 400:
                    raise PluginRpcError(
                        code="plugin.rpc_rejected",
                        message="Plugin RPC request was rejected",
                    )
                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError as error:
                        raise PluginRpcError(
                            code="plugin.rpc_response_invalid",
                            message="Plugin RPC response is invalid",
                        ) from error
                    if declared_length < 0:
                        raise PluginRpcError(
                            code="plugin.rpc_response_invalid",
                            message="Plugin RPC response is invalid",
                        )
                    if declared_length > response_byte_limit:
                        raise PluginRpcError(
                            code="plugin.rpc_response_too_large",
                            message="Plugin RPC response exceeded its byte budget",
                        )
                body_bytes = bytearray()
                for chunk in response.iter_bytes():
                    if len(body_bytes) + len(chunk) > response_byte_limit:
                        raise PluginRpcError(
                            code="plugin.rpc_response_too_large",
                            message="Plugin RPC response exceeded its byte budget",
                        )
                    body_bytes.extend(chunk)
        except PluginRpcError:
            raise
        except httpx.TimeoutException as error:
            raise PluginRpcError(
                code="plugin.rpc_unknown", message="Plugin RPC result is unknown after timeout"
            ) from error
        except httpx.HTTPError as error:
            raise PluginRpcError(
                code="plugin.rpc_unavailable", message="Plugin RPC transport is unavailable"
            ) from error
        try:
            result = json.loads(body_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PluginRpcError(
                code="plugin.rpc_response_invalid", message="Plugin RPC response is invalid"
            ) from error
        if not isinstance(result, dict) or result.get("contract") != "PluginResponse/v1alpha1":
            raise PluginRpcError(
                code="plugin.rpc_response_invalid", message="Plugin RPC response is invalid"
            )
        return cast(Mapping[str, Any], result)


def _encode_json(payload: Mapping[str, object]) -> str:
    return _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _decode_json(segment: str) -> dict[str, Any]:
    payload = json.loads(_b64url_decode(segment))
    if not isinstance(payload, dict):
        raise ValueError("identity segment is not an object")
    return cast(dict[str, Any], payload)


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("plugin identity timestamp must be timezone-aware")
    return value.astimezone(UTC)
