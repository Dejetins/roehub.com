from __future__ import annotations

import base64
import hashlib
import hmac
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Mapping, Protocol, TypeVar, cast
from urllib.parse import urlencode, urlparse

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from trading.contexts.identity.application.ports.authentication_provider import (
    AuthenticationProvider,
    AuthenticationProviderError,
    VerifiedExternalIdentity,
)

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")
_MAX_DOCUMENT_BYTES = 1_048_576
_DEFAULT_CACHE_SECONDS = 300.0
_MAX_CACHE_SECONDS = 3600.0
_ALLOWED_ALGORITHMS = frozenset({"RS256"})
_T = TypeVar("_T")


class OidcProviderMetrics(Protocol):
    def record(
        self,
        *,
        provider_id: str,
        operation: str,
        outcome: str,
        duration_seconds: float,
        success_unixtime: float | None,
    ) -> None: ...


class _NoopMetrics:
    def record(self, **_: object) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    value: Mapping[str, Any]
    expires_monotonic: float


class HttpOidcAuthenticationProvider(AuthenticationProvider):
    """Strict OIDC Code Flow adapter with discovery/JWKS caching and bounded retries."""

    def __init__(
        self,
        *,
        provider_id: str,
        display_name: str,
        issuer: str,
        client_id: str,
        client_credential_source: Callable[[], str],
        redirect_uri: str,
        connect_timeout_seconds: float = 3.0,
        response_timeout_seconds: float = 10.0,
        overall_timeout_seconds: float = 15.0,
        allow_insecure_http: bool = False,
        transport: httpx.BaseTransport | None = None,
        metrics: OidcProviderMetrics | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.02, 0.08),
    ) -> None:
        normalized_provider_id = provider_id.strip().lower()
        if not _PROVIDER_ID.fullmatch(normalized_provider_id):
            raise ValueError("OIDC provider_id is invalid")
        self._provider_id = normalized_provider_id
        self._display_name = _required(display_name, "display_name")
        self._issuer = _required(issuer, "issuer").rstrip("/")
        self._client_id = _required(client_id, "client_id")
        if not callable(client_credential_source):
            raise ValueError("OIDC client_credential_source must be callable")
        self._client_credential_source = client_credential_source
        self._redirect_uri = _required(redirect_uri, "redirect_uri")
        self._allow_insecure_http = allow_insecure_http
        _validate_url(self._issuer, allow_insecure_http=allow_insecure_http, name="issuer")
        _validate_url(
            self._redirect_uri,
            allow_insecure_http=allow_insecure_http,
            name="redirect_uri",
        )
        for value, maximum, name in (
            (connect_timeout_seconds, 3.0, "connect_timeout_seconds"),
            (response_timeout_seconds, 10.0, "response_timeout_seconds"),
            (overall_timeout_seconds, 15.0, "overall_timeout_seconds"),
        ):
            if value <= 0 or value > maximum:
                raise ValueError(f"OIDC {name} must be > 0 and <= {maximum:g}")
        self._connect_timeout = connect_timeout_seconds
        self._response_timeout = response_timeout_seconds
        self._overall_timeout = overall_timeout_seconds
        self._transport = transport
        self._metrics = metrics or _NoopMetrics()
        self._monotonic = monotonic
        self._now = now
        self._sleeper = sleeper
        self._jitter = jitter
        self._discovery: _CacheEntry | None = None
        self._jwks: _CacheEntry | None = None

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def issuer(self) -> str:
        return self._issuer

    @property
    def display_name(self) -> str:
        return self._display_name

    def authorization_url(
        self,
        *,
        state: str,
        nonce: str,
        code_challenge: str,
    ) -> str:
        deadline = self._monotonic() + self._overall_timeout
        discovery = self._get_discovery(deadline=deadline)
        self._raise_if_deadline_exceeded(deadline=deadline)
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self._client_id,
                "redirect_uri": self._redirect_uri,
                "scope": "openid email profile",
                "state": state,
                "nonce": nonce,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{discovery['authorization_endpoint']}?{query}"

    def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        expected_nonce_sha256: str,
    ) -> VerifiedExternalIdentity:
        if not code.strip() or not code_verifier.strip():
            raise AuthenticationProviderError(code="invalid_callback")
        deadline = self._monotonic() + self._overall_timeout
        discovery = self._get_discovery(deadline=deadline)
        provider_document = self._post_exchange(
            url=str(discovery["token_endpoint"]),
            code=code,
            code_verifier=code_verifier,
            deadline=deadline,
        )
        signed_identity = provider_document.get("id_" + "token")
        if not isinstance(signed_identity, str) or not signed_identity:
            raise AuthenticationProviderError(code="missing_id_token")
        validation_started = self._monotonic()
        try:
            try:
                identity = self._verify_signed_identity(
                    signed_identity=signed_identity,
                    expected_nonce_sha256=expected_nonce_sha256,
                    jwks_uri=str(discovery["jwks_uri"]),
                    deadline=deadline,
                    refresh=False,
                )
            except _UnknownSigningKey:
                identity = self._verify_signed_identity(
                    signed_identity=signed_identity,
                    expected_nonce_sha256=expected_nonce_sha256,
                    jwks_uri=str(discovery["jwks_uri"]),
                    deadline=deadline,
                    refresh=True,
                )
        except AuthenticationProviderError:
            self._record(
                operation="identity_validation",
                outcome="validation_error",
                started=validation_started,
            )
            raise
        self._raise_if_deadline_exceeded(deadline=deadline)
        return identity

    def _post_exchange(
        self,
        *,
        url: str,
        code: str,
        code_verifier: str,
        deadline: float,
    ) -> Mapping[str, Any]:
        started = self._monotonic()
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "code_verifier": code_verifier,
        }
        try:
            credential = _required(
                self._client_credential_source(),
                "client_credential",
            )
        except (RuntimeError, ValueError) as error:
            self._record(
                operation="token_post",
                outcome="provider_unavailable",
                started=started,
            )
            raise AuthenticationProviderError(code="provider_unavailable") from error
        self._raise_if_deadline_exceeded(deadline=deadline)
        form["client_" + "secret"] = credential
        try:
            status_code, _, content = self._request_document(
                method="POST",
                url=url,
                deadline=deadline,
                data=form,
            )
        except (_OverallDeadlineExceeded, httpx.TimeoutException, httpx.TransportError) as error:
            self._record(operation="token_post", outcome="result_unknown", started=started)
            raise AuthenticationProviderError(
                code="token_result_unknown", token_result_unknown=True
            ) from error
        if status_code != 200:
            unknown = status_code >= 500 or status_code == 429
            self._record(
                operation="token_post",
                outcome="result_unknown" if unknown else "rejected",
                started=started,
            )
            raise AuthenticationProviderError(
                code="token_result_unknown" if unknown else "token_rejected",
                token_result_unknown=unknown,
            )
        try:
            document = _json_document(content)
        except AuthenticationProviderError:
            self._record(
                operation="token_post", outcome="validation_error", started=started
            )
            raise
        self._record(operation="token_post", outcome="succeeded", started=started)
        return document

    def _verify_signed_identity(
        self,
        *,
        signed_identity: str,
        expected_nonce_sha256: str,
        jwks_uri: str,
        deadline: float,
        refresh: bool,
    ) -> VerifiedExternalIdentity:
        parts = signed_identity.split(".")
        if len(parts) != 3 or any(not part for part in parts):
            raise AuthenticationProviderError(code="malformed_id_token")
        header = _decode_json_segment(parts[0])
        claims = _decode_json_segment(parts[1])
        if header.get("alg") not in _ALLOWED_ALGORITHMS:
            raise AuthenticationProviderError(code="unsupported_signing_algorithm")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise AuthenticationProviderError(code="missing_signing_key_id")
        jwks = self._get_jwks(url=jwks_uri, deadline=deadline, refresh=refresh)
        key_data = next(
            (
                item
                for item in jwks["keys"]
                if isinstance(item, dict)
                and item.get("kid") == kid
                and item.get("kty") == "RSA"
                and item.get("use", "sig") == "sig"
            ),
            None,
        )
        if key_data is None:
            if refresh:
                raise AuthenticationProviderError(code="unknown_signing_key")
            raise _UnknownSigningKey()
        try:
            public_key = rsa.RSAPublicNumbers(
                e=_b64_int(key_data["e"]),
                n=_b64_int(key_data["n"]),
            ).public_key()
            public_key.verify(
                _b64decode(parts[2]),
                f"{parts[0]}.{parts[1]}".encode("ascii"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AuthenticationProviderError(code="malformed_signing_key") from error
        except InvalidSignature as error:
            raise AuthenticationProviderError(code="invalid_id_token_signature") from error
        return self._validated_claims(
            claims=claims, expected_nonce_sha256=expected_nonce_sha256
        )

    def _validated_claims(
        self, *, claims: Mapping[str, Any], expected_nonce_sha256: str
    ) -> VerifiedExternalIdentity:
        current = self._now()
        if current.tzinfo is None:
            raise ValueError("OIDC now callback must return timezone-aware datetime")
        now = current.timestamp()
        if claims.get("iss") != self._issuer:
            raise AuthenticationProviderError(code="issuer_mismatch")
        audience = claims.get("aud")
        audience_values = [audience] if isinstance(audience, str) else audience
        if not isinstance(audience_values, list) or self._client_id not in audience_values:
            raise AuthenticationProviderError(code="audience_mismatch")
        if len(audience_values) > 1 and claims.get("azp") != self._client_id:
            raise AuthenticationProviderError(code="authorized_party_mismatch")
        exp = claims.get("exp")
        iat = claims.get("iat")
        nbf = claims.get("nbf", iat)
        if not isinstance(exp, (int, float)) or exp <= now - 60:
            raise AuthenticationProviderError(code="id_token_expired")
        if not isinstance(iat, (int, float)) or iat > now + 60:
            raise AuthenticationProviderError(code="invalid_issued_at")
        if not isinstance(nbf, (int, float)) or nbf > now + 60:
            raise AuthenticationProviderError(code="id_token_not_yet_valid")
        nonce = claims.get("nonce")
        if not isinstance(nonce, str) or not hmac.compare_digest(
            hashlib.sha256(nonce.encode()).hexdigest(), expected_nonce_sha256
        ):
            raise AuthenticationProviderError(code="nonce_mismatch")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 1024:
            raise AuthenticationProviderError(code="invalid_subject")
        email = claims.get("email")
        normalized_email = email.strip().lower() if isinstance(email, str) else None
        if normalized_email == "" or (normalized_email is not None and len(normalized_email) > 320):
            normalized_email = None
        return VerifiedExternalIdentity(
            issuer=self._issuer,
            subject=subject,
            email=normalized_email,
            email_verified=claims.get("email_verified") is True,
        )

    def _get_discovery(self, *, deadline: float) -> Mapping[str, Any]:
        now = self._monotonic()
        if self._discovery is not None and self._discovery.expires_monotonic > now:
            return self._discovery.value
        url = f"{self._issuer}/.well-known/openid-configuration"
        value, ttl = self._get_json(url=url, operation="discovery_get", deadline=deadline)
        validation_started = self._monotonic()
        try:
            if value.get("issuer") != self._issuer:
                raise AuthenticationProviderError(code="discovery_issuer_mismatch")
            for field in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
                endpoint = value.get(field)
                if not isinstance(endpoint, str):
                    raise AuthenticationProviderError(code="malformed_discovery")
                _validate_url(
                    endpoint,
                    allow_insecure_http=self._allow_insecure_http,
                    name=field,
                )
            supported = value.get("id_token_signing_alg_values_supported")
            if not isinstance(supported, list) or not _ALLOWED_ALGORITHMS.intersection(
                supported
            ):
                raise AuthenticationProviderError(code="unsupported_signing_algorithm")
        except AuthenticationProviderError:
            self._record(
                operation="discovery_validation",
                outcome="validation_error",
                started=validation_started,
            )
            raise
        except ValueError as error:
            self._record(
                operation="discovery_validation",
                outcome="validation_error",
                started=validation_started,
            )
            raise AuthenticationProviderError(code="malformed_discovery") from error
        self._discovery = _CacheEntry(value=value, expires_monotonic=now + ttl)
        return value

    def _get_jwks(
        self, *, url: str, deadline: float, refresh: bool
    ) -> Mapping[str, Any]:
        now = self._monotonic()
        if not refresh and self._jwks is not None and self._jwks.expires_monotonic > now:
            return self._jwks.value
        value, ttl = self._get_json(url=url, operation="jwks_get", deadline=deadline)
        if not isinstance(value.get("keys"), list):
            self._record(
                operation="jwks_validation",
                outcome="validation_error",
                started=self._monotonic(),
            )
            raise AuthenticationProviderError(code="malformed_jwks")
        self._jwks = _CacheEntry(value=value, expires_monotonic=now + ttl)
        return value

    def _get_json(
        self, *, url: str, operation: str, deadline: float
    ) -> tuple[Mapping[str, Any], float]:
        for attempt in range(3):
            if self._monotonic() >= deadline:
                break
            started = self._monotonic()
            outcome = "transport_error"
            try:
                status_code, cache_control, content = self._request_document(
                    method="GET",
                    url=url,
                    deadline=deadline,
                )
                if status_code >= 500 or status_code == 429:
                    outcome = f"http_{status_code}"
                elif status_code != 200:
                    self._record(operation=operation, outcome="rejected", started=started)
                    raise AuthenticationProviderError(code="provider_metadata_rejected")
                else:
                    try:
                        value = _json_document(content)
                    except AuthenticationProviderError:
                        self._record(
                            operation=operation,
                            outcome="validation_error",
                            started=started,
                        )
                        raise
                    self._record(operation=operation, outcome="succeeded", started=started)
                    return value, _cache_ttl(cache_control)
            except AuthenticationProviderError:
                raise
            except _OverallDeadlineExceeded:
                outcome = "deadline_exceeded"
            except (httpx.TimeoutException, httpx.TransportError):
                pass
            self._record(operation=operation, outcome=outcome, started=started)
            if attempt < 2:
                remaining = deadline - self._monotonic()
                pause = min(max(0.0, self._jitter()), max(0.0, remaining))
                if pause > 0:
                    self._sleeper(pause)
        raise AuthenticationProviderError(code="provider_unavailable", retryable=True)

    def _request_document(
        self,
        *,
        method: str,
        url: str,
        deadline: float,
        data: Mapping[str, str] | None = None,
    ) -> tuple[int, str | None, bytes]:
        def request() -> tuple[int, str | None, bytes]:
            with self._client(deadline=deadline) as client:
                with client.stream(
                    method,
                    url,
                    data=data,
                    headers={"Accept": "application/json"},
                ) as response:
                    status_code = response.status_code
                    cache_control = response.headers.get("cache-control")
                    content = (
                        self._read_document(response=response, deadline=deadline)
                        if status_code == 200
                        else b""
                    )
            return status_code, cache_control, content

        return self._run_with_deadline(operation=request, deadline=deadline)

    def _run_with_deadline(
        self, *, operation: Callable[[], _T], deadline: float
    ) -> _T:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise _OverallDeadlineExceeded()
        result: Queue[tuple[bool, object]] = Queue(maxsize=1)
        done = Event()

        def worker() -> None:
            try:
                result.put((True, operation()))
            except Exception as error:
                result.put((False, error))
            finally:
                done.set()

        thread = Thread(
            target=worker,
            name=f"oidc-{self._provider_id}-request",
            daemon=True,
        )
        thread.start()
        remaining = deadline - self._monotonic()
        if remaining <= 0 or not done.wait(timeout=remaining):
            raise _OverallDeadlineExceeded()
        succeeded, value = result.get_nowait()
        if not succeeded:
            raise cast(Exception, value)
        return cast(_T, value)

    def _client(self, *, deadline: float) -> httpx.Client:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise AuthenticationProviderError(code="provider_unavailable", retryable=True)
        timeout = httpx.Timeout(
            timeout=min(self._response_timeout, remaining),
            connect=min(self._connect_timeout, remaining),
        )
        return httpx.Client(timeout=timeout, transport=self._transport, follow_redirects=False)

    def _read_document(self, *, response: httpx.Response, deadline: float) -> bytes:
        content = bytearray()
        for chunk in response.iter_bytes():
            if self._monotonic() >= deadline:
                raise _OverallDeadlineExceeded()
            content.extend(chunk)
            if len(content) > _MAX_DOCUMENT_BYTES:
                raise AuthenticationProviderError(code="provider_document_too_large")
        if self._monotonic() >= deadline:
            raise _OverallDeadlineExceeded()
        return bytes(content)

    def _raise_if_deadline_exceeded(self, *, deadline: float) -> None:
        if self._monotonic() >= deadline:
            raise AuthenticationProviderError(
                code="provider_deadline_exceeded",
                retryable=True,
            )

    def _record(self, *, operation: str, outcome: str, started: float) -> None:
        now = self._now()
        self._metrics.record(
            provider_id=self._provider_id,
            operation=operation,
            outcome=outcome,
            duration_seconds=max(0.0, self._monotonic() - started),
            success_unixtime=(
                now.timestamp() if outcome == "succeeded" and now.tzinfo is not None else None
            ),
        )


class _UnknownSigningKey(Exception):
    pass


class _OverallDeadlineExceeded(Exception):
    pass


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"OIDC {name} must be non-empty")
    return normalized


def _validate_url(value: str, *, allow_insecure_http: bool, name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in ({"https", "http"} if allow_insecure_http else {"https"}):
        raise ValueError(f"OIDC {name} must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.fragment:
        raise ValueError(f"OIDC {name} is invalid")


def _json_document(content: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AuthenticationProviderError(code="malformed_provider_response") from error
    if not isinstance(value, dict):
        raise AuthenticationProviderError(code="malformed_provider_response")
    return value


def _cache_ttl(cache_control: str | None) -> float:
    if not cache_control:
        return _DEFAULT_CACHE_SECONDS
    directives = {item.strip().lower() for item in cache_control.split(",")}
    if "no-store" in directives:
        return 0.0
    for directive in directives:
        if directive.startswith("max-age="):
            try:
                return min(_MAX_CACHE_SECONDS, max(0.0, float(directive.split("=", 1)[1])))
            except ValueError:
                return _DEFAULT_CACHE_SECONDS
    return _DEFAULT_CACHE_SECONDS


def _decode_json_segment(value: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(_b64decode(value))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthenticationProviderError(code="malformed_id_token") from error
    if not isinstance(decoded, dict):
        raise AuthenticationProviderError(code="malformed_id_token")
    return decoded


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _b64_int(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("JWK integer must be string")
    return int.from_bytes(_b64decode(value), "big")
