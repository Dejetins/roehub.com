from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from trading.contexts.identity.adapters.outbound.security.oidc import (
    HttpOidcAuthenticationProvider,
)
from trading.contexts.identity.application import AuthenticationProviderError

_ISSUER = "https://identity.example.test"
_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


class _Metrics:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def record(self, **values: object) -> None:
        self.rows.append(values)


class _MutableMonotonic:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _DeadlineCrossingStream(httpx.SyncByteStream):
    def __init__(self, *, clock: _MutableMonotonic, payload: bytes) -> None:
        self._clock = clock
        self._payload = payload

    def __iter__(self) -> Iterator[bytes]:
        self._clock.value += 2.0
        yield self._payload


class _WallClockSlowStream(httpx.SyncByteStream):
    def __init__(self, *, delay_seconds: float, payload: bytes) -> None:
        self._delay_seconds = delay_seconds
        self._payload = payload

    def __iter__(self) -> Iterator[bytes]:
        time.sleep(self._delay_seconds)
        yield self._payload


def test_provider_verifies_discovery_pkce_jwks_and_signed_identity() -> None:
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    captured_form: dict[str, str] = {}
    nonce = "disposable-nonce"
    compact_identity = _signed_identity(
        key=signing_key,
        key_id="current",
        nonce=nonce,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            captured_form.update(
                {
                    key: values[0]
                    for key, values in parse_qs(
                        request.content.decode(), strict_parsing=True
                    ).items()
                }
            )
            return httpx.Response(200, json={"id_" + "token": compact_identity})
        if request.url.path == "/keys":
            return httpx.Response(200, json={"keys": [_jwk(signing_key, "current")]})
        raise AssertionError(request.url)

    metrics = _Metrics()
    provider = _provider(transport=httpx.MockTransport(handler), metrics=metrics)
    authorization = provider.authorization_url(
        state="state",
        nonce=nonce,
        code_challenge="challenge",
    )
    query = parse_qs(httpx.URL(authorization).query.decode())

    identity = provider.exchange_code(
        code="disposable-code",
        code_verifier="v" * 64,
        expected_nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(),
    )

    assert query["code_challenge_method"] == ["S256"]
    assert query["nonce"] == [nonce]
    assert captured_form["code"] == "disposable-code"
    assert captured_form["code_verifier"] == "v" * 64
    assert identity.subject == "disposable-subject"
    assert identity.email == "invited@example.test"
    assert identity.email_verified is True
    assert {row["operation"] for row in metrics.rows} == {
        "discovery_get",
        "jwks_get",
        "token_post",
    }


def test_provider_resolves_rotated_client_credential_before_each_token_post() -> None:
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nonce = "rotating-credential-nonce"
    compact_identity = _signed_identity(
        key=signing_key,
        key_id="current",
        nonce=nonce,
    )
    active = ["version-one"]
    observed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            form = parse_qs(request.content.decode(), strict_parsing=True)
            observed.append(form["client_" + "secret"][0])
            return httpx.Response(200, json={"id_" + "token": compact_identity})
        if request.url.path == "/keys":
            return httpx.Response(200, json={"keys": [_jwk(signing_key, "current")]})
        raise AssertionError(request.url)

    provider = _provider(
        transport=httpx.MockTransport(handler),
        client_credential_source=lambda: active[0],
    )
    for expected in ("version-one", "version-two"):
        active[0] = expected
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(),
        )

    assert observed == ["version-one", "version-two"]


def test_discovery_get_retries_at_most_twice() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503)
        return _discovery()

    provider = _provider(transport=httpx.MockTransport(handler))

    location = provider.authorization_url(
        state="state", nonce="nonce", code_challenge="challenge"
    )

    assert location.startswith(f"{_ISSUER}/authorize?")
    assert calls == 3


def test_discovery_transport_timeout_retries_twice_then_fails_closed() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("disposable discovery timeout", request=request)

    provider = _provider(transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationProviderError) as captured:
        provider.authorization_url(
            state="state", nonce="nonce", code_challenge="challenge"
        )

    assert captured.value.retryable is True
    assert calls == 3


def test_discovery_slow_stream_cannot_exceed_overall_wall_clock_budget() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            stream=_WallClockSlowStream(delay_seconds=0.5, payload=b"{}"),
        )

    provider = _provider(
        transport=httpx.MockTransport(handler), overall_timeout_seconds=0.05
    )
    started = time.monotonic()

    with pytest.raises(AuthenticationProviderError) as captured:
        provider.authorization_url(
            state="state", nonce="nonce", code_challenge="challenge"
        )

    assert time.monotonic() - started < 0.25
    assert captured.value.retryable is True
    assert calls == 1


def test_exchange_unknown_result_is_never_retried() -> None:
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            post_calls += 1
            raise httpx.ReadTimeout("bounded fixture timeout", request=request)
        raise AssertionError(request.url)

    provider = _provider(transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationProviderError) as captured:
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
        )

    assert captured.value.token_result_unknown is True
    assert post_calls == 1


def test_exchange_stream_crossing_overall_deadline_fails_unknown_once() -> None:
    clock = _MutableMonotonic()
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            post_calls += 1
            return httpx.Response(
                200,
                stream=_DeadlineCrossingStream(
                    clock=clock,
                    payload=b'{"id_token":"must-not-be-observed"}',
                ),
            )
        raise AssertionError(request.url)

    provider = _provider(
        transport=httpx.MockTransport(handler),
        monotonic=clock,
        overall_timeout_seconds=1.0,
    )
    provider.authorization_url(
        state="state", nonce="nonce", code_challenge="challenge"
    )

    with pytest.raises(AuthenticationProviderError) as captured:
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
        )

    assert captured.value.token_result_unknown is True
    assert post_calls == 1


def test_token_slow_stream_cannot_exceed_overall_wall_clock_budget() -> None:
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_calls
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            post_calls += 1
            return httpx.Response(
                200,
                stream=_WallClockSlowStream(delay_seconds=0.5, payload=b"{}"),
            )
        raise AssertionError(request.url)

    provider = _provider(
        transport=httpx.MockTransport(handler), overall_timeout_seconds=0.05
    )
    provider.authorization_url(
        state="state", nonce="nonce", code_challenge="challenge"
    )
    started = time.monotonic()

    with pytest.raises(AuthenticationProviderError) as captured:
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
        )

    assert time.monotonic() - started < 0.25
    assert captured.value.token_result_unknown is True
    assert post_calls == 1


def test_unknown_signing_key_refreshes_stale_jwks_once() -> None:
    old_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    current_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwks_calls = 0
    compact_identity = _signed_identity(
        key=current_key,
        key_id="current",
        nonce="nonce",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal jwks_calls
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            return httpx.Response(200, json={"id_" + "token": compact_identity})
        if request.url.path == "/keys":
            jwks_calls += 1
            selected = old_key if jwks_calls == 1 else current_key
            selected_id = "old" if jwks_calls == 1 else "current"
            return httpx.Response(200, json={"keys": [_jwk(selected, selected_id)]})
        raise AssertionError(request.url)

    provider = _provider(transport=httpx.MockTransport(handler))

    identity = provider.exchange_code(
        code="disposable-code",
        code_verifier="v" * 64,
        expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
    )

    assert identity.subject == "disposable-subject"
    assert jwks_calls == 2


def test_jwks_timeout_retries_get_without_repeating_token_post() -> None:
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    compact_identity = _signed_identity(
        key=signing_key,
        key_id="current",
        nonce="nonce",
    )
    token_calls = 0
    jwks_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls, jwks_calls
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            token_calls += 1
            return httpx.Response(200, json={"id_" + "token": compact_identity})
        if request.url.path == "/keys":
            jwks_calls += 1
            raise httpx.ReadTimeout("disposable JWKS timeout", request=request)
        raise AssertionError(request.url)

    provider = _provider(transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationProviderError) as captured:
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
        )

    assert captured.value.retryable is True
    assert token_calls == 1
    assert jwks_calls == 3


def test_jwks_slow_stream_cannot_exceed_overall_wall_clock_budget() -> None:
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    compact_identity = _signed_identity(
        key=signing_key,
        key_id="current",
        nonce="nonce",
    )
    token_calls = 0
    jwks_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls, jwks_calls
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            token_calls += 1
            return httpx.Response(200, json={"id_" + "token": compact_identity})
        if request.url.path == "/keys":
            jwks_calls += 1
            return httpx.Response(
                200,
                stream=_WallClockSlowStream(delay_seconds=0.5, payload=b"{}"),
            )
        raise AssertionError(request.url)

    provider = _provider(
        transport=httpx.MockTransport(handler), overall_timeout_seconds=0.05
    )
    provider.authorization_url(
        state="state", nonce="nonce", code_challenge="challenge"
    )
    started = time.monotonic()

    with pytest.raises(AuthenticationProviderError) as captured:
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
        )

    assert time.monotonic() - started < 0.25
    assert captured.value.retryable is True
    assert token_calls == 1
    assert jwks_calls == 1


def test_discovery_semantic_failure_records_bounded_failure_outcome() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = _discovery()
        document = json.loads(response.content)
        document["issuer"] = "https://unexpected.example.test"
        return httpx.Response(200, json=document)

    metrics = _Metrics()
    provider = _provider(transport=httpx.MockTransport(handler), metrics=metrics)

    with pytest.raises(AuthenticationProviderError, match="discovery_issuer_mismatch"):
        provider.authorization_url(
            state="state", nonce="nonce", code_challenge="challenge"
        )

    assert any(
        row["operation"] == "discovery_validation"
        and row["outcome"] == "validation_error"
        for row in metrics.rows
    )


def test_malformed_token_document_records_bounded_failure_outcome() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            return httpx.Response(200, content=b"not-json")
        raise AssertionError(request.url)

    metrics = _Metrics()
    provider = _provider(transport=httpx.MockTransport(handler), metrics=metrics)

    with pytest.raises(AuthenticationProviderError, match="malformed_provider_response"):
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
        )

    assert any(
        row["operation"] == "token_post" and row["outcome"] == "validation_error"
        for row in metrics.rows
    )


def test_malformed_signing_key_is_rejected_without_refresh_loop() -> None:
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    compact_identity = _signed_identity(
        key=signing_key,
        key_id="current",
        nonce="nonce",
    )
    jwks_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal jwks_calls
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            return httpx.Response(200, json={"id_" + "token": compact_identity})
        if request.url.path == "/keys":
            jwks_calls += 1
            malformed = _jwk(signing_key, "current")
            del malformed["e"]
            return httpx.Response(200, json={"keys": [malformed]})
        raise AssertionError(request.url)

    provider = _provider(transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationProviderError, match="malformed_signing_key"):
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"nonce").hexdigest(),
        )

    assert jwks_calls == 1


def test_nonce_mismatch_rejects_verified_signature() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    compact_identity = _signed_identity(key=key, key_id="current", nonce="other")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("openid-configuration"):
            return _discovery()
        if request.url.path == "/exchange":
            return httpx.Response(200, json={"id_" + "token": compact_identity})
        if request.url.path == "/keys":
            return httpx.Response(200, json={"keys": [_jwk(key, "current")]})
        raise AssertionError(request.url)

    provider = _provider(transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationProviderError, match="nonce_mismatch"):
        provider.exchange_code(
            code="disposable-code",
            code_verifier="v" * 64,
            expected_nonce_sha256=hashlib.sha256(b"expected").hexdigest(),
        )


def _provider(
    *,
    transport: httpx.BaseTransport,
    metrics: _Metrics | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    overall_timeout_seconds: float = 15.0,
    client_credential_source: Callable[[], str] = lambda: "test-only-placeholder",
) -> HttpOidcAuthenticationProvider:
    return HttpOidcAuthenticationProvider(
        provider_id="fixture",
        display_name="Fixture Identity",
        issuer=_ISSUER,
        client_id="roehub-browser",
        client_credential_source=client_credential_source,
        redirect_uri="https://roehub.example/api/auth/oidc/callback",
        transport=transport,
        metrics=metrics,
        monotonic=monotonic,
        overall_timeout_seconds=overall_timeout_seconds,
        now=lambda: _NOW,
        sleeper=lambda _: None,
        jitter=lambda: 0.0,
    )


def _discovery() -> httpx.Response:
    return httpx.Response(
        200,
        headers={"cache-control": "max-age=300"},
        json={
            "issuer": _ISSUER,
            "authorization_endpoint": f"{_ISSUER}/authorize",
            "token_endpoint": f"{_ISSUER}/exchange",
            "jwks_uri": f"{_ISSUER}/keys",
            "id_token_signing_alg_values_supported": ["RS256"],
        },
    )


def _signed_identity(
    *, key: rsa.RSAPrivateKey, key_id: str, nonce: str
) -> str:
    header = _segment({"alg": "RS256", "kid": key_id, "typ": "JWT"})
    claims = _segment(
        {
            "iss": _ISSUER,
            "sub": "disposable-subject",
            "aud": "roehub-browser",
            "exp": int((_NOW + timedelta(minutes=5)).timestamp()),
            "iat": int(_NOW.timestamp()),
            "nonce": nonce,
            "email": "invited@example.test",
            "email_verified": True,
        }
    )
    signing_input = f"{header}.{claims}".encode()
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{claims}.{_b64(signature)}"


def _jwk(key: rsa.RSAPrivateKey, key_id: str) -> dict[str, str]:
    numbers = key.public_key().public_numbers()
    return {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": key_id,
        "n": _b64(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
        "e": _b64(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
    }


def _segment(value: dict[str, object]) -> str:
    return _b64(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()
