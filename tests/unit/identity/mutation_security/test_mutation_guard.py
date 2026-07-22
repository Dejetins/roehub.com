from __future__ import annotations

from typing import cast

from starlette.requests import Request

from trading.contexts.identity.adapters.inbound.api.mutation_guard import (
    BrowserMutationEnvelope,
    BrowserMutationRequestGuard,
    BrowserMutationTransportReason,
)
from trading.contexts.identity.application.mutation_security import (
    IdempotencyDisposition,
    MutationSecurityDecision,
    MutationSecurityRequest,
    MutationSecurityService,
)


class _Security:
    def __init__(self) -> None:
        self.cookie_authenticated: bool | None = None
        self.transport_accepted: bool | None = None

    def decide(self, *, request, transport_proof) -> MutationSecurityDecision:
        _ = request
        self.cookie_authenticated = transport_proof.cookie_authenticated
        self.transport_accepted = transport_proof.accepted
        return MutationSecurityDecision(
            applicable=transport_proof.cookie_authenticated,
            allowed=transport_proof.accepted,
            reason=None,
            authorization=None,
            idempotency=IdempotencyDisposition.NOT_REQUIRED,
        )


def _request(*, headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/mutation",
            "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )


def test_cookie_authenticated_browser_rejects_foreign_origin_even_with_csrf() -> None:
    guard = BrowserMutationRequestGuard(session_cookie_name="roehub_session_id")
    request = _request(
        headers={
            "Host": "testserver",
            "Origin": "https://evil.example",
            "Cookie": "roehub_session_id=<redacted>; roehub_csrf=<redacted>",
            "X-CSRF-Token": "<redacted>",
        }
    )

    result = guard.check(request=request)

    assert result.cookie_authenticated is True
    assert result.accepted is False
    assert result.reason is BrowserMutationTransportReason.ORIGIN_MISMATCH


def test_cookie_authenticated_browser_requires_origin_and_double_submit_csrf() -> None:
    guard = BrowserMutationRequestGuard(session_cookie_name="roehub_session_id")
    no_origin = guard.check(request=_request(headers={"Cookie": "roehub_session_id=<redacted>"}))
    no_csrf = guard.check(
        request=_request(
            headers={
                "Host": "testserver",
                "Origin": "http://testserver",
                "Cookie": "roehub_session_id=<redacted>",
            }
        )
    )
    mismatch = guard.check(
        request=_request(
            headers={
                "Host": "testserver",
                "Origin": "http://testserver",
                "Cookie": "roehub_session_id=<redacted>; roehub_csrf=<redacted-a>",
                "X-CSRF-Token": "<redacted-b>",
            }
        )
    )
    opaque_origin = guard.check(
        request=_request(
            headers={
                "Host": "testserver",
                "Origin": "null",
                "Cookie": "roehub_session_id=<redacted>; roehub_csrf=<redacted>",
                "X-CSRF-Token": "<redacted>",
            }
        )
    )

    assert no_origin.reason is BrowserMutationTransportReason.ORIGIN_REQUIRED
    assert no_csrf.reason is BrowserMutationTransportReason.CSRF_REQUIRED
    assert mismatch.reason is BrowserMutationTransportReason.CSRF_MISMATCH
    assert opaque_origin.reason is BrowserMutationTransportReason.ORIGIN_MISMATCH


def test_api_client_without_browser_session_cookie_keeps_existing_auth_path() -> None:
    guard = BrowserMutationRequestGuard(session_cookie_name="roehub_session_id")
    request = _request(
        headers={
            "Authorization": "Bearer <redacted>",
            "Origin": "https://api-client.example",
        }
    )

    result = guard.enforce(request=request)

    assert result.cookie_authenticated is False
    assert result.accepted is True
    assert result.reason is None


def test_cookie_authenticated_browser_accepts_same_origin_matching_csrf() -> None:
    guard = BrowserMutationRequestGuard(session_cookie_name="roehub_session_id")
    request = _request(
        headers={
            "Host": "testserver",
            "Origin": "http://testserver",
            "Cookie": "roehub_session_id=<redacted>; roehub_csrf=<redacted>",
            "X-CSRF-Token": "<redacted>",
        }
    )

    result = guard.enforce(request=request)

    assert result.cookie_authenticated is True
    assert result.accepted is True


def test_route_facing_envelope_passes_guard_proof_into_application_boundary() -> None:
    security = _Security()
    envelope = BrowserMutationEnvelope(
        request_guard=BrowserMutationRequestGuard(session_cookie_name="roehub_session_id"),
        security=cast(MutationSecurityService, security),
    )
    request = _request(
        headers={
            "Host": "testserver",
            "Origin": "http://testserver",
            "Cookie": "roehub_session_id=<redacted>; roehub_csrf=<redacted>",
            "X-CSRF-Token": "<redacted>",
        }
    )

    decision = envelope.decide(
        http_request=request,
        mutation=cast(MutationSecurityRequest, object()),
    )

    assert decision.allowed is True
    assert security.cookie_authenticated is True
    assert security.transport_accepted is True
