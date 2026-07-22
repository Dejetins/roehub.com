from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from fastapi import HTTPException
from starlette.requests import Request

from trading.contexts.identity.adapters.inbound.api.csrf import (
    same_origin_rejection_reason,
)
from trading.contexts.identity.application.mutation_security.models import (
    MutationSecurityDecision,
    MutationSecurityRequest,
)
from trading.contexts.identity.application.mutation_security.service import (
    MutationSecurityService,
)


class BrowserMutationTransportReason(StrEnum):
    ORIGIN_REQUIRED = "origin_required"
    ORIGIN_MISMATCH = "origin_mismatch"
    CSRF_REQUIRED = "csrf_required"
    CSRF_MISMATCH = "csrf_mismatch"


@dataclass(frozen=True, slots=True)
class BrowserMutationTransportCheck:
    cookie_authenticated: bool
    accepted: bool
    reason: BrowserMutationTransportReason | None = None


class BrowserMutationRequestGuard:
    """Apply origin and double-submit CSRF checks only to session-cookie clients."""

    def __init__(
        self,
        *,
        session_cookie_name: str,
        csrf_cookie_name: str = "roehub_csrf",
        csrf_header_name: str = "x-csrf-token",
    ) -> None:
        values = (session_cookie_name, csrf_cookie_name, csrf_header_name)
        if any(not value.strip() for value in values):
            raise ValueError("browser mutation cookie and header names must be non-empty")
        self._session_cookie_name = session_cookie_name.strip()
        self._csrf_cookie_name = csrf_cookie_name.strip()
        self._csrf_header_name = csrf_header_name.strip()

    def check(self, *, request: Request) -> BrowserMutationTransportCheck:
        session_cookie = request.cookies.get(self._session_cookie_name)
        if session_cookie is None or not session_cookie.strip():
            return BrowserMutationTransportCheck(
                cookie_authenticated=False,
                accepted=True,
            )
        origin_reason = same_origin_rejection_reason(
            request=request,
            fail_closed_without_origin=True,
        )
        if origin_reason == "csrf_required":
            return BrowserMutationTransportCheck(
                cookie_authenticated=True,
                accepted=False,
                reason=BrowserMutationTransportReason.ORIGIN_REQUIRED,
            )
        if origin_reason is not None:
            return BrowserMutationTransportCheck(
                cookie_authenticated=True,
                accepted=False,
                reason=BrowserMutationTransportReason.ORIGIN_MISMATCH,
            )
        if not _has_valid_origin_shape(request=request):
            return BrowserMutationTransportCheck(
                cookie_authenticated=True,
                accepted=False,
                reason=BrowserMutationTransportReason.ORIGIN_MISMATCH,
            )
        cookie_token = request.cookies.get(self._csrf_cookie_name)
        header_token = request.headers.get(self._csrf_header_name)
        if (
            cookie_token is None
            or header_token is None
            or not cookie_token.strip()
            or not header_token.strip()
        ):
            return BrowserMutationTransportCheck(
                cookie_authenticated=True,
                accepted=False,
                reason=BrowserMutationTransportReason.CSRF_REQUIRED,
            )
        if not secrets.compare_digest(cookie_token.encode(), header_token.encode()):
            return BrowserMutationTransportCheck(
                cookie_authenticated=True,
                accepted=False,
                reason=BrowserMutationTransportReason.CSRF_MISMATCH,
            )
        return BrowserMutationTransportCheck(cookie_authenticated=True, accepted=True)

    def enforce(self, *, request: Request) -> BrowserMutationTransportCheck:
        check = self.check(request=request)
        if check.accepted:
            return check
        raise HTTPException(
            status_code=403,
            detail={
                "error": "browser_mutation_rejected",
                "message": "Browser mutation security proof is required.",
                "reason": check.reason.value if check.reason is not None else "denied",
            },
        )


class BrowserMutationEnvelope:
    """Single route-facing facade binding transport proof to application checks."""

    def __init__(
        self,
        *,
        request_guard: BrowserMutationRequestGuard,
        security: MutationSecurityService,
    ) -> None:
        if request_guard is None:  # type: ignore[truthy-bool]
            raise ValueError("BrowserMutationEnvelope requires request_guard")
        if security is None:  # type: ignore[truthy-bool]
            raise ValueError("BrowserMutationEnvelope requires security")
        self._request_guard = request_guard
        self._security = security

    def decide(
        self,
        *,
        http_request: Request,
        mutation: MutationSecurityRequest,
    ) -> MutationSecurityDecision:
        proof = self._request_guard.check(request=http_request)
        return self._security.decide(
            request=mutation,
            transport_proof=proof,
        )


def _has_valid_origin_shape(*, request: Request) -> bool:
    found = False
    for value in (request.headers.get("origin"), request.headers.get("referer")):
        if value is None:
            continue
        found = True
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
    return found
