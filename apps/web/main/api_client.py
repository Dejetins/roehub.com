from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

_CURRENT_USER_PATH = "/api/auth/current-user"


@dataclass(frozen=True)
class WebCurrentUser:
    """
    WebCurrentUser is an authenticated principal snapshot returned by identity API.

    Docs:
      - docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - apps/web/main/app.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
    """

    user_id: str
    paid_level: str


@dataclass(frozen=True)
class CurrentUserApiResult:
    """
    CurrentUserApiResult stores deterministic outcome of current-user API lookup.

    Docs:
      - docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
    Related:
      - apps/web/main/app.py
      - apps/web/main/api_client.py
    """

    status_code: int
    user: WebCurrentUser | None
    error_message: str | None


class CurrentUserApiClient(Protocol):
    """
    CurrentUserApiClient defines contract for server-side current-user lookup.

    Docs:
      - docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
    Related:
      - apps/web/main/app.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
    """

    def fetch_current_user(self, *, cookie_header: str | None) -> CurrentUserApiResult:
        """
        Fetch authenticated user using opaque forwarded browser cookies.

        Args:
            cookie_header: Raw `Cookie` header copied from incoming browser request.
        Returns:
            CurrentUserApiResult: Deterministic lookup outcome.
        Assumptions:
            Cookie header must be forwarded verbatim without parsing JWT/cookie keys.
        Raises:
            NotImplementedError: Always raised by the contract base.
        Side Effects:
            None.
        """
        ...


class HttpxCurrentUserApiClient(CurrentUserApiClient):
    """
    HttpxCurrentUserApiClient reads `/api/auth/current-user` through `httpx`.

    Docs:
      - docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
    Related:
      - apps/web/main/settings.py
      - apps/web/main/app.py
      - src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
    """

    def __init__(
        self,
        *,
        api_base_url: str,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Initialize API adapter with immutable HTTP settings and optional mock transport.

        Args:
            api_base_url: Absolute API base URL from `WEB_API_BASE_URL`.
            timeout_seconds: HTTP timeout for current-user lookup.
            transport: Optional httpx transport override used in tests.
        Returns:
            None.
        Assumptions:
            `api_base_url` points to server that accepts `/api/auth/current-user`.
        Raises:
            ValueError: If URL is blank or timeout is non-positive.
        Side Effects:
            None.
        """
        normalized_api_base_url = api_base_url.strip().rstrip("/")
        if not normalized_api_base_url:
            raise ValueError("HttpxCurrentUserApiClient requires non-empty api_base_url")
        if timeout_seconds <= 0:
            raise ValueError("HttpxCurrentUserApiClient requires positive timeout_seconds")

        self._api_base_url = normalized_api_base_url
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def fetch_current_user(self, *, cookie_header: str | None) -> CurrentUserApiResult:
        """
        Fetch current user through identity API using forwarded browser cookies.

        Args:
            cookie_header: Raw `Cookie` header from incoming browser request.
        Returns:
            CurrentUserApiResult: Status and optional principal snapshot.
        Assumptions:
            Identity API contract returns `user_id` and `paid_level` on HTTP 200.
        Raises:
            None.
        Side Effects:
            Performs one outbound HTTP request to identity API.
        """
        request_headers = _build_request_headers(cookie_header=cookie_header)
        endpoint_url = f"{self._api_base_url}{_CURRENT_USER_PATH}"
        try:
            with httpx.Client(
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as http_client:
                response = http_client.get(endpoint_url, headers=request_headers)
        except httpx.HTTPError as error:
            return CurrentUserApiResult(
                status_code=503,
                user=None,
                error_message=f"Current-user request failed: {error}",
            )

        if response.status_code == 401:
            return CurrentUserApiResult(status_code=401, user=None, error_message=None)
        if response.status_code == 200:
            return _build_success_result(response=response)
        return CurrentUserApiResult(
            status_code=response.status_code,
            user=None,
            error_message=f"Unexpected current-user status: {response.status_code}",
        )



def _build_request_headers(*, cookie_header: str | None) -> dict[str, str]:
    """
    Build deterministic request headers for identity current-user HTTP call.

    Args:
        cookie_header: Raw incoming browser Cookie header.
    Returns:
        dict[str, str]: Header mapping forwarded to identity API.
    Assumptions:
        Cookie is opaque and MUST be forwarded without parsing.
    Raises:
        None.
    Side Effects:
        None.
    """
    if cookie_header is None:
        return {}
    normalized_cookie = cookie_header.strip()
    if not normalized_cookie:
        return {}
    return {"Cookie": normalized_cookie}



def _build_success_result(*, response: httpx.Response) -> CurrentUserApiResult:
    """
    Parse HTTP 200 current-user payload into deterministic domain-neutral shape.

    Args:
        response: Successful HTTP response from identity API.
    Returns:
        CurrentUserApiResult: Parsed principal payload or parse error details.
    Assumptions:
        Payload is JSON object with `user_id` and `paid_level` string fields.
    Raises:
        None.
    Side Effects:
        None.
    """
    try:
        payload = dict(response.json())
    except (TypeError, ValueError):
        return CurrentUserApiResult(
            status_code=502,
            user=None,
            error_message="Current-user response is not a valid JSON object",
        )

    user_id = str(payload.get("user_id", "")).strip()
    paid_level = str(payload.get("paid_level", "")).strip()
    if not user_id or not paid_level:
        return CurrentUserApiResult(
            status_code=502,
            user=None,
            error_message="Current-user payload is missing required fields",
        )

    return CurrentUserApiResult(
        status_code=200,
        user=WebCurrentUser(user_id=user_id, paid_level=paid_level),
        error_message=None,
    )
