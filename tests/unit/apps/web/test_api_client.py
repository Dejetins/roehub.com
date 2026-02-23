from __future__ import annotations

import httpx

from apps.web.main.api_client import HttpxCurrentUserApiClient


def test_current_user_client_forwards_cookie_header_verbatim() -> None:
    """
    Verify internal API client forwards incoming cookie header without parsing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cookie header is opaque and must be copied as-is to API request.
    Raises:
        AssertionError: If request path or cookie header differs from expected values.
    Side Effects:
        None.
    """
    captured_headers: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """
        Capture request headers/path and return deterministic identity payload.

        Args:
            request: Outbound request built by web internal API client.
        Returns:
            httpx.Response: Deterministic successful identity response.
        Assumptions:
            Handler runs synchronously inside `httpx.MockTransport`.
        Raises:
            None.
        Side Effects:
            Mutates `captured_headers` for test assertions.
        """
        captured_headers["cookie"] = request.headers.get("cookie")
        captured_headers["path"] = request.url.path
        return httpx.Response(
            status_code=200,
            json={"user_id": "00000000-0000-0000-0000-000000000123", "paid_level": "free"},
        )

    client = HttpxCurrentUserApiClient(
        api_base_url="http://api.local",
        transport=httpx.MockTransport(handler),
    )

    result = client.fetch_current_user(cookie_header="session=abc; mode=dev")

    assert captured_headers["cookie"] == "session=abc; mode=dev"
    assert captured_headers["path"] == "/api/auth/current-user"
    assert result.status_code == 200
    assert result.user is not None
    assert result.user.user_id == "00000000-0000-0000-0000-000000000123"
