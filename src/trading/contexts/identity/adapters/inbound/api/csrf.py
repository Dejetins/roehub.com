from __future__ import annotations

from urllib.parse import urlparse

from starlette.requests import Request


def same_origin_rejection_reason(
    *,
    request: Request,
    fail_closed_without_origin: bool,
) -> str | None:
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    if origin is None and referer is None:
        if fail_closed_without_origin:
            return "csrf_required"
        return None

    expected_host = request.headers.get("host", "")
    for candidate in (origin, referer):
        if candidate is None:
            continue
        parsed = urlparse(candidate)
        if parsed.netloc and parsed.netloc != expected_host:
            return "csrf_origin_mismatch"
    return None
