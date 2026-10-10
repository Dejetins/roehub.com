"""Browser marker is identity metadata, never an authentication credential."""

from __future__ import annotations

import re
from datetime import datetime
from hashlib import sha256
from uuid import UUID

from fastapi import Request, Response

from trading.contexts.identity.application.ports.session_repository import SessionRepository

BROWSER_COOKIE = "roehub_browser"


def register_browser(
    *,
    request: Request,
    response: Response,
    repository: SessionRepository,
    session_id: UUID,
    now: datetime,
    secure: bool,
) -> UUID:
    marker = request.cookies.get(BROWSER_COOKIE, "")
    if re.fullmatch(r"[0-9a-f]{64}", marker) is None:
        # Deterministic for concurrent adoption requests; cannot recover the bearer UUID.
        marker = sha256(f"roehub-browser-marker:{session_id}".encode()).hexdigest()
    device_id = repository.bind_device(
        session_id=session_id,
        device_key=sha256(marker.encode()).hexdigest(),
        label=browser_label(request.headers.get("user-agent", "")),
        now=now,
    )
    response.set_cookie(
        BROWSER_COOKIE,
        marker,
        max_age=365 * 86400,
        path="/",
        httponly=True,
        secure=secure,
        samesite="lax",
    )
    response.headers["Cache-Control"] = "no-store"
    return device_id


def browser_label(user_agent: str) -> str:
    """Store a bounded display label; never raw User-Agent, IP or inferred location."""
    ua = user_agent[:2048]
    browser = next(
        (
            name
            for token, name in (
                ("Edg/", "Edge"),
                ("OPR/", "Opera"),
                ("Firefox/", "Firefox"),
                ("FxiOS/", "Firefox"),
                ("Chrome/", "Chrome"),
                ("CriOS/", "Chrome"),
                ("Safari/", "Safari"),
            )
            if token in ua
        ),
        "Browser",
    )
    platform = next(
        (
            name
            for token, name in (
                ("Android", "Android"),
                ("iPhone", "iPhone"),
                ("iPad", "iPad"),
                ("Windows", "Windows"),
                ("Macintosh", "macOS"),
                ("Linux", "Linux"),
            )
            if token in ua
        ),
        None,
    )
    return f"{browser} · {platform}" if platform else browser
