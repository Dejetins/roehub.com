"""PROTOTYPE-only route boundary around the unchanged Roehub FastAPI/Jinja app."""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path
from typing import Final

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from apps.web.main.api_client import CurrentUserApiResult, WebCurrentUser
from apps.web.main.app import create_app as create_ssr_app

_APP_ROOT: Final = Path(__file__).resolve().parent
_DIST_ROOT: Final = _APP_ROOT / "dist"
_SAFE_SESSION: Final = "prototype-safe-session"
_REST_SEQUENCE = itertools.count(1)
_SSE_SEQUENCE = itertools.count(1)


class _PrototypeCurrentUserApiClient:
    """Server-side fixture adapter; the React client never decides authorization."""

    def fetch_current_user(self, *, cookie_header: str | None) -> CurrentUserApiResult:
        if cookie_header is None or f"roehub_session_id={_SAFE_SESSION}" not in cookie_header:
            return CurrentUserApiResult(status_code=401, user=None, error_message=None)
        return CurrentUserApiResult(
            status_code=200,
            user=WebCurrentUser(user_id="prototype-user", paid_level="fixture"),
            error_message=None,
        )


def _fixture_rows(status: str = "running") -> list[dict[str, object]]:
    return [
        {
            "id": "bt-safe-001",
            "name": "BTC daily fixture",
            "status": status,
            "returnPct": 8.42,
        },
        {
            "id": "bt-safe-002",
            "name": "ETH volatility fixture",
            "status": "completed",
            "returnPct": 3.17,
        },
        {
            "id": "bt-safe-003",
            "name": "Empty-market fixture",
            "status": "queued",
            "returnPct": 0.0,
        },
    ]


def create_prototype_app() -> FastAPI:
    """Compose bounded prototype routes before mounting the unchanged SSR gateway."""
    app = FastAPI(
        title="Roehub frontend architecture PROTOTYPE",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.mount(
        "/__prototype/react/assets",
        StaticFiles(directory=_DIST_ROOT / "assets", check_dir=False),
        name="prototype-assets",
    )

    @app.middleware("http")
    async def mark_prototype_boundary(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        if request.url.path.startswith("/__prototype/"):
            response.headers["X-Roehub-Prototype"] = "true"
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/__prototype/api/backtests", include_in_schema=False)
    async def get_backtests(request: Request, latency_ms: int = 36) -> JSONResponse:
        bounded_latency = min(1200, max(0, latency_ms))
        elapsed = 0
        while elapsed < bounded_latency:
            await asyncio.sleep(min(20, bounded_latency - elapsed) / 1000)
            elapsed += 20
            if await request.is_disconnected():
                return JSONResponse({"cancelled": True}, status_code=499)
        return JSONResponse(
            {
                "revision": f"rest-{next(_REST_SEQUENCE):04d}",
                "source": "mock-rest",
                "serverAuthorization": "fixture-server-projection",
                "rows": _fixture_rows(),
            }
        )

    @app.get("/__prototype/events", include_in_schema=False)
    async def stream_events(request: Request) -> StreamingResponse:
        async def events():  # type: ignore[no-untyped-def]
            statuses = ("running", "completed", "queued")
            while not await request.is_disconnected():
                sequence = next(_SSE_SEQUENCE)
                payload = {
                    "revision": f"sse-{sequence:04d}",
                    "rowId": "bt-safe-001",
                    "status": statuses[sequence % len(statuses)],
                }
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                await asyncio.sleep(0.12)

        return StreamingResponse(events(), media_type="text/event-stream")

    def react_index() -> Response:
        if not (_DIST_ROOT / "index.html").is_file():
            return Response("Run `npm run build` before starting the prototype", status_code=503)
        response = FileResponse(_DIST_ROOT / "index.html", media_type="text/html")
        response.set_cookie(
            "roehub_session_id",
            _SAFE_SESSION,
            httponly=True,
            samesite="lax",
            secure=False,
        )
        return response

    app.add_api_route(
        "/__prototype/react",
        react_index,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        "/__prototype/react/",
        react_index,
        methods=["GET"],
        include_in_schema=False,
    )

    @app.get("/__prototype/react/{spa_path:path}", include_in_schema=False)
    def react_history_fallback(spa_path: str) -> Response:
        del spa_path
        return react_index()

    ssr_app = create_ssr_app(
        environ={
            "WEB_API_BASE_URL": "http://prototype.invalid",
            "WEB_API_UPSTREAM_URL": "http://prototype.invalid",
            "WEB_ASSET_VERSION": "prototype",
        }
    )
    ssr_app.state.current_user_api_client = _PrototypeCurrentUserApiClient()
    app.mount("/", ssr_app, name="current-fastapi-jinja-ssr")
    return app


if __name__ == "__main__":
    uvicorn.run(
        create_prototype_app(),
        host="127.0.0.1",
        port=4173,
        log_level="warning",
    )
