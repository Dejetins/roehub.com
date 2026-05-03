from __future__ import annotations

import asyncio
import json
from typing import Callable, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.application.ports import (
    StrategyRealtimeOutputReader,
    StrategyRealtimeStreamMessageV1,
    StrategyRealtimeStreamUnavailableError,
)
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]

_DEFAULT_STREAM_COUNT = 50
_DEFAULT_BLOCK_MS = 1000


class StrategyStreamOwnerScope(Protocol):
    def ensure_strategy_owner(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
    ) -> None:
        ...


def build_strategy_streams_router(
    *,
    current_user_dependency: CurrentUserDependency,
    stream_reader: StrategyRealtimeOutputReader,
    owner_scope: StrategyStreamOwnerScope,
) -> APIRouter:
    """
    Build read-only Strategy SSE bridge over per-user Redis Streams.

    Local contract:
    - browser path: `GET /api/stream/strategies?strategy_id=&last_event_id=`
    - backend path: `GET /stream/strategies`
    - owner scope: auth principal is resolved before stream read;
      optional strategy_id is owner-checked
    - request DTO: optional `strategy_id`, optional `last_event_id`, bounded `count`
    - response DTO: text/event-stream events `status`, `strategy.metric`,
      `strategy.event`, `fallback`
    - status codes: 200 stream, 401 auth, 403/404 owner guard, 422 query validation
    - error payload: non-stream errors use RoehubError envelope;
      stream substrate errors emit fallback event
    - pagination: Redis `last_event_id`; browser may also send `Last-Event-ID` header
    - cache identity: none; live user-scoped stream
    - compatibility: compatible-change, additive `/stream/strategies` surface
    """
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategy_streams_router requires current_user_dependency")
    if stream_reader is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategy_streams_router requires stream_reader")
    if owner_scope is None:  # type: ignore[truthy-bool]
        raise ValueError("build_strategy_streams_router requires owner_scope")

    router = APIRouter(tags=["strategy-streams"])

    def require_stream_user(request: Request) -> CurrentUserPrincipal:
        try:
            return current_user_dependency(request)
        except HTTPException as error:
            if error.status_code == 401:
                raise RoehubError(
                    code="auth.required",
                    message="Authentication is required",
                    details={},
                ) from error
            raise

    @router.get("/stream/strategies")
    async def get_strategy_stream(
        request: Request,
        strategy_id: UUID | None = None,
        last_event_id: str | None = None,
        count: int = Query(default=_DEFAULT_STREAM_COUNT, ge=1, le=_DEFAULT_STREAM_COUNT),
        principal: CurrentUserPrincipal = Depends(require_stream_user),
    ) -> StreamingResponse:
        if strategy_id is not None:
            owner_scope.ensure_strategy_owner(principal=principal, strategy_id=strategy_id)
        start_event_id = _resolve_last_event_id(
            query_last_event_id=last_event_id,
            header_last_event_id=request.headers.get("last-event-id"),
        )

        async def event_generator():
            cursor = start_event_id
            yield _format_sse(
                event="status",
                data={
                    "status": "connected",
                    "polling_fallback": False,
                    "last_event_id": cursor,
                },
            )
            while not await request.is_disconnected():
                try:
                    messages = await asyncio.to_thread(
                        lambda: stream_reader.read_for_user(
                            user_id=principal.user_id,
                            strategy_id=strategy_id,
                            last_event_id=cursor,
                            count=count,
                            block_ms=_DEFAULT_BLOCK_MS,
                        )
                    )
                except StrategyRealtimeStreamUnavailableError as error:
                    yield _format_sse(
                        event="fallback",
                        data={
                            "reason": str(error),
                            "polling_fallback": True,
                        },
                    )
                    return
                if not messages:
                    yield ": heartbeat\n\n"
                    continue
                for message in messages:
                    cursor = message.message_id
                    yield _format_stream_message(message=message)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return router


def _resolve_last_event_id(
    *,
    query_last_event_id: str | None,
    header_last_event_id: str | None,
) -> str:
    for candidate in (query_last_event_id, header_last_event_id):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return "$"


def _format_stream_message(*, message: StrategyRealtimeStreamMessageV1) -> str:
    event_name = "strategy.metric" if message.stream_kind == "metric" else "strategy.event"
    return _format_sse(
        event=event_name,
        event_id=message.message_id,
        data={
            "stream": message.stream,
            "kind": message.stream_kind,
            "message_id": message.message_id,
            "payload": dict(message.payload),
        },
    )


def _format_sse(
    *,
    event: str,
    data: dict[str, object],
    event_id: str | None = None,
) -> str:
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


__all__ = ["StrategyStreamOwnerScope", "build_strategy_streams_router"]
