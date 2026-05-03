from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_strategy_streams_router
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.application.ports import (
    StrategyRealtimeStreamMessageV1,
    StrategyRealtimeStreamUnavailableError,
)
from trading.shared_kernel.primitives import PaidLevel, UserId

"""
Stage 7 stream contract:
- method/path: browser `GET /api/stream/strategies`, backend `GET /stream/strategies`.
- owner scope: auth and optional `strategy_id` owner guard run before stream read.
- request DTO: `strategy_id`, `last_event_id`, bounded `count`.
- response DTO: SSE events with `status`, `strategy.metric`, `strategy.event`, `fallback`.
- status/error: 200 stream, 401 `auth.required`, 403/404 owner guard, 422 query validation.
- pagination/cache: Redis `last_event_id`; no cache identity.
- compatibility: additive compatible-change.
"""


def test_strategy_stream_requires_authenticated_owner_before_read() -> None:
    reader = _FallbackReader()
    client = _build_client(reader=reader)

    response = client.get("/stream/strategies")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"
    assert reader.calls == 0


def test_strategy_stream_owner_scope_runs_before_reader() -> None:
    reader = _FallbackReader()
    client = _build_client(reader=reader, owner_scope=_DenyOwnerScope())

    response = client.get(
        "/stream/strategies?strategy_id=00000000-0000-0000-0000-000000020001",
        headers={"x-user-id": "00000000-0000-0000-0000-000000020002"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert reader.calls == 0


def test_strategy_stream_emits_fallback_event_when_redis_reader_is_unavailable() -> None:
    client = _build_client(reader=_FallbackReader())

    response = client.get(
        "/stream/strategies?last_event_id=9-0",
        headers={"x-user-id": "00000000-0000-0000-0000-000000020003"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: status" in response.text
    assert '"last_event_id":"9-0"' in response.text
    assert "event: fallback" in response.text
    assert '"polling_fallback":true' in response.text


def test_strategy_stream_serializes_metric_events_and_preserves_last_event_id() -> None:
    reader = _OneMessageReader()
    client = _build_client(reader=reader)
    strategy_id = "00000000-0000-0000-0000-000000020004"
    user_id = "00000000-0000-0000-0000-000000020005"

    response = client.get(
        f"/stream/strategies?strategy_id={strategy_id}&last_event_id=1-0",
        headers={"x-user-id": user_id},
    )

    assert response.status_code == 200
    assert reader.seen_user_id == user_id
    assert reader.seen_strategy_id == strategy_id
    assert reader.seen_last_event_ids[0] == "1-0"
    assert "id: 2-0" in response.text
    assert "event: strategy.metric" in response.text
    assert '"metric_type":"lag_seconds"' in response.text
    assert "event: fallback" in response.text


class _HeaderCurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


class _AllowOwnerScope:
    def ensure_strategy_owner(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
    ) -> None:
        _ = (principal, strategy_id)


class _DenyOwnerScope:
    def ensure_strategy_owner(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: UUID,
    ) -> None:
        _ = principal
        from trading.platform.errors import RoehubError

        raise RoehubError(
            code="forbidden",
            message="Strategy does not belong to current user",
            details={"strategy_id": str(strategy_id)},
        )


class _FallbackReader:
    def __init__(self) -> None:
        self.calls = 0

    def read_for_user(self, **kwargs):
        _ = kwargs
        self.calls += 1
        raise StrategyRealtimeStreamUnavailableError("stream unavailable")


class _OneMessageReader:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_user_id: str | None = None
        self.seen_strategy_id: str | None = None
        self.seen_last_event_ids: list[str] = []

    def read_for_user(self, **kwargs):
        self.calls += 1
        self.seen_user_id = str(kwargs["user_id"])
        self.seen_strategy_id = (
            str(kwargs["strategy_id"]) if kwargs["strategy_id"] is not None else None
        )
        self.seen_last_event_ids.append(str(kwargs["last_event_id"]))
        if self.calls > 1:
            raise StrategyRealtimeStreamUnavailableError("stream unavailable")
        return (
            StrategyRealtimeStreamMessageV1(
                stream=f"strategy.metrics.v1.user.{self.seen_user_id}",
                stream_kind="metric",
                message_id="2-0",
                payload={
                    "schema_version": "1",
                    "ts": "2026-05-03T08:00:00Z",
                    "strategy_id": self.seen_strategy_id or "",
                    "run_id": "00000000-0000-0000-0000-000000020006",
                    "metric_type": "lag_seconds",
                    "value": "12",
                    "instrument_key": "binance:spot:BTCUSDT",
                    "timeframe": "15m",
                },
            ),
        )


def _build_client(
    *,
    reader,
    owner_scope=None,
) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_strategy_streams_router(
            current_user_dependency=_HeaderCurrentUserDependency(),
            stream_reader=reader,
            owner_scope=owner_scope or _AllowOwnerScope(),
        )
    )
    return TestClient(app)
