from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_ui_execution_router
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExecutionIntentRepository,
)
from trading.contexts.live_execution.application import ExecutionIngressService
from trading.shared_kernel.primitives import PaidLevel, UserId

_USER_ID = "00000000-0000-0000-0000-000000010501"


class _Clock:
    def __init__(self) -> None:
        self._index = 0

    def now(self) -> datetime:
        value = datetime(2026, 5, 31, 13, 30, tzinfo=UTC) + timedelta(seconds=self._index)
        self._index += 1
        return value


class _CurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        return CurrentUserPrincipal(
            user_id=UserId.from_string(request.headers["x-user-id"]),
            paid_level=PaidLevel.free(),
        )


def test_ui_execution_routes_create_and_dedupe_intent() -> None:
    client = _build_client()

    source_response = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "ops-stage10-ref",
            "source_ref": {"ops_test_id": "stage10"},
            "idempotency_key": "source-key",
        },
    )
    source_replay = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "ops_test",
            "source_event_ref": "ops-stage10-ref",
            "source_ref": {"ops_test_id": "stage10"},
            "idempotency_key": "source-key",
        },
    )
    intent_payload = {
        "source_event_id": source_response.json()["source_event_id"],
        "idempotency_key": "intent-key",
        "exchange_connection_id": "00000000-0000-0000-0000-000000010601",
        "market_type": "spot",
        "instrument_key": "binance:spot:BTCUSDT",
        "order": {
            "order_type": "limit",
            "side": "buy",
            "quantity": "0.01",
            "limit_price": "10000",
        },
    }
    intent_response = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json=intent_payload,
    )
    intent_replay = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json=intent_payload,
    )

    assert source_response.status_code == 201
    assert source_replay.status_code == 200
    assert source_replay.json()["duplicate"] is True
    assert intent_response.status_code == 201
    assert intent_replay.status_code == 200
    assert intent_replay.json()["duplicate"] is True
    assert intent_replay.json()["intent_id"] == intent_response.json()["intent_id"]
    assert intent_response.json()["order_type"] == "limit"
    assert intent_response.json()["status_reason"] == "stage10_recorded_no_dispatch"


def test_ui_execution_route_rejects_unsupported_order_model() -> None:
    client = _build_client()
    source_id = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "manual_request",
            "source_event_ref": "manual-stage10-ref",
            "source_ref": {"manual_request_id": "manual-stage10"},
            "idempotency_key": "manual-source-key",
        },
    ).json()["source_event_id"]

    response = client.post(
        "/ui/execution/intents",
        headers={"x-user-id": _USER_ID},
        json={
            "source_event_id": source_id,
            "idempotency_key": "manual-intent-key",
            "exchange_connection_id": "00000000-0000-0000-0000-000000010601",
            "market_type": "spot",
            "instrument_key": "binance:spot:BTCUSDT",
            "order": {
                "order_type": "market",
                "side": "buy",
                "quote_notional": "25",
                "take_profit": {"price": "11000"},
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "execution.unsupported_order_model"
    assert response.json()["error"]["details"]["reason"] == "tp_sl_not_supported"


def test_ui_execution_route_requires_strategy_signal_id_for_strategy_sources() -> None:
    client = _build_client()

    response = client.post(
        "/ui/execution/source-events",
        headers={"x-user-id": _USER_ID},
        json={
            "source_type": "strategy_signal",
            "source_event_ref": "signal-without-id",
            "source_ref": {"strategy_id": str(UUID(int=1))},
            "idempotency_key": "bad-source-key",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "execution.invalid_source_event"
    assert response.json()["error"]["details"]["reason"] == "strategy_signal_id_required"


def _build_client() -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_execution_router(
            ingress_service=ExecutionIngressService(
                repository=InMemoryExecutionIntentRepository(),
                clock=_Clock(),
            ),
            current_user_dependency=_CurrentUserDependency(),
        )
    )
    return TestClient(app)
