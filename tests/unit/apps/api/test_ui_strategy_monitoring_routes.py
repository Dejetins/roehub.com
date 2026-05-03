import gzip
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_strategies_router, build_ui_strategies_monitoring_router
from apps.api.wiring.modules.ui_strategies_monitoring import StrategyMonitoringQueryService
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyEventRepository,
    InMemoryStrategyRepository,
    InMemoryStrategyRunRepository,
)
from trading.contexts.strategy.adapters.outbound.time import SystemStrategyClock
from trading.contexts.strategy.application import (
    CloneStrategyUseCase,
    CreateStrategyUseCase,
    CurrentUser,
    CurrentUserProvider,
    DeleteStrategyUseCase,
    GetMyStrategyUseCase,
    ListMyStrategiesUseCase,
    RunStrategyUseCase,
    StopStrategyUseCase,
)
from trading.shared_kernel.primitives import PaidLevel, UserId

"""
Stage 7 local endpoint contract:
- method/path: browser `/api/ui/strategies/monitor`, backend `/ui/strategies/monitor`.
- method/path: browser `/api/ui/strategies/{id}/snapshot|positions|fills|equity`,
  backend same paths without `/api`.
- owner scope: identity current-user principal gates every Strategy query.
- request DTO: query params only; `state`, `cursor`, `limit`, `range`, `points`.
- response DTO: bounded monitoring DTOs; no fake positions/fills/equity data.
- status/error: 200, 401 `auth.required`, 403/404 from Strategy owner use-case, 422.
- pagination/cache: offset cursor for monitor v1, bounded payloads, no cache identity.
- compatibility: additive compatible-change; persisted schema none.
"""


def test_strategy_monitor_requires_authenticated_owner() -> None:
    client = _build_client()

    response = client.get("/ui/strategies/monitor")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


def test_strategy_monitor_returns_owner_scoped_bounded_items_and_active_filter() -> None:
    client = _build_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000010001"}
    first_id = _create_strategy(client, headers=headers, symbol="BTCUSDT")
    second_id = _create_strategy(client, headers=headers, symbol="ETHUSDT")

    run_response = client.post(f"/strategies/{first_id}/run", headers=headers)
    assert run_response.status_code == 200

    all_response = client.get("/ui/strategies/monitor?state=all", headers=headers)
    active_response = client.get("/ui/strategies/monitor?state=active", headers=headers)

    assert all_response.status_code == 200
    payload = all_response.json()
    assert payload["source"]["status"] == "available"
    assert payload["poll_interval_seconds"] == 10
    assert payload["limits"]["strategies"] == 50
    assert payload["links"]["stream"] == "/api/stream/strategies"
    assert len(payload["items"]) == 2
    assert {item["strategy_id"] for item in payload["items"]} == {first_id, second_id}
    assert len(gzip.compress(all_response.content)) < 50 * 1024

    assert active_response.status_code == 200
    active_payload = active_response.json()
    assert [item["strategy_id"] for item in active_payload["items"]] == [first_id]
    assert active_payload["items"][0]["state"] == "starting"


def test_strategy_snapshot_and_empty_detail_read_models_do_not_fake_trading_data() -> None:
    client = _build_client()
    headers = {"x-user-id": "00000000-0000-0000-0000-000000010002"}
    strategy_id = _create_strategy(client, headers=headers, symbol="SOLUSDT")

    snapshot = client.get(f"/ui/strategies/{strategy_id}/snapshot", headers=headers)
    positions = client.get(f"/ui/strategies/{strategy_id}/positions?limit=50", headers=headers)
    fills = client.get(f"/ui/strategies/{strategy_id}/fills?limit=50", headers=headers)
    equity = client.get(
        f"/ui/strategies/{strategy_id}/equity?range=1d&points=600",
        headers=headers,
    )

    assert snapshot.status_code == 200
    snapshot_payload = snapshot.json()
    assert snapshot_payload["strategy_id"] == strategy_id
    assert snapshot_payload["run"]["state"] == "idle"
    assert snapshot_payload["links"]["run"] == f"/api/strategies/{strategy_id}/run"
    assert snapshot_payload["metrics"][0] == {
        "key": "run_state",
        "value": "idle",
        "tone": "neutral",
        "updated_at": None,
    }

    assert positions.status_code == 200
    assert positions.json()["source"]["status"] == "unavailable"
    assert positions.json()["items"] == []

    assert fills.status_code == 200
    assert fills.json()["source"]["status"] == "unavailable"
    assert fills.json()["items"] == []

    assert equity.status_code == 200
    assert equity.json()["source"]["status"] == "unavailable"
    assert equity.json()["items"] == []


def test_strategy_monitoring_snapshot_enforces_owner_scope() -> None:
    client = _build_client()
    owner_headers = {"x-user-id": "00000000-0000-0000-0000-000000010003"}
    outsider_headers = {"x-user-id": "00000000-0000-0000-0000-000000010004"}
    strategy_id = _create_strategy(client, headers=owner_headers, symbol="ADAUSDT")

    response = client.get(f"/ui/strategies/{strategy_id}/snapshot", headers=outsider_headers)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


class _SequenceClock:
    def __init__(self) -> None:
        self._values = [
            datetime(2026, 5, 3, 8, 0, tzinfo=timezone.utc) + timedelta(minutes=index)
            for index in range(120)
        ]

    def now(self) -> datetime:
        if not self._values:
            raise ValueError("_SequenceClock exhausted")
        return self._values.pop(0)


class _StrategyHeaderDependency:
    def __call__(self, request: StarletteRequest) -> CurrentUserProvider:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return _StaticCurrentUserProvider(user_id=UserId.from_string(raw_user_id))


class _MonitoringHeaderDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


class _StaticCurrentUserProvider(CurrentUserProvider):
    def __init__(self, *, user_id: UserId) -> None:
        self._user_id = user_id

    def require_current_user(self) -> CurrentUser:
        return CurrentUser(user_id=self._user_id)


def _build_client() -> TestClient:
    strategy_repository = InMemoryStrategyRepository()
    run_repository = InMemoryStrategyRunRepository()
    event_repository = InMemoryStrategyEventRepository()
    clock = _SequenceClock()

    strategy_router = build_strategies_router(
        create_use_case=CreateStrategyUseCase(
            repository=strategy_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        clone_use_case=CloneStrategyUseCase(
            repository=strategy_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        list_use_case=ListMyStrategiesUseCase(repository=strategy_repository),
        get_use_case=GetMyStrategyUseCase(repository=strategy_repository),
        run_use_case=RunStrategyUseCase(
            strategy_repository=strategy_repository,
            run_repository=run_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        stop_use_case=StopStrategyUseCase(
            strategy_repository=strategy_repository,
            run_repository=run_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        delete_use_case=DeleteStrategyUseCase(
            repository=strategy_repository,
            event_repository=event_repository,
            clock=clock,
        ),
        current_user_provider_dependency=_StrategyHeaderDependency(),
    )
    query_service = StrategyMonitoringQueryService(
        list_use_case=ListMyStrategiesUseCase(repository=strategy_repository),
        get_use_case=GetMyStrategyUseCase(repository=strategy_repository),
        run_repository=run_repository,
        clock=SystemStrategyClock(),
    )

    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(strategy_router)
    app.include_router(
        build_ui_strategies_monitoring_router(
            current_user_dependency=_MonitoringHeaderDependency(),
            monitoring_query=query_service,
        )
    )
    return TestClient(app)


def _create_strategy(client: TestClient, *, headers: dict[str, str], symbol: str) -> str:
    response = client.post("/strategies", json=_create_payload(symbol=symbol), headers=headers)
    assert response.status_code == 201
    return str(response.json()["strategy_id"])


def _create_payload(*, symbol: str) -> dict[str, Any]:
    return {
        "instrument_id": {
            "market_id": 1,
            "symbol": symbol,
        },
        "instrument_key": f"binance:spot:{symbol}",
        "market_type": "spot",
        "timeframe": "15m",
        "indicators": [{"id": "ema", "inputs": {"close": "close"}, "params": {"period": 20}}],
        "signal_template": "EMA(20)",
    }
