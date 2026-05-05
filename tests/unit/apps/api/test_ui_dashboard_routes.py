from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_ui_dashboard_router
from apps.api.wiring.modules.ui_dashboard import DashboardSummaryQueryService
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.domain.entities import Strategy, StrategyRun
from trading.contexts.strategy.domain.entities.strategy_spec_v1 import StrategySpecV1
from trading.shared_kernel.primitives import PaidLevel, UserId

_USER_ID = "00000000-0000-0000-0000-000000004004"

_SUMMARY_ENDPOINT_CONTRACT = {
    "method_path": {
        "browser": "GET /api/ui/dashboard/summary",
        "backend": "GET /ui/dashboard/summary",
    },
    "owner_scope": "current identity principal only; no cross-user ids accepted",
    "request_dto": "query refresh=initial|auto|manual, default initial; no body",
    "response_dto": "bounded DashboardSummaryResponse with required panel zones and sources[]",
    "status_codes": "200, 401, 422; manual rate limit is represented in DTO",
    "error_payload": "RoehubError envelope for auth.required and validation_error",
    "pagination": "none for summary; arrays are capped",
    "cache_identity": "none; no persisted request hash or cache key",
    "compatibility": "compatible-change",
}


def test_dashboard_summary_exposes_bounded_degraded_contract() -> None:
    service = DashboardSummaryQueryService(
        strategy_repository=_FakeStrategyRepository(
            strategies=(_strategy(symbol="BTCUSDT"),),
        ),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    response = client.get("/ui/dashboard/summary", headers={"x-user-id": _USER_ID})

    assert response.status_code == 200
    payload = response.json()
    assert _SUMMARY_ENDPOINT_CONTRACT["method_path"]["browser"] == (
        "GET /api/ui/dashboard/summary"
    )
    assert payload["selected_strategy_snapshot"]["strategy_id"] is not None
    assert payload["selected_strategy_snapshot"]["exchange"] == "binance"
    assert payload["selected_strategy_snapshot"]["symbols"] == ["BTCUSDT"]
    assert payload["equity_pnl_series"]["state"] == "unavailable"
    assert payload["open_positions"]["state"] == "unavailable"
    assert payload["recent_executions"]["state"] == "unavailable"
    assert payload["symbol_allocation"]["state"] == "unavailable"
    assert payload["strategy_list"]["totals"]["strategies"] == 1
    assert payload["refresh_control"]["interval_seconds"] == 15
    assert payload["refresh_control"]["preset_key"] == "15s"
    source_statuses = {source["name"]: source["status"] for source in payload["sources"]}
    assert source_statuses["strategy_strategies"] == "available"
    assert source_statuses["portfolio_snapshots"] == "unavailable"
    assert source_statuses["position_snapshots"] == "unavailable"
    compressed = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    assert len(compressed) < 80 * 1024


def test_dashboard_summary_auth_failure_uses_auth_required_code() -> None:
    service = DashboardSummaryQueryService(
        strategy_repository=_FakeStrategyRepository(strategies=()),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    response = client.get("/ui/dashboard/summary")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


def test_dashboard_summary_degrades_source_failure_without_page_failure() -> None:
    service = DashboardSummaryQueryService(
        strategy_repository=_FailingStrategyRepository(),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    response = client.get("/ui/dashboard/summary", headers={"x-user-id": _USER_ID})

    assert response.status_code == 200
    payload = response.json()
    source_statuses = {source["name"]: source for source in payload["sources"]}
    assert source_statuses["strategy_strategies"]["status"] == "degraded"
    assert payload["selected_strategy_snapshot"]["state"] == "empty"
    assert payload["strategy_list"]["state"] == "empty"
    assert payload["health_risk"]["state"] == "warn"


def test_dashboard_manual_refresh_reports_rate_limit_in_dto() -> None:
    service = DashboardSummaryQueryService(
        strategy_repository=_FakeStrategyRepository(strategies=()),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    first = client.get("/ui/dashboard/summary?refresh=manual", headers={"x-user-id": _USER_ID})
    second = client.get("/ui/dashboard/summary?refresh=manual", headers={"x-user-id": _USER_ID})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["refresh_status"] == "rate_limited"
    assert second.json()["retry_after_seconds"] >= 1
    assert second.json()["refresh_control"]["next_allowed_refresh_at"] is not None


class _HeaderCurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


class _FakeStrategyRepository:
    def __init__(self, *, strategies: tuple[Strategy, ...]) -> None:
        self._strategies = strategies

    def list_for_user(
        self,
        *,
        user_id: UserId,
        include_deleted: bool = False,
    ) -> tuple[Strategy, ...]:
        return tuple(
            strategy
            for strategy in self._strategies
            if strategy.user_id == user_id and (include_deleted or not strategy.is_deleted)
        )

    def create(self, *, strategy: Strategy) -> Strategy:
        raise NotImplementedError

    def find_by_strategy_id(self, *, user_id: UserId, strategy_id: UUID) -> Strategy | None:
        raise NotImplementedError

    def find_any_by_strategy_id(self, *, strategy_id: UUID) -> Strategy | None:
        raise NotImplementedError

    def soft_delete(self, *, user_id: UserId, strategy_id: UUID) -> bool:
        raise NotImplementedError


class _FailingStrategyRepository(_FakeStrategyRepository):
    def __init__(self) -> None:
        super().__init__(strategies=())

    def list_for_user(
        self,
        *,
        user_id: UserId,
        include_deleted: bool = False,
    ) -> tuple[Strategy, ...]:
        raise ValueError("strategy source unavailable")


class _FakeRunRepository:
    def __init__(self, *, runs: tuple[StrategyRun, ...] = ()) -> None:
        self._runs = runs

    def find_active_for_strategy(
        self,
        *,
        user_id: UserId,
        strategy_id: UUID,
    ) -> StrategyRun | None:
        for run in self._runs:
            if run.user_id == user_id and run.strategy_id == strategy_id and run.is_active():
                return run
        return None

    def create(self, *, run: StrategyRun) -> StrategyRun:
        raise NotImplementedError

    def update(self, *, run: StrategyRun) -> StrategyRun:
        raise NotImplementedError

    def find_by_run_id(self, *, user_id: UserId, run_id: UUID) -> StrategyRun | None:
        raise NotImplementedError

    def list_for_strategy(self, *, user_id: UserId, strategy_id: UUID) -> tuple[StrategyRun, ...]:
        raise NotImplementedError

    def list_active_runs(self) -> tuple[StrategyRun, ...]:
        raise NotImplementedError


def _build_client(*, service: DashboardSummaryQueryService) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_dashboard_router(
            summary_service=service,
            current_user_dependency=_HeaderCurrentUserDependency(),
        )
    )
    return TestClient(app)


def _strategy(*, symbol: str) -> Strategy:
    user_id = UserId.from_string(_USER_ID)
    spec = StrategySpecV1.from_json(payload=_strategy_spec_payload(symbol=symbol))
    return Strategy.create(
        user_id=user_id,
        spec=spec,
        created_at=datetime(2026, 5, 5, 9, 0, tzinfo=UTC),
        strategy_id=UUID("00000000-0000-0000-0000-000000009001"),
    )


def _strategy_spec_payload(*, symbol: str) -> dict[str, Any]:
    return {
        "instrument_id": {
            "market_id": 1,
            "symbol": symbol,
        },
        "instrument_key": f"binance:spot:{symbol}",
        "market_type": "spot",
        "timeframe": "1m",
        "indicators": [
            {
                "name": "MA",
                "params": {
                    "fast": 20,
                    "slow": 50,
                },
            }
        ],
        "signal_template": "MA(20,50)",
    }
