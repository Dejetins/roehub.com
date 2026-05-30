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
from apps.api.routes import build_ui_strategies_dashboard_router
from apps.api.wiring.modules.ui_strategies_dashboard import StrategyDashboardQueryService
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.domain.entities import Strategy, StrategyRun
from trading.contexts.strategy.domain.entities.strategy_spec_v1 import StrategySpecV1
from trading.shared_kernel.primitives import PaidLevel, UserId

_USER_ID = "00000000-0000-0000-0000-000000006006"
_STRATEGY_ID = UUID("00000000-0000-0000-0000-000000006101")

_STRATEGY_DASHBOARD_ENDPOINT_CONTRACT = {
    "method_path": {
        "browser": "GET /api/ui/strategies/dashboard",
        "backend": "GET /ui/strategies/dashboard",
    },
    "owner_scope": "current identity principal; strategy rows are loaded by owner user id",
    "request_dto": (
        "query strategy_id optional, state=active|all default all, cursor optional, "
        "refresh=initial|auto|manual default initial"
    ),
    "response_dto": (
        "bounded StrategyDashboardResponse with sources[], freshness, selected strategy, "
        "panel states, selector and refresh_control"
    ),
    "status_codes": "200, 401, 422; manual refresh rate limit is represented in DTO",
    "error_payload": "RoehubError envelope for auth.required and validation_error",
    "pagination": (
        "cursor accepted for compatibility; first implementation caps selector/trades arrays"
    ),
    "cache_identity": "none; no persisted request hash or cache key",
    "compatibility": "compatible-change",
}


def test_strategy_dashboard_exposes_reference_panel_inventory_and_degraded_stats() -> None:
    strategy = _strategy(symbol="BTCUSDT")
    run = StrategyRun.start(
        run_id=UUID("00000000-0000-0000-0000-000000006201"),
        user_id=strategy.user_id,
        strategy_id=strategy.strategy_id,
        started_at=datetime(2026, 5, 6, 9, 0, tzinfo=UTC),
        metadata_json={
            "warmup": {
                "algorithm": "numeric_max_param_v1",
                "bars": 5,
                "processed_bars": 3,
                "satisfied": False,
            },
            "rollup": {
                "timeframe": "1m",
                "bucket_open_ts": "2026-05-06T09:00:00Z",
                "bucket_count_1m": 2,
            },
        },
    )
    service = StrategyDashboardQueryService(
        strategy_repository=_FakeStrategyRepository(strategies=(strategy,)),
        run_repository=_FakeRunRepository(runs=(run,)),
    )
    client = _build_client(service=service)

    response = client.get(
        f"/ui/strategies/dashboard?strategy_id={_STRATEGY_ID}&state=active",
        headers={"x-user-id": _USER_ID},
    )

    assert response.status_code == 200
    payload = response.json()
    assert _STRATEGY_DASHBOARD_ENDPOINT_CONTRACT["method_path"]["browser"] == (
        "GET /api/ui/strategies/dashboard"
    )
    assert payload["selected_strategy"]["strategy_id"] == str(_STRATEGY_ID)
    assert payload["selected_strategy"]["status"] == "live"
    assert payload["selected_strategy"]["exchange"] == "binance"
    assert payload["selected_strategy"]["symbols"] == ["BTCUSDT"]
    assert payload["live_profile"]["mode"] == "monitor_only"
    assert payload["live_profile"]["readiness_status"] == "ready"
    assert payload["live_profile"]["readiness_reason"] == "monitor_only_no_exchange_submit"
    assert payload["strategy_selector"]["filters"]["state"] == "active"
    assert payload["strategy_selector"]["totals"]["strategies"] == 1
    assert payload["strategy_selector"]["items"][0]["status"] == "live"
    for panel in [
        "chart",
        "metric_grid",
        "monthly_stats",
        "long_short",
        "drawdown",
        "equity_curve",
        "hourly_results",
        "trades",
    ]:
        assert payload[panel]["state"] == "unavailable"
        assert payload[panel]["source"]
        assert payload[panel]["degradation_reason"]
    assert "summary" not in payload["monthly_stats"]
    assert "symbol_results" not in payload
    assert payload["refresh_control"]["interval_seconds"] == 15
    assert payload["refresh_control"]["preset_key"] == "15s"
    source_statuses = {source["name"]: source["status"] for source in payload["sources"]}
    assert source_statuses["strategy_strategies"] == "available"
    assert source_statuses["strategy_runs"] == "available"
    assert source_statuses["strategy_live_profiles"] == "unavailable"
    assert source_statuses["strategy_run_metadata"] == "available"
    assert source_statuses["strategy_stat_projections"] == "unavailable"
    assert source_statuses["execution_fills"] == "unavailable"
    source_freshness = {source["name"]: source["age_seconds"] for source in payload["sources"]}
    assert isinstance(source_freshness["strategy_strategies"], int)
    assert isinstance(source_freshness["strategy_runs"], int)
    assert isinstance(source_freshness["strategy_run_metadata"], int)
    risk_rows = {row["key"]: row for row in payload["risk_execution"]["rows"]}
    assert payload["risk_execution"]["source"] == "strategy_run_metadata"
    assert payload["risk_execution"]["state"] == "ready"
    assert risk_rows["run_state"]["total_value"] == "starting"
    assert risk_rows["warmup_progress"]["total_value"] == "3/5"
    assert risk_rows["warmup_satisfied"]["total_value"] == "no"
    assert risk_rows["rollup_bucket_count_1m"]["total_value"] == "2"
    compressed = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    assert len(compressed) < 96 * 1024


def test_strategy_dashboard_auth_failure_uses_auth_required_code() -> None:
    service = StrategyDashboardQueryService(
        strategy_repository=_FakeStrategyRepository(strategies=()),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    response = client.get("/ui/strategies/dashboard")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


def test_strategy_dashboard_source_failure_degrades_panels_without_500() -> None:
    service = StrategyDashboardQueryService(
        strategy_repository=_FailingStrategyRepository(),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    response = client.get("/ui/strategies/dashboard", headers={"x-user-id": _USER_ID})

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_strategy"]["state"] == "empty"
    assert payload["strategy_selector"]["state"] == "empty"
    source_statuses = {source["name"]: source for source in payload["sources"]}
    assert source_statuses["strategy_strategies"]["status"] == "degraded"
    assert payload["refresh_status"] == "degraded"


def test_strategy_dashboard_manual_refresh_reports_rate_limit_in_dto() -> None:
    service = StrategyDashboardQueryService(
        strategy_repository=_FakeStrategyRepository(strategies=()),
        run_repository=_FakeRunRepository(),
    )
    client = _build_client(service=service)

    first = client.get("/ui/strategies/dashboard?refresh=manual", headers={"x-user-id": _USER_ID})
    second = client.get("/ui/strategies/dashboard?refresh=manual", headers={"x-user-id": _USER_ID})

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


def _build_client(*, service: StrategyDashboardQueryService) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_strategies_dashboard_router(
            dashboard_service=service,
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
        created_at=datetime(2026, 5, 6, 8, 0, tzinfo=UTC),
        strategy_id=_STRATEGY_ID,
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
