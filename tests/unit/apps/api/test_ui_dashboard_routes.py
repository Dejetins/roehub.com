from __future__ import annotations

import gzip
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.dto import (
    DashboardBacktestsResponse,
    DashboardSourceResponse,
    DashboardStrategiesResponse,
    DashboardStrategyItemResponse,
)
from apps.api.routes import build_ui_dashboard_router
from apps.api.wiring.modules.ui_dashboard import (
    DashboardSummaryQueryService,
    StaticAlertsDashboardProvider,
    UnavailableBacktestsDashboardProvider,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId

"""
Stage 4 local endpoint contract:
- method/path: browser `GET /api/ui/dashboard/summary`, backend `GET /ui/dashboard/summary`.
- owner scope: every source receives the authenticated identity principal.
- request DTO: none; response DTO: bounded `DashboardSummaryResponse`.
- status/error: 200 on partial source success; 401 maps to Roehub `auth.required`.
- pagination/cache: summary has bounded panel slices, no cursor endpoint, no cache identity.
- compatibility: additive compatible-change; persisted schema none.
"""


def test_dashboard_summary_requires_authenticated_owner() -> None:
    client = _build_client()

    response = client.get("/ui/dashboard/summary")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


def test_dashboard_summary_is_owner_scoped_and_uses_bounded_response_contract() -> None:
    strategies = _RecordingStrategiesProvider()
    client = _build_client(strategy_provider=strategies)
    user_id = "00000000-0000-0000-0000-000000000411"

    response = client.get("/ui/dashboard/summary", headers={"x-user-id": user_id})

    assert response.status_code == 200
    payload = response.json()
    assert payload["account"] == {
        "source": {
            "status": "available",
            "code": "account.available",
            "message": "Authenticated account principal is available",
            "updated_at": payload["generated_at"],
        },
        "user_id": user_id,
        "paid_level": "free",
    }
    assert strategies.seen_user_id == user_id
    assert payload["poll_interval_seconds"] == 12
    assert payload["links"]["self"] == "/api/ui/dashboard/summary"
    assert payload["strategies"]["total_count"] == 1
    assert payload["strategies"]["items"][0]["state"] == "running"
    assert len(gzip.compress(response.content)) < 50 * 1024


def test_dashboard_summary_degrades_failed_source_without_breaking_other_panels() -> None:
    client = _build_client(backtests_provider=_SecretFailingBacktestsProvider())

    response = client.get(
        "/ui/dashboard/summary",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000412"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["account"]["source"]["status"] == "available"
    assert payload["strategies"]["source"]["status"] == "available"
    assert payload["backtests"]["source"] == {
        "status": "degraded",
        "code": "backtests.provider_failed",
        "message": "Backtest dashboard source failed",
        "updated_at": None,
    }
    assert "postgresql://" not in response.text
    assert "secret" not in response.text


def test_dashboard_summary_uses_unavailable_state_for_unwired_sources() -> None:
    client = _build_client(
        backtests_provider=UnavailableBacktestsDashboardProvider(
            code="backtests.unconfigured",
            message="Backtest jobs storage is not configured for dashboard reads",
        )
    )

    response = client.get(
        "/ui/dashboard/summary",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000413"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backtests"]["source"]["status"] == "unavailable"
    assert payload["backtests"]["active_count"] is None
    assert payload["backtests"]["items"] == []
    assert payload["alerts"]["source"]["status"] == "unavailable"


class _HeaderCurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


class _RecordingStrategiesProvider:
    def __init__(self) -> None:
        self.seen_user_id: str | None = None

    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardStrategiesResponse:
        self.seen_user_id = str(principal.user_id)
        return DashboardStrategiesResponse(
            source=DashboardSourceResponse(
                status="available",
                code="strategies.available",
                message="Strategy owner read-model is available",
            ),
            total_count=1,
            active_count=1,
            items=[
                DashboardStrategyItemResponse(
                    strategy_id="00000000-0000-0000-0000-000000000501",
                    name="BTCUSDT / 15m",
                    state="running",
                    instrument_key="binance:spot:BTCUSDT",
                    timeframe="15m",
                    updated_at="2026-05-03T08:00:00Z",
                )
            ],
        )


class _SecretFailingBacktestsProvider:
    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardBacktestsResponse:
        _ = principal
        raise RuntimeError("postgresql://secret@localhost:5432/roehub")


def _build_client(
    *,
    strategy_provider: Any | None = None,
    backtests_provider: Any | None = None,
    alerts_provider: Any | None = None,
) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    query = DashboardSummaryQueryService(
        strategy_provider=strategy_provider or _RecordingStrategiesProvider(),
        backtests_provider=backtests_provider or _EmptyBacktestsProvider(),
        alerts_provider=alerts_provider or StaticAlertsDashboardProvider(),
    )
    app.include_router(
        build_ui_dashboard_router(
            current_user_dependency=_HeaderCurrentUserDependency(),
            summary_query=query,
        )
    )
    return TestClient(app)


class _EmptyBacktestsProvider:
    def get_panel(self, *, principal: CurrentUserPrincipal) -> DashboardBacktestsResponse:
        _ = principal
        return DashboardBacktestsResponse(
            source=DashboardSourceResponse(
                status="available",
                code="backtests.available",
                message="Backtest jobs read-model is available",
            ),
            active_count=0,
            items=[],
            next_cursor=None,
        )
