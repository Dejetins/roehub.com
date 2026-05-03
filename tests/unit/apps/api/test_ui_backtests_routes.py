from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.ui_backtests import build_ui_backtests_router
from trading.contexts.backtest.application.dto import BacktestJobCountersResult
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_get_ui_backtests_counters_returns_owner_scoped_toolbar_contract() -> None:
    """
    Contract:
    method/path: browser `/api/ui/backtests/counters`, router `/ui/backtests/counters`;
    owner scope: current user from identity dependency;
    request DTO: none;
    response DTO: active/max counts, can_create, links;
    status codes: 200, 401, 503 when jobs service is unavailable;
    error payload: RoehubError envelope;
    pagination/cache identity: none;
    compatibility: compatible-change.
    """
    use_case = _CountersUseCase()
    client = _build_client(use_case=cast(BacktestJobsUseCase, use_case))

    response = client.get(
        "/ui/backtests/counters",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000901"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "active_jobs": 2,
        "max_active_jobs": 5,
        "max_active_jobs_global": 20,
        "can_create": True,
        "links": {
            "history": "/backtests/jobs",
            "create": "/backtests/jobs",
        },
    }
    assert use_case.user_ids == ("00000000-0000-0000-0000-000000000901",)


def test_get_ui_backtests_counters_auth_failure_uses_auth_required_code() -> None:
    client = _build_client(use_case=cast(BacktestJobsUseCase, _CountersUseCase()))

    response = client.get("/ui/backtests/counters")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


def test_get_ui_backtests_counters_without_jobs_service_returns_503() -> None:
    client = _build_client(use_case=None)

    response = client.get(
        "/ui/backtests/counters",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000902"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backtest.queue_saturated"


class _HeaderCurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(status_code=401, detail={"message": "Authentication required"})
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


@dataclass
class _CountersUseCase:
    user_ids: tuple[str, ...] = ()

    def counters(self, *, user_id: UserId) -> BacktestJobCountersResult:
        self.user_ids = (*self.user_ids, str(user_id))
        return BacktestJobCountersResult(
            active_jobs=2,
            max_active_jobs=5,
            max_active_jobs_global=20,
            can_create=True,
        )


def _build_client(*, use_case: BacktestJobsUseCase | None) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_backtests_router(
            current_user_dependency=_HeaderCurrentUserDependency(),
            jobs_use_case=use_case,
        )
    )
    return TestClient(app)
