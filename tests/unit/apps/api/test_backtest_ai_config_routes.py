from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.backtest_ai_config import build_backtest_ai_config_router
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_ai_config_router_registers_no_retired_job_endpoints() -> None:
    router = build_backtest_ai_config_router(
        current_user_dependency=_CurrentUserDependency(),
        jobs_use_case=object(),
    )

    paths = {getattr(route, "path", "") for route in router.routes}

    assert not any("/backtests/ai-config" in path for path in paths)


def test_retired_ai_config_job_endpoints_are_not_active() -> None:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtest_ai_config_router(
            current_user_dependency=_CurrentUserDependency(),
            jobs_use_case=object(),
        )
    )
    client = TestClient(app)
    retired_endpoint = "/backtests" + "/ai-config" + "/jobs"

    response = client.post(
        retired_endpoint,
        json={"mode": "assistant_v1", "locale": "en", "message": "Create RSI config"},
    )

    assert response.status_code == 404


class _CurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        _ = request
        return CurrentUserPrincipal(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000901"),
            paid_level=PaidLevel.free(),
        )
