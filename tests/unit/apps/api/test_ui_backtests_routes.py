from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.ui_backtests import build_ui_backtests_router
from apps.api.wiring.modules.ui_backtests import (
    BacktestWorkstationManualRefreshLimiter,
    BacktestWorkstationQueryService,
)
from tests.unit.apps.api.test_backtests_routes import (
    _build_jobs_use_case,
    _complete_job,
    _FakeJobRepository,
    _HeaderCurrentUserDependency,
    _valid_request,
)
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.services.v2 import (
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)
from trading.shared_kernel.primitives import UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000000901")


def test_get_backtest_workstation_returns_bounded_read_model_without_trades() -> None:
    repository = _FakeJobRepository()
    jobs_use_case = _build_jobs_use_case(repository=repository)
    client = _build_client(jobs_use_case=jobs_use_case)

    created = jobs_use_case.create(
        user_id=_USER_ID,
        payload=_valid_request(),
        idempotency_key="workstation-key",
    )
    _complete_job(repository=repository, job_id=UUID(created.job.job_id))

    response = client.get(
        "/ui/backtests/workstation?state=succeeded&query=mean",
        headers={"x-user-id": str(_USER_ID)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_defaults"]["supported_timeframes"] == ["15m"]
    assert "preset" not in payload["config_draft"]
    assert payload["ai_configurator_state"]["enabled"] is False
    assert payload["instrument_universe"]["selected_symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert payload["indicator_catalog"]["items"]
    assert payload["optimization_overview"]["completed_jobs"] == 1
    assert payload["recent_events"]["items"]
    assert payload["job_table"]["filters"]["state"] == "succeeded"
    assert payload["job_table"]["filters"]["query"] == "mean"
    assert payload["job_table"]["items"][0]["job_id"] == created.job.job_id
    assert payload["refresh_control"]["manual"] is True
    assert payload["refresh_control"]["default_preset"] == "15s"
    assert "trades" not in payload["job_table"]["items"][0]


def test_get_backtest_workstation_manual_refresh_rate_limit() -> None:
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(repository=_FakeJobRepository()),
        refresh_limiter=BacktestWorkstationManualRefreshLimiter(interval_seconds=30)
    )
    headers = {"x-user-id": str(_USER_ID)}

    first = client.get("/ui/backtests/workstation?refresh=manual", headers=headers)
    second = client.get("/ui/backtests/workstation?refresh=manual", headers=headers)

    assert first.status_code == 200
    assert first.json()["refresh_status"] == "fresh"
    assert first.json()["next_allowed_refresh_at"] is not None
    assert second.status_code == 200
    assert second.json()["refresh_status"] == "rate_limited"
    assert second.json()["retry_after_seconds"] > 0


def test_get_backtest_workstation_degrades_when_jobs_repository_is_unconfigured() -> None:
    client = _build_client(jobs_use_case=None)

    response = client.get(
        "/ui/backtests/workstation",
        headers={"x-user-id": str(_USER_ID)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_table"]["state"] == "unavailable"
    assert payload["footer_status"]["worker"] == "unavailable"
    assert payload["refresh_status"] == "degraded"


def _build_client(
    *,
    jobs_use_case=None,
    refresh_limiter: BacktestWorkstationManualRefreshLimiter | None = None,
) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_backtests_router(
            workstation_service=BacktestWorkstationQueryService(
                runtime_defaults_service=_runtime_defaults_service(),
                jobs_use_case=jobs_use_case,
                refresh_limiter=refresh_limiter,
            ),
            current_user_dependency=_HeaderCurrentUserDependency(),  # type: ignore[arg-type]
        )
    )
    return TestClient(app)


def _runtime_defaults_service() -> BacktestRuntimeDefaultsService:
    return BacktestRuntimeDefaultsService(
        defaults_provider=YamlBacktestGridDefaultsProvider.from_yaml(
            config_path="configs/prod/indicators.yaml"
        ),
        runtime_config=BacktestRuntimeConfig(
            hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
            hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
            artifact_config_hash="a" * 64,
        ),
    )
