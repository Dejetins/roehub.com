from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes import build_backtests_router
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
)
from trading.contexts.backtest.application.ports import BacktestArtifactContextUnavailable
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_get_backtest_runtime_defaults_returns_public_contract() -> None:
    client = _build_client()

    response = client.get(
        "/backtests/runtime-defaults",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000201"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported_timeframes"] == ["15m"]
    assert payload["risk_modes"] == ["none", "tp_sl_grid"]
    assert payload["direction_modes"] == ["long_only", "long_short_reversal"]
    assert "fixed_equity_pct_max_quote" in payload["sizing_modes"]
    assert "total_return_pct" in payload["ranking_metrics"]
    assert payload["top_n_default"] == 100
    assert payload["guardrails"]["max_top_n"] == 100


def test_post_backtest_preflight_returns_normalized_result_without_job_creation() -> None:
    resolver = _FakeArtifactResolver()
    client = _build_client(resolver=resolver)

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000202"},
        json=_valid_request(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_request"]["timeframe"] == "15m"
    assert payload["normalized_request"]["coordinates"]["symbol"] == "BTCUSDT"
    assert len(payload["request_hash"]) == 64
    assert payload["artifact_metadata"]["artifact_slot"] == "slot_a"
    assert payload["cost_estimate"]["candidate_combinations"] == 6
    assert payload["errors"] == []
    assert resolver.coordinates == (BacktestCoordinates("binance", "spot", "BTCUSDT"),)


def test_post_backtest_preflight_invalid_indicator_returns_backtest_422() -> None:
    client = _build_client()
    request = _valid_request()
    request["indicators"][0]["indicator_id"] = "ma.nope"

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000203"},
        json=request,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "backtest.invalid_request"
    assert response.json()["error"]["details"]["errors"][0]["path"] == (
        "indicators.0.indicator_id"
    )


def test_post_backtest_preflight_artifacts_unavailable_returns_backtest_503() -> None:
    client = _build_client(resolver=_UnavailableArtifactResolver())

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000204"},
        json=_valid_request(),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backtest.artifacts_unavailable"
    assert response.json()["error"]["details"]["retryable"] is True


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


@dataclass
class _FakeArtifactResolver:
    coordinates: tuple[BacktestCoordinates, ...] = ()

    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        self.coordinates = (*self.coordinates, coordinates)
        return BacktestArtifactMetadata(
            artifact_slot="slot_a",
            artifact_slot_generation=4,
            artifact_manifest_hash="a" * 64,
            artifact_asof_date="2026-03-25",
            hit_times_manifest_hash="b" * 64,
            published_at_utc="2026-03-25T02:00:00Z",
        )


class _UnavailableArtifactResolver:
    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        raise BacktestArtifactContextUnavailable("current pointer missing")


def _build_client(
    *,
    resolver: _FakeArtifactResolver | _UnavailableArtifactResolver | None = None,
) -> TestClient:
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
    )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtests_router(
            runtime_defaults_service=BacktestRuntimeDefaultsService(
                defaults_provider=defaults_provider,
                runtime_config=runtime_config,
            ),
            preflight_service=BacktestPreflightService(
                defaults_provider=defaults_provider,
                artifact_context_resolver=resolver or _FakeArtifactResolver(),
                runtime_config=runtime_config,
            ),
            current_user_dependency=_HeaderCurrentUserDependency(),  # type: ignore[arg-type]
        )
    )
    return TestClient(app)


def _valid_request() -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "2020-01-11T20:08:00Z",
            "end": "2026-04-11T20:08:00Z",
        },
        "indicators": [
            {
                "indicator_id": "ma.dema",
                "sources": ["close"],
                "window": {"start": 5, "stop": 10, "step": 1},
            }
        ],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {
            "primary_metric": "total_return_pct",
            "direction": "desc",
        },
        "top_n": 100,
    }
