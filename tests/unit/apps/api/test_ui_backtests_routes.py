from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
    SUPPORTED_BACKTEST_TIMEFRAMES_V1,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)
from trading.contexts.market_data.application.dto.reference_api import EnabledMarketReference
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000000901")


def test_get_backtest_workstation_returns_bounded_read_model_without_trades() -> None:
    repository = _FakeJobRepository()
    jobs_use_case = _build_jobs_use_case(repository=repository)
    client = _build_client(jobs_use_case=jobs_use_case)

    request = dict(_valid_request())
    request["strategy_name"] = "dema-1h-long-short-a1b2c3"
    created = jobs_use_case.create(
        user_id=_USER_ID,
        payload=request,
        idempotency_key="workstation-key",
    )
    _complete_job(repository=repository, job_id=UUID(created.job.job_id))

    response = client.get(
        "/ui/backtests/workstation?state=succeeded&query=dema",
        headers={"x-user-id": str(_USER_ID)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_defaults"]["supported_timeframes"] == list(
        SUPPORTED_BACKTEST_TIMEFRAMES_V1
    )
    assert "preset" not in payload["config_draft"]
    assert payload["config_draft"]["top_n"] == 10
    assert payload["config_draft"]["timeframe"] == "1h"
    assert payload["config_draft"]["execution"]["direction_mode"] == "long_short_reversal"
    assert payload["config_draft"]["execution"]["sizing"] == {"mode": "all_in"}
    assert payload["ai_configurator_state"]["enabled"] is False
    assert payload["instrument_universe"]["source"] == "market_data_reference"
    assert payload["instrument_universe"]["markets"] == [
        {"value": "binance", "label": "Binance", "status": "available"},
        {"value": "bybit", "label": "Bybit", "status": "available"},
    ]
    assert payload["instrument_universe"]["market_types"] == [
        {"value": "spot", "label": "Spot", "status": "available"},
        {"value": "futures", "label": "Futures", "status": "available"},
    ]
    assert payload["instrument_universe"]["selected_symbols"] == ["BTCUSDT"]
    assert payload["config_draft"]["time_range"]["end"].startswith(
        (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    )
    assert payload["config_draft"]["indicators"] == [
        {
            "indicator_id": "ma.dema",
            "params": {"window": {"start": 5, "stop": 200, "step": 1}},
            "window": {"start": 5, "stop": 200, "step": 1},
            "sources": ["close"],
        }
    ]
    assert payload["indicator_catalog"]["items"]
    assert payload["indicator_catalog"]["items"][0]["family"]
    assert payload["indicator_catalog"]["items"][0]["param_specs"]["params"]
    ma_ema = next(
        row for row in payload["indicator_catalog"]["items"] if row["indicator_id"] == "ma.ema"
    )
    assert ma_ema["label"] == "EMA"
    trend_adx = next(
        row for row in payload["indicator_catalog"]["items"] if row["indicator_id"] == "trend.adx"
    )
    assert trend_adx["sources"] == []
    assert payload["optimization_overview"]["completed_jobs"] == 1
    assert payload["recent_events"]["items"]
    assert payload["job_table"]["filters"]["state"] == "succeeded"
    assert payload["job_table"]["filters"]["query"] == "dema"
    assert payload["job_table"]["items"][0]["job_id"] == created.job.job_id
    assert payload["job_table"]["items"][0]["strategy"] == "dema-1h-long-short-a1b2c3"
    assert payload["job_table"]["items"][0]["exchange"] == "binance"
    assert payload["job_table"]["items"][0]["market_type"] == "spot"
    assert payload["job_table"]["items"][0]["direction"] == "long_short_reversal"
    assert payload["refresh_control"]["manual"] is True
    assert payload["refresh_control"]["default_preset"] == "15s"
    assert "trades" not in payload["job_table"]["items"][0]


def test_get_backtest_workstation_exposes_ai_configurator_unavailable_state() -> None:
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(repository=_FakeJobRepository()),
    )

    response = client.get(
        "/ui/backtests/workstation",
        headers={"x-user-id": str(_USER_ID)},
    )

    assert response.status_code == 200
    state = response.json()["ai_configurator_state"]
    assert state["enabled"] is False
    assert state["state"] == "unavailable"
    assert state["stage"] == "assistant-v1-ui"
    assert state["endpoints"] == {}


def test_get_backtest_workstation_filters_jobs_by_exchange_market_symbol_date() -> None:
    repository = _FakeJobRepository()
    jobs_use_case = _build_jobs_use_case(repository=repository)
    client = _build_client(jobs_use_case=jobs_use_case)

    btc = jobs_use_case.create(
        user_id=_USER_ID,
        payload=_valid_request(),
        idempotency_key="workstation-btc",
    )
    _complete_job(repository=repository, job_id=UUID(btc.job.job_id))
    eth_request = _valid_request()
    eth_request["coordinates"]["symbol"] = "ETHUSDT"
    eth = jobs_use_case.create(
        user_id=_USER_ID,
        payload=eth_request,
        idempotency_key="workstation-eth",
    )
    _complete_job(repository=repository, job_id=UUID(eth.job.job_id))
    launched_date = eth.job.created_at.date().isoformat()

    response = client.get(
        (
            "/ui/backtests/workstation"
            f"?exchange=binance&market_type=spot&symbol=ETHUSDT"
            f"&launched_from={launched_date}&launched_to={launched_date}"
        ),
        headers={"x-user-id": str(_USER_ID)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_table"]["filters"]["exchange"] == "binance"
    assert payload["job_table"]["filters"]["market_type"] == "spot"
    assert payload["job_table"]["filters"]["symbol"] == "ETHUSDT"
    assert payload["job_table"]["filters"]["launched_from"] == launched_date
    assert payload["job_table"]["filters"]["launched_to"] == launched_date
    assert [row["job_id"] for row in payload["job_table"]["items"]] == [eth.job.job_id]
    assert payload["job_table"]["items"][0]["symbol"] == "ETHUSDT"
    assert payload["job_table"]["items"][0]["created_at"].startswith(launched_date)


def test_get_backtest_workstation_filters_instruments_by_reference_market() -> None:
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=_FakeJobRepository()))

    response = client.get(
        (
            "/ui/backtests/workstation"
            "?instrument_exchange=bybit&instrument_market_type=futures"
        ),
        headers={"x-user-id": str(_USER_ID)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["instrument_universe"]["source"] == "market_data_reference"
    assert payload["instrument_universe"]["market_types"] == [
        {"value": "futures", "label": "Futures", "status": "available"}
    ]
    assert [row["value"] for row in payload["instrument_universe"]["symbols"]] == [
        "BTCUSDT",
        "ETHUSDT",
    ]


def test_get_backtest_workstation_manual_refresh_rate_limit() -> None:
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(repository=_FakeJobRepository()),
        refresh_limiter=BacktestWorkstationManualRefreshLimiter(interval_seconds=30),
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
    ai_configurator_state: dict[str, object] | None = None,
) -> TestClient:
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_ui_backtests_router(
            workstation_service=BacktestWorkstationQueryService(
                runtime_defaults_service=_runtime_defaults_service(),
                jobs_use_case=jobs_use_case,
                refresh_limiter=refresh_limiter,
                list_enabled_markets_use_case=_FakeListEnabledMarketsUseCase(),
                search_enabled_tradable_instruments_use_case=(
                    _FakeSearchEnabledTradableInstrumentsUseCase()
                ),
                ai_configurator_state=ai_configurator_state,
            ),
            current_user_dependency=_HeaderCurrentUserDependency(),  # type: ignore[arg-type]
        )
    )
    return TestClient(app)


class _FakeListEnabledMarketsUseCase:
    def execute(self) -> tuple[EnabledMarketReference, ...]:
        return (
            EnabledMarketReference(
                market_id=MarketId(1),
                exchange_name="binance",
                market_type="spot",
                market_code="binance_spot",
            ),
            EnabledMarketReference(
                market_id=MarketId(2),
                exchange_name="binance",
                market_type="futures",
                market_code="binance_futures",
            ),
            EnabledMarketReference(
                market_id=MarketId(3),
                exchange_name="bybit",
                market_type="futures",
                market_code="bybit_futures",
            ),
        )


class _FakeSearchEnabledTradableInstrumentsUseCase:
    def execute(
        self,
        *,
        market_id: MarketId,
        q: str | None = None,
        limit: int | None = None,
    ) -> tuple[InstrumentId, ...]:
        symbols_by_market = {
            1: ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            2: ("BNBUSDT",),
            3: ("BTCUSDT", "ETHUSDT"),
        }
        return tuple(
            InstrumentId(market_id=market_id, symbol=Symbol(symbol))
            for symbol in symbols_by_market.get(int(market_id.value), ())
        )


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
