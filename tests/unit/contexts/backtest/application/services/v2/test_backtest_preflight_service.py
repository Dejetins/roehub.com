from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestRuntimeGuardrails,
)
from trading.contexts.backtest.application.services.v2 import (
    BACKTEST_ERROR_INVALID_REQUEST,
    BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE,
    BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED,
    SUPPORTED_BACKTEST_TIMEFRAMES_V1,
    BacktestPreflightRejected,
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)


def test_runtime_defaults_expose_iteration_1_public_contract() -> None:
    service = BacktestRuntimeDefaultsService(
        defaults_provider=_defaults_provider(),
        runtime_config=_runtime_config(),
    )

    response = service.execute().as_mapping()

    assert response["supported_timeframes"] == list(SUPPORTED_BACKTEST_TIMEFRAMES_V1)
    assert response["risk_modes"] == ["none", "tp_sl_grid"]
    assert response["direction_modes"] == ["long_only", "long_short_reversal"]
    assert response["sizing_modes"] == [
        "all_in",
        "fixed_quote",
        "fixed_equity_pct",
        "fixed_equity_pct_min_quote",
        "fixed_equity_pct_max_quote",
    ]
    assert "total_return_pct" in response["ranking_metrics"]
    assert response["top_n_default"] == 50
    assert response["guardrails"]["max_candidate_combinations"] == 10_000_000_000_000
    assert response["quality_constraints_default"] == {
        "min_closed_trades_policy": "timeframe_sqrt_v1",
        "base_trades_per_year_at_1h": 24,
        "min_trades_per_year": 12,
        "max_trades_per_year": 365,
    }
    assert "ma.dema" in response["supported_indicator_ids"]


def test_preflight_success_returns_normalized_request_hash_artifact_and_cost() -> None:
    resolver = _FakeArtifactResolver()
    service = _service(resolver=resolver)

    first = service.execute(_valid_request())
    second = service.execute(_valid_request())

    assert first.errors == ()
    assert first.normalized_request["coordinates"] == {
        "exchange": "binance",
        "market_type": "spot",
        "symbol": "BTCUSDT",
    }
    assert first.normalized_request["timeframe"] == "15m"
    assert first.normalized_request["quality_constraints"] == {"min_closed_trades": 300}
    assert first.normalized_request["execution"]["sizing"] == {
        "mode": "fixed_equity_pct",
        "equity_pct": 10.0,
    }
    assert first.request_hash == second.request_hash
    assert first.result_config_hash == second.result_config_hash
    assert len(first.request_hash) == 64
    assert first.artifact_metadata.artifact_slot == "slot_a"
    assert first.artifact_metadata.hit_times_manifest_hash == "b" * 64
    assert first.cost_estimate.as_mapping() == {
        "indicator_rows": 6,
        "candidate_combinations": 6,
        "tp_sl_cells": 0,
        "cost_class": "small",
        "estimated_combinations_upper_bound": 6,
        "estimated_combinations": 6,
        "arity": 1,
        "row_count_upper_bounds_by_indicator": {"ma.dema": 6},
        "risk_mode": "none",
        "requested_range": {
            "start": "2020-01-11T20:08:00Z",
            "end": "2026-04-11T20:08:00Z",
        },
        "requested_top_n": 50,
        "scheduling_class": "heavy",
    }
    assert resolver.coordinates == (BacktestCoordinates("binance", "spot", "BTCUSDT"),) * 2


def test_preflight_accepts_explicit_min_closed_trades_override() -> None:
    request = _valid_request()
    request["quality_constraints"] = {"min_closed_trades": 37}

    result = _service().execute(request)

    assert result.normalized_request["quality_constraints"] == {"min_closed_trades": 37}


@pytest.mark.parametrize(
    ("sizing", "expected"),
    [
        ({"mode": "all_in"}, {"mode": "all_in"}),
        (
            {"mode": "fixed_quote", "quote_amount": 250.0},
            {"mode": "fixed_quote", "quote_amount": 250.0},
        ),
        (
            {"mode": "fixed_equity_pct", "equity_pct": 15.0},
            {"mode": "fixed_equity_pct", "equity_pct": 15.0},
        ),
        (
            {
                "mode": "fixed_equity_pct_min_quote",
                "equity_pct": 5.0,
                "min_quote": 50.0,
            },
            {
                "mode": "fixed_equity_pct_min_quote",
                "equity_pct": 5.0,
                "min_quote": 50.0,
            },
        ),
        (
            {
                "mode": "fixed_equity_pct_max_quote",
                "equity_pct": 50.0,
                "max_quote": 500.0,
            },
            {
                "mode": "fixed_equity_pct_max_quote",
                "equity_pct": 50.0,
                "max_quote": 500.0,
            },
        ),
    ],
)
def test_preflight_normalizes_public_sizing_fields(
    sizing: dict[str, float | str],
    expected: dict[str, float | str],
) -> None:
    request = _valid_request()
    request["execution"]["sizing"] = sizing

    result = _service().execute(request)

    assert result.normalized_request["execution"]["sizing"] == expected


@pytest.mark.parametrize(
    ("mutation", "path", "issue_code"),
    [
        (lambda request: request.update({"timeframe": "1m"}), "timeframe", "unsupported_timeframe"),
        (lambda request: request.update({"timeframe": "5m"}), "timeframe", "unsupported_timeframe"),
        (
            lambda request: request["coordinates"].update({"exchange": "unsupported"}),
            "coordinates",
            "unsupported_market",
        ),
        (
            lambda request: request["time_range"].update({"end": "2020-01-11T20:08:00Z"}),
            "time_range",
            "invalid_range",
        ),
        (
            lambda request: request["indicators"][0].update({"indicator_id": "ma.nope"}),
            "indicators.0.indicator_id",
            "unknown_indicator",
        ),
        (
            lambda request: request["indicators"][0].update({"sources": ["not_a_source"]}),
            "indicators.0.sources.0",
            "invalid_source",
        ),
        (
            lambda request: request["indicators"][0].update(
                {"window": {"start": 201, "stop": 205, "step": 1}}
            ),
            "indicators.0.window",
            "invalid_window",
        ),
        (
            lambda request: request["execution"].update({"direction_mode": "short_only"}),
            "execution.direction_mode",
            "unsupported_value",
        ),
        (
            lambda request: request["execution"].update({"sizing": {"mode": "unknown"}}),
            "execution.sizing.mode",
            "unsupported_value",
        ),
    ],
)
def test_preflight_invalid_request_cases_are_deterministic(
    mutation: Any,
    path: str,
    issue_code: str,
) -> None:
    request = _valid_request()
    mutation(request)

    with pytest.raises(BacktestPreflightRejected) as exc_info:
        _service().execute(request)

    assert exc_info.value.error_code == BACKTEST_ERROR_INVALID_REQUEST
    assert exc_info.value.issues[0].path == path
    assert exc_info.value.issues[0].code == issue_code


def test_preflight_rejects_request_too_expensive() -> None:
    request = _valid_request()
    request["indicators"] = [
        {
            "indicator_id": "ma.dema",
            "sources": ["close", "high", "hlc3", "low", "ohlc4", "open"],
            "window": {"start": 5, "stop": 200, "step": 1},
        }
    ]

    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
        guardrails=BacktestRuntimeGuardrails(max_indicator_rows=1_000),
    )

    with pytest.raises(BacktestPreflightRejected) as exc_info:
        _service(runtime_config=runtime_config).execute(request)

    assert exc_info.value.error_code == BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE
    assert exc_info.value.issues[0].code == "max_indicator_rows"


def test_preflight_classifies_obvious_196_pow_5_grid_as_heavy() -> None:
    request = _valid_request()
    request["indicators"] = [
        {
            "indicator_id": "ma.dema",
            "sources": ["close"],
            "window": {"start": 5, "stop": 200, "step": 1},
        }
        for _ in range(5)
    ]

    result = _service().execute(request)

    assert result.cost_estimate.estimated_combinations_upper_bound == 289_254_654_976
    assert result.cost_estimate.arity == 5
    assert result.cost_estimate.requested_top_n == 50
    assert result.cost_estimate.risk_mode == "none"
    assert result.cost_estimate.scheduling_class == "heavy"
    assert result.cost_estimate.row_count_upper_bounds_by_indicator == {
        "ma.dema": 196,
        "ma.dema#1": 196,
        "ma.dema#2": 196,
        "ma.dema#3": 196,
        "ma.dema#4": 196,
    }


def test_preflight_tp_sl_grid_validates_cells_and_configured_coverage() -> None:
    request = _valid_request()
    request["risk"] = {
        "mode": "tp_sl_grid",
        "tp": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
        "sl": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
    }

    result = _service().execute(request)

    assert result.normalized_request["risk"]["mode"] == "tp_sl_grid"
    assert result.cost_estimate.tp_sl_cells == 2209


def test_preflight_rejects_tp_sl_grid_outside_configured_coverage() -> None:
    request = _valid_request()
    request["risk"] = {
        "mode": "tp_sl_grid",
        "tp": {"start_pct": 25.5, "stop_pct": 25.5, "step_pct": 0.5},
        "sl": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
    }
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 51)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
    )

    with pytest.raises(BacktestPreflightRejected) as exc_info:
        _service(runtime_config=runtime_config).execute(request)

    assert exc_info.value.error_code == BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED
    assert exc_info.value.issues[0].code == "tp_sl_grid_not_covered"


def test_preflight_rejects_tp_sl_grid_above_cell_guardrail() -> None:
    request = _valid_request()
    request["risk"] = {
        "mode": "tp_sl_grid",
        "tp": {"start_pct": 0.5, "stop_pct": 50.0, "step_pct": 0.5},
        "sl": {"start_pct": 0.5, "stop_pct": 25.0, "step_pct": 0.5},
    }

    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
        guardrails=BacktestRuntimeGuardrails(max_tp_sl_cells=2_209),
    )

    with pytest.raises(BacktestPreflightRejected) as exc_info:
        _service(runtime_config=runtime_config).execute(request)

    assert exc_info.value.error_code == BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE
    assert exc_info.value.issues[0].code == "max_tp_sl_cells"


def test_preflight_does_not_pass_request_paths_to_artifact_resolver() -> None:
    resolver = _FakeArtifactResolver()
    request = _valid_request()
    request["artifact_root"] = "/tmp/user-supplied-root"

    result = _service(resolver=resolver).execute(request)

    assert result.artifact_metadata.artifact_manifest_hash == "a" * 64
    assert resolver.coordinates == (BacktestCoordinates("binance", "spot", "BTCUSDT"),)


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


def _service(
    *,
    resolver: _FakeArtifactResolver | None = None,
    runtime_config: BacktestRuntimeConfig | None = None,
) -> BacktestPreflightService:
    return BacktestPreflightService(
        defaults_provider=_defaults_provider(),
        artifact_context_resolver=resolver or _FakeArtifactResolver(),
        runtime_config=runtime_config or _runtime_config(),
    )


def _defaults_provider() -> YamlBacktestGridDefaultsProvider:
    return YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )


def _runtime_config() -> BacktestRuntimeConfig:
    return BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
    )


def _valid_request() -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "btcusdt",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "2020-01-11T20:08:00Z",
            "end": "2026-04-11T20:08:00Z",
        },
        "indicators": [
            {
                "indicator_id": "MA.DEMA",
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
        "top_n": 50,
    }
