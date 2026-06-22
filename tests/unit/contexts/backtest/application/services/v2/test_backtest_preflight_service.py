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
    BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
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
    assert response["direction_modes"] == ["long_only", "short", "long_short_reversal"]
    assert response["sizing_modes"] == [
        "all_in",
        "fixed_quote",
        "fixed_equity_pct",
        "fixed_equity_pct_min_quote",
        "fixed_equity_pct_max_quote",
    ]
    assert "total_return_pct" in response["ranking_metrics"]
    assert "total_return_pct_net_of_funding" in response["ranking_metrics"]
    assert response["top_n_default"] == 10
    assert response["guardrails"]["max_candidate_combinations"] == 10_000_000_000_000
    assert response["quality_constraints_default"] == {
        "min_closed_trades_policy": "timeframe_sqrt_v1",
        "base_trades_per_year_at_1h": 24,
        "min_trades_per_year": 12,
        "max_trades_per_year": 365,
    }
    assert response["execution_defaults"]["funding"] == {
        "mode": "include_when_futures",
        "coverage_policy": "degraded_with_warning",
    }
    compatibility = response["direction_market_compatibility"]
    assert compatibility["markets"]["spot"]["allowed_direction_modes"] == ["long_only"]
    assert compatibility["markets"]["spot"]["rejected_direction_modes"] == {
        "short": BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
        "long_short_reversal": BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
    }
    assert compatibility["markets"]["futures"]["allowed_direction_modes"] == [
        "long_only",
        "short",
        "long_short_reversal",
    ]
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
    assert first.normalized_request["execution"]["funding"] == {
        "mode": "off",
        "coverage_policy": "degraded_with_warning",
    }
    assert first.normalized_request["ranking"] == {
        "primary_metric": "total_return_pct",
        "requested_primary_metric": "total_return_pct",
        "effective_primary_metric": "total_return_pct",
        "direction": "desc",
    }
    assert first.funding_readiness == {
        "status": "not_applicable",
        "coverage_policy": "not_applicable",
        "warning_codes": [],
        "coverage_ratio": None,
        "window": {
            "start": "2020-01-11T20:08:00Z",
            "end": "2026-04-11T20:08:00Z",
        },
        "funding_manifest_hash": None,
        "rows_count": 0,
        "expected_event_count": 0,
        "missing_event_count": 0,
    }
    assert first.direction_market_compatibility == {
        "market_type": "spot",
        "direction_mode": "long_only",
        "compatible": True,
        "reason_codes": [],
        "required_market_type": None,
        "funding_default": {
            "mode": "off",
            "coverage_policy": "degraded_with_warning",
        },
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
        "requested_top_n": 10,
        "scheduling_class": "heavy",
    }
    assert resolver.coordinates == (BacktestCoordinates("binance", "spot", "BTCUSDT"),) * 2


def test_preflight_futures_short_defaults_to_funding_include_and_ready() -> None:
    request = _valid_request()
    request["coordinates"]["market_type"] = "futures"
    request["execution"]["direction_mode"] = "short"
    resolver = _FakeArtifactResolver(
        funding_coverage_status="ready",
        funding_coverage_policy="ready",
        funding_manifest_hash="c" * 64,
        funding_rows_count=6,
        funding_expected_event_count=6,
        funding_missing_event_count=0,
    )

    result = _service(resolver=resolver).execute(request)

    assert result.normalized_request["execution"]["funding"] == {
        "mode": "include_when_futures",
        "coverage_policy": "degraded_with_warning",
    }
    assert result.normalized_request["ranking"] == {
        "primary_metric": "total_return_pct",
        "requested_primary_metric": "total_return_pct",
        "effective_primary_metric": "total_return_pct_net_of_funding",
        "direction": "desc",
    }
    assert result.funding_readiness == {
        "status": "ready",
        "coverage_policy": "degraded_with_warning",
        "warning_codes": [],
        "coverage_ratio": 1.0,
        "window": {
            "start": "2020-01-11T20:08:00Z",
            "end": "2026-04-11T20:08:00Z",
        },
        "funding_manifest_hash": "c" * 64,
        "rows_count": 6,
        "expected_event_count": 6,
        "missing_event_count": 0,
    }
    assert result.direction_market_compatibility["compatible"] is True
    assert result.direction_market_compatibility["required_market_type"] == "futures"


@pytest.mark.parametrize(
    ("status", "reason_codes", "expected_ratio", "warning_code"),
    [
        ("degraded", ("missing_trailing_coverage",), 0.75, "funding_readiness_degraded"),
        ("unavailable", ("no_funding_rows",), 0.0, "funding_readiness_unavailable"),
    ],
)
def test_preflight_futures_funding_readiness_warns_without_blocking(
    status: str,
    reason_codes: tuple[str, ...],
    expected_ratio: float,
    warning_code: str,
) -> None:
    request = _valid_request()
    request["coordinates"]["market_type"] = "futures"
    request["execution"]["direction_mode"] = "long_only"
    resolver = _FakeArtifactResolver(
        funding_coverage_status=status,
        funding_coverage_policy=status,
        funding_manifest_hash="c" * 64,
        funding_rows_count=3 if status == "degraded" else 0,
        funding_expected_event_count=4,
        funding_missing_event_count=1 if status == "degraded" else 4,
        funding_reason_codes=reason_codes,
    )

    result = _service(resolver=resolver).execute(request)

    assert result.errors == ()
    assert result.funding_readiness["status"] == status
    assert result.funding_readiness["coverage_policy"] == "degraded_with_warning"
    assert result.funding_readiness["warning_codes"] == list(reason_codes)
    assert result.funding_readiness["coverage_ratio"] == expected_ratio
    assert warning_code in {warning.code for warning in result.warnings}


def test_preflight_futures_missing_funding_manifest_reports_unavailable() -> None:
    request = _valid_request()
    request["coordinates"]["market_type"] = "futures"
    request["execution"]["direction_mode"] = "long_only"

    result = _service(resolver=_FakeArtifactResolver()).execute(request)

    assert result.funding_readiness["status"] == "unavailable"
    assert result.funding_readiness["coverage_policy"] == "degraded_with_warning"
    assert result.funding_readiness["warning_codes"] == ["funding_artifacts_unavailable"]
    assert result.funding_readiness["funding_manifest_hash"] is None


def test_preflight_rejects_new_spot_short_like_requests() -> None:
    for direction_mode in ("short", "long_short_reversal"):
        request = _valid_request()
        request["execution"]["direction_mode"] = direction_mode

        with pytest.raises(BacktestPreflightRejected) as exc_info:
            _service().execute(request)

        assert exc_info.value.error_code == BACKTEST_ERROR_INVALID_REQUEST
        assert exc_info.value.issues[0].path == "execution.direction_mode"
        assert exc_info.value.issues[0].code == (
            BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET
        )


def test_preflight_request_hash_includes_normalized_funding_config() -> None:
    default_request = _valid_request()
    default_request["coordinates"]["market_type"] = "futures"
    default_request["execution"]["direction_mode"] = "long_only"
    funding_off_request = _valid_request()
    funding_off_request["coordinates"]["market_type"] = "futures"
    funding_off_request["execution"]["direction_mode"] = "long_only"
    funding_off_request["execution"]["funding"] = {
        "mode": "off",
        "coverage_policy": "degraded_with_warning",
    }

    service = _service(
        resolver=_FakeArtifactResolver(
            funding_coverage_status="ready",
            funding_coverage_policy="ready",
            funding_manifest_hash="c" * 64,
            funding_rows_count=6,
            funding_expected_event_count=6,
            funding_missing_event_count=0,
        )
    )

    assert service.execute(default_request).request_hash != service.execute(
        funding_off_request
    ).request_hash


def test_preflight_accepts_explicit_min_closed_trades_override() -> None:
    request = _valid_request()
    request["quality_constraints"] = {"min_closed_trades": 37}

    result = _service().execute(request)

    assert result.normalized_request["quality_constraints"] == {"min_closed_trades": 37}


def test_preflight_uses_top_n_10_when_request_omits_top_n() -> None:
    request = _valid_request()
    request.pop("top_n")

    result = _service().execute(request)

    assert result.normalized_request["top_n"] == 10
    assert result.cost_estimate.requested_top_n == 10


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
    assert result.cost_estimate.requested_top_n == 10
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


def test_preflight_tp_sl_futures_funding_defaults_to_net_effective_ranking() -> None:
    request = _valid_request()
    request["coordinates"]["market_type"] = "futures"
    request["risk"] = {
        "mode": "tp_sl_grid",
        "tp": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
        "sl": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
    }
    resolver = _FakeArtifactResolver(
        funding_coverage_status="ready",
        funding_coverage_policy="ready",
        funding_manifest_hash="c" * 64,
        funding_rows_count=6,
        funding_expected_event_count=6,
        funding_missing_event_count=0,
    )

    result = _service(resolver=resolver).execute(request)

    assert result.normalized_request["execution"]["funding"] == {
        "mode": "include_when_futures",
        "coverage_policy": "degraded_with_warning",
    }
    assert result.normalized_request["ranking"] == {
        "primary_metric": "total_return_pct",
        "requested_primary_metric": "total_return_pct",
        "effective_primary_metric": "total_return_pct_net_of_funding",
        "direction": "desc",
    }
    assert result.cost_estimate.risk_mode == "tp_sl_grid"


def test_preflight_tp_sl_grid_accepts_one_sided_risk() -> None:
    request = _valid_request()
    request["risk"] = {
        "mode": "tp_sl_grid",
        "tp": {"enabled": False},
        "sl": {"enabled": True, "start_pct": 0.5, "stop_pct": 25.0, "step_pct": 0.5},
    }

    result = _service().execute(request)

    assert result.normalized_request["risk"] == {
        "mode": "tp_sl_grid",
        "tp": {"enabled": False},
        "sl": {"enabled": True, "start_pct": 0.5, "stop_pct": 25.0, "step_pct": 0.5},
    }
    assert result.cost_estimate.tp_sl_cells == 50


def test_preflight_tp_sl_grid_rejects_both_sides_disabled() -> None:
    request = _valid_request()
    request["risk"] = {
        "mode": "tp_sl_grid",
        "tp": {"enabled": False},
        "sl": {"enabled": False},
    }

    with pytest.raises(BacktestPreflightRejected) as exc_info:
        _service().execute(request)

    assert exc_info.value.error_code == BACKTEST_ERROR_INVALID_REQUEST
    assert exc_info.value.issues[0].path == "risk"
    assert exc_info.value.issues[0].code == "empty"


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


def test_preflight_rejects_end_date_after_artifact_asof_date() -> None:
    request = _valid_request()
    request["time_range"]["end"] = "2026-03-26T00:00:00Z"

    with pytest.raises(BacktestPreflightRejected) as exc_info:
        _service(resolver=_FakeArtifactResolver(artifact_asof_date="2026-03-25")).execute(
            request
        )

    assert exc_info.value.error_code == BACKTEST_ERROR_INVALID_REQUEST
    assert exc_info.value.issues[0].path == "time_range.end"
    assert exc_info.value.issues[0].code == "after_artifact_asof_date"


@dataclass
class _FakeArtifactResolver:
    coordinates: tuple[BacktestCoordinates, ...] = ()
    artifact_asof_date: str = "2026-04-11"
    funding_manifest_hash: str | None = None
    funding_coverage_status: str | None = None
    funding_coverage_policy: str | None = None
    funding_rows_count: int | None = None
    funding_expected_event_count: int | None = None
    funding_missing_event_count: int | None = None
    funding_reason_codes: tuple[str, ...] = ()

    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        self.coordinates = (*self.coordinates, coordinates)
        return BacktestArtifactMetadata(
            artifact_slot="slot_a",
            artifact_slot_generation=4,
            artifact_manifest_hash="a" * 64,
            artifact_asof_date=self.artifact_asof_date,
            hit_times_manifest_hash="b" * 64,
            published_at_utc="2026-03-25T02:00:00Z",
            funding_manifest_hash=self.funding_manifest_hash,
            funding_coverage_status=self.funding_coverage_status,
            funding_coverage_policy=self.funding_coverage_policy,
            funding_rows_count=self.funding_rows_count,
            funding_expected_event_count=self.funding_expected_event_count,
            funding_missing_event_count=self.funding_missing_event_count,
            funding_reason_codes=self.funding_reason_codes,
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
            "direction_mode": "long_only",
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
        "top_n": 10,
    }
