from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from apps.api.dto import (
    BacktestsPostRequest,
    BacktestsVariantReportPostRequest,
    build_backtest_run_request,
    build_backtest_variant_report_payload,
    build_backtest_variant_report_run_request,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.domain.specifications import RangeValuesSpec


def test_build_backtest_run_request_preserves_int_range_axes() -> None:
    """Ensure range axis values do not get coerced to float.

    This protects integer indicator params (e.g. MA `window`) from failing grid validation
    with: `axis 'window' expects integer values`.

    Related:
      - apps/api/dto/backtests.py
      - src/trading/contexts/indicators/application/services/grid_builder.py
      - configs/prod/indicators.yaml
    """

    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "source": {"mode": "explicit", "values": ["close"]},
                        "params": {
                            "window": {
                                "mode": "range",
                                "start": 5,
                                "stop_incl": 100,
                                "step": 1,
                            }
                        },
                    }
                ],
            },
        }
    )

    built = build_backtest_run_request(request=request)
    assert built.template is not None
    assert len(built.template.indicator_grids) == 1

    window_spec = built.template.indicator_grids[0].params["window"]
    assert isinstance(window_spec, RangeValuesSpec)
    assert isinstance(window_spec.start, int)
    assert isinstance(window_spec.stop_inclusive, int)
    assert isinstance(window_spec.step, int)

    materialized = window_spec.materialize()
    assert len(materialized) > 0
    assert all(isinstance(item, int) for item in materialized)


def test_build_backtest_run_request_normalizes_ranking_metrics() -> None:
    """
    Verify ranking request block is accepted and normalized into application DTO.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Ranking metric identifiers are case-insensitive and normalized to lowercase literals.
    Raises:
        AssertionError: If ranking block is not converted or normalized deterministically.
    Side Effects:
        None.
    """
    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "source": {"mode": "explicit", "values": ["close"]},
                        "params": {
                            "window": {
                                "mode": "range",
                                "start": 5,
                                "stop_incl": 20,
                                "step": 5,
                            }
                        },
                    }
                ],
            },
            "ranking": {
                "primary_metric": "RETURN_OVER_MAX_DRAWDOWN",
                "secondary_metric": "PROFIT_FACTOR",
            },
        }
    )

    built = build_backtest_run_request(request=request)
    assert built.ranking is not None
    assert built.ranking.primary_metric == "return_over_max_drawdown"
    assert built.ranking.secondary_metric == "profit_factor"


def test_backtests_post_request_rejects_unknown_ranking_metric() -> None:
    """
    Verify strict request validation rejects unsupported ranking metric literal.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed ranking metrics are fixed by v1 contract.
    Raises:
        AssertionError: If unsupported metric value is accepted.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="must be one of"):
        BacktestsPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "ranking": {
                    "primary_metric": "total_return",
                },
            }
        )


def test_backtests_post_request_rejects_duplicate_ranking_metrics() -> None:
    """
    Verify strict request validation forbids duplicate primary and secondary ranking metrics.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Secondary metric duplicates primary metric only when both identifiers normalize equally.
    Raises:
        AssertionError: If duplicate metrics are accepted.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="secondary_metric must be different"):
        BacktestsPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "ranking": {
                    "primary_metric": "total_return_pct",
                    "secondary_metric": "TOTAL_RETURN_PCT",
                },
            }
        )


def test_build_backtest_variant_report_run_request_reuses_mode_validation() -> None:
    """
    Verify variant-report run-context mapper reuses `strategy_id xor template` validation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Variant-report endpoint shares sync mode contract with `POST /backtests`.
    Raises:
        AssertionError: If mode conflict is not rejected deterministically.
    Side Effects:
        None.
    """
    request = BacktestsVariantReportPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
            },
            "strategy_id": "00000000-0000-0000-0000-000000000123",
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {
                            "window": {"mode": "explicit", "values": [20]},
                        },
                    }
                ],
            },
            "variant": _variant_payload_request(),
        }
    )

    with pytest.raises(
        BacktestValidationError,
        match="requires exactly one mode",
    ):
        build_backtest_variant_report_run_request(request=request)


def test_build_backtest_variant_report_payload_normalizes_payload_deterministically() -> None:
    """
    Verify variant payload mapper sorts keys and lowercases signal identifiers deterministically.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Request payload can arrive in arbitrary key order from browser-side cache.
    Raises:
        AssertionError: If normalized payload ordering differs from deterministic contract.
    Side Effects:
        None.
    """
    payload = build_backtest_variant_report_payload(
        request=BacktestsVariantReportPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "variant": {
                    "indicator_selections": [
                        {
                            "indicator_id": "ma.sma",
                            "inputs": {"source": "close"},
                            "params": {"window": 20},
                        }
                    ],
                    "signal_params": {"MA.SMA": {"Cross_Up": 0.5}},
                    "risk_params": {"tp_enabled": True, "sl_enabled": True, "sl_pct": 2.0},
                    "execution_params": {
                        "slippage_pct": 0.01,
                        "fee_pct": 0.075,
                        "init_cash_quote": 10000.0,
                    },
                    "direction_mode": "long-short",
                    "sizing_mode": "all_in",
                },
            }
        ).variant
    )

    assert payload.signal_params is not None
    assert payload.risk_params is not None
    assert payload.execution_params is not None
    assert tuple(payload.signal_params.keys()) == ("ma.sma",)
    assert tuple(payload.signal_params["ma.sma"].keys()) == ("cross_up",)
    assert tuple(payload.risk_params.keys()) == ("sl_enabled", "sl_pct", "tp_enabled")
    assert tuple(payload.execution_params.keys()) == (
        "fee_pct",
        "init_cash_quote",
        "slippage_pct",
    )


def test_build_backtest_variant_report_payload_rejects_boolean_indicator_values() -> None:
    """
    Verify variant payload mapper rejects booleans in explicit indicator selection scalars.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Indicator selection scalars follow indicators variant key contract (`int|float|str`).
    Raises:
        AssertionError: If boolean scalar is accepted in selection mapping.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="must be int, float, or string"):
        BacktestsVariantReportPostRequest.model_validate(
            {
                "time_range": {
                    "start": datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
                    "end": datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc),
                },
                "template": {
                    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                    "timeframe": "1m",
                    "indicator_grids": [
                        {
                            "indicator_id": "ma.sma",
                            "params": {
                                "window": {"mode": "explicit", "values": [20]},
                            },
                        }
                    ],
                },
                "variant": {
                    "indicator_selections": [
                        {
                            "indicator_id": "ma.sma",
                            "inputs": {"source": "close"},
                            "params": {"window": True},
                        }
                    ],
                    "signal_params": {"ma.sma": {"cross_up": 0.5}},
                    "risk_params": {"sl_enabled": True},
                    "execution_params": {"fee_pct": 0.075},
                    "direction_mode": "long-short",
                    "sizing_mode": "all_in",
                },
            }
        )


def _variant_payload_request() -> dict[str, object]:
    """
    Build minimal explicit variant payload used by DTO variant-report mapping tests.

    Args:
        None.
    Returns:
        dict[str, object]: Explicit variant payload JSON object.
    Assumptions:
        One indicator selection is sufficient for mode-validation tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "indicator_selections": [
            {
                "indicator_id": "ma.sma",
                "inputs": {"source": "close"},
                "params": {"window": 20},
            }
        ],
        "signal_params": {"ma.sma": {"cross_up": 0.5}},
        "risk_params": {"sl_enabled": True},
        "execution_params": {"fee_pct": 0.075},
        "direction_mode": "long-short",
        "sizing_mode": "all_in",
    }
