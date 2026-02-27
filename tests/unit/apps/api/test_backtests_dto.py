from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from apps.api.dto import BacktestsPostRequest, build_backtest_run_request
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
