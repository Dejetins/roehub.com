from __future__ import annotations

from datetime import datetime, timezone

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
