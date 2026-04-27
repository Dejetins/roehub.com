from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningConfig,
    BacktestNoRiskPriceContext,
    BacktestPreparePoolsResult,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparePoolsTiming,
)
from trading.contexts.backtest.application.services.v2 import (
    EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_NO_RISK_BACKEND,
    STREAMING_2_NO_RISK_BACKEND,
    BacktestComboPlanningService,
    BacktestNoRiskExactScoringService,
    build_signal_segments,
)
from trading.contexts.backtest.application.services.v2 import no_risk_exact as no_risk_module


def test_event_segments_two_no_risk_matches_slow_reference_and_metrics() -> None:
    prepared = _prepared_result(("alpha", "beta"))
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=_normalized_request(direction_mode="long_short_reversal"),
    )

    result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=_normalized_request(direction_mode="long_short_reversal"),
        price_context=_price_context(),
    )

    assert planning.backend.backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND
    assert result.self_check["passed"] is True
    assert result.self_check["trade_count_equal"] is True
    assert result.telemetry.benchmark_top_k == 5
    assert result.telemetry.top_results_count == 5
    assert _metric_keys().issubset(result.top_results[0].keys())
    assert "_local_indices" not in result.top_results[0]
    assert "_proxy_pending" not in result.top_results[0]


@pytest.mark.parametrize("arity", [1, 3, 10])
def test_event_segments_n_no_risk_supports_generic_arities(arity: int) -> None:
    indicator_ids = tuple(f"i{idx}" for idx in range(arity))
    prepared = _prepared_result(indicator_ids, rows_per_indicator=1)
    request = _normalized_request(direction_mode="long_short_reversal")
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )

    result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        price_context=_price_context(),
    )

    assert planning.backend.backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert result.self_check["passed"] is True
    assert result.telemetry.exact_backend_display_name == f"event_segments_{arity}_no_risk"
    assert result.telemetry.top_results_count == 1
    assert result.top_results[0]["trade_count"] >= 1


def test_streaming_two_no_risk_fallback_matches_default_top_results() -> None:
    prepared = _prepared_result(("alpha", "beta"))
    request = _normalized_request(direction_mode="long_short_reversal")
    default_planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    streaming_planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
        requested_backend_id=STREAMING_2_NO_RISK_BACKEND,
    )

    default_result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=default_planning,
        normalized_request=request,
        price_context=_price_context(),
    )
    streaming_result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=streaming_planning,
        normalized_request=request,
        price_context=_price_context(),
    )

    assert streaming_planning.backend.backend_id == STREAMING_2_NO_RISK_BACKEND
    assert _top_return_vector(streaming_result) == pytest.approx(
        _top_return_vector(default_result)
    )


def test_direction_modes_change_close_and_reversal_semantics() -> None:
    prepared = _prepared_result(("alpha", "beta"), rows_per_indicator=1)
    long_only_request = _normalized_request(direction_mode="long_only")
    reversal_request = _normalized_request(direction_mode="long_short_reversal")
    planning_long_only = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=long_only_request,
    )
    planning_reversal = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=reversal_request,
    )

    long_only = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning_long_only,
        normalized_request=long_only_request,
        price_context=_price_context(),
    )
    reversal = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning_reversal,
        normalized_request=reversal_request,
        price_context=_price_context(),
    )

    assert long_only.self_check["passed"] is True
    assert reversal.self_check["passed"] is True
    assert long_only.top_results[0]["trade_count"] == 1
    assert reversal.top_results[0]["trade_count"] == 2


def test_request_top_n_100_does_not_change_canonical_benchmark_top_k() -> None:
    prepared = _prepared_result(("alpha", "beta", "gamma"))
    request_top_100 = _normalized_request(top_n=100)
    request_top_3 = _normalized_request(top_n=3)
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request_top_100,
    )
    service = BacktestNoRiskExactScoringService()

    top_100_result = service.execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request_top_100,
        price_context=_price_context(),
    )
    top_3_result = service.execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request_top_3,
        price_context=_price_context(),
    )

    assert top_100_result.telemetry.request_top_n == 100
    assert top_100_result.telemetry.benchmark_top_k == 5
    assert top_100_result.telemetry.heap_capacity == 5
    assert top_100_result.telemetry.top_results_count == 5
    assert top_3_result.telemetry.request_top_n == 3
    assert top_3_result.telemetry.benchmark_top_k == 5
    assert _top_return_vector(top_3_result) == pytest.approx(_top_return_vector(top_100_result))


def test_top_result_proxy_fill_recomputes_only_final_pass_through_top_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(("alpha", "beta", "gamma"))
    request = _normalized_request(top_n=100)
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    calls: list[int] = []
    original = no_risk_module.proxy_for_indicator_rows

    def proxy_spy(**kwargs: Any) -> tuple[int, float]:
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(no_risk_module, "proxy_for_indicator_rows", proxy_spy)

    result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        price_context=_price_context(),
    )

    assert planning.proxy_context.active is False
    assert result.telemetry.top_results_count == 5
    assert len(calls) == 5


def test_active_proxy_metadata_is_preserved_without_final_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(("alpha", "beta"))
    request = _normalized_request(top_n=100)
    planning_service = BacktestComboPlanningService(
        config=BacktestComboPlanningConfig(combo_top_frac=0.75, combo_min_confirm=1)
    )
    planning = planning_service.execute(
        prepared_result=prepared,
        normalized_request=request,
    )

    def fail_proxy_fill(**_: Any) -> tuple[int, float]:
        raise AssertionError("pass-through proxy recompute should not run for active proxy rows")

    monkeypatch.setattr(no_risk_module, "proxy_for_indicator_rows", fail_proxy_fill)

    result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        price_context=_price_context(),
    )

    assert planning.proxy_context.active is True
    assert result.top_results
    assert all(int(item["confirm_count"]) >= 1 for item in result.top_results)


def _prepared_result(
    indicator_ids: Sequence[str],
    *,
    rows_per_indicator: int = 3,
) -> BacktestPreparePoolsResult:
    pools = tuple(
        _pool(indicator_id=indicator_id, rows_per_indicator=rows_per_indicator)
        for indicator_id in indicator_ids
    )
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=tuple(indicator_ids),
        indicator_pools=pools,
        signal_returns_15m=np.asarray([0.10, -0.10, 0.05, -0.05, 0.02], dtype=np.float32),
        execution_mapping=PreparedExecutionMapping(
            signal_entry_exec_idx_15m=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32),
            run_bar_open_1m_idx_15m=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint32),
            run_bar_close_1m_idx_15m=np.asarray([1, 2, 3, 4, 5, 5], dtype=np.uint32),
            t_exec_limit_1m=6,
        ),
        time_slice_start_15m=0,
        time_slice_stop_15m=6,
        trade_T_length=6,
        eval_T_length=5,
        row_metadata_order_hash="b" * 64,
        timing=PreparePoolsTiming(
            stage_name="prepare_pools_core",
            wall_time_s=0.0,
            subsegments={"prepare_pools_core": 0.0},
        ),
    )


def _pool(*, indicator_id: str, rows_per_indicator: int) -> PreparedIndicatorPool:
    row_templates = (
        [1, 1, -1, -1, 0, 1],
        [1, 0, 1, 1, -1, -1],
        [-1, -1, -1, 1, 1, 0],
    )
    trade_rows = np.asarray(row_templates[:rows_per_indicator], dtype=np.int8)
    row_ids = np.arange(rows_per_indicator, dtype=np.int32) + _indicator_row_offset(indicator_id)
    segments = build_signal_segments(trade_rows)
    metadata = tuple(
        PreparedIndicatorRowMetadata(
            indicator_id=indicator_id,
            row_id=int(row_ids[row_idx]),
            source="close",
            window=5 + row_idx,
        )
        for row_idx in range(rows_per_indicator)
    )
    eval_t = np.ascontiguousarray(trade_rows[:, :-1])
    return PreparedIndicatorPool(
        indicator_id=indicator_id,
        row_ids=row_ids,
        filtered_row_ids=row_ids,
        trade_T=trade_rows,
        eval_T=eval_t,
        segments=segments,
        row_score=np.zeros(rows_per_indicator, dtype=np.float32),
        score_adj=np.zeros(rows_per_indicator, dtype=np.float32),
        nonzero=np.count_nonzero(eval_t, axis=1).astype(np.int32),
        proxy=np.zeros(rows_per_indicator, dtype=np.float32),
        change_count=segments.change_count,
        metadata=metadata,
    )


def _indicator_row_offset(indicator_id: str) -> np.int32:
    return np.int32((sum(ord(char) for char in indicator_id) % 97) * 100)


def _price_context() -> BacktestNoRiskPriceContext:
    return BacktestNoRiskPriceContext(
        execution_open_1m=np.asarray([100.0, 100.0, 108.0, 104.0, 96.0, 90.0], dtype=np.float32),
        execution_close_1m=np.asarray([100.0, 101.0, 107.0, 103.0, 95.0, 88.0], dtype=np.float32),
    )


def _normalized_request(
    *,
    direction_mode: str = "long_short_reversal",
    top_n: int = 100,
) -> dict[str, Any]:
    return {
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": direction_mode,
            "fee_rate": 0.0,
            "slippage_rate": 0.0,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "all_in", "fixed_quote": 100.0},
            "profit_lock": {"enabled": False, "safe_profit_percent": 30.0},
            "close_on_end": True,
        },
        "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
        "top_n": top_n,
    }


def _metric_keys() -> set[str]:
    return {
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "trade_count",
        "sharpe_trades",
        "win_rate_pct",
        "avg_trade_ret_pct",
        "avg_trade_exec_bars",
        "exposure_pct",
    }


def _top_return_vector(result: Any) -> list[float]:
    return [float(item["total_return_pct"]) for item in result.top_results]
