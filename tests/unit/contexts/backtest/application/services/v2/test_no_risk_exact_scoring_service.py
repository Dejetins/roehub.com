from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningConfig,
    BacktestNoRiskExactScoringConfig,
    BacktestNoRiskExecutionPrices,
    BacktestPreparePoolsResult,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparePoolsTiming,
)
from trading.contexts.backtest.application.services.v2 import (
    EXACT_SCORING_STAGE_NAME,
    HEAP_UPDATE_STAGE_NAME,
    SELF_CHECK_STAGE_NAME,
    STREAMING_2_NO_RISK_BACKEND,
    TOP_RESULT_PROXY_FILL_STAGE_NAME,
    TOTAL_WITHOUT_WARMUP_STAGE_NAME,
    BacktestComboPlanningService,
    BacktestNoRiskExactScoringRejected,
    BacktestNoRiskExactScoringService,
    build_persisted_top_n_summary_rows,
    build_signal_segments,
    evaluate_no_risk_exact_chunk,
    no_risk_execution_config_from_normalized,
)


def test_event_segments_two_no_risk_matches_streaming_and_slow_reference() -> None:
    prepared = _prepared_result(("alpha", "beta"))
    prices = _execution_prices()
    request = _normalized_request(direction_mode="long_short_reversal")
    planning_service = BacktestComboPlanningService()
    default_plan = planning_service.execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    streaming_plan = planning_service.execute(
        prepared_result=prepared,
        normalized_request=request,
        requested_backend_id=STREAMING_2_NO_RISK_BACKEND,
    )
    selected = {
        "alpha": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "beta": np.asarray([0, 1, 0, 1], dtype=np.int32),
    }
    execution_config = no_risk_execution_config_from_normalized(normalized_request=request)

    event_scores = evaluate_no_risk_exact_chunk(
        selected_rows_by_indicator=selected,
        prepared_result=prepared,
        combo_planning_result=default_plan,
        execution_config=execution_config,
        execution_prices=prices,
    )
    streaming_scores = evaluate_no_risk_exact_chunk(
        selected_rows_by_indicator=selected,
        prepared_result=prepared,
        combo_planning_result=streaming_plan,
        execution_config=execution_config,
        execution_prices=prices,
    )
    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactScoringConfig(top_n=3),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=default_plan,
        normalized_request=request,
        execution_prices=prices,
    )

    assert event_scores.trade_count.tolist() == streaming_scores.trade_count.tolist()
    assert event_scores.total_return_pct.tolist() == pytest.approx(
        streaming_scores.total_return_pct.tolist()
    )
    assert result.telemetry.self_check.checked == 2
    assert result.telemetry.self_check.passed is True
    assert set(result.telemetry.stage_timings) == {
        SELF_CHECK_STAGE_NAME,
        EXACT_SCORING_STAGE_NAME,
        HEAP_UPDATE_STAGE_NAME,
        TOP_RESULT_PROXY_FILL_STAGE_NAME,
        TOTAL_WITHOUT_WARMUP_STAGE_NAME,
    }


def test_long_only_closes_on_non_long_consensus_without_opening_short() -> None:
    prepared = _prepared_result(("alpha", "beta"))
    prices = _execution_prices()
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=_normalized_request(direction_mode="long_only"),
    )
    selected = {
        "alpha": np.asarray([0], dtype=np.int32),
        "beta": np.asarray([1], dtype=np.int32),
    }
    execution_config = no_risk_execution_config_from_normalized(
        normalized_request=_normalized_request(direction_mode="long_only")
    )

    scores = evaluate_no_risk_exact_chunk(
        selected_rows_by_indicator=selected,
        prepared_result=prepared,
        combo_planning_result=planning,
        execution_config=execution_config,
        execution_prices=prices,
    )

    assert scores.trade_count.tolist() == [1]
    assert scores.exposure_pct.tolist() == pytest.approx([50.0])


def test_event_segments_n_supports_arity_one_and_generic_n() -> None:
    prices = _execution_prices()
    service = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactScoringConfig(top_n=2),
    )

    arity_one = _prepared_result(("alpha",))
    arity_one_request = _normalized_request(direction_mode="long_short_reversal")
    arity_one_plan = BacktestComboPlanningService().execute(
        prepared_result=arity_one,
        normalized_request=arity_one_request,
    )
    arity_one_result = service.execute(
        prepared_result=arity_one,
        combo_planning_result=arity_one_plan,
        normalized_request=arity_one_request,
        execution_prices=prices,
    )

    arity_three = _prepared_result(("alpha", "beta", "gamma"))
    arity_three_request = _normalized_request(direction_mode="long_short_reversal")
    arity_three_plan = BacktestComboPlanningService().execute(
        prepared_result=arity_three,
        normalized_request=arity_three_request,
    )
    arity_three_result = service.execute(
        prepared_result=arity_three,
        combo_planning_result=arity_three_plan,
        normalized_request=arity_three_request,
        execution_prices=prices,
    )

    assert arity_one_result.telemetry.exact_candidates_evaluated == 2
    assert arity_one_result.top_rows[0].summary_metrics.trade_count >= 1
    assert arity_three_result.telemetry.exact_candidates_evaluated == 8
    assert arity_three_result.telemetry.self_check.checked == 2


@pytest.mark.parametrize("arity", [8, 9, 10])
def test_event_segments_n_arity_8_to_10_correctness_smoke(arity: int) -> None:
    prepared = _prepared_many(arity)
    request = _normalized_request(direction_mode="long_short_reversal")
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactScoringConfig(top_n=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        execution_prices=_execution_prices(),
    )

    assert result.telemetry.exact_candidates_evaluated == 1
    assert result.telemetry.self_check.checked == 1
    assert result.top_rows[0].summary_metrics.trade_count == 1


def test_heap_update_keeps_deterministic_top_n_and_proxy_fill_for_pass_through() -> None:
    prepared = _prepared_result(("alpha", "beta"))
    request = _normalized_request(direction_mode="long_short_reversal")
    request["top_n"] = 3
    planning = BacktestComboPlanningService(
        config=BacktestComboPlanningConfig(combo_chunk_size=2),
    ).execute(
        prepared_result=prepared,
        normalized_request=request,
    )

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactScoringConfig(top_n=3, combo_chunk_size=2),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        execution_prices=_flat_execution_prices(),
    )

    assert [row.variant_index for row in result.top_rows] == [0, 1, 2]
    assert [dict(row.row_ids_by_indicator) for row in result.top_rows] == [
        {"alpha": 0, "beta": 0},
        {"alpha": 0, "beta": 1},
        {"alpha": 1, "beta": 0},
    ]
    assert result.top_rows[0].confirm_count == 1
    assert result.top_rows[0].proxy_score == pytest.approx(1.0)
    assert result.telemetry.top_result_proxy_filled == 3


def test_identity_mapping_builds_summary_only_persistence_rows() -> None:
    prepared = _prepared_result(("alpha", "beta"))
    request = _normalized_request(direction_mode="long_short_reversal")
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactScoringConfig(top_n=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=request,
        execution_prices=_execution_prices(),
    )

    persisted = build_persisted_top_n_summary_rows(
        job_id=UUID("00000000-0000-0000-0000-000000001004"),
        top_rows=result.top_rows,
        updated_at=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
    )

    top_row = result.top_rows[0]
    persisted_row = persisted[0]
    assert top_row.public_variant_key.startswith("no-risk/v1|")
    assert top_row.public_variant_key != top_row.variant_hash
    assert len(top_row.variant_hash) == 64
    assert persisted_row.variant_key == top_row.variant_hash
    assert persisted_row.indicator_variant_key == top_row.indicator_variant_hash
    assert persisted_row.payload_json["variant_key"] == top_row.public_variant_key
    assert persisted_row.payload_json["public_variant_key"] == top_row.public_variant_key
    assert persisted_row.payload_json["variant_hash"] == top_row.variant_hash
    assert persisted_row.payload_json["storage_variant_key"] == top_row.variant_hash
    assert persisted_row.report_table_md is None
    assert persisted_row.trades_json is None
    assert set(persisted_row.summary_metrics_json) >= {
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


def test_close_on_end_false_leaves_final_open_trade_unclosed() -> None:
    prepared = _prepared_constant_long()
    planning = BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=_normalized_request(close_on_end=False),
    )

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactScoringConfig(top_n=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=_normalized_request(close_on_end=False),
        execution_prices=_execution_prices(),
    )

    assert result.top_rows[0].summary_metrics.trade_count == 0
    assert result.top_rows[0].summary_metrics.total_return_pct == 0.0


def test_unsupported_sizing_mode_fails_explicitly() -> None:
    request = _normalized_request()
    request["execution"]["sizing"] = {"mode": "fixed_equity_pct", "equity_pct": 10.0}

    with pytest.raises(BacktestNoRiskExactScoringRejected, match="supports sizing modes"):
        no_risk_execution_config_from_normalized(normalized_request=request)


def _prepared_result(indicator_ids: Sequence[str]) -> BacktestPreparePoolsResult:
    pools_by_id = {
        "alpha": _pool(
            indicator_id="alpha",
            trade_rows=[
                [1, 1, 0, -1],
                [-1, -1, 1, 1],
            ],
        ),
        "beta": _pool(
            indicator_id="beta",
            trade_rows=[
                [1, 0, 0, -1],
                [1, 1, -1, 0],
            ],
        ),
        "gamma": _pool(
            indicator_id="gamma",
            trade_rows=[
                [1, 1, -1, -1],
                [-1, -1, 1, 1],
            ],
        ),
    }
    return _prepared_from_pools(tuple(pools_by_id[indicator_id] for indicator_id in indicator_ids))


def _prepared_many(arity: int) -> BacktestPreparePoolsResult:
    pools = tuple(
        _pool(
            indicator_id=f"i{index}",
            trade_rows=[[1, 1, 1, 1]],
        )
        for index in range(arity)
    )
    return _prepared_from_pools(pools)


def _prepared_constant_long() -> BacktestPreparePoolsResult:
    return _prepared_from_pools(
        (
            _pool(
                indicator_id="alpha",
                trade_rows=[[1, 1, 1, 1]],
            ),
        )
    )


def _prepared_from_pools(
    pools: tuple[PreparedIndicatorPool, ...],
) -> BacktestPreparePoolsResult:
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=tuple(pool.indicator_id for pool in pools),
        indicator_pools=pools,
        signal_returns_15m=np.asarray([1.0, 2.0, -2.0], dtype=np.float32),
        execution_mapping=PreparedExecutionMapping(
            signal_entry_exec_idx_15m=np.asarray([1, 2, 3, 4], dtype=np.int32),
            run_bar_open_1m_idx_15m=np.asarray([0, 1, 2, 3], dtype=np.int32),
            run_bar_close_1m_idx_15m=np.asarray([1, 2, 3, 4], dtype=np.int32),
            t_exec_limit_1m=4,
        ),
        time_slice_start_15m=0,
        time_slice_stop_15m=4,
        trade_T_length=4,
        eval_T_length=3,
        row_metadata_order_hash="a" * 64,
        timing=PreparePoolsTiming(
            stage_name="prepare_pools_core",
            wall_time_s=0.0,
            subsegments={"prepare_pools_core": 0.0},
        ),
    )


def _pool(
    *,
    indicator_id: str,
    trade_rows: Sequence[Sequence[int]],
) -> PreparedIndicatorPool:
    trade_t = np.asarray(trade_rows, dtype=np.int8)
    eval_t = np.ascontiguousarray(trade_t[:, :3])
    row_ids = np.arange(trade_t.shape[0], dtype=np.int32)
    segments = build_signal_segments(trade_t)
    metadata = tuple(
        PreparedIndicatorRowMetadata(
            indicator_id=indicator_id,
            row_id=int(row_id),
            source="close",
            window=5 + int(row_id),
        )
        for row_id in row_ids
    )
    return PreparedIndicatorPool(
        indicator_id=indicator_id,
        row_ids=row_ids,
        filtered_row_ids=row_ids,
        trade_T=trade_t,
        eval_T=eval_t,
        segments=segments,
        row_score=np.zeros(trade_t.shape[0], dtype=np.float32),
        score_adj=np.zeros(trade_t.shape[0], dtype=np.float32),
        nonzero=np.count_nonzero(eval_t, axis=1).astype(np.int32),
        proxy=np.zeros(trade_t.shape[0], dtype=np.float32),
        change_count=segments.change_count,
        metadata=metadata,
    )


def _execution_prices() -> BacktestNoRiskExecutionPrices:
    return BacktestNoRiskExecutionPrices(
        open_1m=np.asarray([100.0, 110.0, 120.0, 90.0], dtype=np.float32),
        close_1m=np.asarray([101.0, 111.0, 119.0, 95.0], dtype=np.float32),
    )


def _flat_execution_prices() -> BacktestNoRiskExecutionPrices:
    return BacktestNoRiskExecutionPrices(
        open_1m=np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float32),
        close_1m=np.asarray([100.0, 100.0, 100.0, 100.0], dtype=np.float32),
    )


def _normalized_request(
    *,
    direction_mode: str = "long_short_reversal",
    close_on_end: bool = True,
) -> dict[str, Any]:
    return {
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": direction_mode,
            "fee_rate": 0.0,
            "slippage_rate": 0.0,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "all_in"},
            "profit_lock": {"enabled": False},
            "close_on_end": close_on_end,
        },
        "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
        "top_n": 100,
    }
