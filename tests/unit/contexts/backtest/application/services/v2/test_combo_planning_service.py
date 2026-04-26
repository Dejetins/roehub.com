from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningConfig,
    BacktestPreparePoolsResult,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparePoolsTiming,
)
from trading.contexts.backtest.application.services.v2 import (
    BUILD_EXACT_CONTEXT_STAGE_NAME,
    BUILD_PROXY_CONTEXT_STAGE_NAME,
    COMBO_ITERATION_STAGE_NAME,
    EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
    PROXY_FILTER_STAGE_NAME,
    STREAMING_2_NO_RISK_BACKEND,
    BacktestBackendRegistry,
    BacktestComboPlanningRejected,
    BacktestComboPlanningService,
    build_signal_segments,
    cartesian_combo_count,
    iter_combo_chunks,
)


def test_backend_registry_selects_supported_v1_backends() -> None:
    registry = BacktestBackendRegistry.default()

    assert registry.select(
        risk_mode="none",
        arity=2,
        direction_mode="long_only",
    ).backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND
    assert registry.select(
        risk_mode="none",
        arity=2,
        direction_mode="long_short_reversal",
        requested_backend_id=STREAMING_2_NO_RISK_BACKEND,
    ).backend_id == STREAMING_2_NO_RISK_BACKEND
    assert registry.select(
        risk_mode="none",
        arity=1,
        direction_mode="long_only",
    ).backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert registry.select(
        risk_mode="none",
        arity=10,
        direction_mode="long_only",
    ).backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert registry.select(
        risk_mode="tp_sl_grid",
        arity=10,
        direction_mode="long_short_reversal",
    ).backend_id == EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND


def test_backend_registry_rejects_unsupported_combinations() -> None:
    registry = BacktestBackendRegistry.default()

    with pytest.raises(BacktestComboPlanningRejected, match="does not support"):
        registry.select(
            risk_mode="none",
            arity=2,
            direction_mode="long_only",
            requested_backend_id=EVENT_SEGMENTS_N_NO_RISK_BACKEND,
        )
    with pytest.raises(BacktestComboPlanningRejected, match="Unsupported indicator arity"):
        registry.select(risk_mode="none", arity=11, direction_mode="long_only")
    with pytest.raises(BacktestComboPlanningRejected, match="Unsupported risk_mode"):
        registry.select(risk_mode="unknown", arity=1, direction_mode="long_only")
    with pytest.raises(BacktestComboPlanningRejected, match="Unsupported direction_mode"):
        registry.select(risk_mode="none", arity=1, direction_mode="sideways")


def test_build_exact_context_packs_arity_first_segment_arrays() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta", "gamma"))
    service = BacktestComboPlanningService()
    backend = BacktestBackendRegistry.default().select(
        risk_mode="none",
        arity=3,
        direction_mode="long_only",
    )

    exact_context = service.build_exact_context(
        prepared_result=prepared,
        backend=backend,
    )

    assert exact_context.required is True
    assert exact_context.materialized is True
    assert exact_context.row_counts == (2, 3, 2)
    assert exact_context.starts is not None
    assert exact_context.ends is not None
    assert exact_context.values is not None
    assert exact_context.counts is not None
    assert exact_context.starts.shape == (3, 3, 3)
    assert exact_context.ends.shape == (3, 3, 3)
    assert exact_context.values.shape == (3, 3, 3)
    assert exact_context.counts.shape == (3, 3)
    assert exact_context.starts.dtype == np.int32
    assert exact_context.values.dtype == np.int8
    assert exact_context.starts.flags.c_contiguous
    assert exact_context.values.flags.c_contiguous
    assert exact_context.starts[0, 0].tolist() == [0, 2, 3]
    assert exact_context.ends[0, 0].tolist() == [2, 3, 4]
    assert exact_context.values[0, 0].tolist() == [1, 0, -1]
    assert exact_context.counts[1].tolist() == [2, 3, 2]
    assert exact_context.counts[2].tolist() == [1, 3, 0]


def test_build_exact_context_keeps_specialized_two_no_risk_near_zero() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    service = BacktestComboPlanningService()
    backend = BacktestBackendRegistry.default().select(
        risk_mode="none",
        arity=2,
        direction_mode="long_short_reversal",
    )

    exact_context = service.build_exact_context(
        prepared_result=prepared,
        backend=backend,
    )

    assert exact_context.required is False
    assert exact_context.materialized is False
    assert exact_context.starts is None
    assert exact_context.ends is None
    assert exact_context.values is None
    assert exact_context.counts is None
    assert exact_context.row_counts == (2, 3)


def test_combo_iteration_matches_cartesian_order_and_records_counts() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta", "gamma"))
    service = BacktestComboPlanningService(
        config=BacktestComboPlanningConfig(combo_chunk_size=4),
    )
    local_row_pools = {
        "alpha": np.arange(2, dtype=np.int32),
        "beta": np.arange(3, dtype=np.int32),
        "gamma": np.arange(2, dtype=np.int32),
    }

    chunks = list(
        iter_combo_chunks(
            indicator_ids=("alpha", "beta", "gamma"),
            local_row_pools=local_row_pools,
            chunk_size=4,
        )
    )
    result = service.execute(
        prepared_result=prepared,
        normalized_request=_normalized_request(risk_mode="none"),
    )

    assert [chunk.size for chunk in chunks] == [4, 4, 4]
    assert chunks[0].rows_by_indicator["alpha"].tolist() == [0, 0, 0, 0]
    assert chunks[0].rows_by_indicator["beta"].tolist() == [0, 0, 1, 1]
    assert chunks[0].rows_by_indicator["gamma"].tolist() == [0, 1, 0, 1]
    assert chunks[1].rows_by_indicator["alpha"].tolist() == [0, 0, 1, 1]
    assert chunks[1].rows_by_indicator["beta"].tolist() == [2, 2, 0, 0]
    assert chunks[1].rows_by_indicator["gamma"].tolist() == [0, 1, 0, 1]
    assert cartesian_combo_count(
        indicator_ids=("alpha", "beta", "gamma"),
        local_row_pools=local_row_pools,
    ) == 12

    assert result.backend.backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert set(result.telemetry.stage_timings) == {
        BUILD_EXACT_CONTEXT_STAGE_NAME,
        BUILD_PROXY_CONTEXT_STAGE_NAME,
        COMBO_ITERATION_STAGE_NAME,
        PROXY_FILTER_STAGE_NAME,
    }
    assert result.telemetry.cartesian_combinations == 12
    assert result.telemetry.combo_chunks_processed == 3
    assert result.telemetry.exact_candidates_evaluated == 12
    assert result.telemetry.proxy_candidates_seen == 12
    assert result.telemetry.proxy_candidates_valid == 12
    assert result.telemetry.proxy_candidates_selected == 12


def test_build_proxy_context_pass_through_avoids_heavy_arrays() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    service = BacktestComboPlanningService()

    proxy_context = service.build_proxy_context(prepared_result=prepared, fee_rate=0.0)
    chunk = next(
        iter_combo_chunks(
            indicator_ids=prepared.indicator_ids,
            local_row_pools={
                "alpha": np.arange(2, dtype=np.int32),
                "beta": np.arange(3, dtype=np.int32),
            },
            chunk_size=6,
        )
    )
    filter_result = service.proxy_filter(combo_chunk=chunk, proxy_context=proxy_context)

    assert proxy_context.active is False
    assert proxy_context.context_type == "pass_through"
    assert proxy_context.confirm_matrix is None
    assert proxy_context.proxy_matrix is None
    assert proxy_context.eval_stack is None
    assert filter_result.selected_indexes.tolist() == [0, 1, 2, 3, 4, 5]
    assert filter_result.selected_rows_by_indicator["alpha"].tolist() == [0, 0, 0, 1, 1, 1]
    assert filter_result.selected_rows_by_indicator["beta"].tolist() == [0, 1, 2, 0, 1, 2]
    assert filter_result.input_candidate_count == 6
    assert filter_result.valid_candidate_count == 6
    assert filter_result.selected_candidate_count == 6
    assert filter_result.confirm is None
    assert filter_result.proxy is None


def test_active_two_indicator_proxy_filter_uses_matrix_cache_and_top_fraction() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    service = BacktestComboPlanningService(
        config=BacktestComboPlanningConfig(combo_top_frac=0.5, combo_min_confirm=1),
    )

    proxy_context = service.build_proxy_context(prepared_result=prepared, fee_rate=0.0)
    chunk = next(
        iter_combo_chunks(
            indicator_ids=prepared.indicator_ids,
            local_row_pools={
                "alpha": np.arange(2, dtype=np.int32),
                "beta": np.arange(3, dtype=np.int32),
            },
            chunk_size=6,
        )
    )
    filter_result = service.proxy_filter(combo_chunk=chunk, proxy_context=proxy_context)

    assert proxy_context.active is True
    assert proxy_context.context_type == "matrix_two"
    assert proxy_context.confirm_matrix is not None
    assert proxy_context.proxy_matrix is not None
    assert proxy_context.confirm_matrix.shape == (2, 3)
    assert proxy_context.proxy_matrix.shape == (2, 3)
    assert filter_result.selected_indexes.tolist() == [1, 4]
    assert filter_result.selected_rows_by_indicator["alpha"].tolist() == [0, 1]
    assert filter_result.selected_rows_by_indicator["beta"].tolist() == [1, 1]
    assert filter_result.input_candidate_count == 6
    assert filter_result.valid_candidate_count == 4
    assert filter_result.selected_candidate_count == 2
    assert filter_result.confirm is not None
    assert filter_result.proxy is not None
    assert filter_result.confirm.tolist() == [2, 1]
    assert filter_result.proxy.tolist() == pytest.approx([3.0, 2.0])


def test_active_generic_n_proxy_filter_uses_eval_stack_and_min_confirm() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta", "gamma"))
    service = BacktestComboPlanningService(
        config=BacktestComboPlanningConfig(combo_top_frac=1.0, combo_min_confirm=2),
    )

    proxy_context = service.build_proxy_context(prepared_result=prepared, fee_rate=0.0)
    chunk = next(
        iter_combo_chunks(
            indicator_ids=prepared.indicator_ids,
            local_row_pools={
                "alpha": np.arange(2, dtype=np.int32),
                "beta": np.arange(3, dtype=np.int32),
                "gamma": np.arange(2, dtype=np.int32),
            },
            chunk_size=12,
        )
    )
    filter_result = service.proxy_filter(combo_chunk=chunk, proxy_context=proxy_context)

    assert proxy_context.active is True
    assert proxy_context.context_type == "generic_n"
    assert proxy_context.eval_stack is not None
    assert proxy_context.eval_stack.shape == (3, 3, 3)
    assert filter_result.selected_indexes.tolist() == [2, 11]
    assert filter_result.selected_rows_by_indicator["alpha"].tolist() == [0, 1]
    assert filter_result.selected_rows_by_indicator["beta"].tolist() == [1, 2]
    assert filter_result.selected_rows_by_indicator["gamma"].tolist() == [0, 1]
    assert filter_result.input_candidate_count == 12
    assert filter_result.valid_candidate_count == 2
    assert filter_result.selected_candidate_count == 2
    assert filter_result.confirm is not None
    assert filter_result.proxy is not None
    assert filter_result.confirm.tolist() == [2, 2]
    assert filter_result.proxy.tolist() == pytest.approx([3.0, 1.0])


def _prepared_result(*, indicator_ids: Sequence[str]) -> BacktestPreparePoolsResult:
    pools_by_id = {
        "alpha": _pool(
            indicator_id="alpha",
            trade_rows=[
                [1, 1, 0, -1],
                [-1, 0, -1, -1],
            ],
            eval_rows=[
                [1, 1, 0],
                [-1, 0, -1],
            ],
        ),
        "beta": _pool(
            indicator_id="beta",
            trade_rows=[
                [1, 0, 0, 0],
                [1, 1, -1, 0],
                [-1, -1, 0, 0],
            ],
            eval_rows=[
                [1, 0, 0],
                [1, 1, -1],
                [-1, 0, -1],
            ],
        ),
        "gamma": _pool(
            indicator_id="gamma",
            trade_rows=[
                [1, 1, 1, 1],
                [-1, 0, -1, -1],
            ],
            eval_rows=[
                [1, 1, -1],
                [-1, 0, -1],
            ],
        ),
    }
    pools = tuple(pools_by_id[indicator_id] for indicator_id in indicator_ids)
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=tuple(indicator_ids),
        indicator_pools=pools,
        signal_returns_15m=np.asarray([1.0, 2.0, -2.0], dtype=np.float32),
        execution_mapping=PreparedExecutionMapping(
            signal_entry_exec_idx_15m=np.asarray([1, 2, 3, 4], dtype=np.int32),
            run_bar_open_1m_idx_15m=np.asarray([0, 1, 2, 3], dtype=np.uint32),
            run_bar_close_1m_idx_15m=np.asarray([1, 2, 3, 4], dtype=np.uint32),
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
    eval_rows: Sequence[Sequence[int]],
) -> PreparedIndicatorPool:
    trade_T = np.asarray(trade_rows, dtype=np.int8)
    eval_T = np.asarray(eval_rows, dtype=np.int8)
    row_ids = np.arange(trade_T.shape[0], dtype=np.int32)
    segments = build_signal_segments(trade_T)
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
        trade_T=trade_T,
        eval_T=eval_T,
        segments=segments,
        row_score=np.zeros(trade_T.shape[0], dtype=np.float32),
        score_adj=np.zeros(trade_T.shape[0], dtype=np.float32),
        nonzero=np.count_nonzero(eval_T, axis=1).astype(np.int32),
        proxy=np.zeros(trade_T.shape[0], dtype=np.float32),
        change_count=segments.change_count,
        metadata=metadata,
    )


def _normalized_request(
    *,
    risk_mode: str = "none",
    direction_mode: str = "long_short_reversal",
    fee_rate: float = 0.0,
) -> dict[str, Any]:
    return {
        "risk": {"mode": risk_mode},
        "execution": {
            "direction_mode": direction_mode,
            "fee_rate": fee_rate,
        },
    }
