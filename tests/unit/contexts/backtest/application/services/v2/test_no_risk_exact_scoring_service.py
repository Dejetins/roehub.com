from __future__ import annotations

import gc
import weakref
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pytest

import trading.contexts.backtest.application.services.v2.no_risk_exact as no_risk_exact_module
from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningResult,
    BacktestComboPlanningTelemetry,
    BacktestExactContext,
    BacktestNoRiskExactConfig,
    BacktestNoRiskTopResult,
    BacktestPreparePoolsResult,
    BacktestProxyContext,
    BacktestSelectedBackend,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparePoolsTiming,
)
from trading.contexts.backtest.application.services.v2 import (
    EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_NO_RISK_BACKEND,
    NO_RISK_EXACT_BOUNDARY_STAGE_NAME,
    NO_RISK_EXACT_SCORED_STATUS,
    NO_RISK_EXACT_SCORING_STAGE_NAME,
    NO_RISK_METRIC_NAMES,
    NO_RISK_SELF_CHECK_NOT_RUN_STATUS,
    NO_RISK_SELF_CHECK_PASSED_STATUS,
    NO_RISK_SELF_CHECK_STAGE_NAME,
    STREAMING_2_NO_RISK_BACKEND,
    BacktestNoRiskExactRejected,
    BacktestNoRiskExactScoringService,
    BacktestNoRiskSelfCheckFailed,
    build_segment_stack,
    build_signal_segments,
)


def test_no_risk_exact_boundary_rejects_non_none_risk_mode() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    planning = _combo_planning_result(prepared=prepared, risk_mode="tp_sl_grid")

    with pytest.raises(BacktestNoRiskExactRejected, match="risk.mode='none'"):
        BacktestNoRiskExactScoringService().execute(
            prepared_result=prepared,
            combo_planning_result=planning,
            normalized_request=_normalized_request(risk_mode="tp_sl_grid"),
        )


def test_no_risk_exact_boundary_scores_compact_internal_telemetry() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta", "gamma"))
    planning = _combo_planning_result(prepared=prepared)
    service = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=5, default_request_top_n=100),
    )

    result = service.execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=_normalized_request(top_n=100),
    )

    assert result.top_results == ()
    assert result.execution_context.as_mapping() == {
        "timeframe": "15m",
        "execution_timeframe": "1m",
        "time_slice_start_15m": 0,
        "time_slice_stop_15m": 4,
        "trade_T_length": 4,
        "eval_T_length": 3,
        "t_exec_limit_1m": 4,
    }
    assert result.telemetry.request_top_n == 100
    assert result.telemetry.benchmark_top_k == 5
    assert result.telemetry.heap_capacity == 5
    assert result.telemetry.top_results_count == 0
    assert result.telemetry.exact_candidates_evaluated == 12
    assert result.telemetry.risk_mode == "none"
    assert result.telemetry.direction_mode == "long_short_reversal"
    assert result.telemetry.backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert result.telemetry.backend_logical_name == "event_segments_3_no_risk"
    assert result.telemetry.backend_implementation_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert result.telemetry.arity == 3
    assert result.telemetry.status == NO_RISK_EXACT_SCORED_STATUS
    assert set(result.telemetry.stage_timings) == {
        NO_RISK_EXACT_BOUNDARY_STAGE_NAME,
        NO_RISK_EXACT_SCORING_STAGE_NAME,
    }
    assert result.telemetry.metric_names == NO_RISK_METRIC_NAMES
    assert set(result.telemetry.sample_metrics or {}) == set(NO_RISK_METRIC_NAMES)
    assert result.self_check.status == NO_RISK_SELF_CHECK_NOT_RUN_STATUS
    assert result.memory_cleanup_evidence.result_is_compact is True

    mapping = result.as_mapping()
    assert mapping["telemetry"]["request_top_n"] == 100
    assert mapping["telemetry"]["benchmark_top_k"] == 5
    assert mapping["telemetry"]["top_results_count"] == 0
    assert mapping["telemetry"]["backend_logical_name"] == "event_segments_3_no_risk"
    assert mapping["telemetry"]["backend_implementation_id"] == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert set(mapping["telemetry"]["sample_metrics"]) == set(NO_RISK_METRIC_NAMES)
    assert mapping["memory_cleanup_evidence"]["result_is_compact"] is True


def test_no_risk_exact_dispatch_records_specialized_two_backend() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    planning = _combo_planning_result(
        prepared=prepared,
        backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    )

    result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=_normalized_request(),
    )

    assert result.telemetry.backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND
    assert result.telemetry.backend_logical_name == EVENT_SEGMENTS_2_NO_RISK_BACKEND
    assert result.telemetry.backend_implementation_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND
    assert result.telemetry.exact_candidates_evaluated == 6


def test_no_risk_streaming_two_matches_event_segments_two_sample_metrics() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    event_result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
        ),
        normalized_request=_normalized_request(),
    )
    streaming_result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=STREAMING_2_NO_RISK_BACKEND,
        ),
        normalized_request=_normalized_request(),
    )

    assert streaming_result.telemetry.backend_id == STREAMING_2_NO_RISK_BACKEND
    assert streaming_result.telemetry.sample_metrics is not None
    assert event_result.telemetry.sample_metrics is not None
    for metric_name in NO_RISK_METRIC_NAMES:
        assert streaming_result.telemetry.sample_metrics[metric_name] == pytest.approx(
            event_result.telemetry.sample_metrics[metric_name],
        )


def test_no_risk_direction_modes_change_long_only_reversal_semantics() -> None:
    prepared = _direction_semantics_prepared_result()

    long_only = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
            direction_mode="long_only",
        ),
        normalized_request=_normalized_request(direction_mode="long_only"),
    )
    long_short = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
            direction_mode="long_short_reversal",
        ),
        normalized_request=_normalized_request(direction_mode="long_short_reversal"),
    )

    assert long_only.telemetry.sample_metrics is not None
    assert long_short.telemetry.sample_metrics is not None
    assert long_only.telemetry.sample_metrics["trade_count"] == 1.0
    assert long_short.telemetry.sample_metrics["trade_count"] == 2.0


@pytest.mark.parametrize(
    "indicator_ids",
    [
        ("alpha",),
        ("alpha", "beta"),
        ("alpha", "beta", "gamma"),
    ],
)
def test_no_risk_self_check_passes_for_arity_one_two_three(
    indicator_ids: tuple[str, ...],
) -> None:
    prepared = _prepared_result(indicator_ids=indicator_ids)
    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=2),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(),
    )

    assert result.self_check.status == NO_RISK_SELF_CHECK_PASSED_STATUS
    assert result.self_check.sample_size == min(2, result.telemetry.exact_candidates_evaluated)
    assert result.self_check.trade_count_equal is True
    assert result.self_check.max_abs_diff == pytest.approx(0.0)
    assert NO_RISK_SELF_CHECK_STAGE_NAME in result.telemetry.stage_timings


def test_no_risk_self_check_fails_fast_on_metric_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    original_evaluate = no_risk_exact_module.evaluate_no_risk_exact_chunk

    def drifted_evaluate(**kwargs: Any) -> None:
        original_evaluate(**kwargs)
        kwargs["buffers"].total_return_pct[0] += 1.0

    monkeypatch.setattr(no_risk_exact_module, "evaluate_no_risk_exact_chunk", drifted_evaluate)

    with pytest.raises(BacktestNoRiskSelfCheckFailed, match="self-check failed"):
        BacktestNoRiskExactScoringService(
            config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=2),
        ).execute(
            prepared_result=prepared,
            combo_planning_result=_combo_planning_result(prepared=prepared),
            normalized_request=_normalized_request(),
        )


def test_no_risk_top_result_rejects_non_compact_metadata() -> None:
    metadata = cast(
        Mapping[str, Any],
        {"candidate_rows": np.asarray([1, 2, 3], dtype=np.int32)},
    )
    with pytest.raises(TypeError, match="compact scalar"):
        BacktestNoRiskTopResult(
            rank=1,
            score=1.0,
            indicator_rows={"alpha": 0},
            metrics={"total_return_pct": 1.0},
            metadata=metadata,
        )


def test_no_risk_exact_result_does_not_retain_heavy_array_references() -> None:
    result, refs = _execute_and_return_heavy_refs()

    assert result.memory_cleanup_evidence.result_is_compact is True
    assert _contains_ndarray(result.as_mapping()) is False

    gc.collect()
    assert {name: ref() for name, ref in refs.items()} == {
        "trade_T": None,
        "signal_returns_15m": None,
        "exact_context_starts": None,
        "proxy_context_eval_stack": None,
    }


def _execute_and_return_heavy_refs() -> tuple[Any, dict[str, weakref.ReferenceType[np.ndarray]]]:
    prepared = _prepared_result(indicator_ids=("alpha", "beta", "gamma"))
    planning = _combo_planning_result(prepared=prepared, materialize_heavy_contexts=True)
    refs = {
        "trade_T": weakref.ref(prepared.indicator_pools[0].trade_T),
        "signal_returns_15m": weakref.ref(prepared.signal_returns_15m),
        "exact_context_starts": weakref.ref(_required_array(planning.exact_context.starts)),
        "proxy_context_eval_stack": weakref.ref(_required_array(planning.proxy_context.eval_stack)),
    }
    result = BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=planning,
        normalized_request=_normalized_request(),
    )
    return result, refs


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
        execution_open_1m=np.asarray([100.0, 101.0, 103.0, 99.0, 98.0], dtype=np.float32),
        execution_close_1m=np.asarray([100.5, 102.0, 100.0, 98.0, 97.0], dtype=np.float32),
    )


def _direction_semantics_prepared_result() -> BacktestPreparePoolsResult:
    pools = (
        _pool(
            indicator_id="alpha",
            trade_rows=[[1, 1, -1, -1]],
            eval_rows=[[1, 1, -1]],
        ),
        _pool(
            indicator_id="beta",
            trade_rows=[[1, 1, -1, -1]],
            eval_rows=[[1, 1, -1]],
        ),
    )
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=("alpha", "beta"),
        indicator_pools=pools,
        signal_returns_15m=np.asarray([1.0, -2.0, -3.0], dtype=np.float32),
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
        row_metadata_order_hash="b" * 64,
        timing=PreparePoolsTiming(
            stage_name="prepare_pools_core",
            wall_time_s=0.0,
            subsegments={"prepare_pools_core": 0.0},
        ),
        execution_open_1m=np.asarray([100.0, 100.0, 90.0, 80.0], dtype=np.float32),
        execution_close_1m=np.asarray([100.0, 95.0, 85.0, 70.0], dtype=np.float32),
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


def _combo_planning_result(
    *,
    prepared: BacktestPreparePoolsResult,
    risk_mode: str = "none",
    backend_id: str | None = None,
    direction_mode: str = "long_short_reversal",
    materialize_heavy_contexts: bool = False,
) -> BacktestComboPlanningResult:
    arity = len(prepared.indicator_ids)
    resolved_backend_id = backend_id or (
        EVENT_SEGMENTS_2_NO_RISK_BACKEND if arity == 2 else EVENT_SEGMENTS_N_NO_RISK_BACKEND
    )
    requires_exact_context = (
        resolved_backend_id == EVENT_SEGMENTS_N_NO_RISK_BACKEND or materialize_heavy_contexts
    )
    eval_stack = None
    ret_15m = None
    exact_context = build_segment_stack(
        indicator_ids=prepared.indicator_ids,
        indicator_pools=prepared.indicator_pools,
    ) if requires_exact_context else BacktestExactContext(
        indicator_ids=prepared.indicator_ids,
        required=False,
        starts=None,
        ends=None,
        values=None,
        counts=None,
        row_counts=tuple(int(pool.trade_T.shape[0]) for pool in prepared.indicator_pools),
        max_rows=max(int(pool.trade_T.shape[0]) for pool in prepared.indicator_pools),
        max_segments=max(int(pool.segments.starts.shape[1]) for pool in prepared.indicator_pools),
    )
    if materialize_heavy_contexts:
        eval_stack = np.zeros((arity, 3, 3), dtype=np.int8)
        ret_15m = np.asarray([1.0, 2.0, -2.0], dtype=np.float32)
    return BacktestComboPlanningResult(
        backend=BacktestSelectedBackend(
            backend_id=resolved_backend_id,
            risk_mode=risk_mode,
            arity=arity,
            direction_mode=direction_mode,
            requires_exact_context=requires_exact_context,
            role=_backend_role(resolved_backend_id),
        ),
        exact_context=exact_context,
        proxy_context=BacktestProxyContext(
            indicator_ids=prepared.indicator_ids,
            active=materialize_heavy_contexts,
            context_type="generic_n" if materialize_heavy_contexts else "pass_through",
            combo_top_frac=1.0,
            combo_min_confirm=1,
            fee_penalty_per_confirm=np.float32(0.0),
            eval_stack=eval_stack,
            ret_15m=ret_15m,
        ),
        telemetry=BacktestComboPlanningTelemetry(
            stage_timings={"combo_iteration": 0.0},
            cartesian_combinations=12,
            combo_chunks_processed=1,
            exact_candidates_evaluated=12,
            proxy_candidates_seen=12,
            proxy_candidates_valid=12,
            proxy_candidates_selected=12,
        ),
    )


def _normalized_request(
    *,
    risk_mode: str = "none",
    direction_mode: str = "long_short_reversal",
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    top_n: int = 100,
    sizing_mode: str = "all_in",
) -> dict[str, Any]:
    return {
        "top_n": top_n,
        "risk": {"mode": risk_mode},
        "execution": {
            "direction_mode": direction_mode,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": sizing_mode, "fixed_quote": 100.0},
            "profit_lock": {"enabled": False, "safe_profit_percent": 30.0},
            "close_on_end": True,
        },
    }


def _backend_role(backend_id: str) -> str:
    if backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND:
        return "default"
    if backend_id == STREAMING_2_NO_RISK_BACKEND:
        return "fallback"
    return "generic"


def _required_array(value: np.ndarray | None) -> np.ndarray:
    assert value is not None
    return value


def _contains_ndarray(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, Mapping):
        return any(_contains_ndarray(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_ndarray(item) for item in value)
    return False
