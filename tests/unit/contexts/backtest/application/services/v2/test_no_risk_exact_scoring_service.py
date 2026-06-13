from __future__ import annotations

import gc
import json
import math
import weakref
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pytest

import trading.contexts.backtest.application.services.v2.combo_planning as combo_planning_module
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
    canonical_no_risk_top_results_hash,
    canonical_no_risk_top_results_payload,
)
from trading.contexts.backtest.application.services.v2 import (
    COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND,
    EVENT_SEGMENTS_2_NO_RISK_BACKEND,
    EVENT_SEGMENTS_N_NO_RISK_BACKEND,
    MATRIX_BITSET_NO_RISK_V1_BACKEND,
    NO_RISK_EXACT_BOUNDARY_STAGE_NAME,
    NO_RISK_EXACT_SCORED_STATUS,
    NO_RISK_EXACT_SCORING_STAGE_NAME,
    NO_RISK_FULL_METRIC_SECOND_PASS_STAGE_NAME,
    NO_RISK_HEAP_UPDATE_STAGE_NAME,
    NO_RISK_MATRIX_BITSET_PACK_STAGE_NAME,
    NO_RISK_METRIC_NAMES,
    NO_RISK_SELF_CHECK_NOT_RUN_STATUS,
    NO_RISK_SELF_CHECK_PASSED_STATUS,
    NO_RISK_SELF_CHECK_STAGE_NAME,
    NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME,
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

    assert len(result.top_results) == 12
    assert result.top_results[0].rank == 1
    assert result.top_results[0].score == pytest.approx(2.970297029702987)
    assert dict(result.top_results[0].indicator_rows) == {
        "alpha": 1,
        "beta": 2,
        "gamma": 1,
    }
    assert result.top_results[0].metadata["confirm_count"] == 2
    assert result.top_results[0].metadata["proxy_score"] == pytest.approx(1.0)
    assert "_proxy_pending" not in result.top_results[0].metadata
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
    assert result.telemetry.heap_capacity == 100
    assert result.telemetry.top_results_count == 12
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
        NO_RISK_FULL_METRIC_SECOND_PASS_STAGE_NAME,
        NO_RISK_HEAP_UPDATE_STAGE_NAME,
        NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME,
    }
    assert result.telemetry.metric_names == NO_RISK_METRIC_NAMES
    assert set(result.telemetry.sample_metrics or {}) == set(NO_RISK_METRIC_NAMES)
    assert result.telemetry.numba_num_threads is not None
    assert result.telemetry.numba_num_threads > 0
    assert result.telemetry.numba_thread_source
    assert result.self_check.status == NO_RISK_SELF_CHECK_NOT_RUN_STATUS
    assert result.memory_cleanup_evidence.result_is_compact is True
    assert result.memory_cleanup_evidence.cleanup_duration_s is not None
    assert result.memory_cleanup_evidence.cleanup_duration_s >= 0.0

    mapping = result.as_mapping()
    assert mapping["telemetry"]["request_top_n"] == 100
    assert mapping["telemetry"]["benchmark_top_k"] == 5
    assert mapping["telemetry"]["top_results_count"] == 12
    assert len(mapping["top_results"]) == 12
    assert mapping["telemetry"]["backend_logical_name"] == "event_segments_3_no_risk"
    assert mapping["telemetry"]["backend_implementation_id"] == EVENT_SEGMENTS_N_NO_RISK_BACKEND
    assert set(mapping["telemetry"]["sample_metrics"]) == set(NO_RISK_METRIC_NAMES)
    assert mapping["telemetry"]["numba_num_threads"] == result.telemetry.numba_num_threads
    assert mapping["telemetry"]["numba_thread_source"] == result.telemetry.numba_thread_source
    assert mapping["memory_cleanup_evidence"]["result_is_compact"] is True
    assert mapping["memory_cleanup_evidence"]["cleanup_duration_s"] >= 0.0


def test_no_risk_quality_gate_filters_final_trade_count_only() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta", "gamma"))
    request = _normalized_request(top_n=100)
    request["quality_constraints"] = {"min_closed_trades": 2}

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(default_request_top_n=100),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=request,
    )

    assert result.top_results == ()
    assert result.telemetry.min_closed_trades == 2
    assert result.telemetry.exact_candidates_evaluated == 12
    assert result.telemetry.quality_candidates_below_min_trades == 12
    assert result.telemetry.quality_candidates_heap_eligible == 0


def test_no_risk_heap_capacity_uses_request_top_n_not_benchmark_top_k() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta", "gamma"))
    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=1, default_request_top_n=100),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(top_n=4),
    )

    assert result.telemetry.request_top_n == 4
    assert result.telemetry.benchmark_top_k == 1
    assert result.telemetry.heap_capacity == 4
    assert result.telemetry.exact_candidates_evaluated == 12
    assert result.telemetry.top_results_count == 4
    assert len(result.top_results) == 4


def test_no_risk_request_top_n_50_can_return_50_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _many_rows_prepared_result(row_count=8)
    _patch_exact_scores(monkeypatch, scores=tuple(float(index) for index in range(64)))

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=5, default_request_top_n=50),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(top_n=50),
    )

    assert result.telemetry.request_top_n == 50
    assert result.telemetry.benchmark_top_k == 5
    assert result.telemetry.heap_capacity == 50
    assert result.telemetry.exact_candidates_evaluated == 64
    assert result.telemetry.top_results_count == 50
    assert len(result.top_results) == 50


def test_no_risk_large_selected_batches_do_not_enter_legacy_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_product(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("itertools.product must not run for large full jobs")

    monkeypatch.setattr(combo_planning_module.itertools, "product", forbidden_product)
    prepared = _large_prepared_result(arity=5, row_count=196)
    batch = next(
        no_risk_exact_module._iter_selected_candidate_batches(
            prepared_result=prepared,
            combo_planning_result=_combo_planning_result(prepared=prepared),
        )
    )

    assert no_risk_exact_module._selected_size(batch.rows_by_indicator) == 4096
    assert batch.rows_by_indicator["indicator_0"][:3].tolist() == [0, 0, 0]
    assert batch.rows_by_indicator["indicator_4"][:3].tolist() == [0, 1, 2]


def test_no_risk_heap_orders_by_score_then_original_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    _patch_exact_scores(monkeypatch, scores=(1.0, 5.0, 5.0, 2.0, 3.0, 4.0))

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=3),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(top_n=3),
    )

    assert [
        (top_result.score, dict(top_result.indicator_rows))
        for top_result in result.top_results
    ] == [
        (5.0, {"alpha": 0, "beta": 2}),
        (5.0, {"alpha": 0, "beta": 1}),
        (4.0, {"alpha": 1, "beta": 2}),
    ]


def test_no_risk_heap_does_not_materialize_metadata_for_rejected_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    _patch_exact_scores(monkeypatch, scores=(6.0, 5.0, 4.0, 3.0, 2.0, 1.0))
    metadata_calls: list[tuple[str, int]] = []
    original_as_mapping = PreparedIndicatorRowMetadata.as_mapping

    def counted_as_mapping(self: PreparedIndicatorRowMetadata) -> dict[str, Any]:
        metadata_calls.append((self.indicator_id, self.row_id))
        return original_as_mapping(self)

    monkeypatch.setattr(PreparedIndicatorRowMetadata, "as_mapping", counted_as_mapping)

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(top_n=1),
    )

    assert result.telemetry.exact_candidates_evaluated == 6
    assert result.telemetry.top_results_count == 1
    assert metadata_calls == [("alpha", 0), ("beta", 0)]


def test_no_risk_arity_one_heap_materializes_metadata_only_for_retained_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(indicator_ids=("alpha",))
    _patch_exact_scores(monkeypatch, scores=(6.0, 5.0))
    metadata_calls: list[tuple[str, int]] = []
    original_as_mapping = PreparedIndicatorRowMetadata.as_mapping

    def counted_as_mapping(self: PreparedIndicatorRowMetadata) -> dict[str, Any]:
        metadata_calls.append((self.indicator_id, self.row_id))
        return original_as_mapping(self)

    monkeypatch.setattr(PreparedIndicatorRowMetadata, "as_mapping", counted_as_mapping)

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_only",
        ),
        normalized_request=_normalized_request(top_n=1, direction_mode="long_only"),
    )

    assert result.telemetry.request_top_n == 1
    assert result.telemetry.benchmark_top_k == 1
    assert result.telemetry.top_results_count == 1
    assert [
        (top_result.score, dict(top_result.indicator_rows))
        for top_result in result.top_results
    ] == [(6.0, {"alpha": 0})]
    assert metadata_calls == [("alpha", 0)]


def test_no_risk_arity_one_heap_entry_does_not_retain_metric_buffers() -> None:
    prepared = _prepared_result(indicator_ids=("alpha",))
    buffers = no_risk_exact_module._allocate_metric_buffers(1)
    buffers.total_return_pct[:] = 12.5
    buffers.max_drawdown_pct[:] = 1.5
    buffers.return_over_max_drawdown[:] = 8.0
    buffers.profit_factor[:] = 2.0
    buffers.trade_count[:] = 3
    buffers.sharpe_trades[:] = 0.75
    buffers.win_rate_pct[:] = 66.0
    buffers.avg_trade_ret_pct[:] = 4.0
    buffers.avg_trade_exec_bars[:] = 5.0
    buffers.exposure_pct[:] = 42.0
    total_return_ref = weakref.ref(buffers.total_return_pct)

    entry = no_risk_exact_module._materialize_heap_entry_arity1(
        top_k_context=no_risk_exact_module._top_k_context_from_prepared(prepared),
        local_index=0,
        original_row=0,
        score=12.5,
        buffers=buffers,
        result_index=0,
        confirm=None,
        proxy=None,
    )

    assert entry.metric_buffers is None
    assert entry.metric_index == -1
    assert entry.metric_values == pytest.approx(
        (12.5, 1.5, 8.0, 2.0, 3.0, 0.75, 66.0, 4.0, 5.0, 42.0)
    )

    del buffers
    gc.collect()
    assert total_return_ref() is None


def test_top_result_proxy_fill_recomputes_only_final_pending_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    _patch_exact_scores(monkeypatch, scores=(6.0, 5.0, 4.0, 3.0, 2.0, 1.0))
    proxy_calls: list[tuple[tuple[int, ...], ...]] = []

    def counted_proxy_fill(**kwargs: Any) -> tuple[int, float]:
        eval_rows = cast(tuple[np.ndarray, ...], kwargs["eval_rows"])
        proxy_calls.append(tuple(tuple(int(value) for value in row) for row in eval_rows))
        return 77, 12.5

    monkeypatch.setattr(
        no_risk_exact_module,
        "proxy_for_indicator_rows",
        counted_proxy_fill,
    )

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(top_n=1),
    )

    assert result.telemetry.top_results_count == 1
    assert proxy_calls == [((1, 1, 0), (1, 0, 0))]
    assert result.top_results[0].metadata["confirm_count"] == 77
    assert result.top_results[0].metadata["proxy_score"] == pytest.approx(12.5)


def test_proxy_for_indicator_rows_dispatches_arity_two_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fast_path(
        left_eval_row: np.ndarray,
        right_eval_row: np.ndarray,
        ret_15m: np.ndarray,
        min_confirm: np.int32,
        fee_penalty_per_confirm: np.float32,
    ) -> tuple[np.int32, np.float32]:
        calls.append((left_eval_row, right_eval_row))
        assert ret_15m.dtype == np.float32
        assert int(min_confirm) == 2
        assert float(fee_penalty_per_confirm) == pytest.approx(0.25)
        return np.int32(9), np.float32(3.5)

    monkeypatch.setattr(no_risk_exact_module, "proxy_for_two_rows", fast_path)

    confirm_count, proxy_score = no_risk_exact_module.proxy_for_indicator_rows(
        eval_rows=(
            np.asarray([1, -1, 0], dtype=np.int8),
            np.asarray([1, -1, 1], dtype=np.int8),
        ),
        ret_15m=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        min_confirm=2,
        fee_penalty_per_confirm=np.float32(0.25),
    )

    assert confirm_count == 9
    assert proxy_score == pytest.approx(3.5)
    assert len(calls) == 1


def test_proxy_for_two_rows_matches_notebook_scalar_path() -> None:
    confirm_count, proxy_score = no_risk_exact_module.proxy_for_two_rows(
        np.asarray([1, 1, -1, 0], dtype=np.int8),
        np.asarray([1, 0, -1, -1], dtype=np.int8),
        np.asarray([1.0, 2.0, -3.0, 4.0], dtype=np.float32),
        np.int32(1),
        np.float32(0.5),
    )

    assert int(confirm_count) == 2
    assert float(proxy_score) == pytest.approx(3.0)


def test_top_result_metadata_removes_proxy_fill_internal_fields() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(top_n=1),
    )

    metadata = result.top_results[0].metadata
    assert "_local_indices" not in metadata
    assert "_proxy_pending" not in metadata
    assert all(not key.endswith(".local_index") for key in metadata)


def test_no_risk_canonical_top_result_payload_matches_notebook_shape() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(benchmark_top_k=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        normalized_request=_normalized_request(top_n=1),
    )

    payload = result.canonical_top_results_payload()

    assert result.canonical_top_results_hash() == canonical_no_risk_top_results_hash(
        result.top_results
    )
    assert len(payload) == 1
    row = payload[0]
    assert set(row) == {
        *NO_RISK_METRIC_NAMES,
        "confirm_count",
        "proxy_score",
        "alpha",
        "beta",
    }
    assert "rank" not in row
    assert "score" not in row
    assert "indicator_rows" not in row
    assert "metrics" not in row
    assert "metadata" not in row
    assert "_local_indices" not in row
    assert "_proxy_pending" not in row
    assert isinstance(row["trade_count"], int)
    assert isinstance(row["confirm_count"], int)
    assert row["alpha"] == {
        "indicator_id": "alpha",
        "row_id": result.top_results[0].indicator_rows["alpha"],
        "source": "close",
        "window": 5 + result.top_results[0].indicator_rows["alpha"],
    }
    assert row["beta"] == {
        "indicator_id": "beta",
        "row_id": result.top_results[0].indicator_rows["beta"],
        "source": "close",
        "window": 5 + result.top_results[0].indicator_rows["beta"],
    }
    assert all(not key.startswith("alpha.") for key in row)
    assert all(not key.startswith("beta.") for key in row)


def test_no_risk_canonical_hash_matches_arity_one_two_evidence_fixture() -> None:
    fixture_path = Path(
        "docs/architecture/backtest/benchmark_iterations/"
        "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    checked: list[tuple[int, str]] = []

    for run in payload["runs"]:
        if run.get("risk_mode") != "none":
            continue
        arity = len(run["indicator_ids"])
        if arity not in (1, 2):
            continue
        checked.append((arity, run["direction_mode"]))
        canonical_payload = canonical_no_risk_top_results_payload(run["top_results"])
        assert canonical_no_risk_top_results_hash(canonical_payload) == run["result_hash"]
        for row in canonical_payload:
            assert "trade_count" in row
            assert isinstance(row["trade_count"], int)
            assert "_local_indices" not in row
            assert "_proxy_pending" not in row

    assert sorted(checked) == [
        (1, "long_only"),
        (1, "long_short_reversal"),
        (2, "long_only"),
        (2, "long_short_reversal"),
    ]


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


@pytest.mark.parametrize(
    "indicator_ids,current_backend_id",
    [
        (("alpha", "beta"), EVENT_SEGMENTS_2_NO_RISK_BACKEND),
        (("alpha", "beta", "gamma"), EVENT_SEGMENTS_N_NO_RISK_BACKEND),
    ],
)
def test_no_risk_matrix_bitset_mvp_matches_current_long_only_top_results(
    indicator_ids: tuple[str, ...],
    current_backend_id: str,
) -> None:
    prepared = _prepared_result(indicator_ids=indicator_ids)
    request = _normalized_request(direction_mode="long_only", top_n=12)

    current_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=3),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=current_backend_id,
            direction_mode="long_only",
        ),
        normalized_request=request,
    )
    matrix_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=3),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=MATRIX_BITSET_NO_RISK_V1_BACKEND,
            direction_mode="long_only",
        ),
        normalized_request=request,
    )

    assert matrix_result.telemetry.backend_id == MATRIX_BITSET_NO_RISK_V1_BACKEND
    assert matrix_result.telemetry.backend_logical_name == MATRIX_BITSET_NO_RISK_V1_BACKEND
    assert matrix_result.telemetry.backend_implementation_id == MATRIX_BITSET_NO_RISK_V1_BACKEND
    assert NO_RISK_MATRIX_BITSET_PACK_STAGE_NAME in matrix_result.telemetry.stage_timings
    assert matrix_result.telemetry.exact_candidates_evaluated == (
        current_result.telemetry.exact_candidates_evaluated
    )
    assert canonical_no_risk_top_results_payload(matrix_result.top_results) == (
        canonical_no_risk_top_results_payload(current_result.top_results)
    )
    assert canonical_no_risk_top_results_hash(
        canonical_no_risk_top_results_payload(matrix_result.top_results)
    ) == canonical_no_risk_top_results_hash(
        canonical_no_risk_top_results_payload(current_result.top_results)
    )
    assert matrix_result.self_check.status == NO_RISK_SELF_CHECK_PASSED_STATUS


@pytest.mark.parametrize(
    "signal_row",
    [
        pytest.param([1, 1, -1, -1], id="long -> short"),
        pytest.param([-1, -1, 1, 1], id="short -> long"),
        pytest.param([1, 1, 0, 0], id="long -> flat"),
        pytest.param([-1, -1, 0, 0], id="short -> flat"),
        pytest.param([0, 1, 1, 1], id="flat -> long"),
        pytest.param([0, -1, -1, -1], id="flat -> short"),
    ],
)
def test_no_risk_matrix_bitset_stage_05_matches_current_reversal_transitions(
    signal_row: Sequence[int],
) -> None:
    prepared = _single_signal_prepared_result(signal_row=signal_row, arity=2)
    request = _normalized_request(direction_mode="long_short_reversal", top_n=1)

    current_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
            direction_mode="long_short_reversal",
        ),
        normalized_request=request,
    )
    matrix_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=MATRIX_BITSET_NO_RISK_V1_BACKEND,
            direction_mode="long_short_reversal",
        ),
        normalized_request=request,
    )

    assert matrix_result.self_check.status == NO_RISK_SELF_CHECK_PASSED_STATUS
    assert canonical_no_risk_top_results_payload(matrix_result.top_results) == (
        canonical_no_risk_top_results_payload(current_result.top_results)
    )


def test_no_risk_matrix_bitset_stage_05_matches_current_arity6_reversal_top_results() -> None:
    indicator_ids = tuple(f"indicator_{index}" for index in range(6))
    prepared = _single_signal_prepared_result(
        signal_row=[1, 1, -1, -1, 0, 1, 1, -1],
        arity=6,
        indicator_ids=indicator_ids,
    )
    request = _normalized_request(direction_mode="long_short_reversal", top_n=1)

    current_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_N_NO_RISK_BACKEND,
            direction_mode="long_short_reversal",
        ),
        normalized_request=request,
    )
    matrix_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=MATRIX_BITSET_NO_RISK_V1_BACKEND,
            direction_mode="long_short_reversal",
        ),
        normalized_request=request,
    )

    assert matrix_result.telemetry.backend_id == MATRIX_BITSET_NO_RISK_V1_BACKEND
    assert matrix_result.self_check.status == NO_RISK_SELF_CHECK_PASSED_STATUS
    assert canonical_no_risk_top_results_payload(matrix_result.top_results) == (
        canonical_no_risk_top_results_payload(current_result.top_results)
    )


def test_no_risk_matrix_bitset_stage_05_matches_current_arity6_long_only_top_results() -> None:
    indicator_ids = tuple(f"indicator_{index}" for index in range(6))
    prepared = _single_signal_prepared_result(
        signal_row=[1, 1, 0, 1, 1, 0, 1, 1],
        arity=6,
        indicator_ids=indicator_ids,
    )
    request = _normalized_request(direction_mode="long_only", top_n=1)

    current_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_N_NO_RISK_BACKEND,
            direction_mode="long_only",
        ),
        normalized_request=request,
    )
    matrix_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=MATRIX_BITSET_NO_RISK_V1_BACKEND,
            direction_mode="long_only",
        ),
        normalized_request=request,
    )

    assert matrix_result.self_check.status == NO_RISK_SELF_CHECK_PASSED_STATUS
    assert canonical_no_risk_top_results_payload(matrix_result.top_results) == (
        canonical_no_risk_top_results_payload(current_result.top_results)
    )


def test_no_risk_compiled_prefix_stage_12_matches_current_arity7_top_results() -> None:
    indicator_ids = tuple(f"indicator_{index}" for index in range(7))
    prepared = _single_signal_prepared_result(
        signal_row=[1, 1, -1, -1, 0, 1, 1, -1],
        arity=7,
        indicator_ids=indicator_ids,
    )
    request = _normalized_request(direction_mode="long_short_reversal", top_n=1)
    request["quality_constraints"] = {"min_closed_trades": 2}

    current_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_N_NO_RISK_BACKEND,
            direction_mode="long_short_reversal",
        ),
        normalized_request=request,
    )
    compiled_result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND,
            direction_mode="long_short_reversal",
        ),
        normalized_request=request,
    )

    prefix = compiled_result.telemetry.prefix_traversal
    assert prefix is not None
    assert prefix["prefix_candidates_selected"] == 1
    assert prefix["selectivity_order"] == list(range(7))
    assert compiled_result.telemetry.backend_id == COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND
    assert compiled_result.self_check.status == NO_RISK_SELF_CHECK_PASSED_STATUS
    assert canonical_no_risk_top_results_payload(compiled_result.top_results) == (
        canonical_no_risk_top_results_payload(current_result.top_results)
    )


def test_no_risk_matrix_bitset_mvp_is_requestable_but_not_default() -> None:
    prepared = _prepared_result(indicator_ids=("alpha", "beta"))
    request = _normalized_request(direction_mode="long_only")

    default_planning = combo_planning_module.BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    matrix_planning = combo_planning_module.BacktestComboPlanningService().execute(
        prepared_result=prepared,
        normalized_request=request,
        requested_backend_id=MATRIX_BITSET_NO_RISK_V1_BACKEND,
    )

    assert default_planning.backend.backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND
    assert matrix_planning.backend.backend_id == MATRIX_BITSET_NO_RISK_V1_BACKEND
    assert matrix_planning.backend.requires_exact_context is False
    assert matrix_planning.backend.role == "matrix_mvp"


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
    "sizing",
    [
        {"mode": "all_in"},
        {"mode": "fixed_quote", "quote_amount": 100.0},
        {"mode": "fixed_equity_pct", "equity_pct": 25.0},
        {
            "mode": "fixed_equity_pct_min_quote",
            "equity_pct": 5.0,
            "min_quote": 100.0,
        },
        {
            "mode": "fixed_equity_pct_max_quote",
            "equity_pct": 50.0,
            "max_quote": 100.0,
        },
    ],
)
@pytest.mark.parametrize("profit_lock_enabled", [False, True])
def test_no_risk_execution_sizing_modes_pass_compiled_self_check(
    sizing: Mapping[str, float | str],
    profit_lock_enabled: bool,
) -> None:
    prepared = _execution_sizing_prepared_result()

    result = BacktestNoRiskExactScoringService(
        config=BacktestNoRiskExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
        ),
        normalized_request=_normalized_request(
            sizing=sizing,
            profit_lock_enabled=profit_lock_enabled,
            direction_mode="long_short_reversal",
            initial_cash_quote=1000.0,
            close_on_end=True,
        ),
    )

    assert result.self_check.status == NO_RISK_SELF_CHECK_PASSED_STATUS
    assert result.top_results[0].metrics["trade_count"] == 3.0
    assert math.isfinite(result.top_results[0].metrics["total_return_pct"])


def test_no_risk_fixed_equity_pct_uses_current_equity_after_wins() -> None:
    prepared = _execution_sizing_prepared_result()
    fixed_first_quote = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 500.0},
    )
    equity_pct = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_equity_pct", "equity_pct": 50.0},
    )

    assert equity_pct > fixed_first_quote


def test_no_risk_min_max_and_available_quote_clamps_are_deterministic() -> None:
    prepared = _execution_sizing_prepared_result()

    min_quote = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={
            "mode": "fixed_equity_pct_min_quote",
            "equity_pct": 5.0,
            "min_quote": 500.0,
        },
    )
    fixed_500 = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 500.0},
    )
    max_quote = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={
            "mode": "fixed_equity_pct_max_quote",
            "equity_pct": 90.0,
            "max_quote": 100.0,
        },
    )
    fixed_100 = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 100.0},
    )
    capped_to_available = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 10_000.0},
    )
    all_in = _score_no_risk_execution_sizing(
        prepared=prepared,
        sizing={"mode": "all_in"},
    )

    assert min_quote == pytest.approx(fixed_500)
    assert max_quote == pytest.approx(fixed_100)
    assert capped_to_available == pytest.approx(all_in)


def test_no_risk_close_on_end_false_leaves_final_position_open() -> None:
    prepared = _execution_sizing_prepared_result()
    close_true = _execute_no_risk_execution_sizing(prepared=prepared, close_on_end=True)
    close_false = _execute_no_risk_execution_sizing(prepared=prepared, close_on_end=False)

    assert close_true.top_results[0].metrics["trade_count"] == 3.0
    assert close_false.top_results[0].metrics["trade_count"] == 2.0
    assert close_true.top_results[0].metrics["total_return_pct"] > (
        close_false.top_results[0].metrics["total_return_pct"]
    )


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


def _many_rows_prepared_result(*, row_count: int) -> BacktestPreparePoolsResult:
    return _prepared_from_pools(
        indicator_ids=("alpha", "beta"),
        pools=(
            _pool(
                indicator_id="alpha",
                trade_rows=[[1, 0, -1, 1] for _ in range(row_count)],
                eval_rows=[[1, 0, -1] for _ in range(row_count)],
            ),
            _pool(
                indicator_id="beta",
                trade_rows=[[1, 0, -1, 1] for _ in range(row_count)],
                eval_rows=[[1, 0, -1] for _ in range(row_count)],
            ),
        ),
        row_metadata_order_hash="m" * 64,
    )


def _large_prepared_result(*, arity: int, row_count: int) -> BacktestPreparePoolsResult:
    indicator_ids = tuple(f"indicator_{index}" for index in range(arity))
    return _prepared_from_pools(
        indicator_ids=indicator_ids,
        pools=tuple(
            _pool(
                indicator_id=indicator_id,
                trade_rows=[[1] for _ in range(row_count)],
                eval_rows=[[1] for _ in range(row_count)],
            )
            for indicator_id in indicator_ids
        ),
        row_metadata_order_hash="l" * 64,
    )


def _prepared_from_pools(
    *,
    indicator_ids: Sequence[str],
    pools: Sequence[PreparedIndicatorPool],
    row_metadata_order_hash: str,
) -> BacktestPreparePoolsResult:
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=tuple(indicator_ids),
        indicator_pools=tuple(pools),
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
        row_metadata_order_hash=row_metadata_order_hash,
        timing=PreparePoolsTiming(
            stage_name="prepare_pools_core",
            wall_time_s=0.0,
            subsegments={"prepare_pools_core": 0.0},
        ),
        execution_open_1m=np.asarray([100.0, 101.0, 103.0, 99.0, 98.0], dtype=np.float32),
        execution_close_1m=np.asarray([100.5, 102.0, 100.0, 98.0, 97.0], dtype=np.float32),
    )


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


def _execution_sizing_prepared_result() -> BacktestPreparePoolsResult:
    pools = (
        _pool(
            indicator_id="alpha",
            trade_rows=[[1, -1, 1, -1]],
            eval_rows=[[1, -1, 1]],
        ),
        _pool(
            indicator_id="beta",
            trade_rows=[[1, -1, 1, -1]],
            eval_rows=[[1, -1, 1]],
        ),
    )
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=("alpha", "beta"),
        indicator_pools=pools,
        signal_returns_15m=np.asarray([10.0, -10.0, 21.0], dtype=np.float32),
        execution_mapping=PreparedExecutionMapping(
            signal_entry_exec_idx_15m=np.asarray([1, 2, 3, 4], dtype=np.int32),
            run_bar_open_1m_idx_15m=np.asarray([0, 1, 2, 3], dtype=np.uint32),
            run_bar_close_1m_idx_15m=np.asarray([0, 1, 2, 3], dtype=np.uint32),
            t_exec_limit_1m=4,
        ),
        time_slice_start_15m=0,
        time_slice_stop_15m=4,
        trade_T_length=4,
        eval_T_length=3,
        row_metadata_order_hash="c" * 64,
        timing=PreparePoolsTiming(
            stage_name="prepare_pools_core",
            wall_time_s=0.0,
            subsegments={"prepare_pools_core": 0.0},
        ),
        execution_open_1m=np.asarray([100.0, 100.0, 110.0, 99.0], dtype=np.float32),
        execution_close_1m=np.asarray([100.0, 100.0, 110.0, 120.0], dtype=np.float32),
    )


def _single_signal_prepared_result(
    *,
    signal_row: Sequence[int],
    arity: int,
    indicator_ids: Sequence[str] | None = None,
) -> BacktestPreparePoolsResult:
    resolved_indicator_ids = tuple(
        indicator_ids or tuple(f"indicator_{index}" for index in range(arity))
    )
    assert len(resolved_indicator_ids) == arity
    pools = tuple(
        _pool(
            indicator_id=indicator_id,
            trade_rows=[signal_row],
            eval_rows=[signal_row[:-1]],
        )
        for indicator_id in resolved_indicator_ids
    )
    signal_length = len(signal_row)
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=resolved_indicator_ids,
        indicator_pools=pools,
        signal_returns_15m=np.zeros(max(signal_length - 1, 1), dtype=np.float32),
        execution_mapping=PreparedExecutionMapping(
            signal_entry_exec_idx_15m=np.arange(signal_length, dtype=np.int32),
            run_bar_open_1m_idx_15m=np.arange(signal_length, dtype=np.uint32),
            run_bar_close_1m_idx_15m=np.arange(signal_length, dtype=np.uint32),
            t_exec_limit_1m=signal_length,
        ),
        time_slice_start_15m=0,
        time_slice_stop_15m=signal_length,
        trade_T_length=signal_length,
        eval_T_length=max(signal_length - 1, 0),
        row_metadata_order_hash="d" * 64,
        timing=PreparePoolsTiming(
            stage_name="prepare_pools_core",
            wall_time_s=0.0,
            subsegments={"prepare_pools_core": 0.0},
        ),
        execution_open_1m=np.linspace(
            100.0,
            100.0 + float(signal_length - 1),
            signal_length,
            dtype=np.float32,
        ),
        execution_close_1m=np.linspace(
            100.5,
            100.5 + float(signal_length - 1),
            signal_length,
            dtype=np.float32,
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
    initial_cash_quote: float = 10000.0,
    top_n: int = 100,
    sizing_mode: str = "all_in",
    sizing: Mapping[str, float | str] | None = None,
    profit_lock_enabled: bool = False,
    close_on_end: bool = True,
) -> dict[str, Any]:
    sizing_payload = dict(sizing or {"mode": sizing_mode, "quote_amount": 100.0})
    return {
        "top_n": top_n,
        "risk": {"mode": risk_mode},
        "execution": {
            "direction_mode": direction_mode,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "initial_cash_quote": initial_cash_quote,
            "sizing": sizing_payload,
            "profit_lock": {
                "enabled": profit_lock_enabled,
                "safe_profit_percent": 30.0,
            },
            "close_on_end": close_on_end,
        },
    }


def _execute_no_risk_execution_sizing(
    *,
    prepared: BacktestPreparePoolsResult,
    sizing: Mapping[str, float | str] | None = None,
    profit_lock_enabled: bool = False,
    close_on_end: bool = True,
) -> Any:
    return BacktestNoRiskExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            backend_id=EVENT_SEGMENTS_2_NO_RISK_BACKEND,
        ),
        normalized_request=_normalized_request(
            sizing=sizing or {"mode": "fixed_equity_pct", "equity_pct": 50.0},
            profit_lock_enabled=profit_lock_enabled,
            initial_cash_quote=1000.0,
            close_on_end=close_on_end,
        ),
    )


def _score_no_risk_execution_sizing(
    *,
    prepared: BacktestPreparePoolsResult,
    sizing: Mapping[str, float | str],
) -> float:
    result = _execute_no_risk_execution_sizing(prepared=prepared, sizing=sizing)
    return float(result.top_results[0].metrics["total_return_pct"])


def _patch_exact_scores(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scores: Sequence[float],
) -> None:
    def fixed_evaluate(**kwargs: Any) -> None:
        buffers = kwargs["buffers"]
        prepared = cast(BacktestPreparePoolsResult, kwargs["prepared_result"])
        selected_rows_by_indicator = cast(
            Mapping[str, np.ndarray],
            kwargs["selected_rows_by_indicator"],
        )
        pool_sizes = tuple(
            int(
                prepared.indicator_pools[
                    prepared.indicator_ids.index(indicator_id)
                ].trade_T.shape[0]
            )
            for indicator_id in prepared.indicator_ids
        )
        selected_scores = np.empty(buffers.size, dtype=np.float64)
        for row_idx in range(buffers.size):
            ordinal = 0
            for indicator_pos, indicator_id in enumerate(prepared.indicator_ids):
                ordinal *= pool_sizes[indicator_pos]
                ordinal += int(selected_rows_by_indicator[indicator_id][row_idx])
            selected_scores[row_idx] = float(scores[ordinal])
        buffers.total_return_pct[:] = selected_scores
        buffers.max_drawdown_pct[:] = 0.0
        buffers.return_over_max_drawdown[:] = 0.0
        buffers.profit_factor[:] = 0.0
        buffers.trade_count[:] = 1
        buffers.sharpe_trades[:] = 0.0
        buffers.win_rate_pct[:] = 0.0
        buffers.avg_trade_ret_pct[:] = 0.0
        buffers.avg_trade_exec_bars[:] = 0.0
        buffers.exposure_pct[:] = 0.0

    monkeypatch.setattr(no_risk_exact_module, "evaluate_no_risk_exact_chunk", fixed_evaluate)


def _backend_role(backend_id: str) -> str:
    if backend_id == EVENT_SEGMENTS_2_NO_RISK_BACKEND:
        return "default"
    if backend_id == STREAMING_2_NO_RISK_BACKEND:
        return "fallback"
    if backend_id == MATRIX_BITSET_NO_RISK_V1_BACKEND:
        return "matrix_mvp"
    if backend_id == COMPILED_PREFIX_PRODUCT_TRAVERSAL_V1_BACKEND:
        return "compiled_prefix_traversal"
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
