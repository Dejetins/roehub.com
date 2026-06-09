from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import trading.contexts.backtest.application.services.v2.tp_sl_exact as tp_sl_exact_module
from trading.contexts.backtest.application.dto import (
    BacktestComboPlanningResult,
    BacktestComboPlanningTelemetry,
    BacktestPreparePoolsResult,
    BacktestProxyContext,
    BacktestSelectedBackend,
    BacktestTpSlExactConfig,
    BacktestTpSlGridEvidence,
    BacktestTpSlGridResolution,
    BacktestTpSlHitTimesCleanupEvidence,
    BacktestTpSlHitTimesResult,
    BacktestTpSlHitTimesSubset,
    BacktestTpSlHitTimesTiming,
    BacktestTpSlRequestedGrid,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparePoolsTiming,
)
from trading.contexts.backtest.application.services.v2 import (
    EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
    MATRIX_CELL_TP_SL_V1_BACKEND,
    TP_SL_EXACT_SCORED_STATUS,
    TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME,
    TP_SL_EXACT_SCORING_STAGE_NAME,
    TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME,
    TP_SL_HEAP_UPDATE_STAGE_NAME,
    TP_SL_SELF_CHECK_PASSED_STATUS,
    TP_SL_SELF_CHECK_STAGE_NAME,
    BacktestTpSlExactScoringService,
    BacktestTpSlSelfCheckFailed,
    build_segment_stack,
    build_signal_segments,
)
from trading.contexts.backtest.application.services.v2.matrix_backend.tp_sl_cells import (
    SL_WINS_TIE_RULE_LITERAL,
    build_tp_sl_selected_cell_shadow,
)


def test_tp_sl_exact_applies_same_bar_sl_tie_rule() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
        open_1m=[100.0, 100.0, 100.0, 101.0],
        close_1m=[100.0, 100.0, 100.0, 101.0],
    )
    hit_times = _hit_times_result(
        tp_values=(0.10,),
        sl_values=(0.05,),
        long_tp=[[4, 4, 2, 4]],
        long_sl=[[4, 4, 2, 4]],
    )

    result = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_only",
        ),
        hit_times_result=hit_times,
        normalized_request=_normalized_request(direction_mode="long_only", fee_rate=0.0),
    )

    top = result.top_results[0]
    assert top.metrics["total_return_pct"] == pytest.approx(-5.0, abs=1e-5)
    assert top.metrics["best_tp_pct"] == pytest.approx(10.0)
    assert top.metrics["best_sl_pct"] == pytest.approx(5.0)
    assert top.metrics["trade_count"] == 1.0


def test_tp_sl_exact_selects_best_cell_and_records_canonical_top_fields() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
        open_1m=[100.0, 100.0, 100.0, 101.0],
        close_1m=[100.0, 100.0, 100.0, 101.0],
    )
    hit_times = _hit_times_result(
        tp_values=(0.02, 0.10),
        sl_values=(0.02, 0.05),
        long_tp=[
            [4, 4, 2, 4],
            [4, 4, 2, 4],
        ],
        long_sl=[
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
    )

    result = BacktestTpSlExactScoringService(
        config=BacktestTpSlExactConfig(benchmark_top_k=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_only",
        ),
        hit_times_result=hit_times,
        normalized_request=_normalized_request(direction_mode="long_only", fee_rate=0.0),
    )

    top = result.top_results[0]
    assert top.best_tp_idx == 1
    assert top.best_sl_idx == 0
    assert top.metrics["total_return_pct"] == pytest.approx(10.0, abs=1e-5)
    assert result.canonical_top_results_payload() == [
        {
            "best_sl_pct": pytest.approx(2.0),
            "best_tp_pct": pytest.approx(10.0),
            "total_return_pct": pytest.approx(10.0, abs=1e-5),
            "trade_count": 1,
        }
    ]


def test_tp_sl_exact_preserves_direction_mode_semantics() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, -1, 0, 0]]},
        open_1m=[100.0, 100.0, 100.0, 100.0],
        close_1m=[100.0, 100.0, 100.0, 100.0],
    )
    hit_times = _hit_times_result(
        tp_values=(0.10,),
        sl_values=(0.05,),
        long_tp=[[4, 4, 4, 4]],
        long_sl=[[4, 4, 4, 4]],
        short_tp=[[4, 4, 4, 3]],
        short_sl=[[4, 4, 4, 4]],
    )

    long_only = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_only",
        ),
        hit_times_result=hit_times,
        normalized_request=_normalized_request(direction_mode="long_only", fee_rate=0.0),
    )
    long_short = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_short_reversal",
        ),
        hit_times_result=hit_times,
        normalized_request=_normalized_request(
            direction_mode="long_short_reversal",
            fee_rate=0.0,
        ),
    )

    assert long_only.telemetry.sample_metrics is not None
    assert long_short.telemetry.sample_metrics is not None
    assert long_only.telemetry.sample_metrics["trade_count"] == 1.0
    assert long_short.telemetry.sample_metrics["trade_count"] == 2.0
    assert long_short.top_results[0].metrics["total_return_pct"] > (
        long_only.top_results[0].metrics["total_return_pct"]
    )


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
def test_tp_sl_execution_sizing_modes_pass_compiled_self_check(
    sizing: Mapping[str, float | str],
    profit_lock_enabled: bool,
) -> None:
    prepared = _tp_sl_execution_sizing_prepared_result()

    result = BacktestTpSlExactScoringService(
        config=BacktestTpSlExactConfig(run_self_check=True, self_check_sample_size=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_short_reversal",
        ),
        hit_times_result=_no_hit_times_result(),
        normalized_request=_normalized_request(
            sizing=sizing,
            profit_lock_enabled=profit_lock_enabled,
            direction_mode="long_short_reversal",
            initial_cash_quote=1000.0,
            close_on_end=True,
        ),
    )

    assert result.self_check.status == TP_SL_SELF_CHECK_PASSED_STATUS
    assert result.top_results[0].metrics["trade_count"] == 3.0
    assert np.isfinite(result.top_results[0].metrics["total_return_pct"])


def test_tp_sl_fixed_equity_pct_uses_current_equity_after_wins() -> None:
    prepared = _tp_sl_execution_sizing_prepared_result()
    fixed_first_quote = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 500.0},
    )
    equity_pct = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_equity_pct", "equity_pct": 50.0},
    )

    assert equity_pct > fixed_first_quote


def test_tp_sl_min_max_and_available_quote_clamps_are_deterministic() -> None:
    prepared = _tp_sl_execution_sizing_prepared_result()

    min_quote = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={
            "mode": "fixed_equity_pct_min_quote",
            "equity_pct": 5.0,
            "min_quote": 500.0,
        },
    )
    fixed_500 = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 500.0},
    )
    max_quote = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={
            "mode": "fixed_equity_pct_max_quote",
            "equity_pct": 90.0,
            "max_quote": 100.0,
        },
    )
    fixed_100 = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 100.0},
    )
    capped_to_available = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={"mode": "fixed_quote", "quote_amount": 10_000.0},
    )
    all_in = _score_tp_sl_execution_sizing(
        prepared=prepared,
        sizing={"mode": "all_in"},
    )

    assert min_quote == pytest.approx(fixed_500)
    assert max_quote == pytest.approx(fixed_100)
    assert capped_to_available == pytest.approx(all_in)


def test_tp_sl_close_on_end_false_leaves_final_position_open() -> None:
    prepared = _tp_sl_execution_sizing_prepared_result()
    close_true = _execute_tp_sl_execution_sizing(prepared=prepared, close_on_end=True)
    close_false = _execute_tp_sl_execution_sizing(prepared=prepared, close_on_end=False)

    assert close_true.top_results[0].metrics["trade_count"] == 3.0
    assert close_false.top_results[0].metrics["trade_count"] == 2.0
    assert close_true.top_results[0].metrics["total_return_pct"] > (
        close_false.top_results[0].metrics["total_return_pct"]
    )


def test_tp_sl_self_check_passes_and_reports_summary() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha", "beta"),
        trade_rows_by_id={
            "alpha": [[1, 1, 0, 0], [1, -1, 0, 0]],
            "beta": [[1, 1, 0, 0], [1, -1, 0, 0]],
        },
    )

    result = BacktestTpSlExactScoringService(
        config=BacktestTpSlExactConfig(run_self_check=True, self_check_sample_size=2),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        hit_times_result=_hit_times_result(),
        normalized_request=_normalized_request(fee_rate=0.0),
    )

    assert result.self_check.status == TP_SL_SELF_CHECK_PASSED_STATUS
    assert result.self_check.sample_size == 2
    assert result.self_check.trade_count_equal is True
    assert result.self_check.valid_tp_sl_indexes is True
    assert TP_SL_SELF_CHECK_STAGE_NAME in result.telemetry.stage_timings


def test_tp_sl_self_check_fails_fast_on_metric_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
    )
    original_evaluate = tp_sl_exact_module.evaluate_tp_sl_exact_chunk

    def drifted_evaluate(**kwargs: Any) -> None:
        original_evaluate(**kwargs)
        kwargs["buffers"].total_return_pct[0] += 1.0

    monkeypatch.setattr(tp_sl_exact_module, "evaluate_tp_sl_exact_chunk", drifted_evaluate)

    with pytest.raises(BacktestTpSlSelfCheckFailed, match="self-check failed"):
        BacktestTpSlExactScoringService(
            config=BacktestTpSlExactConfig(run_self_check=True, self_check_sample_size=1),
        ).execute(
            prepared_result=prepared,
            combo_planning_result=_combo_planning_result(prepared=prepared),
            hit_times_result=_hit_times_result(
                tp_values=(0.10, 0.20),
                long_tp=((4, 4, 2, 4), (4, 4, 4, 4)),
            ),
            normalized_request=_normalized_request(fee_rate=0.0),
        )


def test_tp_sl_self_check_fails_fast_on_best_cell_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
    )
    original_evaluate = tp_sl_exact_module.evaluate_tp_sl_exact_chunk

    def drifted_evaluate(**kwargs: Any) -> None:
        original_evaluate(**kwargs)
        kwargs["buffers"].best_tp_idx[0] = 1
        kwargs["buffers"].total_return_pct[0] += 1.0

    monkeypatch.setattr(tp_sl_exact_module, "evaluate_tp_sl_exact_chunk", drifted_evaluate)

    with pytest.raises(BacktestTpSlSelfCheckFailed, match="best_cell_equal=False"):
        BacktestTpSlExactScoringService(
            config=BacktestTpSlExactConfig(run_self_check=True, self_check_sample_size=1),
        ).execute(
            prepared_result=prepared,
            combo_planning_result=_combo_planning_result(prepared=prepared),
            hit_times_result=_hit_times_result(
                tp_values=(0.10, 0.20),
                long_tp=((4, 4, 2, 4), (4, 4, 4, 4)),
            ),
            normalized_request=_normalized_request(fee_rate=0.0),
        )


def test_tp_sl_heap_uses_request_top_n_and_not_benchmark_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha", "beta"),
        trade_rows_by_id={
            "alpha": [[1, 1, 0, 0], [1, 1, 0, 0]],
            "beta": [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
        },
    )
    _patch_tp_sl_scores(
        monkeypatch,
        scores=(1.0, 5.0, 5.0, 2.0, 3.0, 4.0),
        best_tp=(0, 0, 1, 0, 0, 0),
        best_sl=(0, 0, 0, 0, 0, 0),
    )

    result = BacktestTpSlExactScoringService(
        config=BacktestTpSlExactConfig(benchmark_top_k=3, default_request_top_n=100),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        hit_times_result=_hit_times_result(tp_values=(0.02, 0.10), sl_values=(0.02,)),
        normalized_request=_normalized_request(top_n=100, fee_rate=0.0),
    )

    assert result.telemetry.request_top_n == 100
    assert result.telemetry.benchmark_top_k == 3
    assert result.telemetry.heap_capacity == 100
    assert result.telemetry.top_results_count == 6
    assert [
        (
            top.score,
            top.metrics["best_tp_pct"],
            dict(top.indicator_rows),
        )
        for top in result.top_results[:3]
    ] == [
        (5.0, pytest.approx(10.0), {"alpha": 0, "beta": 2}),
        (5.0, pytest.approx(2.0), {"alpha": 0, "beta": 1}),
        (4.0, pytest.approx(2.0), {"alpha": 1, "beta": 2}),
    ]


def test_tp_sl_matrix_cell_full_grid_matches_reference_backend() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha", "beta"),
        trade_rows_by_id={
            "alpha": [[1, 1, 0, 0], [1, -1, 0, 0]],
            "beta": [[1, 1, 0, 0], [1, -1, 0, 0]],
        },
        open_1m=[100.0, 100.0, 110.0, 99.0],
        close_1m=[100.0, 100.0, 110.0, 120.0],
    )
    hit_times = _hit_times_result(
        tp_values=(0.02, 0.05, 0.10),
        sl_values=(0.02, 0.05),
        long_tp=[
            [4, 4, 2, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
        long_sl=[
            [4, 4, 4, 4],
            [4, 4, 3, 4],
        ],
        short_tp=[
            [4, 4, 4, 3],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
        short_sl=[
            [4, 4, 4, 4],
            [4, 4, 2, 4],
        ],
    )
    request = _normalized_request(
        direction_mode="long_short_reversal",
        fee_rate=0.0,
        top_n=4,
    )

    reference = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_short_reversal",
        ),
        hit_times_result=hit_times,
        normalized_request=request,
    )
    matrix = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_short_reversal",
            backend_id=MATRIX_CELL_TP_SL_V1_BACKEND,
        ),
        hit_times_result=hit_times,
        normalized_request=request,
    )

    assert matrix.canonical_top_results_payload() == pytest.approx(
        reference.canonical_top_results_payload()
    )
    assert [
        dict(item.indicator_rows) for item in matrix.top_results
    ] == [dict(item.indicator_rows) for item in reference.top_results]
    cell_backend = matrix.telemetry.cell_backend
    assert cell_backend is not None
    assert cell_backend["backend_id"] == MATRIX_CELL_TP_SL_V1_BACKEND
    assert cell_backend["cell_block_shape"] == "16 x 16"
    assert cell_backend["tp_sl_cells"] == 6
    assert cell_backend["sl_wins_tie_rule"] == "SL wins"


def test_tp_sl_quality_gate_filters_final_trade_count_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha", "beta"),
        trade_rows_by_id={
            "alpha": [[1, 1, 0, 0], [1, 1, 0, 0]],
            "beta": [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
        },
    )
    _patch_tp_sl_scores(
        monkeypatch,
        scores=(1.0, 5.0, 5.0, 2.0, 3.0, 4.0),
        trade_counts=(1, 1, 1, 1, 1, 1),
        best_tp=(0, 0, 1, 0, 0, 0),
        best_sl=(0, 0, 0, 0, 0, 0),
    )
    request = _normalized_request(top_n=100, fee_rate=0.0)
    request["quality_constraints"] = {"min_closed_trades": 2}

    result = BacktestTpSlExactScoringService(
        config=BacktestTpSlExactConfig(benchmark_top_k=3, default_request_top_n=100),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        hit_times_result=_hit_times_result(tp_values=(0.02, 0.10), sl_values=(0.02,)),
        normalized_request=request,
    )

    assert result.top_results == ()
    assert result.telemetry.min_closed_trades == 2
    assert result.telemetry.exact_candidates_evaluated == 6
    assert result.telemetry.quality_candidates_below_min_trades == 6
    assert result.telemetry.quality_candidates_heap_eligible == 0


def test_tp_sl_full_metrics_second_pass_is_bounded_and_service_only() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
        open_1m=[100.0, 100.0, 100.0, 100.0],
        close_1m=[100.0, 100.0, 100.0, 100.0],
    )
    hit_times = _hit_times_result(
        tp_values=(0.10,),
        sl_values=(0.05,),
        long_tp=[[4, 4, 2, 4]],
        long_sl=[[4, 4, 4, 4]],
    )

    result = BacktestTpSlExactScoringService(
        config=BacktestTpSlExactConfig(benchmark_top_k=1),
    ).execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_only",
        ),
        hit_times_result=hit_times,
        normalized_request=_normalized_request(direction_mode="long_only", fee_rate=0.0),
    )

    top = result.top_results[0]
    assert result.telemetry.stage_timings[TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME] >= 0.0
    assert result.telemetry.stage_timings[TP_SL_EXACT_SCORING_STAGE_NAME] == pytest.approx(
        result.telemetry.stage_timings[TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME]
    )
    assert top.metrics["max_drawdown_pct"] == pytest.approx(0.0)
    assert top.metrics["profit_factor"] == float("inf")
    assert top.metrics["win_rate_pct"] == pytest.approx(100.0)
    assert top.metrics["avg_trade_ret_pct"] == pytest.approx(10.0, abs=1e-5)
    assert top.metrics["exposure_pct"] == pytest.approx(25.0)


def test_tp_sl_result_does_not_build_iteration_7_identity_fields() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
    )
    result = BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(prepared=prepared),
        hit_times_result=_hit_times_result(),
        normalized_request=_normalized_request(fee_rate=0.0),
    )

    mapping = result.as_mapping()
    serialized_keys = _all_mapping_keys(mapping)
    assert "variant_key" not in serialized_keys
    assert "variant_hash" not in serialized_keys
    assert "indicator_variant_hash" not in serialized_keys
    assert result.memory_cleanup_evidence.result_is_compact is True
    assert result.telemetry.status == TP_SL_EXACT_SCORED_STATUS
    assert TP_SL_HEAP_UPDATE_STAGE_NAME in result.telemetry.stage_timings


def test_tp_sl_selected_cell_shadow_validates_long_and_short_cells() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, -1, 0, 0]]},
        open_1m=[100.0, 100.0, 100.0, 100.0],
        close_1m=[100.0, 100.0, 100.0, 100.0],
    )
    hit_times = _hit_times_result(
        tp_values=(0.02, 0.10),
        sl_values=(0.02, 0.05),
        long_tp=[
            [4, 4, 2, 4],
            [4, 4, 2, 4],
        ],
        long_sl=[
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
        short_tp=[
            [4, 4, 4, 3],
            [4, 4, 4, 3],
        ],
        short_sl=[
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
    )

    validation = build_tp_sl_selected_cell_shadow(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_short_reversal",
        ),
        hit_times_result=hit_times,
        normalized_request=_normalized_request(
            direction_mode="long_short_reversal",
            fee_rate=0.0,
        ),
    )
    mapping = validation.as_mapping()

    assert mapping["status"] == "passed"
    assert mapping["parity_pass"] is True
    assert mapping["tp_count"] == 2
    assert mapping["sl_count"] == 2
    assert mapping["selected_cell_scores"] == 4
    assert mapping["trade_tape"]["long_trade_count"] == 1
    assert mapping["trade_tape"]["short_trade_count"] == 1
    assert mapping["by_entry_layout"]["arrays"]["long_tp_by_entry.u32.npy"][
        "shape"
    ] == [2, 2]
    assert mapping["production_topn_feed"] == "current_path_only"


def test_tp_sl_selected_cell_shadow_records_sl_wins_tie_rule() -> None:
    prepared = _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, 1, 0, 0]]},
        open_1m=[100.0, 100.0, 100.0, 101.0],
        close_1m=[100.0, 100.0, 100.0, 101.0],
    )
    hit_times = _hit_times_result(
        tp_values=(0.10,),
        sl_values=(0.05,),
        long_tp=[[4, 4, 2, 4]],
        long_sl=[[4, 4, 2, 4]],
    )

    validation = build_tp_sl_selected_cell_shadow(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_only",
        ),
        hit_times_result=hit_times,
        normalized_request=_normalized_request(direction_mode="long_only", fee_rate=0.0),
    )

    assert validation.status == "passed"
    assert validation.parity_pass is True
    assert validation.sl_wins_tie_rule == SL_WINS_TIE_RULE_LITERAL


def _prepared_result(
    *,
    indicator_ids: Sequence[str],
    trade_rows_by_id: Mapping[str, Sequence[Sequence[int]]],
    open_1m: Sequence[float] | None = None,
    close_1m: Sequence[float] | None = None,
) -> BacktestPreparePoolsResult:
    pools = tuple(
        _pool(
            indicator_id=indicator_id,
            trade_rows=trade_rows_by_id[indicator_id],
        )
        for indicator_id in indicator_ids
    )
    return BacktestPreparePoolsResult(
        timeframe="15m",
        indicator_ids=tuple(indicator_ids),
        indicator_pools=pools,
        signal_returns_15m=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
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
        execution_open_1m=np.asarray(
            [100.0, 100.0, 100.0, 101.0] if open_1m is None else open_1m,
            dtype=np.float32,
        ),
        execution_close_1m=np.asarray(
            [100.0, 100.0, 100.0, 101.0] if close_1m is None else close_1m,
            dtype=np.float32,
        ),
    )


def _tp_sl_execution_sizing_prepared_result() -> BacktestPreparePoolsResult:
    return _prepared_result(
        indicator_ids=("alpha",),
        trade_rows_by_id={"alpha": [[1, -1, 1, -1]]},
        open_1m=[100.0, 100.0, 110.0, 99.0],
        close_1m=[100.0, 100.0, 110.0, 120.0],
    )


def _pool(
    *,
    indicator_id: str,
    trade_rows: Sequence[Sequence[int]],
) -> PreparedIndicatorPool:
    trade_T = np.asarray(trade_rows, dtype=np.int8)
    eval_T = np.ascontiguousarray(trade_T[:, :3])
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
    direction_mode: str = "long_short_reversal",
    backend_id: str = EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND,
) -> BacktestComboPlanningResult:
    exact_context = build_segment_stack(
        indicator_ids=prepared.indicator_ids,
        indicator_pools=prepared.indicator_pools,
    )
    return BacktestComboPlanningResult(
        backend=BacktestSelectedBackend(
            backend_id=backend_id,
            risk_mode="tp_sl_grid",
            arity=len(prepared.indicator_ids),
            direction_mode=direction_mode,
            requires_exact_context=True,
            role="generic",
        ),
        exact_context=exact_context,
        proxy_context=BacktestProxyContext(
            indicator_ids=prepared.indicator_ids,
            active=False,
            context_type="pass_through",
            combo_top_frac=1.0,
            combo_min_confirm=1,
            fee_penalty_per_confirm=np.float32(0.0),
        ),
        telemetry=BacktestComboPlanningTelemetry(
            stage_timings={
                "build_exact_context": 0.0,
                "build_proxy_context": 0.0,
                "combo_iteration": 0.0,
                "proxy_filter": 0.0,
            },
            cartesian_combinations=1,
            combo_chunks_processed=1,
            exact_candidates_evaluated=1,
            proxy_candidates_seen=1,
            proxy_candidates_valid=1,
            proxy_candidates_selected=1,
        ),
    )


def _hit_times_result(
    *,
    tp_values: Sequence[float] = (0.10,),
    sl_values: Sequence[float] = (0.05,),
    long_tp: Sequence[Sequence[int]] | None = None,
    long_sl: Sequence[Sequence[int]] | None = None,
    short_tp: Sequence[Sequence[int]] | None = None,
    short_sl: Sequence[Sequence[int]] | None = None,
) -> BacktestTpSlHitTimesResult:
    tp_array = np.asarray(tp_values, dtype=np.float32)
    sl_array = np.asarray(sl_values, dtype=np.float32)
    sentinel = 4
    subset = BacktestTpSlHitTimesSubset(
        tp_values=tp_array,
        sl_values=sl_array,
        long_tp=np.asarray(
            long_tp if long_tp is not None else [[4, 4, 2, 4]] * len(tp_values),
            dtype=np.uint32,
        ),
        long_sl=np.asarray(
            long_sl if long_sl is not None else [[4, 4, 4, 4]] * len(sl_values),
            dtype=np.uint32,
        ),
        short_tp=np.asarray(
            short_tp if short_tp is not None else [[4, 4, 4, 3]] * len(tp_values),
            dtype=np.uint32,
        ),
        short_sl=np.asarray(
            short_sl if short_sl is not None else [[4, 4, 4, 4]] * len(sl_values),
            dtype=np.uint32,
        ),
        sentinel_index=sentinel,
    )
    requested = BacktestTpSlRequestedGrid(
        tp_levels_pct=tuple(float(value) * 100.0 for value in tp_values),
        sl_levels_pct=tuple(float(value) * 100.0 for value in sl_values),
    )
    resolution = BacktestTpSlGridResolution(
        requested_grid=requested,
        tp_indexes=np.arange(len(tp_values), dtype=np.int32),
        sl_indexes=np.arange(len(sl_values), dtype=np.int32),
        tp_values=tp_array,
        sl_values=sl_array,
        evidence=BacktestTpSlGridEvidence(
            artifact_path="hit_times/15m",
            timeframe="15m",
            target_grid={"covered_by_artifact": True},
            artifact_grid={"covered_by_artifact": True},
            requested_grid=requested,
            resolved_tp_indexes=tuple(range(len(tp_values))),
            resolved_sl_indexes=tuple(range(len(sl_values))),
        ),
    )
    return BacktestTpSlHitTimesResult(
        hit_times_manifest_hash="a" * 64,
        resolution=resolution,
        hit_times=subset,
        timing=BacktestTpSlHitTimesTiming(wall_time_s=0.0, subsegments={}),
        cleanup_evidence=BacktestTpSlHitTimesCleanupEvidence(
            status="success",
            retained_hit_times_grid_arrays=False,
            retained_hit_times_table_arrays=False,
            retained_materialized_subset=True,
        ),
    )


def _no_hit_times_result() -> BacktestTpSlHitTimesResult:
    return _hit_times_result(
        tp_values=(0.50,),
        sl_values=(0.50,),
        long_tp=[[4, 4, 4, 4]],
        long_sl=[[4, 4, 4, 4]],
        short_tp=[[4, 4, 4, 4]],
        short_sl=[[4, 4, 4, 4]],
    )


def _normalized_request(
    *,
    direction_mode: str = "long_short_reversal",
    fee_rate: float = 0.0,
    initial_cash_quote: float = 10000.0,
    top_n: int = 100,
    sizing: Mapping[str, float | str] | None = None,
    profit_lock_enabled: bool = False,
    close_on_end: bool = True,
) -> dict[str, Any]:
    sizing_payload = dict(sizing or {"mode": "all_in", "quote_amount": 100.0})
    return {
        "top_n": top_n,
        "risk": {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 10.0, "stop_pct": 10.0, "step_pct": 1.0},
            "sl": {"start_pct": 5.0, "stop_pct": 5.0, "step_pct": 1.0},
        },
        "execution": {
            "direction_mode": direction_mode,
            "fee_rate": fee_rate,
            "slippage_rate": 0.0,
            "initial_cash_quote": initial_cash_quote,
            "sizing": sizing_payload,
            "profit_lock": {
                "enabled": profit_lock_enabled,
                "safe_profit_percent": 30.0,
            },
            "close_on_end": close_on_end,
        },
    }


def _execute_tp_sl_execution_sizing(
    *,
    prepared: BacktestPreparePoolsResult,
    sizing: Mapping[str, float | str] | None = None,
    profit_lock_enabled: bool = False,
    close_on_end: bool = True,
) -> Any:
    return BacktestTpSlExactScoringService().execute(
        prepared_result=prepared,
        combo_planning_result=_combo_planning_result(
            prepared=prepared,
            direction_mode="long_short_reversal",
        ),
        hit_times_result=_no_hit_times_result(),
        normalized_request=_normalized_request(
            sizing=sizing or {"mode": "fixed_equity_pct", "equity_pct": 50.0},
            profit_lock_enabled=profit_lock_enabled,
            initial_cash_quote=1000.0,
            close_on_end=close_on_end,
        ),
    )


def _score_tp_sl_execution_sizing(
    *,
    prepared: BacktestPreparePoolsResult,
    sizing: Mapping[str, float | str],
) -> float:
    result = _execute_tp_sl_execution_sizing(prepared=prepared, sizing=sizing)
    return float(result.top_results[0].metrics["total_return_pct"])


def _patch_tp_sl_scores(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scores: Sequence[float],
    trade_counts: Sequence[int] | None = None,
    best_tp: Sequence[int],
    best_sl: Sequence[int],
) -> None:
    def fixed_evaluate(**kwargs: Any) -> None:
        buffers = kwargs["buffers"]
        assert buffers.size == len(scores)
        buffers.total_return_pct[:] = np.asarray(scores, dtype=np.float64)
        buffers.trade_count[:] = np.asarray(
            trade_counts if trade_counts is not None else [1] * len(scores),
            dtype=np.int32,
        )
        buffers.best_tp_idx[:] = np.asarray(best_tp, dtype=np.int32)
        buffers.best_sl_idx[:] = np.asarray(best_sl, dtype=np.int32)

    monkeypatch.setattr(tp_sl_exact_module, "evaluate_tp_sl_exact_chunk", fixed_evaluate)


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_all_mapping_keys(item))
        return keys
    if isinstance(value, (tuple, list)):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_mapping_keys(item))
        return keys
    return set()
