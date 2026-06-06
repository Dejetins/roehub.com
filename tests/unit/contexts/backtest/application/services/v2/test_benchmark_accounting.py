from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backtest.run_api_runner_benchmark_parity import (
    _INSTRUMENTATION_COUNTER_FIELDS,
    _instrumentation_summary,
    _merged_instrumentation_counters,
)
from trading.contexts.backtest.application.services.v2.benchmark_accounting import (
    CANONICAL_STAGE_ORDER,
    SERVICE_ONLY_TELEMETRY_FIELDS,
    BenchmarkAccountingError,
    build_benchmark_accounting_record,
    normalize_canonical_timers,
    validate_canonical_benchmark_json,
)

CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)


def test_canonical_json_stage_accounting_normalizes_notebook_aliases() -> None:
    summary = validate_canonical_benchmark_json(CANONICAL_JSON)

    assert summary["run_count"] == 28
    assert summary["request"]["top_n"] == 100
    assert summary["benchmark_top_k"] == 5
    assert summary["sample_warmup_top_k"] == 1
    assert summary["top_results_count_values"] == [5]
    assert summary["heap_capacity"] == 5
    assert summary["canonical_stage_order"] == list(CANONICAL_STAGE_ORDER)
    assert summary["prepare_pools_alias_normalized"] is True
    assert summary["total_alias_is_historical"] is True
    assert summary["service_total_compared_to_canonical"] is False
    assert "prepare_pools" not in summary["stage_presence"]
    assert "total" not in summary["stage_presence"]
    assert "service_total_without_warmup" not in summary["stage_presence"]
    assert "cleanup" not in summary["canonical_stage_order"]
    for service_only_stage in SERVICE_ONLY_TELEMETRY_FIELDS:
        assert service_only_stage not in summary["canonical_stage_order"]


def test_accounting_record_separates_notebook_and_service_totals() -> None:
    record = build_benchmark_accounting_record(
        timers={
            "prepare_pools": 1.0,
            "build_exact_context": 2.0,
            "build_proxy_context": 3.0,
            "combo_iteration": 4.0,
            "proxy_filter": 5.0,
            "self_check": 6.0,
            "exact_scoring": 7.0,
            "heap_update": 8.0,
            "top_result_proxy_fill": 9.0,
            "artifact_context_resolve": 10.0,
            "artifact_array_open": 11.0,
            "request_slice_prepare": 12.0,
            "prepare_pools_total": 13.0,
            "top_result_assembly": 14.0,
            "persist_top_n_io": 15.0,
            "service_total_without_warmup": 100.0,
        },
        risk_mode="none",
        request_top_n=100,
        benchmark_top_k=5,
        sample_warmup_top_k=1,
        top_results_count=5,
        heap_capacity=5,
    )

    assert record["request"]["top_n"] == 100
    assert record["benchmark_top_k"] == 5
    assert record["sample_warmup_top_k"] == 1
    assert record["top_results_count"] == 5
    assert record["heap_capacity"] == 5
    assert record["total_without_warmup"] == pytest.approx(45.0)
    assert record["service_total_without_warmup"] == pytest.approx(100.0)
    assert record["service_total_compared_to_canonical"] is False
    assert "prepare_pools_core" in record["canonical_timers"]
    assert "prepare_pools" not in record["canonical_timers"]
    assert "service_total_without_warmup" not in record["canonical_timers"]
    assert "service_total_without_warmup" in record["service_only_telemetry"]


def test_total_alias_does_not_create_separate_canonical_stage() -> None:
    timers = normalize_canonical_timers(
        {
            "prepare_pools": 1.0,
            "total": 2.0,
            "total_without_warmup": 2.0,
        }
    )

    assert timers == {
        "prepare_pools_core": 1.0,
        "total_without_warmup": 2.0,
    }


def test_accounting_fails_closed_on_unknown_stage() -> None:
    with pytest.raises(BenchmarkAccountingError, match="unknown benchmark stage"):
        normalize_canonical_timers({"prepare_pools": 1.0, "cleanup": 2.0})


def test_accounting_rejects_missing_required_total_target() -> None:
    with pytest.raises(BenchmarkAccountingError, match="missing required"):
        build_benchmark_accounting_record(
            timers={
                "prepare_pools_core": 1.0,
                "build_exact_context": 2.0,
                "build_proxy_context": 3.0,
                "combo_iteration": 4.0,
                "proxy_filter": 5.0,
                "self_check": 6.0,
                "exact_scoring": 7.0,
                "heap_update": 8.0,
                "service_total_without_warmup": 100.0,
            },
            risk_mode="none",
            request_top_n=100,
            benchmark_top_k=5,
            sample_warmup_top_k=1,
            top_results_count=5,
            heap_capacity=5,
        )


def test_stage_01_instrumentation_counters_are_ordered_and_nulls_are_present() -> None:
    counters = _merged_instrumentation_counters(
        [
            {
                "instrumentation_counters": {
                    "trade_cell_evals_per_sec": None,
                    "artifact_load_ms": 12.5,
                    "signals_pack_ms": None,
                    "combo_iteration_ms": 3.0,
                    "proxy_filter_ms": 4.0,
                    "exact_scoring_ms": 5.0,
                    "tp_sl_exact_scoring_ms": None,
                    "top_result_assembly_ms": 6.0,
                    "rows_before_prefilter": None,
                    "rows_after_prefilter": 426,
                    "row_signature_ms": 7.0,
                    "unique_rows_after_dedup": 36,
                    "duplicate_signal_row_ids": {"ma.ema": [2, 3]},
                    "row_signature_collision_count": 0,
                    "consensus_signature_count": 2_176_782_336,
                    "consensus_signature_mode": "upper_bound_unique_row_product",
                    "candidate_upper_bound_after_row_dedup": 2_176_782_336,
                    "combo_count_planned": 1000,
                    "candidates_after_proxy": 50,
                    "exact_candidates": 50,
                    "avg_segments_per_candidate": None,
                    "avg_trades_per_candidate": None,
                    "tp_count": None,
                    "sl_count": None,
                    "tp_sl_cells": 0,
                    "exact_candidates_per_sec": 10.0,
                }
            }
        ]
    )

    assert tuple(counters) == _INSTRUMENTATION_COUNTER_FIELDS
    summary = _instrumentation_summary(
        [{"job_name": "none/arity_6/long_only", "instrumentation_counters": counters}]
    )
    row = summary["rows"][0]
    assert summary["pass"] is True
    assert row["missing_fields"] == []
    assert "signals_pack_ms" in row["null_fields"]
    assert row["counters"]["unique_rows_after_dedup"] == 36
    assert row["counters"]["duplicate_signal_row_ids"] == "{'ma.ema': [2, 3]}"
