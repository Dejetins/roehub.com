from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from apps.worker.backtest_job_runner.wiring.modules.child_ipc import child_success_to_mapping
from apps.worker.backtest_job_runner.wiring.modules.child_process import _write_result_evidence
from scripts.backtest import run_api_runner_benchmark_parity as runner
from scripts.backtest.full_result_parity import CONTEXT_KEYS, assess_full_parity, metric_equal
from trading.contexts.backtest.application.dto import BacktestNoRiskTopResult, BacktestTpSlTopResult
from trading.contexts.backtest.application.dto.backtest_jobs import build_top_variant_read_model
from trading.contexts.backtest.application.services.v2.top_result_assembly import (
    BacktestTopResultAssemblyService,
)

CONTEXT = {key: f"fixture-{key}-v1" for key in CONTEXT_KEYS}


def fixture(top_n=50, count=50, direction="desc", tp_sl=False):
    # Independent prescribed rows: tied scores, then indicator ordinal ascending.
    # This is an oracle for comparison/transport, not a financial-engine measurement.
    order = sorted(range(count), key=lambda i: (i // 2 if direction == "asc" else -(i // 2), i))
    expected = []
    source = []
    request = {
        "risk": {"mode": "tp_sl_grid" if tp_sl else "none"},
        "execution": {"funding": {"mode": "off"}},
        "ranking": {"primary_metric": "total_return_pct", "direction": direction},
    }
    for rank, i in enumerate(order, 1):
        params = {
            "schema_version": 1,
            "indicators": [
                {"indicator_id": "alpha", "row_id": i, "source": "close", "window": i + 1}
            ],
            **request,
        }
        metrics = {"total_return_pct": float(i // 2), "trade_count": 2.0, "profit_factor": None}
        if tp_sl:
            params["risk"] = {"mode": "tp_sl_grid", "best_tp_pct": 2.0, "best_sl_pct": 1.0}
            metrics.update(best_tp_pct=2.0, best_sl_pct=1.0)
        expected.append(
            {
                "rank": rank,
                "variant_hash": hashlib.sha256(
                    json.dumps(
                        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                    ).encode()
                ).hexdigest(),
                "canonical_variant_params": params,
                "summary_metrics": metrics,
                "best_tp_pct": 2.0 if tp_sl else None,
                "best_sl_pct": 1.0 if tp_sl else None,
            }
        )
        source.append(
            BacktestNoRiskTopResult(
                rank=rank,
                score=float(i // 2),
                indicator_rows={"alpha": i},
                metrics={
                    "total_return_pct": float(i // 2),
                    "trade_count": 2,
                    "profit_factor": float("inf"),
                },
                metadata={"alpha.source": "close", "alpha.window": i + 1},
            )
        )
    if tp_sl:
        source = [
            BacktestTpSlTopResult(
                rank=row.rank,
                score=float(row.score),
                indicator_rows=row.indicator_rows,
                best_tp_idx=0,
                best_sl_idx=0,
                metadata=row.metadata,
                metrics={**row.metrics, "best_tp_pct": 2.0, "best_sl_pct": 1.0},
            )
            for row in source
        ]
    reference = {
        "schema": "backtest_full_top_reference_v1",
        "complete": True,
        "requested_top_n": top_n,
        "expected_count": count,
        "context": CONTEXT,
        "provenance": {
            "method": "independent_expectations",
            "source": __file__,
            "revision": "fixture-v1",
        },
        "items": expected,
    }
    assembly = BacktestTopResultAssemblyService().assemble(
        job_id=uuid4(), normalized_request=request, top_results=source, updated_at=datetime.now(UTC)
    )
    api: dict[str, Any] = {
        "items": [
            build_top_variant_read_model(job_id=str(row.job_id), row=row).as_mapping()
            for row in assembly.top_variants
        ]
    }
    return api, reference, assembly


def assess(api, ref, **kwargs):
    return assess_full_parity(
        api_top=api,
        reference=ref,
        context=CONTEXT,
        requested_top_n=ref["requested_top_n"],
        available_count=ref["expected_count"],
        **kwargs,
    )


@pytest.mark.parametrize("top_n,count", [(1, 1), (50, 50), (50, 8), (50, 0), (7, 7)])
@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_complete_rows_and_valid_empty(top_n, count, direction):
    api, ref, _ = fixture(top_n, count, direction)
    result = assess(api, ref)
    assert result["status"] == "passed"
    assert result["complete"] and result["compared_count"] == count


@pytest.mark.parametrize(
    "damage",
    [
        "tail_metric",
        "identity",
        "tie_order",
        "missing",
        "duplicate",
        "extra",
        "cell",
        "missing_metric",
        "rank",
    ],
)
def test_full_top_50_negative_controls(damage):
    api, ref, _ = fixture()
    rows = api["items"]
    if damage == "tail_metric":
        rows[49]["summary_metrics"]["total_return_pct"] += 1
    elif damage == "identity":
        rows[30]["canonical_variant_params"]["indicators"][0]["row_id"] = 999
    elif damage == "tie_order":
        rows[28], rows[29] = rows[29], rows[28]
    elif damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[30] = copy.deepcopy(rows[29])
    elif damage == "extra":
        rows.append(copy.deepcopy(rows[0]))
    elif damage == "cell":
        rows[30]["best_tp_pct"] = 5.0
    elif damage == "missing_metric":
        del rows[30]["summary_metrics"]["trade_count"]
    elif damage == "rank":
        rows[30]["rank"] = 999
    assert assess(api, ref)["status"] == "failed"


@pytest.mark.parametrize(
    "gap",
    [
        "reference_tail",
        "pagination",
        "context",
        "provenance",
        "reference_metric",
        "legacy",
        "unknown_count",
    ],
)
def test_unavailable_prerequisites_are_not_assessed(gap):
    api, ref, _ = fixture()
    if gap == "reference_tail":
        ref["items"] = ref["items"][:5]
    elif gap == "pagination":
        api["next_cursor"] = "next-page"
    elif gap == "context":
        ref["context"] = {**CONTEXT, "request_hash": "different-quality-or-ranking"}
    elif gap == "provenance":
        del ref["provenance"]
    elif gap == "reference_metric":
        del ref["items"][20]["summary_metrics"]["total_return_pct"]
    elif gap == "legacy":
        del ref["schema"]
    else:
        del ref["expected_count"]
    assert (
        assess_full_parity(
            api_top=api, reference=ref, context=CONTEXT, requested_top_n=50, available_count=50
        )["status"]
        == "not_assessed"
    )


def test_transport_producer_to_report_and_acceptance(tmp_path):
    api, ref, assembly = fixture()
    diagnostics: dict[str, Any] = {
        "telemetry": {
            "risk_mode": "none",
            "arity": 1,
            "direction_mode": "long_only",
            "exact_candidates_evaluated": 50,
            "request_top_n": 50,
            "benchmark_top_k": 5,
            "top_results_count": 50,
        }
    }
    diagnostics["top_results_sample"] = [
        dict(row.payload_json["source_top_result"]) for row in assembly.top_variants[:5]
    ]
    payload = child_success_to_mapping(
        result=SimpleNamespace(
            top_variants=assembly.top_variants,
            stage_timings={},
            summary_hash=assembly.summary_hash,
            cleanup_evidence={},
            exact_diagnostics=diagnostics,
            instrumentation_counters={},
        )
    )
    env = {
        "ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR": str(tmp_path),
        "ROEHUB_BACKTEST_BENCHMARK_FULL_TOP": "1",
    }
    _write_result_evidence(env=env, job_id=uuid4(), payload=payload, process_evidence={})
    evidence = [json.loads(next(tmp_path.glob("*.json")).read_text())]
    assert evidence[0]["benchmark_full_top"]["count"] == 50
    reference_run = {**diagnostics["telemetry"], "full_top_reference": ref}
    parity = runner._compare_reference_results(
        api_top=api,
        child_evidence=evidence,
        reference_run=reference_run,
        comparison_context=CONTEXT,
    )
    assert parity["status"] == "passed"
    assert parity["transport_consistency"]["compared_count"] == 50
    jobs = [{"job_name": "fixture", "parity": parity}]
    report: dict[str, Any] = {
        key: {"pass": True}
        for key in (
            "api_runner_path",
            "performance",
            "instrumentation",
            "memory_release",
            "mixed_scheduler_smoke",
            "lazy_cache_hit_memory",
            "legacy_path_absence",
            "dead_code_audit",
            "docs_drift_audit",
        )
    }
    report["parity"] = runner._parity_summary(jobs)
    assert runner._exit_code(report) == 0
    # A complete transport match alone cannot bless the legacy sample reference.
    del reference_run["full_top_reference"]
    parity = runner._compare_reference_results(
        api_top=api,
        child_evidence=evidence,
        reference_run=reference_run,
        comparison_context=CONTEXT,
    )
    assert parity["status"] == "not_assessed"
    jobs[0]["parity"] = parity
    report["parity"] = runner._parity_summary(jobs)
    report["benchmark_jobs"] = jobs
    assert runner._exit_code(report) == 1
    assert runner._exit_code(report, diagnostic_only=True) == 0
    assert "not_assessed" in runner._render_json(report)
    assert "not_assessed" in runner._render_summary(payload=report)
    assert report["parity"]["pass"] is False


@pytest.mark.parametrize("shadow", [False, True])
def test_quality_and_shadow_cannot_bypass_missing_oracle(shadow):
    result = runner._compare_reference_results(
        api_top={"items": []},
        reference_run={},
        child_evidence=[
            {
                "exact_diagnostics": {
                    "telemetry": {
                        "min_closed_trades": 1,
                        "top_results_count": 0,
                        "request_top_n": 50,
                        "quality_candidates_heap_eligible": 0,
                    }
                }
            }
        ],
        stage_08_tp_sl_selected_cells=shadow,
    )
    assert result["full_reference"]["status"] == "not_assessed"
    assert result["pass"] is False


def test_nonfinite_normalization_and_fixed_tolerance():
    assert metric_equal(float("inf"), None)
    assert metric_equal(float("nan"), None)
    assert metric_equal(1.0, 1.0 + 1e-6)
    assert not metric_equal(1.0, 1.0 + 1e-4)
    assert not metric_equal(None, 0)
    assert not metric_equal("1", 1)


def test_empty_job_list_cannot_pass():
    assert runner._parity_summary([])["status"] == "not_assessed"


def test_full_child_evidence_is_opt_in_and_capped(tmp_path):
    _, _, assembly = fixture(count=51)
    payload = child_success_to_mapping(
        result=SimpleNamespace(
            top_variants=assembly.top_variants,
            stage_timings={},
            summary_hash=assembly.summary_hash,
            cleanup_evidence={},
            exact_diagnostics={},
            instrumentation_counters={},
        )
    )
    env = {"ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR": str(tmp_path)}
    _write_result_evidence(env=env, job_id=uuid4(), payload=payload, process_evidence={})
    first = next(tmp_path.glob("*.json"))
    assert "benchmark_full_top" not in json.loads(first.read_text())
    _write_result_evidence(
        env={**env, "ROEHUB_BACKTEST_BENCHMARK_FULL_TOP": "1"},
        job_id=uuid4(),
        payload=payload,
        process_evidence={},
    )
    second = next(p for p in tmp_path.glob("*.json") if p != first)
    full = json.loads(second.read_text())["benchmark_full_top"]
    assert full["count"] == 51 and len(full["items"]) == 50 and full["complete"] is False


def test_assessed_legacy_mismatch_cannot_contradict_aggregate_pass():
    api, ref, _ = fixture()
    telemetry = {
        "risk_mode": "none",
        "arity": 1,
        "direction_mode": "long_only",
        "exact_candidates_evaluated": 50,
        "request_top_n": 50,
        "benchmark_top_k": 5,
        "top_results_count": 50,
    }
    samples = [
        {"rank": rank, "metrics": row["summary_metrics"]}
        for rank, row in enumerate(api["items"][:5], 1)
    ]
    legacy = copy.deepcopy(samples)
    legacy[0]["metrics"]["total_return_pct"] += 1
    child = {
        "exact_diagnostics": {"telemetry": telemetry, "top_results_sample": samples},
        "benchmark_full_top": {"items": api["items"], "count": 50, "complete": True},
    }
    result = runner._compare_reference_results(
        api_top=api,
        child_evidence=[child],
        reference_run={**telemetry, "full_top_reference": ref, "top_results": legacy},
        comparison_context=CONTEXT,
    )
    assert result["full_reference"]["status"] == "passed"
    assert result["partial_sample_comparison"]["status"] == "failed"
    assert result["status"] == "failed" and result["pass"] is False


@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_tp_sl_selected_cell_with_unchanged_return(direction):
    api, ref, _ = fixture(direction=direction, tp_sl=True)
    assert assess(api, ref)["status"] == "passed"
    api["items"][49]["best_sl_pct"] = 1.5
    api["items"][49]["summary_metrics"]["best_sl_pct"] = 1.5
    api["items"][49]["canonical_variant_params"]["risk"]["best_sl_pct"] = 1.5
    assert assess(api, ref)["status"] == "failed"


@pytest.mark.parametrize("top_n,count", [(1, 1), (50, 8), (50, 0)])
def test_aggregate_accepts_only_complete_independent_reference(top_n, count):
    api, ref, _ = fixture(top_n, count)
    telemetry = {
        "risk_mode": "none",
        "arity": 1,
        "direction_mode": "long_only",
        "exact_candidates_evaluated": count,
        "request_top_n": top_n,
        "benchmark_top_k": 5,
        "top_results_count": count,
    }
    child = {
        "exact_diagnostics": {"telemetry": telemetry},
        "benchmark_full_top": {"items": api["items"], "count": count, "complete": True},
    }
    result = runner._compare_reference_results(
        api_top=api,
        child_evidence=[child],
        reference_run={**telemetry, "full_top_reference": ref},
        comparison_context=CONTEXT,
        requested_top_n=top_n,
    )
    assert result["status"] == "passed"
    assert result["full_reference"]["compared_count"] == count
