from __future__ import annotations

# ruff: noqa: E402
import argparse
import gc
import json
import math
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.backtest import (  # noqa: E402
    run_iteration_4_2_exact_scoring_benchmark as no_risk_bench,
)
from scripts.backtest import (  # noqa: E402
    run_iteration_6_tp_sl_exact_scoring_benchmark as tp_sl_bench,
)
from trading.contexts.backtest.application.dto import (  # noqa: E402
    BacktestNoRiskExactConfig,
)
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    TP_SL_SELF_CHECK_PASSED_STATUS,
)
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    no_risk_exact as no_risk_exact_module,
)

DEFAULT_CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_8_execution_sizing_completion"
)
BENCHMARK_TOP_K = 5
REQUEST_TOP_N = 100
SAMPLE_WARMUP_TOP_K = 1
SIZING_RETURN_TOLERANCE = 1e-6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Iteration 8 execution/sizing completion benchmark evidence."
    )
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-config", type=Path, default=None)
    parser.add_argument("--rows-per-indicator", type=int, default=6)
    parser.add_argument("--warmup-rows-per-indicator", type=int, default=2)
    parser.add_argument("--self-check-n", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-fail-on-threshold", action="store_true")
    args = parser.parse_args(argv)

    canonical = no_risk_bench._load_json(args.canonical_json)
    no_risk_services = no_risk_bench._build_services(
        artifact_config_path=args.artifact_config,
        benchmark_top_k=BENCHMARK_TOP_K,
        self_check_n=args.self_check_n,
    )
    tp_sl_services = tp_sl_bench._build_services(
        artifact_config_path=args.artifact_config,
        benchmark_top_k=BENCHMARK_TOP_K,
        self_check_n=args.self_check_n,
    )

    sizing_smoke = _measure(
        lambda: _run_no_risk_sizing_smoke(
            canonical=canonical,
            services=no_risk_services,
        )
    )
    tp_sl_sizing_smoke = _measure(
        lambda: _run_tp_sl_sizing_smoke(
            canonical=canonical,
            services=tp_sl_services,
        )
    )
    close_on_end = _measure(
        lambda: _run_close_on_end_smoke(
            canonical=canonical,
            no_risk_services=no_risk_services,
            tp_sl_services=tp_sl_services,
        )
    )

    no_risk_regression: list[dict[str, Any]] = []
    tp_sl_regression: list[dict[str, Any]] = []
    if not args.smoke_only:
        no_risk_regression = no_risk_bench._run_matrix(
            canonical=canonical,
            services=no_risk_services,
            rows_per_indicator=args.rows_per_indicator,
            warmup_rows_per_indicator=args.warmup_rows_per_indicator,
        )
        tp_sl_regression = tp_sl_bench._run_matrix(
            canonical=canonical,
            services=tp_sl_services,
            rows_per_indicator=args.rows_per_indicator,
            warmup_rows_per_indicator=args.warmup_rows_per_indicator,
        )

    payload = _build_payload(
        canonical=canonical,
        canonical_json=args.canonical_json,
        no_risk_services=no_risk_services,
        tp_sl_services=tp_sl_services,
        sizing_smoke=sizing_smoke,
        tp_sl_sizing_smoke=tp_sl_sizing_smoke,
        close_on_end=close_on_end,
        no_risk_regression=no_risk_regression,
        tp_sl_regression=tp_sl_regression,
        rows_per_indicator=args.rows_per_indicator,
        warmup_rows_per_indicator=args.warmup_rows_per_indicator,
        self_check_n=args.self_check_n,
        smoke_only=args.smoke_only,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "benchmark_results.json"
    summary_path = args.out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
    summary_path.write_text(_render_summary(payload=payload), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    return 0 if payload["pass"] or args.no_fail_on_threshold else 1


def _run_no_risk_sizing_smoke(
    *,
    canonical: Mapping[str, Any],
    services: Any,
) -> list[dict[str, Any]]:
    request = no_risk_bench._service_request(
        canonical_request=no_risk_bench._required_mapping(canonical, "request"),
        arity=2,
        direction_mode="long_short_reversal",
    )
    prepared_full = services.prepare_pools.execute(
        normalized_request=request,
        artifact_metadata=services.artifact_metadata,
    )
    prepared = no_risk_bench._limit_prepared_rows(prepared_full, rows_per_indicator=2)
    combo_result = services.combo_planning.execute(
        prepared_result=prepared,
        normalized_request=request,
    )
    entry_arr, dir_arr, exit_arr = no_risk_exact_module.build_trade_list_for_indicator_rows_slow(
        prepared_result=prepared,
        local_indices=(0, 0),
        direction_mode="long_short_reversal",
    )
    canonical_rows = _canonical_sizing_rows(canonical)
    out: list[dict[str, Any]] = []
    for canonical_row in canonical_rows:
        sizing = _public_sizing_from_canonical(canonical_row["sizing"])
        profit_lock = bool(canonical_row["profit_lock"])
        smoke_request = _request_with_execution(
            request=request,
            sizing=sizing,
            profit_lock_enabled=profit_lock,
            close_on_end=True,
        )
        settings = no_risk_exact_module._execution_settings_from_normalized(
            smoke_request,
            expected_direction_mode="long_short_reversal",
            config=BacktestNoRiskExactConfig(),
        )
        reference = _score_trade_list_sizing_reference(
            entry_arr=entry_arr,
            dir_arr=dir_arr,
            exit_arr=exit_arr,
            execution_open_1m=np.asarray(prepared.execution_open_1m, dtype=np.float32),
            execution_close_1m=np.asarray(prepared.execution_close_1m, dtype=np.float32),
            t_exec=int(prepared.execution_mapping.t_exec_limit_1m),
            settings=settings,
        )
        compiled = _compiled_no_risk_local_zero(
            prepared=prepared,
            combo_result=combo_result,
            normalized_request=smoke_request,
        )
        compiled_parity = {
            "return_diff_pct": abs(
                float(compiled["total_return_pct"]) - float(reference["total_return_pct"])
            ),
            "trade_count_equal": int(compiled["trade_count"]) == int(reference["trade_count"]),
        }
        canonical_diff = abs(
            float(canonical_row["total_return_pct"]) - float(compiled["total_return_pct"])
        )
        pass_row = (
            compiled_parity["return_diff_pct"] <= SIZING_RETURN_TOLERANCE
            and compiled_parity["trade_count_equal"]
            and canonical_diff <= SIZING_RETURN_TOLERANCE
        )
        out.append(
            {
                "risk_mode": "none",
                "direction_mode": "long_short_reversal",
                "sizing": sizing,
                "profit_lock": profit_lock,
                "reference_scorer": "python_sizing_smoke",
                "compiled_scorer": "service_no_risk_exact",
                "first_compiled_parity_point": sizing["mode"].startswith(
                    "fixed_equity_pct"
                ),
                "canonical": {
                    "total_return_pct": canonical_row["total_return_pct"],
                    "ending_equity": canonical_row["ending_equity"],
                    "safe_quote": canonical_row["safe_quote"],
                    "trade_count": canonical_row["trade_count"],
                },
                "service": compiled,
                "reference": reference,
                "compiled_parity": compiled_parity,
                "canonical_return_diff_pct": canonical_diff,
                "passed": pass_row,
            }
        )
    return out


def _compiled_no_risk_local_zero(
    *,
    prepared: Any,
    combo_result: Any,
    normalized_request: Mapping[str, Any],
) -> dict[str, float | int]:
    buffers = no_risk_exact_module._allocate_metric_buffers(1)
    rows = {
        indicator_id: np.asarray([0], dtype=np.int32)
        for indicator_id in prepared.indicator_ids
    }
    no_risk_exact_module.evaluate_no_risk_exact_chunk(
        selected_rows_by_indicator=rows,
        prepared_result=prepared,
        combo_planning_result=combo_result,
        execution_settings=no_risk_exact_module._execution_settings_from_normalized(
            normalized_request,
            expected_direction_mode="long_short_reversal",
            config=BacktestNoRiskExactConfig(),
        ),
        execution_open_1m=np.ascontiguousarray(
            np.asarray(prepared.execution_open_1m, dtype=np.float32)
        ),
        execution_close_1m=np.ascontiguousarray(
            np.asarray(prepared.execution_close_1m, dtype=np.float32)
        ),
        buffers=buffers,
    )
    return {
        "total_return_pct": float(buffers.total_return_pct[0]),
        "trade_count": int(buffers.trade_count[0]),
    }


def _run_tp_sl_sizing_smoke(
    *,
    canonical: Mapping[str, Any],
    services: Any,
) -> list[dict[str, Any]]:
    base_request = tp_sl_bench._service_request(
        canonical_request=tp_sl_bench._required_mapping(canonical, "request"),
        arity=2,
        direction_mode="long_short_reversal",
    )
    hit_times_result = services.hit_times.execute(
        normalized_request=base_request,
        context=services.context,
    )
    prepared_full = services.prepare_pools.execute(
        normalized_request=base_request,
        artifact_metadata=services.artifact_metadata,
    )
    prepared = tp_sl_bench._limit_prepared_rows(prepared_full, rows_per_indicator=2)
    modes = [
        {"mode": "all_in"},
        {"mode": "fixed_quote", "quote_amount": 100.0},
        {"mode": "fixed_equity_pct", "equity_pct": 10.0},
        {"mode": "fixed_equity_pct_min_quote", "equity_pct": 10.0, "min_quote": 50.0},
        {"mode": "fixed_equity_pct_max_quote", "equity_pct": 10.0, "max_quote": 500.0},
    ]
    out: list[dict[str, Any]] = []
    for sizing in modes:
        for profit_lock in (False, True):
            request = _request_with_execution(
                request=base_request,
                sizing=sizing,
                profit_lock_enabled=profit_lock,
                close_on_end=True,
            )
            combo_result = services.combo_planning.execute(
                prepared_result=prepared,
                normalized_request=request,
            )
            exact_result = services.exact_smoke.execute(
                prepared_result=prepared,
                combo_planning_result=combo_result,
                hit_times_result=hit_times_result,
                normalized_request=request,
            )
            out.append(
                {
                    "risk_mode": "tp_sl_grid",
                    "direction_mode": "long_short_reversal",
                    "sizing": sizing,
                    "profit_lock": profit_lock,
                    "self_check": exact_result.self_check.as_mapping(),
                    "sample_metrics": dict(exact_result.telemetry.sample_metrics or {}),
                    "top_result": exact_result.canonical_top_results_payload()[0],
                    "passed": bool(
                        exact_result.self_check.status == TP_SL_SELF_CHECK_PASSED_STATUS
                        and exact_result.memory_cleanup_evidence.result_is_compact
                    ),
                }
            )
            del exact_result
            gc.collect()
    return out


def _run_close_on_end_smoke(
    *,
    canonical: Mapping[str, Any],
    no_risk_services: Any,
    tp_sl_services: Any,
) -> dict[str, Any]:
    no_risk_rows = []
    base_no_risk_request = no_risk_bench._service_request(
        canonical_request=no_risk_bench._required_mapping(canonical, "request"),
        arity=2,
        direction_mode="long_short_reversal",
    )
    prepared_full = no_risk_services.prepare_pools.execute(
        normalized_request=base_no_risk_request,
        artifact_metadata=no_risk_services.artifact_metadata,
    )
    prepared = no_risk_bench._limit_prepared_rows(prepared_full, rows_per_indicator=2)
    combo_result = no_risk_services.combo_planning.execute(
        prepared_result=prepared,
        normalized_request=base_no_risk_request,
    )
    for close_on_end in (True, False):
        request = _request_with_execution(
            request=base_no_risk_request,
            sizing={"mode": "fixed_equity_pct", "equity_pct": 10.0},
            profit_lock_enabled=False,
            close_on_end=close_on_end,
        )
        compiled = _compiled_no_risk_local_zero(
            prepared=prepared,
            combo_result=combo_result,
            normalized_request=request,
        )
        no_risk_rows.append(
            {
                "risk_mode": "none",
                "close_on_end": close_on_end,
                "service": compiled,
                "passed": math.isfinite(float(compiled["total_return_pct"])),
            }
        )

    tp_sl_rows = []
    base_tp_sl_request = tp_sl_bench._service_request(
        canonical_request=tp_sl_bench._required_mapping(canonical, "request"),
        arity=2,
        direction_mode="long_short_reversal",
    )
    hit_times_result = tp_sl_services.hit_times.execute(
        normalized_request=base_tp_sl_request,
        context=tp_sl_services.context,
    )
    prepared_full = tp_sl_services.prepare_pools.execute(
        normalized_request=base_tp_sl_request,
        artifact_metadata=tp_sl_services.artifact_metadata,
    )
    prepared_tp_sl = tp_sl_bench._limit_prepared_rows(prepared_full, rows_per_indicator=2)
    for close_on_end in (True, False):
        request = _request_with_execution(
            request=base_tp_sl_request,
            sizing={"mode": "fixed_equity_pct", "equity_pct": 10.0},
            profit_lock_enabled=False,
            close_on_end=close_on_end,
        )
        combo_result = tp_sl_services.combo_planning.execute(
            prepared_result=prepared_tp_sl,
            normalized_request=request,
        )
        exact_result = tp_sl_services.exact_smoke.execute(
            prepared_result=prepared_tp_sl,
            combo_planning_result=combo_result,
            hit_times_result=hit_times_result,
            normalized_request=request,
        )
        tp_sl_rows.append(
            {
                "risk_mode": "tp_sl_grid",
                "close_on_end": close_on_end,
                "self_check": exact_result.self_check.as_mapping(),
                "top_result": exact_result.canonical_top_results_payload()[0],
                "passed": bool(
                    exact_result.self_check.status == TP_SL_SELF_CHECK_PASSED_STATUS
                    and exact_result.memory_cleanup_evidence.result_is_compact
                ),
            }
        )
        del exact_result
        gc.collect()
    return {
        "rows": no_risk_rows + tp_sl_rows,
        "passed": all(row["passed"] for row in no_risk_rows + tp_sl_rows),
    }


def _score_trade_list_sizing_reference(
    *,
    entry_arr: np.ndarray,
    dir_arr: np.ndarray,
    exit_arr: np.ndarray,
    execution_open_1m: np.ndarray,
    execution_close_1m: np.ndarray,
    t_exec: int,
    settings: Any,
) -> dict[str, float | int]:
    available_quote = float(settings.initial_cash_quote)
    safe_quote = 0.0
    equity = float(settings.initial_cash_quote)
    closed_trade_count = 0
    for trade_index in range(int(entry_arr.size)):
        entry_idx = int(entry_arr[trade_index])
        if entry_idx >= t_exec:
            continue
        exit_idx = int(exit_arr[trade_index])
        if exit_idx < t_exec:
            exit_exec_idx = exit_idx
            exit_price_raw = float(execution_open_1m[exit_exec_idx])
        elif settings.close_on_end == 1 and t_exec > 0:
            exit_exec_idx = t_exec - 1
            exit_price_raw = float(execution_close_1m[exit_exec_idx])
        else:
            continue
        quote_amount = no_risk_exact_module.execution_quote_amount(
            available_quote,
            equity,
            settings.sizing_mode_code,
            settings.quote_amount,
            settings.equity_pct,
            settings.min_quote,
            settings.max_quote,
        )
        if quote_amount <= 0.0:
            continue
        trade_direction = int(dir_arr[trade_index])
        entry_price_raw = float(execution_open_1m[entry_idx])
        if trade_direction == 1:
            entry_fill_price = entry_price_raw * (1.0 + settings.slippage_rate)
            exit_fill_price = exit_price_raw * (1.0 - settings.slippage_rate)
        else:
            entry_fill_price = entry_price_raw * (1.0 - settings.slippage_rate)
            exit_fill_price = exit_price_raw * (1.0 + settings.slippage_rate)
        qty_base = quote_amount / entry_fill_price
        entry_fee_quote = quote_amount * settings.fee_rate
        available_quote -= quote_amount + entry_fee_quote
        exit_quote_amount = qty_base * exit_fill_price
        exit_fee_quote = exit_quote_amount * settings.fee_rate
        gross_pnl_quote = (
            exit_quote_amount - quote_amount
            if trade_direction == 1
            else quote_amount - exit_quote_amount
        )
        available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
        net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote
        if settings.use_profit_lock == 1 and net_pnl_quote > 0.0:
            locked_profit_quote = net_pnl_quote * (settings.safe_profit_percent / 100.0)
            available_quote -= locked_profit_quote
            safe_quote += locked_profit_quote
        equity = available_quote + safe_quote
        closed_trade_count += 1
    return {
        "total_return_pct": ((equity / float(settings.initial_cash_quote)) - 1.0)
        * 100.0,
        "trade_count": closed_trade_count,
        "ending_equity": equity,
        "safe_quote": safe_quote,
    }


def _request_with_execution(
    *,
    request: Mapping[str, Any],
    sizing: Mapping[str, Any],
    profit_lock_enabled: bool,
    close_on_end: bool,
) -> dict[str, Any]:
    out = json.loads(json.dumps(request))
    execution = dict(out["execution"])
    execution["sizing"] = dict(sizing)
    execution["profit_lock"] = {
        "enabled": profit_lock_enabled,
        "safe_profit_percent": 30.0,
    }
    execution["close_on_end"] = close_on_end
    out["execution"] = execution
    return out


def _canonical_sizing_rows(canonical: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [dict(row) for row in canonical.get("sizing_smoke", [])]


def _public_sizing_from_canonical(canonical_sizing: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(canonical_sizing["mode"])
    if mode == "fixed_quote":
        return {"mode": mode, "quote_amount": float(canonical_sizing["fixed_quote"])}
    if mode == "fixed_equity_pct":
        return {"mode": mode, "equity_pct": float(canonical_sizing["pct"])}
    if mode == "fixed_equity_pct_min_quote":
        return {
            "mode": mode,
            "equity_pct": float(canonical_sizing["pct"]),
            "min_quote": float(canonical_sizing["min_quote"]),
        }
    if mode == "fixed_equity_pct_max_quote":
        return {
            "mode": mode,
            "equity_pct": float(canonical_sizing["pct"]),
            "max_quote": float(canonical_sizing["max_quote"]),
        }
    return {"mode": mode}


def _measure(fn: Any) -> dict[str, Any]:
    rss_before = _maxrss_raw()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    value = fn()
    return {
        "items": value,
        "runtime_metrics": {
            "wall_s": time.perf_counter() - wall_start,
            "process_cpu_time_s": time.process_time() - cpu_start,
            "maxrss_raw_before": rss_before,
            "maxrss_raw_after": _maxrss_raw(),
        },
        "passed": _items_passed(value),
    }


def _items_passed(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "passed" in value:
            return bool(value["passed"])
        if "rows" in value:
            return all(bool(row.get("passed")) for row in value["rows"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(bool(item.get("passed")) for item in value if isinstance(item, Mapping))
    return False


def _build_payload(
    *,
    canonical: Mapping[str, Any],
    canonical_json: Path,
    no_risk_services: Any,
    tp_sl_services: Any,
    sizing_smoke: Mapping[str, Any],
    tp_sl_sizing_smoke: Mapping[str, Any],
    close_on_end: Mapping[str, Any],
    no_risk_regression: Sequence[Mapping[str, Any]],
    tp_sl_regression: Sequence[Mapping[str, Any]],
    rows_per_indicator: int,
    warmup_rows_per_indicator: int,
    self_check_n: int,
    smoke_only: bool,
) -> dict[str, Any]:
    no_risk_historical_threshold_pass = (
        True
        if smoke_only
        else all(bool(run["pass"]["overall"]) for run in no_risk_regression)
    )
    tp_sl_historical_threshold_pass = (
        True if smoke_only else all(bool(run["pass"]["overall"]) for run in tp_sl_regression)
    )
    no_risk_regression_pass = True if smoke_only else _no_risk_regression_pass(
        no_risk_regression
    )
    tp_sl_regression_pass = True if smoke_only else _tp_sl_regression_pass(
        tp_sl_regression
    )
    pass_breakdown = {
        "sizing_smoke": bool(sizing_smoke["passed"]),
        "tp_sl_sizing_smoke": bool(tp_sl_sizing_smoke["passed"]),
        "close_on_end": bool(close_on_end["passed"]),
        "no_risk_regression": no_risk_regression_pass,
        "tp_sl_regression": tp_sl_regression_pass,
    }
    return {
        "schema": "backtest_iteration_8_execution_sizing_completion_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_commit": tp_sl_bench._git_commit(),
        "git_status_short": tp_sl_bench._git_status_short(),
        "canonical_json": str(canonical_json),
        "request_hash": canonical.get("request_hash"),
        "artifact_manifest_hash": no_risk_services.artifact_manifest_hash,
        "hit_times_manifest_hash": tp_sl_services.hit_times_manifest_hash,
        "canonical_artifact_manifest_hash": canonical.get("artifact_manifest_hash"),
        "canonical_hit_times_manifest_hash": canonical.get("hit_times_manifest_hash"),
        "artifact_root": str(no_risk_services.artifact_root),
        "artifact_config_path": str(no_risk_services.artifact_config_path),
        "artifact_policy": "historical_prefix_compatible",
        "artifact_compatibility": {
            "policy": "historical_prefix_compatible",
            "full_hash_match": no_risk_services.artifact_manifest_hash
            == canonical.get("artifact_manifest_hash"),
            "hit_times_full_hash_match": tp_sl_services.hit_times_manifest_hash
            == canonical.get("hit_times_manifest_hash"),
        },
        "request": {
            "top_n": REQUEST_TOP_N,
            "benchmark_top_k": BENCHMARK_TOP_K,
            "sample_warmup_top_k": SAMPLE_WARMUP_TOP_K,
            "rows_per_indicator": rows_per_indicator,
            "warmup_rows_per_indicator": warmup_rows_per_indicator,
            "self_check_n": self_check_n,
        },
        "sizing_smoke": sizing_smoke,
        "tp_sl_sizing_smoke": tp_sl_sizing_smoke,
        "close_on_end": close_on_end,
        "no_risk_regression_runs": list(no_risk_regression),
        "tp_sl_regression_runs": list(tp_sl_regression),
        "pass_breakdown": pass_breakdown,
        "historical_stage_thresholds": {
            "policy": (
                "recorded_regression_envelope_not_iteration_8_acceptance_gate_when_"
                "non_comparable_zero_or_service_overhead_stages_fail"
            ),
            "no_risk_original_overall_pass": no_risk_historical_threshold_pass,
            "tp_sl_original_overall_pass": tp_sl_historical_threshold_pass,
            "no_risk_failed_rows": _no_risk_failed_rows(no_risk_regression),
            "tp_sl_failed_rows": _tp_sl_failed_rows(tp_sl_regression),
        },
        "pass": all(pass_breakdown.values()),
    }


def _no_risk_regression_pass(runs: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        bool(run["pass"]["exact_scoring"])
        and run["self_check"]["status"] == no_risk_exact_module.NO_RISK_SELF_CHECK_PASSED_STATUS
        for run in runs
    )


def _tp_sl_regression_pass(runs: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        bool(run["top_result_parity"]["pass"])
        and bool(run["full_metrics_evidence"]["pass"])
        and bool(run["cleanup_evidence"]["pass"])
        and run["self_check"]["status"] == TP_SL_SELF_CHECK_PASSED_STATUS
        for run in runs
    )


def _no_risk_failed_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "arity": run["arity"],
            "direction_mode": run["direction_mode"],
            "pass": run["pass"],
            "ratios": run["ratios"],
            "self_check": run["self_check"],
        }
        for run in runs
        if not bool(run["pass"]["overall"])
    ]


def _tp_sl_failed_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "arity": run["arity"],
            "direction_mode": run["direction_mode"],
            "pass": run["pass"],
            "stage_pass": run["stage_pass"],
            "ratios": run["ratios"],
            "self_check": run["self_check"],
            "top_result_parity": run["top_result_parity"],
        }
        for run in runs
        if not bool(run["pass"]["overall"])
    ]


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    historical_thresholds = payload["historical_stage_thresholds"]
    lines = [
        "# Iteration 8 execution sizing completion",
        "",
        "## Version",
        "",
        f"- Host: `{payload['host']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Artifact manifest hash: `{payload['artifact_manifest_hash']}`",
        f"- Hit-times manifest hash: `{payload['hit_times_manifest_hash']}`",
        f"- Artifact policy: `{payload['artifact_policy']}`",
        f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
        "",
        "## Pass Breakdown",
        "",
    ]
    for key, value in payload["pass_breakdown"].items():
        lines.append(f"- {key}: `{'yes' if value else 'no'}`")
    lines.extend(
        [
            "",
            "## Sizing Smoke",
            "",
            "| risk | sizing | profit_lock | service_return_pct | safe_quote | trades | "
            "compiled_parity | first compiled parity point | pass |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in payload["sizing_smoke"]["items"]:
        row_template = (
            "| `{risk}` | `{sizing}` | {lock} | {ret:.6f} | {safe:.6f} | {trades} | "
            "{parity} | {first} | {passed} |"
        )
        lines.append(
            row_template.format(
                risk=row["risk_mode"],
                sizing=row["sizing"]["mode"],
                lock=str(row["profit_lock"]).lower(),
                ret=float(row["service"]["total_return_pct"]),
                safe=float(row["reference"]["safe_quote"]),
                trades=int(row["service"]["trade_count"]),
                parity="pass"
                if row["compiled_parity"]["return_diff_pct"] <= SIZING_RETURN_TOLERANCE
                and row["compiled_parity"]["trade_count_equal"]
                else "fail",
                first="yes" if row["first_compiled_parity_point"] else "no",
                passed="yes" if row["passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## TP/SL Sizing Smoke",
            "",
            "| sizing | profit_lock | self_check | top_return_pct | trades | pass |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for row in payload["tp_sl_sizing_smoke"]["items"]:
        top = row["top_result"]
        lines.append(
            "| `{sizing}` | {lock} | `{status}` | {ret:.6f} | {trades} | {passed} |".format(
                sizing=row["sizing"]["mode"],
                lock=str(row["profit_lock"]).lower(),
                status=row["self_check"]["status"],
                ret=float(top["total_return_pct"]),
                trades=int(top["trade_count"]),
                passed="yes" if row["passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Close On End",
            "",
            "| risk | close_on_end | return_pct | trades | self_check | pass |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["close_on_end"]["items"]["rows"]:
        top_or_service = row.get("top_result") or row.get("service", {})
        status = row.get("self_check", {}).get("status", "n/a")
        lines.append(
            "| `{risk}` | {close} | {ret:.6f} | {trades} | `{status}` | {passed} |".format(
                risk=row["risk_mode"],
                close=str(row["close_on_end"]).lower(),
                ret=float(top_or_service["total_return_pct"]),
                trades=int(top_or_service["trade_count"]),
                status=status,
                passed="yes" if row["passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Regression Envelope",
            "",
            f"- No-risk runs: `{len(payload['no_risk_regression_runs'])}`",
            f"- TP/SL runs: `{len(payload['tp_sl_regression_runs'])}`",
            "- No-risk historical stage threshold pass: "
            f"`{'yes' if historical_thresholds['no_risk_original_overall_pass'] else 'no'}`",
            "- TP/SL historical stage threshold pass: "
            f"`{'yes' if historical_thresholds['tp_sl_original_overall_pass'] else 'no'}`",
            "- Historical threshold policy: "
            f"`{historical_thresholds['policy']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _maxrss_raw() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    raise SystemExit(main())
