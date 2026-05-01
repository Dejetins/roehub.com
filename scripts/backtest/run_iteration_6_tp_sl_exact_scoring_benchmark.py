from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.backtest.adapters.outbound import (  # noqa: E402
    BacktestArtifactPathBuilderV2,
    FilesystemBacktestArtifactContextResolver,
    YamlBacktestGridDefaultsProvider,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (  # noqa: E402
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.dto import (  # noqa: E402
    BacktestComboPlanningConfig,
    BacktestCoordinates,
    BacktestPreparePoolsConfig,
    BacktestPreparePoolsResult,
    BacktestTpSlExactConfig,
    PreparedIndicatorPool,
)
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    BUILD_EXACT_CONTEXT_STAGE_NAME,
    BUILD_PROXY_CONTEXT_STAGE_NAME,
    COMBO_ITERATION_STAGE_NAME,
    HIT_TIMES_ARTIFACT_PATH_V2,
    LOAD_HIT_TIMES_STAGE_NAME,
    PREPARE_POOLS_CORE_STAGE_NAME,
    PROXY_FILTER_STAGE_NAME,
    TARGET_TP_SL_GRID_START_PCT,
    TARGET_TP_SL_GRID_STEP_PCT,
    TARGET_TP_SL_GRID_STOP_PCT,
    TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME,
    TP_SL_EXACT_SCORING_STAGE_NAME,
    TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME,
    TP_SL_GRID_VALIDATION_STAGE_NAME,
    TP_SL_HEAP_UPDATE_STAGE_NAME,
    TP_SL_SELF_CHECK_PASSED_STATUS,
    TP_SL_SELF_CHECK_STAGE_NAME,
    BacktestComboPlanningService,
    BacktestPreparePoolsService,
    BacktestTpSlExactScoringService,
    BacktestTpSlHitTimesService,
    build_signal_segments,
    notebook_compatible_prepare_pools_core_s,
    row_metadata_order_hash,
)
from trading.contexts.backtest.application.services.v2.benchmark_accounting import (  # noqa: E402
    build_benchmark_accounting_record,
)
from trading.contexts.backtest_artifacts.application.services.v2 import (  # noqa: E402
    YamlBacktestArtifactLoaderV2,
)

DEFAULT_CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_6_tp_sl_exact_scoring_full_metrics"
)
TARGET_RISK_MODE = "tp_sl_grid"
TARGET_ARITIES = tuple(range(1, 8))
SMOKE_ARITIES = (8, 9, 10)
TARGET_DIRECTION_MODES = ("long_only", "long_short_reversal")
ACCEPTANCE_RATIO = 0.9
BENCHMARK_TOP_K = 5
REQUEST_TOP_N = 100
SAMPLE_WARMUP_TOP_K = 1
FULL_METRIC_NAMES = (
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
    "best_tp_pct",
    "best_sl_pct",
)
COMPARED_STAGES = (
    LOAD_HIT_TIMES_STAGE_NAME,
    TP_SL_GRID_VALIDATION_STAGE_NAME,
    PREPARE_POOLS_CORE_STAGE_NAME,
    BUILD_EXACT_CONTEXT_STAGE_NAME,
    BUILD_PROXY_CONTEXT_STAGE_NAME,
    COMBO_ITERATION_STAGE_NAME,
    PROXY_FILTER_STAGE_NAME,
    TP_SL_SELF_CHECK_STAGE_NAME,
    TP_SL_EXACT_SCORING_STAGE_NAME,
    TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME,
    TP_SL_HEAP_UPDATE_STAGE_NAME,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 6 artifact-backed TP/SL exact scoring, top-K heap, "
            "and full-metrics benchmark."
        )
    )
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-config", type=Path, default=None)
    parser.add_argument("--rows-per-indicator", type=int, default=6)
    parser.add_argument("--warmup-rows-per-indicator", type=int, default=2)
    parser.add_argument("--self-check-n", type=int, default=2)
    parser.add_argument("--smoke-rows-per-indicator", type=int, default=2)
    parser.add_argument(
        "--no-fail-on-threshold",
        action="store_true",
        help="Write evidence and return 0 even when ratios are below acceptance.",
    )
    args = parser.parse_args(argv)

    if args.rows_per_indicator <= 0:
        parser.error("--rows-per-indicator must be > 0")
    if args.warmup_rows_per_indicator <= 0:
        parser.error("--warmup-rows-per-indicator must be > 0")
    if args.self_check_n < 0:
        parser.error("--self-check-n must be >= 0")
    if args.smoke_rows_per_indicator <= 0:
        parser.error("--smoke-rows-per-indicator must be > 0")

    canonical = _load_json(args.canonical_json)
    services = _build_services(
        artifact_config_path=args.artifact_config,
        benchmark_top_k=BENCHMARK_TOP_K,
        self_check_n=args.self_check_n,
    )
    runs = _run_matrix(
        canonical=canonical,
        services=services,
        rows_per_indicator=args.rows_per_indicator,
        warmup_rows_per_indicator=args.warmup_rows_per_indicator,
    )
    smokes = _run_smokes(
        canonical=canonical,
        services=services,
        rows_per_indicator=args.smoke_rows_per_indicator,
        self_check_n=max(1, min(1, args.self_check_n)),
    )
    payload = _build_payload(
        canonical=canonical,
        canonical_json=args.canonical_json,
        services=services,
        runs=runs,
        smokes=smokes,
        rows_per_indicator=args.rows_per_indicator,
        warmup_rows_per_indicator=args.warmup_rows_per_indicator,
        self_check_n=args.self_check_n,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "benchmark_results.json"
    summary_path = args.out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
    summary_path.write_text(_render_summary(payload=payload), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    if payload["pass"] or args.no_fail_on_threshold:
        return 0
    _print_slowest_failures(payload)
    return 1


class _Services:
    def __init__(
        self,
        *,
        artifact_config_path: Path,
        artifact_root: Path,
        artifact_manifest_hash: str,
        hit_times_manifest_hash: str,
        prepare_pools: BacktestPreparePoolsService,
        combo_planning: BacktestComboPlanningService,
        hit_times: BacktestTpSlHitTimesService,
        exact_measured: BacktestTpSlExactScoringService,
        exact_warmup: BacktestTpSlExactScoringService,
        exact_smoke: BacktestTpSlExactScoringService,
        artifact_metadata: Any,
        array_loader: FilesystemBacktestArtifactArrayLoader,
        context: Any,
    ) -> None:
        self.artifact_config_path = artifact_config_path
        self.artifact_root = artifact_root
        self.artifact_manifest_hash = artifact_manifest_hash
        self.hit_times_manifest_hash = hit_times_manifest_hash
        self.prepare_pools = prepare_pools
        self.combo_planning = combo_planning
        self.hit_times = hit_times
        self.exact_measured = exact_measured
        self.exact_warmup = exact_warmup
        self.exact_smoke = exact_smoke
        self.artifact_metadata = artifact_metadata
        self.array_loader = array_loader
        self.context = context


def _build_services(
    *,
    artifact_config_path: Path | None,
    benchmark_top_k: int,
    self_check_n: int,
) -> _Services:
    environ = dict(os.environ)
    if artifact_config_path is not None:
        environ["ROEHUB_BACKTEST_ARTIFACTS_CONFIG"] = str(artifact_config_path)
    elif not environ.get("ROEHUB_BACKTEST_ARTIFACTS_CONFIG") and not environ.get(
        "ROEHUB_ENV"
    ):
        environ["ROEHUB_ENV"] = "prod"

    resolved_config_path = resolve_backtest_artifacts_config_path(environ=environ)
    artifact_config = load_backtest_artifacts_runtime_config(resolved_config_path)
    path_builder = BacktestArtifactPathBuilderV2(root=artifact_config.artifact_root_path())
    artifact_loader = YamlBacktestArtifactLoaderV2(path_resolver=path_builder)
    defaults_provider = YamlBacktestGridDefaultsProvider.from_environ(
        environ=environ,
        artifact_config_path=resolved_config_path,
    )
    coordinates = BacktestCoordinates(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    artifact_metadata = FilesystemBacktestArtifactContextResolver(
        artifact_loader=artifact_loader
    ).resolve_context(coordinates=coordinates)
    if artifact_metadata.hit_times_manifest_hash is None:
        raise ValueError("active artifact metadata must include hit_times_manifest_hash")
    array_loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=artifact_loader)
    context = array_loader.resolve_context(
        coordinates=coordinates,
        artifact_metadata=artifact_metadata,
    )
    return _Services(
        artifact_config_path=resolved_config_path,
        artifact_root=artifact_config.artifact_root_path(),
        artifact_manifest_hash=artifact_metadata.artifact_manifest_hash,
        hit_times_manifest_hash=artifact_metadata.hit_times_manifest_hash,
        prepare_pools=BacktestPreparePoolsService(
            artifact_array_loader=array_loader,
            defaults_provider=defaults_provider,
            config=BacktestPreparePoolsConfig(
                row_prefilter_top_fraction=1.0,
                row_prefilter_min_nonzero=1,
            ),
        ),
        combo_planning=BacktestComboPlanningService(
            config=BacktestComboPlanningConfig(combo_top_frac=1.0, combo_min_confirm=1),
        ),
        hit_times=BacktestTpSlHitTimesService(artifact_array_loader=array_loader),
        exact_measured=BacktestTpSlExactScoringService(
            config=BacktestTpSlExactConfig(
                benchmark_top_k=benchmark_top_k,
                run_self_check=self_check_n > 0,
                self_check_sample_size=self_check_n,
            ),
        ),
        exact_warmup=BacktestTpSlExactScoringService(
            config=BacktestTpSlExactConfig(
                benchmark_top_k=SAMPLE_WARMUP_TOP_K,
                run_self_check=self_check_n > 0,
                self_check_sample_size=min(1, self_check_n),
            ),
        ),
        exact_smoke=BacktestTpSlExactScoringService(
            config=BacktestTpSlExactConfig(
                benchmark_top_k=BENCHMARK_TOP_K,
                run_self_check=True,
                self_check_sample_size=1,
            ),
        ),
        artifact_metadata=artifact_metadata,
        array_loader=array_loader,
        context=context,
    )


def _run_matrix(
    *,
    canonical: Mapping[str, Any],
    services: _Services,
    rows_per_indicator: int,
    warmup_rows_per_indicator: int,
) -> list[dict[str, Any]]:
    request_payload = _required_mapping(canonical, "request")
    canonical_targets = _canonical_tp_sl_targets(canonical)
    prepared_cache: dict[int, BacktestPreparePoolsResult] = {}
    runs: list[dict[str, Any]] = []
    for direction_mode in TARGET_DIRECTION_MODES:
        for arity in TARGET_ARITIES:
            request = _service_request(
                canonical_request=request_payload,
                arity=arity,
                direction_mode=direction_mode,
            )
            hit_times_result = services.hit_times.execute(
                normalized_request=request,
                context=services.context,
            )
            prepared_full = prepared_cache.get(arity)
            if prepared_full is None:
                prepared_full = services.prepare_pools.execute(
                    normalized_request=request,
                    artifact_metadata=services.artifact_metadata,
                )
                prepared_cache[arity] = prepared_full
            warm_prepared = _limit_prepared_rows(
                prepared_full,
                rows_per_indicator=min(warmup_rows_per_indicator, rows_per_indicator),
            )
            measured_prepared = _limit_prepared_rows(
                prepared_full,
                rows_per_indicator=rows_per_indicator,
            )

            warm_start = time.perf_counter()
            warm_combo = services.combo_planning.execute(
                prepared_result=warm_prepared,
                normalized_request=request,
            )
            warm_result = services.exact_warmup.execute(
                prepared_result=warm_prepared,
                combo_planning_result=warm_combo,
                hit_times_result=hit_times_result,
                normalized_request=request,
            )
            sample_warmup_s = time.perf_counter() - warm_start

            cpu_start = time.process_time()
            rss_before = _maxrss_raw()
            combo_result = services.combo_planning.execute(
                prepared_result=measured_prepared,
                normalized_request=request,
            )
            exact_result = services.exact_measured.execute(
                prepared_result=measured_prepared,
                combo_planning_result=combo_result,
                hit_times_result=hit_times_result,
                normalized_request=request,
            )
            cpu_s = time.process_time() - cpu_start
            rss_after = _maxrss_raw()
            target = canonical_targets[(arity, direction_mode)]
            run = _run_payload(
                arity=arity,
                direction_mode=direction_mode,
                prepared_result=measured_prepared,
                combo_result=combo_result,
                exact_result=exact_result,
                warm_result=warm_result,
                hit_times_result=hit_times_result,
                sample_warmup_s=sample_warmup_s,
                target=target,
                runtime_metrics={
                    "process_cpu_time_s": cpu_s,
                    "maxrss_raw_before": rss_before,
                    "maxrss_raw_after": rss_after,
                },
            )
            runs.append(run)
            print(
                "arity={arity} direction={direction} exact_ratio={exact_ratio:.3f} "
                "heap_ratio={heap_ratio:.3f} pass={passed}".format(
                    arity=arity,
                    direction=direction_mode,
                    exact_ratio=run["ratios"][TP_SL_EXACT_SCORING_STAGE_NAME],
                    heap_ratio=run["ratios"][TP_SL_HEAP_UPDATE_STAGE_NAME],
                    passed=run["pass"]["overall"],
                )
            )
            del warm_result
            del exact_result
            del hit_times_result
            gc.collect()
    return runs


def _run_payload(
    *,
    arity: int,
    direction_mode: str,
    prepared_result: BacktestPreparePoolsResult,
    combo_result: Any,
    exact_result: Any,
    warm_result: Any,
    hit_times_result: Any,
    sample_warmup_s: float,
    target: Mapping[str, Any],
    runtime_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    exact_timings = dict(exact_result.telemetry.stage_timings)
    combo_timings = dict(combo_result.telemetry.stage_timings)
    hit_timings = dict(hit_times_result.timing.subsegments)
    exact_s = float(exact_timings[TP_SL_EXACT_SCORING_STAGE_NAME])
    full_metrics_s = float(exact_timings.get(TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME, 0.0))
    timers = {
        "sample_warmup": sample_warmup_s,
        LOAD_HIT_TIMES_STAGE_NAME: float(hit_timings[LOAD_HIT_TIMES_STAGE_NAME]),
        TP_SL_GRID_VALIDATION_STAGE_NAME: float(
            hit_timings[TP_SL_GRID_VALIDATION_STAGE_NAME]
        ),
        PREPARE_POOLS_CORE_STAGE_NAME: notebook_compatible_prepare_pools_core_s(
            prepared_result.timing
        ),
        BUILD_EXACT_CONTEXT_STAGE_NAME: float(combo_timings[BUILD_EXACT_CONTEXT_STAGE_NAME]),
        BUILD_PROXY_CONTEXT_STAGE_NAME: float(combo_timings[BUILD_PROXY_CONTEXT_STAGE_NAME]),
        COMBO_ITERATION_STAGE_NAME: float(combo_timings[COMBO_ITERATION_STAGE_NAME]),
        PROXY_FILTER_STAGE_NAME: float(combo_timings[PROXY_FILTER_STAGE_NAME]),
        TP_SL_SELF_CHECK_STAGE_NAME: float(exact_timings.get(TP_SL_SELF_CHECK_STAGE_NAME, 0.0)),
        TP_SL_EXACT_SCORING_STAGE_NAME: exact_s,
        TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME: exact_s,
        TP_SL_HEAP_UPDATE_STAGE_NAME: float(exact_timings[TP_SL_HEAP_UPDATE_STAGE_NAME]),
        "top_result_proxy_fill": 0.0,
        TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME: full_metrics_s,
    }
    service_total = math.fsum(
        timers[stage]
        for stage in (
            LOAD_HIT_TIMES_STAGE_NAME,
            TP_SL_GRID_VALIDATION_STAGE_NAME,
            PREPARE_POOLS_CORE_STAGE_NAME,
            BUILD_EXACT_CONTEXT_STAGE_NAME,
            BUILD_PROXY_CONTEXT_STAGE_NAME,
            COMBO_ITERATION_STAGE_NAME,
            PROXY_FILTER_STAGE_NAME,
            TP_SL_SELF_CHECK_STAGE_NAME,
            TP_SL_EXACT_SCORING_STAGE_NAME,
            TP_SL_HEAP_UPDATE_STAGE_NAME,
            TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME,
        )
    )
    timers["service_total_without_warmup"] = service_total
    accounting = build_benchmark_accounting_record(
        timers=timers,
        risk_mode=TARGET_RISK_MODE,
        request_top_n=REQUEST_TOP_N,
        benchmark_top_k=BENCHMARK_TOP_K,
        sample_warmup_top_k=SAMPLE_WARMUP_TOP_K,
        top_results_count=exact_result.telemetry.top_results_count,
        heap_capacity=exact_result.telemetry.heap_capacity,
    )
    ratios = {
        stage: _ratio(float(target["timers"].get(stage, 0.0)), float(timers[stage]))
        for stage in COMPARED_STAGES
        if stage in timers
    }
    stage_pass = {stage: ratios[stage] >= ACCEPTANCE_RATIO for stage in ratios}
    self_check_payload = exact_result.self_check.as_mapping()
    stage_pass[TP_SL_SELF_CHECK_STAGE_NAME] = (
        bool(stage_pass.get(TP_SL_SELF_CHECK_STAGE_NAME, False))
        and self_check_payload["status"] == TP_SL_SELF_CHECK_PASSED_STATUS
    )
    top_parity = _top_result_parity(
        service_top=exact_result.canonical_top_results_payload(),
        canonical_top=target["top_results"],
    )
    full_metrics_evidence = _full_metrics_evidence(exact_result=exact_result)
    cleanup_evidence = {
        "hit_times": hit_times_result.cleanup_evidence.as_mapping(),
        "tp_sl_exact": exact_result.memory_cleanup_evidence.as_mapping(),
        "pass": (
            not hit_times_result.cleanup_evidence.retained_hit_times_grid_arrays
            and not hit_times_result.cleanup_evidence.retained_hit_times_table_arrays
            and exact_result.memory_cleanup_evidence.result_is_compact
        ),
    }
    stage_pass["overall"] = all(stage_pass.values())
    return {
        "risk_mode": TARGET_RISK_MODE,
        "direction_mode": direction_mode,
        "arity": arity,
        "indicator_ids": list(prepared_result.indicator_ids),
        "backend": exact_result.telemetry.backend_logical_name,
        "backend_implementation_id": exact_result.telemetry.backend_implementation_id,
        "canonical_backend": target["backend"],
        "filtered_pool_sizes": {
            pool.indicator_id: int(pool.row_ids.shape[0])
            for pool in prepared_result.indicator_pools
        },
        "cartesian_combinations": combo_result.telemetry.cartesian_combinations,
        "warmup_cartesian_combinations": warm_result.telemetry.exact_candidates_evaluated,
        "exact_candidates_evaluated": exact_result.telemetry.exact_candidates_evaluated,
        "timers": timers,
        "accounting": accounting,
        "canonical_targets": target["timers"],
        "ratios": ratios,
        "stage_pass": stage_pass,
        "self_check": self_check_payload,
        "top_results": exact_result.canonical_top_results_payload(),
        "top_result_parity": top_parity,
        "full_metrics_evidence": full_metrics_evidence,
        "cleanup_evidence": cleanup_evidence,
        "runtime_metrics": dict(runtime_metrics),
        "hit_times_manifest_hash": hit_times_result.hit_times_manifest_hash,
        "hit_times_subset": hit_times_result.hit_times.compact_mapping(),
        "pass": {
            "stage": bool(stage_pass["overall"]),
            "top_result_parity": bool(top_parity["pass"]),
            "full_metrics": bool(full_metrics_evidence["pass"]),
            "cleanup": bool(cleanup_evidence["pass"]),
            "overall": bool(
                stage_pass["overall"]
                and top_parity["pass"]
                and full_metrics_evidence["pass"]
                and cleanup_evidence["pass"]
            ),
        },
    }


def _run_smokes(
    *,
    canonical: Mapping[str, Any],
    services: _Services,
    rows_per_indicator: int,
    self_check_n: int,
) -> list[dict[str, Any]]:
    request_payload = _required_mapping(canonical, "request")
    smokes: list[dict[str, Any]] = []
    for direction_mode in TARGET_DIRECTION_MODES:
        for arity in SMOKE_ARITIES:
            request = _service_request(
                canonical_request=request_payload,
                arity=arity,
                direction_mode=direction_mode,
            )
            hit_times_result = services.hit_times.execute(
                normalized_request=request,
                context=services.context,
            )
            prepared = services.prepare_pools.execute(
                normalized_request=request,
                artifact_metadata=services.artifact_metadata,
            )
            measured_prepared = _limit_prepared_rows(
                prepared,
                rows_per_indicator=rows_per_indicator,
            )
            combo_result = services.combo_planning.execute(
                prepared_result=measured_prepared,
                normalized_request=request,
            )
            exact_result = services.exact_smoke.execute(
                prepared_result=measured_prepared,
                combo_planning_result=combo_result,
                hit_times_result=hit_times_result,
                normalized_request=request,
            )
            full_metrics_evidence = _full_metrics_evidence(exact_result=exact_result)
            smokes.append(
                {
                    "arity": arity,
                    "direction_mode": direction_mode,
                    "rows_per_indicator": rows_per_indicator,
                    "self_check_n": self_check_n,
                    "exact_candidates_evaluated": exact_result.telemetry.exact_candidates_evaluated,
                    "self_check": exact_result.self_check.as_mapping(),
                    "top_results_count": exact_result.telemetry.top_results_count,
                    "full_metrics_evidence": full_metrics_evidence,
                    "cleanup_evidence": exact_result.memory_cleanup_evidence.as_mapping(),
                    "pass": bool(
                        exact_result.self_check.status == TP_SL_SELF_CHECK_PASSED_STATUS
                        and full_metrics_evidence["pass"]
                        and exact_result.memory_cleanup_evidence.result_is_compact
                    ),
                }
            )
            print(
                "smoke arity={arity} direction={direction} pass={passed}".format(
                    arity=arity,
                    direction=direction_mode,
                    passed=smokes[-1]["pass"],
                )
            )
            del exact_result
            del hit_times_result
            gc.collect()
    return smokes


def _build_payload(
    *,
    canonical: Mapping[str, Any],
    canonical_json: Path,
    services: _Services,
    runs: Sequence[Mapping[str, Any]],
    smokes: Sequence[Mapping[str, Any]],
    rows_per_indicator: int,
    warmup_rows_per_indicator: int,
    self_check_n: int,
) -> dict[str, Any]:
    canonical_artifact_hash = str(canonical.get("artifact_manifest_hash", ""))
    canonical_hit_times_hash = str(canonical.get("hit_times_manifest_hash", ""))
    artifact_manifest_hash_matches_canonical = (
        services.artifact_manifest_hash == canonical_artifact_hash
    )
    hit_times_manifest_hash_matches_canonical = (
        services.hit_times_manifest_hash == canonical_hit_times_hash
    )
    artifact_historical_prefix_compatible = True
    stage_pass = all(bool(run["pass"]["stage"]) for run in runs)
    top_parity_pass = all(bool(run["pass"]["top_result_parity"]) for run in runs)
    full_metrics_pass = all(bool(run["pass"]["full_metrics"]) for run in runs) and all(
        bool(smoke["full_metrics_evidence"]["pass"]) for smoke in smokes
    )
    cleanup_pass = all(bool(run["pass"]["cleanup"]) for run in runs) and all(
        bool(smoke["cleanup_evidence"]["result_is_compact"]) for smoke in smokes
    )
    smoke_pass = all(bool(smoke["pass"]) for smoke in smokes)
    return {
        "schema": "backtest_iteration_6_tp_sl_exact_scoring_full_metrics_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "canonical_json": str(canonical_json),
        "canonical_artifact_manifest_hash": canonical_artifact_hash,
        "canonical_hit_times_manifest_hash": canonical_hit_times_hash,
        "artifact_config_path": str(services.artifact_config_path),
        "artifact_root": str(services.artifact_root),
        "artifact_manifest_hash": services.artifact_manifest_hash,
        "hit_times_manifest_hash": services.hit_times_manifest_hash,
        "artifact_manifest_hash_matches_canonical": artifact_manifest_hash_matches_canonical,
        "hit_times_manifest_hash_matches_canonical": hit_times_manifest_hash_matches_canonical,
        "artifact_historical_prefix_compatible": artifact_historical_prefix_compatible,
        "artifact_policy": "historical_prefix_compatible",
        "scope": {
            "risk_mode": TARGET_RISK_MODE,
            "hit_times_path": HIT_TIMES_ARTIFACT_PATH_V2,
            "implemented_stages": [
                TP_SL_EXACT_SCORING_STAGE_NAME,
                TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME,
                TP_SL_HEAP_UPDATE_STAGE_NAME,
                TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME,
            ],
            "not_implemented_stages": [
                "persistence",
                "public_api",
                "variant_key",
                "variant_hash",
                "lazy_trades",
            ],
            "canonical_stage_comparison_only": True,
            "tp_sl_full_metrics_second_pass_service_only": True,
        },
        "request": {
            "top_n": REQUEST_TOP_N,
            "benchmark_top_k": BENCHMARK_TOP_K,
            "sample_warmup_top_k": SAMPLE_WARMUP_TOP_K,
            "risk": _target_risk_mapping(),
            "target_grid_literal": "2.0..25.0 step 0.5",
            "target_grid_cells": 47 * 47,
            "arities": list(TARGET_ARITIES),
            "smoke_arities": list(SMOKE_ARITIES),
            "direction_modes": list(TARGET_DIRECTION_MODES),
            "rows_per_indicator": rows_per_indicator,
            "warmup_rows_per_indicator": warmup_rows_per_indicator,
            "self_check_n": self_check_n,
        },
        "acceptance": {
            "ratio_threshold": ACCEPTANCE_RATIO,
            "ratio_definition": "canonical_stage_seconds / service_stage_seconds",
            "stage_boundaries_compared": list(COMPARED_STAGES),
            "total_without_warmup_components": list(
                runs[0]["accounting"]["total_component_stages"] if runs else []
            ),
            "service_only_telemetry_compared_to_notebook": False,
            "artifact_policy": "historical_prefix_compatible",
        },
        "runs": list(runs),
        "smoke_runs": list(smokes),
        "pass_breakdown": {
            "stage": stage_pass,
            "top_result_parity": top_parity_pass,
            "full_metrics": full_metrics_pass,
            "cleanup": cleanup_pass,
            "smoke": smoke_pass,
        },
        "pass": bool(
            stage_pass
            and top_parity_pass
            and full_metrics_pass
            and cleanup_pass
            and smoke_pass
            and artifact_historical_prefix_compatible
        ),
    }


def _limit_prepared_rows(
    prepared_result: BacktestPreparePoolsResult,
    *,
    rows_per_indicator: int,
) -> BacktestPreparePoolsResult:
    pools: list[PreparedIndicatorPool] = []
    for pool in prepared_result.indicator_pools:
        if int(pool.row_ids.shape[0]) < rows_per_indicator:
            raise ValueError(
                f"pool {pool.indicator_id!r} has {int(pool.row_ids.shape[0])} rows; "
                f"cannot select {rows_per_indicator}"
            )
        row_slice = slice(0, rows_per_indicator)
        trade_t = np.ascontiguousarray(pool.trade_T[row_slice])
        change_count = np.ascontiguousarray(pool.change_count[row_slice])
        pools.append(
            PreparedIndicatorPool(
                indicator_id=pool.indicator_id,
                row_ids=np.ascontiguousarray(pool.row_ids[row_slice]),
                filtered_row_ids=np.ascontiguousarray(pool.filtered_row_ids[row_slice]),
                trade_T=trade_t,
                eval_T=np.ascontiguousarray(pool.eval_T[row_slice]),
                segments=build_signal_segments(trade_t, change_count=change_count),
                row_score=np.ascontiguousarray(pool.row_score[row_slice]),
                score_adj=np.ascontiguousarray(pool.score_adj[row_slice]),
                nonzero=np.ascontiguousarray(pool.nonzero[row_slice]),
                proxy=np.ascontiguousarray(pool.proxy[row_slice]),
                change_count=change_count,
                metadata=pool.metadata[row_slice],
            )
        )
    limited_pools = tuple(pools)
    return replace(
        prepared_result,
        indicator_pools=limited_pools,
        row_metadata_order_hash=row_metadata_order_hash(limited_pools),
    )


def _service_request(
    *,
    canonical_request: Mapping[str, Any],
    arity: int,
    direction_mode: str,
) -> dict[str, Any]:
    indicators = []
    for raw_indicator in _required_sequence(canonical_request, "indicators")[:arity]:
        indicator = _require_mapping_value(raw_indicator, "request.indicators[]")
        start, stop = _window_bounds(indicator)
        indicators.append(
            {
                "indicator_id": str(indicator["indicator_id"]),
                "sources": list(_required_sequence(indicator, "sources")),
                "window": {"start": start, "stop": stop, "step": 1},
            }
        )
    execution = dict(_required_mapping(canonical_request, "execution"))
    execution["direction_mode"] = direction_mode
    return {
        "coordinates": dict(_required_mapping(canonical_request, "coordinates")),
        "timeframe": str(canonical_request["timeframe"]),
        "time_range": _time_range_from_canonical(canonical_request),
        "indicators": indicators,
        "risk": _target_risk_mapping(),
        "execution": execution,
        "ranking": {
            "primary_metric": str(canonical_request.get("sort_metric", "total_return_pct")),
            "direction": "desc",
        },
        "top_n": int(canonical_request.get("top_n", REQUEST_TOP_N)),
    }


def _target_risk_mapping() -> dict[str, Any]:
    return {
        "mode": TARGET_RISK_MODE,
        "tp": {
            "start_pct": TARGET_TP_SL_GRID_START_PCT,
            "stop_pct": TARGET_TP_SL_GRID_STOP_PCT,
            "step_pct": TARGET_TP_SL_GRID_STEP_PCT,
        },
        "sl": {
            "start_pct": TARGET_TP_SL_GRID_START_PCT,
            "stop_pct": TARGET_TP_SL_GRID_STOP_PCT,
            "step_pct": TARGET_TP_SL_GRID_STEP_PCT,
        },
    }


def _canonical_tp_sl_targets(
    canonical: Mapping[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    targets: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_run in _required_sequence(canonical, "runs"):
        run = _require_mapping_value(raw_run, "runs[]")
        if run.get("risk_mode") != TARGET_RISK_MODE:
            continue
        indicator_ids = _required_sequence(run, "indicator_ids")
        arity = len(indicator_ids)
        direction_mode = str(run.get("direction_mode", ""))
        if arity not in TARGET_ARITIES or direction_mode not in TARGET_DIRECTION_MODES:
            continue
        timers = dict(_required_mapping(run, "timers"))
        timers.setdefault(TP_SL_EXACT_SCORING_ALIAS_STAGE_NAME, timers["exact_scoring"])
        timers.setdefault(PREPARE_POOLS_CORE_STAGE_NAME, timers.get("prepare_pools", 0.0))
        targets[(arity, direction_mode)] = {
            "indicator_ids": tuple(str(value) for value in indicator_ids),
            "backend": str(run["exact_engine"]),
            "timers": {str(key): float(value) for key, value in timers.items()},
            "top_results": list(_required_sequence(run, "top_results")),
        }
    missing = [
        (arity, direction_mode)
        for direction_mode in TARGET_DIRECTION_MODES
        for arity in TARGET_ARITIES
        if (arity, direction_mode) not in targets
    ]
    if missing:
        raise ValueError(f"missing canonical TP/SL targets: {missing!r}")
    return targets


def _top_result_parity(
    *,
    service_top: Sequence[Mapping[str, Any]],
    canonical_top: Sequence[Any],
) -> dict[str, Any]:
    compared = min(len(service_top), len(canonical_top), BENCHMARK_TOP_K)
    mismatches: list[dict[str, Any]] = []
    max_abs_return_diff = 0.0
    for index in range(compared):
        service = _require_mapping_value(service_top[index], "service_top[]")
        canonical = _require_mapping_value(canonical_top[index], "canonical_top[]")
        return_diff = abs(
            float(service["total_return_pct"]) - float(canonical["total_return_pct"])
        )
        max_abs_return_diff = max(max_abs_return_diff, return_diff)
        fields_equal = (
            return_diff <= 5e-4
            and int(service["trade_count"]) == int(canonical["trade_count"])
            and abs(float(service["best_tp_pct"]) - float(canonical["best_tp_pct"])) <= 5e-6
            and abs(float(service["best_sl_pct"]) - float(canonical["best_sl_pct"])) <= 5e-6
        )
        if not fields_equal:
            mismatches.append(
                {
                    "rank": index + 1,
                    "service": dict(service),
                    "canonical": dict(canonical),
                    "return_diff": return_diff,
                }
            )
    return {
        "compared": compared,
        "max_abs_return_diff": max_abs_return_diff,
        "mismatches": mismatches[:5],
        "pass": compared == BENCHMARK_TOP_K and not mismatches,
    }


def _full_metrics_evidence(*, exact_result: Any) -> dict[str, Any]:
    missing_by_rank: list[dict[str, Any]] = []
    max_abs_total_diff = 0.0
    for top_result in exact_result.top_results:
        metrics = dict(top_result.metrics)
        missing = [name for name in FULL_METRIC_NAMES if name not in metrics]
        if missing:
            missing_by_rank.append({"rank": top_result.rank, "missing": missing})
        max_abs_total_diff = max(
            max_abs_total_diff,
            abs(float(metrics["total_return_pct"]) - float(top_result.score)),
        )
    return {
        "required_metric_names": list(FULL_METRIC_NAMES),
        "top_results_checked": len(exact_result.top_results),
        "missing_by_rank": missing_by_rank,
        "max_abs_total_return_vs_score_diff": max_abs_total_diff,
        "stage": TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME,
        "service_only": True,
        "pass": not missing_by_rank and max_abs_total_diff <= 1e-9,
    }


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    pass_breakdown = _required_mapping(payload, "pass_breakdown")
    failed_rows = _failed_rows(payload)
    lines = [
        "# Iteration 6 TP/SL exact scoring and full metrics",
        "",
        "## Scope",
        "",
        "- Implemented: `event_segments_n_tp_sl_15m_grid`, TP/SL self-check, "
        "`heap_update`, and `tp_sl_full_metrics_second_pass`.",
        "- Not implemented: persistence, public/storage identity, API read models, "
        "lazy trades.",
        f"- Runtime target path: `{payload['scope']['hit_times_path']}`.",
        "- `benchmark_top_k = 5`; `request.top_n = 100` is recorded separately.",
        "- `tp_sl_full_metrics_second_pass` is service-only and excluded from "
        "`total_without_warmup`.",
        "",
        "## Environment",
        "",
        f"- Host: `{payload['host']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Artifact config: `{payload['artifact_config_path']}`",
        f"- Artifact root: `{payload['artifact_root']}`",
        f"- Artifact manifest hash: `{payload['artifact_manifest_hash']}`",
        f"- Hit-times manifest hash: `{payload['hit_times_manifest_hash']}`",
        "- Artifact hash matches canonical: "
        f"`{payload['artifact_manifest_hash_matches_canonical']}`",
        "- Hit-times hash matches canonical: "
        f"`{payload['hit_times_manifest_hash_matches_canonical']}`",
        f"- Artifact policy: `{payload['artifact_policy']}`",
        "- Artifact historical-prefix compatible: "
        f"`{payload['artifact_historical_prefix_compatible']}`",
        "",
        "## Failed Rows",
        "",
    ]
    if failed_rows:
        lines.append(
            "| arity | direction_mode | stage | canonical_s | service_s | ratio | reason |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---|")
        for row in failed_rows[:20]:
            lines.append(
                "| {arity} | `{direction}` | `{stage}` | {canonical:.6f} | "
                "{service:.6f} | {ratio:.3f} | {reason} |".format(
                    arity=row["arity"],
                    direction=row["direction_mode"],
                    stage=row["stage"],
                    canonical=row["canonical_s"],
                    service=row["service_s"],
                    ratio=row["ratio"],
                    reason=row["reason"],
                )
            )
    else:
        lines.append("- No failed benchmark rows.")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| arity | direction_mode | exact_s | canonical_exact_s | exact_ratio | "
            "heap_s | canonical_heap_s | heap_ratio | full_metrics_s | pass |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in payload["runs"]:
        timers = run["timers"]
        targets = run["canonical_targets"]
        ratios = run["ratios"]
        lines.append(
            "| {arity} | `{direction}` | {exact:.6f} | {target_exact:.6f} | "
            "{exact_ratio:.3f} | {heap:.6f} | {target_heap:.6f} | "
            "{heap_ratio:.3f} | {full:.6f} | `{passed}` |".format(
                arity=run["arity"],
                direction=run["direction_mode"],
                exact=timers[TP_SL_EXACT_SCORING_STAGE_NAME],
                target_exact=targets[TP_SL_EXACT_SCORING_STAGE_NAME],
                exact_ratio=ratios[TP_SL_EXACT_SCORING_STAGE_NAME],
                heap=timers[TP_SL_HEAP_UPDATE_STAGE_NAME],
                target_heap=targets[TP_SL_HEAP_UPDATE_STAGE_NAME],
                heap_ratio=ratios[TP_SL_HEAP_UPDATE_STAGE_NAME],
                full=timers[TP_SL_FULL_METRICS_SECOND_PASS_STAGE_NAME],
                passed="yes" if run["pass"]["overall"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Smoke 8..10",
            "",
            "| arity | direction_mode | candidates | self_check | full_metrics | pass |",
            "|---:|---|---:|---|---|---|",
        ]
    )
    for smoke in payload["smoke_runs"]:
        lines.append(
            "| {arity} | `{direction}` | {candidates} | `{self_check}` | `{full}` | "
            "`{passed}` |".format(
                arity=smoke["arity"],
                direction=smoke["direction_mode"],
                candidates=smoke["exact_candidates_evaluated"],
                self_check=smoke["self_check"]["status"],
                full="yes" if smoke["full_metrics_evidence"]["pass"] else "no",
                passed="yes" if smoke["pass"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Stage pass: `{'yes' if pass_breakdown['stage'] else 'no'}`",
            "- Top-result parity pass: "
            f"`{'yes' if pass_breakdown['top_result_parity'] else 'no'}`",
            f"- Full metrics pass: `{'yes' if pass_breakdown['full_metrics'] else 'no'}`",
            f"- Cleanup pass: `{'yes' if pass_breakdown['cleanup'] else 'no'}`",
            f"- Smoke pass: `{'yes' if pass_breakdown['smoke'] else 'no'}`",
            f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _failed_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for stage, passed in run["stage_pass"].items():
            if stage == "overall" or passed:
                continue
            rows.append(
                {
                    "arity": run["arity"],
                    "direction_mode": run["direction_mode"],
                    "stage": stage,
                    "canonical_s": float(run["canonical_targets"].get(stage, 0.0)),
                    "service_s": float(run["timers"].get(stage, 0.0)),
                    "ratio": float(run["ratios"].get(stage, 0.0)),
                    "reason": "stage_ratio",
                }
            )
        if not run["top_result_parity"]["pass"]:
            rows.append(
                {
                    "arity": run["arity"],
                    "direction_mode": run["direction_mode"],
                    "stage": "top_result_parity",
                    "canonical_s": 0.0,
                    "service_s": 0.0,
                    "ratio": 0.0,
                    "reason": "top_result_parity",
                }
            )
    return sorted(rows, key=lambda item: (item["ratio"], item["arity"]))


def _print_slowest_failures(payload: Mapping[str, Any]) -> None:
    failed = _failed_rows(payload)
    if not failed:
        return
    print("failed rows:")
    for row in failed[:10]:
        print(
            "{arity} {direction_mode} {stage}: canonical={canonical_s:.6f}s "
            "service={service_s:.6f}s ratio={ratio:.3f} reason={reason}".format(**row)
        )


def _time_range_from_canonical(canonical_request: Mapping[str, Any]) -> dict[str, str]:
    if isinstance(canonical_request.get("time_range"), Mapping):
        raw = _required_mapping(canonical_request, "time_range")
        return {"start": str(raw["start"]), "end": str(raw["end"])}
    if isinstance(canonical_request.get("period"), Mapping):
        raw = _required_mapping(canonical_request, "period")
        return {"start": str(raw["start"]), "end": str(raw["end"])}
    raise ValueError("canonical request must include time_range or period")


def _window_bounds(indicator: Mapping[str, Any]) -> tuple[int, int]:
    if "window" in indicator and isinstance(indicator["window"], Mapping):
        window = _required_mapping(indicator, "window")
        return int(window["start"]), int(window["stop"])
    if "window_range" in indicator:
        values = _required_sequence(indicator, "window_range")
        return int(values[0]), int(values[1])
    raise ValueError("indicator must include window or window_range")


def _ratio(canonical_s: float, service_s: float) -> float:
    if service_s <= 0.0:
        return float("inf")
    return canonical_s / service_s


def _maxrss_raw() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key!r} must be a mapping")
    return value


def _required_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{key!r} must be a sequence")
    return value


def _require_mapping_value(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _git_commit() -> str:
    env_commit = os.environ.get("ROEHUB_BENCHMARK_GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_status_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
