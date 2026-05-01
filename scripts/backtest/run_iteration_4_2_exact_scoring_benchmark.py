from __future__ import annotations

import argparse
import json
import math
import os
import platform
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
    BacktestNoRiskExactConfig,
    BacktestPreparePoolsConfig,
    BacktestPreparePoolsResult,
    PreparedIndicatorPool,
)
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    BUILD_EXACT_CONTEXT_STAGE_NAME,
    BUILD_PROXY_CONTEXT_STAGE_NAME,
    COMBO_ITERATION_STAGE_NAME,
    NO_RISK_EXACT_SCORING_STAGE_NAME,
    NO_RISK_SELF_CHECK_PASSED_STATUS,
    NO_RISK_SELF_CHECK_STAGE_NAME,
    PROXY_FILTER_STAGE_NAME,
    BacktestComboPlanningService,
    BacktestNoRiskExactScoringService,
    BacktestPreparePoolsService,
    build_signal_segments,
    notebook_compatible_prepare_pools_core_s,
    row_metadata_order_hash,
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
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_4_2_exact_scoring_self_check"
)
TARGET_RISK_MODE = "none"
TARGET_ARITIES = tuple(range(1, 8))
TARGET_DIRECTION_MODES = ("long_only", "long_short_reversal")
ACCEPTANCE_RATIO = 0.9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 4.2 exact-only no-risk service benchmark against the "
            "canonical notebook stage targets."
        )
    )
    parser.add_argument(
        "--canonical-json",
        type=Path,
        default=DEFAULT_CANONICAL_JSON,
        help="Canonical notebook benchmark_results.json path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Evidence directory for benchmark_results.json and benchmark_summary.md.",
    )
    parser.add_argument(
        "--artifact-config",
        type=Path,
        default=None,
        help=(
            "Optional backtest_artifacts.yaml path. Defaults to "
            "ROEHUB_BACKTEST_ARTIFACTS_CONFIG, then ROEHUB_ENV=prod."
        ),
    )
    parser.add_argument(
        "--rows-per-indicator",
        type=int,
        default=6,
        help="Measured rows per indicator. Canonical target uses 6.",
    )
    parser.add_argument(
        "--warmup-rows-per-indicator",
        type=int,
        default=2,
        help="Warmup rows per indicator. Canonical target uses 2.",
    )
    parser.add_argument(
        "--self-check-n",
        type=int,
        default=2,
        help="Measured self-check sample size. Canonical target uses 2.",
    )
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

    canonical = _load_json(args.canonical_json)
    benchmark_top_k = 5
    services = _build_services(
        artifact_config_path=args.artifact_config,
        benchmark_top_k=benchmark_top_k,
        self_check_n=args.self_check_n,
    )
    runs = _run_matrix(
        canonical=canonical,
        services=services,
        rows_per_indicator=args.rows_per_indicator,
        warmup_rows_per_indicator=args.warmup_rows_per_indicator,
    )
    payload = _build_payload(
        canonical=canonical,
        canonical_json=args.canonical_json,
        services=services,
        runs=runs,
        rows_per_indicator=args.rows_per_indicator,
        warmup_rows_per_indicator=args.warmup_rows_per_indicator,
        self_check_n=args.self_check_n,
        benchmark_top_k=benchmark_top_k,
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
    return 1


class _Services:
    def __init__(
        self,
        *,
        artifact_config_path: Path,
        artifact_root: Path,
        artifact_manifest_hash: str,
        prepare_pools: BacktestPreparePoolsService,
        combo_planning: BacktestComboPlanningService,
        exact_measured: BacktestNoRiskExactScoringService,
        exact_warmup: BacktestNoRiskExactScoringService,
        artifact_metadata: Any,
    ) -> None:
        self.artifact_config_path = artifact_config_path
        self.artifact_root = artifact_root
        self.artifact_manifest_hash = artifact_manifest_hash
        self.prepare_pools = prepare_pools
        self.combo_planning = combo_planning
        self.exact_measured = exact_measured
        self.exact_warmup = exact_warmup
        self.artifact_metadata = artifact_metadata


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
    return _Services(
        artifact_config_path=resolved_config_path,
        artifact_root=artifact_config.artifact_root_path(),
        artifact_manifest_hash=artifact_metadata.artifact_manifest_hash,
        prepare_pools=BacktestPreparePoolsService(
            artifact_array_loader=FilesystemBacktestArtifactArrayLoader(
                artifact_loader=artifact_loader
            ),
            defaults_provider=defaults_provider,
            config=BacktestPreparePoolsConfig(
                row_prefilter_top_fraction=1.0,
                row_prefilter_min_nonzero=1,
            ),
        ),
        combo_planning=BacktestComboPlanningService(
            config=BacktestComboPlanningConfig(
                combo_top_frac=1.0,
                combo_min_confirm=1,
            ),
        ),
        exact_measured=BacktestNoRiskExactScoringService(
            config=BacktestNoRiskExactConfig(
                benchmark_top_k=benchmark_top_k,
                run_self_check=self_check_n > 0,
                self_check_sample_size=self_check_n,
            ),
        ),
        exact_warmup=BacktestNoRiskExactScoringService(
            config=BacktestNoRiskExactConfig(
                benchmark_top_k=1,
                run_self_check=self_check_n > 0,
                self_check_sample_size=min(1, self_check_n),
            ),
        ),
        artifact_metadata=artifact_metadata,
    )


def _run_matrix(
    *,
    canonical: Mapping[str, Any],
    services: _Services,
    rows_per_indicator: int,
    warmup_rows_per_indicator: int,
) -> list[dict[str, Any]]:
    request_payload = _required_mapping(canonical, "request")
    canonical_targets = _canonical_no_risk_targets(canonical)
    prepared_cache: dict[int, BacktestPreparePoolsResult] = {}
    runs: list[dict[str, Any]] = []

    for direction_mode in TARGET_DIRECTION_MODES:
        for arity in TARGET_ARITIES:
            request = _service_request(
                canonical_request=request_payload,
                arity=arity,
                direction_mode=direction_mode,
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
                normalized_request=request,
            )
            sample_warmup_s = time.perf_counter() - warm_start

            combo_result = services.combo_planning.execute(
                prepared_result=measured_prepared,
                normalized_request=request,
            )
            exact_result = services.exact_measured.execute(
                prepared_result=measured_prepared,
                combo_planning_result=combo_result,
                normalized_request=request,
            )
            target = canonical_targets[(arity, direction_mode)]
            run = _run_payload(
                arity=arity,
                direction_mode=direction_mode,
                prepared_result=measured_prepared,
                combo_result=combo_result,
                exact_result=exact_result,
                warm_result=warm_result,
                sample_warmup_s=sample_warmup_s,
                target=target,
            )
            runs.append(run)
            print(
                "arity={arity} direction={direction} exact_ratio={exact_ratio:.3f} "
                "self_ratio={self_ratio:.3f} self_check={self_check}".format(
                    arity=arity,
                    direction=direction_mode,
                    exact_ratio=run["ratios"]["exact_scoring"],
                    self_ratio=run["ratios"]["self_check"],
                    self_check=run["self_check"]["status"],
                )
            )

    return runs


def _run_payload(
    *,
    arity: int,
    direction_mode: str,
    prepared_result: BacktestPreparePoolsResult,
    combo_result: Any,
    exact_result: Any,
    warm_result: Any,
    sample_warmup_s: float,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    stage_timings = dict(exact_result.telemetry.stage_timings)
    combo_timings = dict(combo_result.telemetry.stage_timings)
    exact_s = float(stage_timings[NO_RISK_EXACT_SCORING_STAGE_NAME])
    self_check_s = float(stage_timings.get(NO_RISK_SELF_CHECK_STAGE_NAME, 0.0))
    target_exact_s = float(target["exact_scoring_s"])
    target_self_check_s = float(target["self_check_s"])
    exact_ratio = _ratio(target_exact_s, exact_s)
    self_check_ratio = _ratio(target_self_check_s, self_check_s)
    self_check_payload = exact_result.self_check.as_mapping()
    pass_payload = {
        "exact_scoring": exact_ratio >= ACCEPTANCE_RATIO,
        "self_check": (
            self_check_ratio >= ACCEPTANCE_RATIO
            and self_check_payload["status"] == NO_RISK_SELF_CHECK_PASSED_STATUS
        ),
    }
    pass_payload["overall"] = all(pass_payload.values())
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
        "request_slice": {
            "time_slice_start_15m": prepared_result.time_slice_start_15m,
            "time_slice_stop_15m": prepared_result.time_slice_stop_15m,
            "trade_T_length": prepared_result.trade_T_length,
            "eval_T_length": prepared_result.eval_T_length,
            "row_identity": {
                pool.indicator_id: [metadata.as_mapping() for metadata in pool.metadata]
                for pool in prepared_result.indicator_pools
            },
        },
        "cartesian_combinations": combo_result.telemetry.cartesian_combinations,
        "warmup_cartesian_combinations": warm_result.telemetry.exact_candidates_evaluated,
        "exact_candidates_evaluated": exact_result.telemetry.exact_candidates_evaluated,
        "timers": {
            "sample_warmup": sample_warmup_s,
            "prepare_pools_core": notebook_compatible_prepare_pools_core_s(
                prepared_result.timing
            ),
            BUILD_EXACT_CONTEXT_STAGE_NAME: combo_timings[BUILD_EXACT_CONTEXT_STAGE_NAME],
            BUILD_PROXY_CONTEXT_STAGE_NAME: combo_timings[BUILD_PROXY_CONTEXT_STAGE_NAME],
            COMBO_ITERATION_STAGE_NAME: combo_timings[COMBO_ITERATION_STAGE_NAME],
            PROXY_FILTER_STAGE_NAME: combo_timings[PROXY_FILTER_STAGE_NAME],
            NO_RISK_SELF_CHECK_STAGE_NAME: self_check_s,
            NO_RISK_EXACT_SCORING_STAGE_NAME: exact_s,
        },
        "canonical_targets": {
            "exact_scoring": target_exact_s,
            "self_check": target_self_check_s,
        },
        "ratios": {
            "exact_scoring": exact_ratio,
            "self_check": self_check_ratio,
        },
        "pass": pass_payload,
        "self_check": self_check_payload,
        "sample_metrics": None
        if exact_result.telemetry.sample_metrics is None
        else dict(exact_result.telemetry.sample_metrics),
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
        "risk": {"mode": TARGET_RISK_MODE},
        "execution": execution,
        "ranking": {
            "primary_metric": str(canonical_request.get("sort_metric", "total_return_pct")),
            "direction": "desc",
        },
        "top_n": int(canonical_request.get("top_n", 100)),
    }


def _canonical_no_risk_targets(
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
        timers = _required_mapping(run, "timers")
        targets[(arity, direction_mode)] = {
            "indicator_ids": tuple(str(value) for value in indicator_ids),
            "backend": str(run["exact_engine"]),
            "exact_scoring_s": float(timers["exact_scoring"]),
            "self_check_s": float(timers["self_check"]),
        }

    missing = [
        (arity, direction_mode)
        for direction_mode in TARGET_DIRECTION_MODES
        for arity in TARGET_ARITIES
        if (arity, direction_mode) not in targets
    ]
    if missing:
        raise ValueError(f"canonical JSON is missing no-risk target(s): {missing!r}")
    return targets


def _build_payload(
    *,
    canonical: Mapping[str, Any],
    canonical_json: Path,
    services: _Services,
    runs: Sequence[Mapping[str, Any]],
    rows_per_indicator: int,
    warmup_rows_per_indicator: int,
    self_check_n: int,
    benchmark_top_k: int,
) -> dict[str, Any]:
    canonical_request = _required_mapping(canonical, "request")
    artifact_manifest_hash_matches_canonical = (
        services.artifact_manifest_hash == str(canonical.get("artifact_manifest_hash", ""))
    )
    stage_pass = all(bool(run["pass"]["overall"]) for run in runs)
    request_slice_identity_pass = _request_slice_identity_pass(
        canonical_request=canonical_request,
        runs=runs,
        rows_per_indicator=rows_per_indicator,
    )
    artifact_historical_prefix_compatible = (
        artifact_manifest_hash_matches_canonical or request_slice_identity_pass
    )
    return {
        "schema": "backtest_iteration_4_2_exact_scoring_self_check_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "canonical_json": str(canonical_json),
        "canonical_artifact_manifest_hash": str(
            canonical.get("artifact_manifest_hash", "")
        ),
        "artifact_config_path": str(services.artifact_config_path),
        "artifact_root": str(services.artifact_root),
        "artifact_manifest_hash": services.artifact_manifest_hash,
        "artifact_manifest_hash_matches_canonical": artifact_manifest_hash_matches_canonical,
        "artifact_historical_prefix_compatible": artifact_historical_prefix_compatible,
        "artifact_acceptance": {
            "policy": "historical_prefix_compatible",
            "full_manifest_hash_match_required": False,
            "historical_prefix_invariant": "artifact historical-prefix invariant",
            "compatibility_evidence": (
                "full manifest hash match"
                if artifact_manifest_hash_matches_canonical
                else (
                    "canonical request-slice row identity and 15m length matched "
                    "for all 14 exact/self-check runs"
                )
            ),
        },
        "scope": {
            "risk_mode": TARGET_RISK_MODE,
            "stage_scope": "exact_only",
            "implemented_stages": [
                NO_RISK_SELF_CHECK_STAGE_NAME,
                NO_RISK_EXACT_SCORING_STAGE_NAME,
            ],
            "not_implemented_stages": [
                "heap_update",
                "top_result_proxy_fill",
                "result_hash_normalization",
                "persistence",
                "public_api_identity",
            ],
        },
        "request": {
            "top_n": int(canonical_request.get("top_n", 100)),
            "benchmark_top_k": benchmark_top_k,
            "rows_per_indicator": rows_per_indicator,
            "warmup_rows_per_indicator": warmup_rows_per_indicator,
            "self_check_n": self_check_n,
            "expected_trade_T_length": _expected_trade_t_length(canonical_request),
            "request_slice_identity_pass": request_slice_identity_pass,
        },
        "acceptance": {
            "ratio_threshold": ACCEPTANCE_RATIO,
            "ratio_definition": "canonical_stage_seconds / service_stage_seconds",
            "stage_boundaries_compared": [
                NO_RISK_SELF_CHECK_STAGE_NAME,
                NO_RISK_EXACT_SCORING_STAGE_NAME,
            ],
            "requires_artifact_manifest_hash_match": False,
            "requires_artifact_historical_prefix_compatibility": True,
            "artifact_policy": "historical_prefix_compatible",
        },
        "runs": list(runs),
        "stage_pass": stage_pass,
        "request_slice_identity_pass": request_slice_identity_pass,
        "pass": stage_pass and artifact_historical_prefix_compatible,
    }


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    lines = [
        "# Iteration 4.2 exact scoring self-check benchmark",
        "",
        "## Scope",
        "",
        "- Compared stages: `self_check`, `exact_scoring`.",
        "- Not compared: `heap_update`, `top_result_proxy_fill`, persistence, public API identity.",
        f"- Acceptance ratio threshold: `{payload['acceptance']['ratio_threshold']}`.",
        "- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.",
        "- `benchmark_top_k = 5` is telemetry only.",
        "",
        "## Environment",
        "",
        f"- Host: `{payload['host']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Artifact config: `{payload['artifact_config_path']}`",
        f"- Artifact manifest hash: `{payload['artifact_manifest_hash']}`",
        "- Artifact hash matches canonical: "
        f"`{payload['artifact_manifest_hash_matches_canonical']}`",
        f"- Artifact policy: `{payload['artifact_acceptance']['policy']}`",
        "- Artifact historical-prefix compatible: "
        f"`{payload['artifact_historical_prefix_compatible']}`",
        "- Artifact compatibility evidence: "
        f"`{payload['artifact_acceptance']['compatibility_evidence']}`",
        "",
        "## Results",
        "",
        "| arity | direction_mode | backend | exact_s | canonical_exact_s | exact_ratio | "
        "self_check_s | canonical_self_check_s | self_ratio | self_check | pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in payload["runs"]:
        timers = run["timers"]
        targets = run["canonical_targets"]
        ratios = run["ratios"]
        lines.append(
            "| {arity} | `{direction}` | `{backend}` | {exact:.6f} | "
            "{target_exact:.6f} | {exact_ratio:.3f} | {self_check:.6f} | "
            "{target_self:.6f} | {self_ratio:.3f} | `{check}` | `{passed}` |".format(
                arity=run["arity"],
                direction=run["direction_mode"],
                backend=run["backend"],
                exact=timers[NO_RISK_EXACT_SCORING_STAGE_NAME],
                target_exact=targets[NO_RISK_EXACT_SCORING_STAGE_NAME],
                exact_ratio=ratios[NO_RISK_EXACT_SCORING_STAGE_NAME],
                self_check=timers[NO_RISK_SELF_CHECK_STAGE_NAME],
                target_self=targets[NO_RISK_SELF_CHECK_STAGE_NAME],
                self_ratio=ratios[NO_RISK_SELF_CHECK_STAGE_NAME],
                check=run["self_check"]["status"],
                passed="yes" if run["pass"]["overall"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Stage pass: `{'yes' if payload['stage_pass'] else 'no'}`",
            "- Request-slice identity pass: "
            f"`{'yes' if payload['request_slice_identity_pass'] else 'no'}`",
            "- Artifact hash matches canonical: "
            f"`{payload['artifact_manifest_hash_matches_canonical']}`",
            "- Artifact historical-prefix compatible: "
            f"`{payload['artifact_historical_prefix_compatible']}`",
            f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _ratio(canonical_seconds: float, service_seconds: float) -> float:
    if service_seconds <= 0.0:
        return math.inf
    return canonical_seconds / service_seconds


def _time_range_from_canonical(canonical_request: Mapping[str, Any]) -> dict[str, str]:
    period = _required_mapping(canonical_request, "period")
    return {
        "start": str(period["start"]).replace(" ", "T"),
        "end": str(period["end"]).replace(" ", "T"),
    }


def _request_slice_identity_pass(
    *,
    canonical_request: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rows_per_indicator: int,
) -> bool:
    expected_trade_t_length = _expected_trade_t_length(canonical_request)
    expected_eval_t_length = expected_trade_t_length - 1
    expected_windows = _expected_windows(
        canonical_request=canonical_request,
        rows_per_indicator=rows_per_indicator,
    )
    expected_row_ids = tuple(range(rows_per_indicator))
    for run in runs:
        request_slice = _required_mapping(run, "request_slice")
        if int(request_slice["trade_T_length"]) != expected_trade_t_length:
            return False
        if int(request_slice["eval_T_length"]) != expected_eval_t_length:
            return False
        row_identity = _required_mapping(request_slice, "row_identity")
        for indicator_id in _required_sequence(run, "indicator_ids"):
            raw_rows = row_identity.get(str(indicator_id))
            if not isinstance(raw_rows, Sequence) or isinstance(
                raw_rows,
                (str, bytes, bytearray),
            ):
                return False
            if len(raw_rows) != rows_per_indicator:
                return False
            for offset, raw_row in enumerate(raw_rows):
                row = _require_mapping_value(raw_row, "request_slice.row_identity[]")
                if int(row.get("row_id", -1)) != expected_row_ids[offset]:
                    return False
                if str(row.get("source", "")).strip().lower() != "close":
                    return False
                if int(row.get("window", -1)) != expected_windows[offset]:
                    return False
    return True


def _expected_trade_t_length(canonical_request: Mapping[str, Any]) -> int:
    time_range = _time_range_from_canonical(canonical_request)
    start = _parse_datetime(time_range["start"])
    end = _parse_datetime(time_range["end"])
    seconds = (end - start).total_seconds()
    if seconds <= 0 or seconds % 900 != 0:
        raise ValueError("canonical request period must align to 15m bars")
    return int(seconds // 900)


def _expected_windows(
    *,
    canonical_request: Mapping[str, Any],
    rows_per_indicator: int,
) -> tuple[int, ...]:
    first_indicator = _require_mapping_value(
        _required_sequence(canonical_request, "indicators")[0],
        "request.indicators[0]",
    )
    start, _stop = _window_bounds(first_indicator)
    return tuple(range(start, start + rows_per_indicator))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _window_bounds(indicator: Mapping[str, Any]) -> tuple[int, int]:
    raw_window_range = indicator.get("window_range")
    if not isinstance(raw_window_range, Sequence) or isinstance(
        raw_window_range,
        (str, bytes, bytearray),
    ):
        raise ValueError("canonical indicator.window_range must be a sequence")
    if len(raw_window_range) != 2:
        raise ValueError("canonical indicator.window_range must have two items")
    return int(raw_window_range[0]), int(raw_window_range[1])


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping field {key!r}")
    return value


def _required_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"expected sequence field {key!r}")
    return value


def _require_mapping_value(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping value at {path}")
    return value


def _git_commit() -> str:
    env_commit = os.environ.get("ROEHUB_BENCHMARK_GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit
    return _git_output(["git", "rev-parse", "HEAD"])


def _git_status_short() -> str:
    env_commit = os.environ.get("ROEHUB_BENCHMARK_GIT_COMMIT", "").strip()
    if env_commit:
        return "runtime-copy-no-git"
    return _git_output(["git", "status", "--short"])


def _git_output(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
