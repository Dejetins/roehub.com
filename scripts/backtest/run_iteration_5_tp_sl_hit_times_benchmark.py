from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.backtest.adapters.outbound import (  # noqa: E402
    BacktestArtifactPathBuilderV2,
    FilesystemBacktestArtifactContextResolver,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (  # noqa: E402
    FilesystemBacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.dto import BacktestCoordinates  # noqa: E402
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE,
    BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED,
    HIT_TIMES_ARTIFACT_PATH_V2,
    LOAD_HIT_TIMES_STAGE_NAME,
    TARGET_TP_SL_GRID_START_PCT,
    TARGET_TP_SL_GRID_STEP_PCT,
    TARGET_TP_SL_GRID_STOP_PCT,
    TP_SL_GRID_VALIDATION_STAGE_NAME,
    BacktestTpSlHitTimesRejected,
    BacktestTpSlHitTimesService,
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
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_5_tp_sl_hit_times_loading_validation"
)
TARGET_RISK_MODE = "tp_sl_grid"
TARGET_ARITIES = tuple(range(1, 8))
TARGET_DIRECTION_MODES = ("long_only", "long_short_reversal")
ACCEPTANCE_RATIO = 0.9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 5 artifact-backed TP/SL hit-times loading and "
            "grid-validation benchmark."
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
        "--no-fail-on-threshold",
        action="store_true",
        help="Write evidence and return 0 even when ratios are below acceptance.",
    )
    args = parser.parse_args(argv)

    canonical = _load_json(args.canonical_json)
    services = _build_services(artifact_config_path=args.artifact_config)
    warmup = _run_warmup(canonical=canonical, services=services)
    runs = _run_matrix(canonical=canonical, services=services)
    validation_failure = _run_missing_level_probe(canonical=canonical, services=services)
    load_failure = _run_failed_load_probe(canonical=canonical, services=services)
    payload = _build_payload(
        canonical=canonical,
        canonical_json=args.canonical_json,
        services=services,
        warmup=warmup,
        runs=runs,
        validation_failure=validation_failure,
        load_failure=load_failure,
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


@dataclass(frozen=True, slots=True)
class _Services:
    artifact_config_path: Path
    artifact_root: Path
    artifact_manifest_hash: str
    hit_times_manifest_hash: str
    array_loader: FilesystemBacktestArtifactArrayLoader
    hit_times: BacktestTpSlHitTimesService
    context: Any


def _build_services(*, artifact_config_path: Path | None) -> _Services:
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
        array_loader=array_loader,
        hit_times=BacktestTpSlHitTimesService(artifact_array_loader=array_loader),
        context=context,
    )


def _run_warmup(*, canonical: Mapping[str, Any], services: _Services) -> dict[str, Any]:
    canonical_run = _risk_on_canonical_runs(canonical)[0]
    request = _service_request(
        canonical_request=_required_mapping(canonical, "request"),
        canonical_run=canonical_run,
    )
    start = time.perf_counter()
    result = services.hit_times.execute(
        normalized_request=request,
        context=services.context,
    )
    warmup = {
        "wall_time_s": time.perf_counter() - start,
        "timing": result.timing.as_mapping(),
        "hit_times_subset": result.hit_times.compact_mapping(),
    }
    del result
    gc.collect()
    return warmup


def _run_matrix(*, canonical: Mapping[str, Any], services: _Services) -> list[dict[str, Any]]:
    canonical_request = _required_mapping(canonical, "request")
    runs: list[dict[str, Any]] = []
    for canonical_run in _risk_on_canonical_runs(canonical):
        request = _service_request(
            canonical_request=canonical_request,
            canonical_run=canonical_run,
        )
        cpu_start = time.process_time()
        rss_before = _maxrss_raw()
        result = services.hit_times.execute(
            normalized_request=request,
            context=services.context,
        )
        cpu_s = time.process_time() - cpu_start
        rss_after = _maxrss_raw()
        timers = dict(result.timing.subsegments)
        targets = _canonical_stage_targets(canonical_run)
        ratios = {
            stage: _ratio(targets[stage], timers[stage])
            for stage in (LOAD_HIT_TIMES_STAGE_NAME, TP_SL_GRID_VALIDATION_STAGE_NAME)
        }
        stage_pass = {
            stage: ratios[stage] >= ACCEPTANCE_RATIO
            for stage in (LOAD_HIT_TIMES_STAGE_NAME, TP_SL_GRID_VALIDATION_STAGE_NAME)
        }
        stage_pass["overall"] = all(stage_pass.values())
        grid_evidence = result.resolution.evidence.as_mapping()
        run = {
            "arity": len(canonical_run["indicator_ids"]),
            "direction_mode": str(canonical_run["direction_mode"]),
            "backend": str(canonical_run["exact_engine"]),
            "risk_mode": TARGET_RISK_MODE,
            "timers": timers,
            "canonical_targets": targets,
            "ratios": ratios,
            "stage_pass": stage_pass,
            "grid_evidence": grid_evidence,
            "hit_times_manifest_hash": result.hit_times_manifest_hash,
            "hit_times_subset": result.hit_times.compact_mapping(),
            "cleanup_evidence": result.cleanup_evidence.as_mapping(),
            "runtime_metrics": {
                "process_cpu_time_s": cpu_s,
                "maxrss_raw_before": rss_before,
                "maxrss_raw_after": rss_after,
            },
            "pass": (
                bool(stage_pass["overall"])
                and result.hit_times_manifest_hash == services.hit_times_manifest_hash
                and bool(grid_evidence["target_grid"]["covered_by_artifact"])
            ),
        }
        runs.append(run)
        del result
        gc.collect()
    return runs


def _run_missing_level_probe(
    *,
    canonical: Mapping[str, Any],
    services: _Services,
) -> dict[str, Any]:
    canonical_run = _risk_on_canonical_runs(canonical)[0]
    request = _service_request(
        canonical_request=_required_mapping(canonical, "request"),
        canonical_run=canonical_run,
    )
    request["risk"]["tp"] = {"start_pct": 1000.0, "stop_pct": 1000.0, "step_pct": 0.5}
    try:
        services.hit_times.execute(normalized_request=request, context=services.context)
    except BacktestTpSlHitTimesRejected as error:
        return {
            "probe": "missing_tp_level",
            "expected_error_code": BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED,
            "error_code": error.error_code,
            "pass": error.error_code == BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED
            and error.cleanup_evidence.retained_materialized_subset is False
            and error.cleanup_evidence.retained_hit_times_table_arrays is False,
            "details": error.details(),
        }
    raise AssertionError("missing TP level probe unexpectedly succeeded")


def _run_failed_load_probe(
    *,
    canonical: Mapping[str, Any],
    services: _Services,
) -> dict[str, Any]:
    canonical_run = _risk_on_canonical_runs(canonical)[0]
    request = _service_request(
        canonical_request=_required_mapping(canonical, "request"),
        canonical_run=canonical_run,
    )
    failing_service = BacktestTpSlHitTimesService(
        artifact_array_loader=_FailingTableLoader(services.array_loader)
    )
    try:
        failing_service.execute(normalized_request=request, context=services.context)
    except BacktestTpSlHitTimesRejected as error:
        return {
            "probe": "failed_table_load",
            "expected_error_code": BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE,
            "error_code": error.error_code,
            "pass": error.error_code == BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE
            and error.cleanup_evidence.retained_materialized_subset is False
            and error.cleanup_evidence.retained_hit_times_table_arrays is False,
            "details": error.details(),
        }
    raise AssertionError("failed table load probe unexpectedly succeeded")


class _FailingTableLoader:
    def __init__(self, inner: FilesystemBacktestArtifactArrayLoader) -> None:
        self._inner = inner

    def resolve_context(self, **kwargs: Any) -> Any:
        return self._inner.resolve_context(**kwargs)

    def load_price_arrays(self, **kwargs: Any) -> Any:
        return self._inner.load_price_arrays(**kwargs)

    def load_mapping_arrays(self, **kwargs: Any) -> Any:
        return self._inner.load_mapping_arrays(**kwargs)

    def load_funding_arrays(self, **kwargs: Any) -> Any:
        return self._inner.load_funding_arrays(**kwargs)

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        return self._inner.load_signal_matrix(**kwargs)

    def load_signal_rows(self, **kwargs: Any) -> Any:
        return self._inner.load_signal_rows(**kwargs)

    def load_hit_times_grid_arrays(self, **kwargs: Any) -> Any:
        return self._inner.load_hit_times_grid_arrays(**kwargs)

    def load_hit_times_table_arrays(self, **kwargs: Any) -> Any:
        raise FileNotFoundError("intentional Iteration 5 failed-load cleanup probe")


def _build_payload(
    *,
    canonical: Mapping[str, Any],
    canonical_json: Path,
    services: _Services,
    warmup: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    validation_failure: Mapping[str, Any],
    load_failure: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_artifact_hash = str(canonical.get("artifact_manifest_hash", ""))
    canonical_hit_times_hash = str(canonical.get("hit_times_manifest_hash", ""))
    artifact_manifest_hash_matches_canonical = (
        services.artifact_manifest_hash == canonical_artifact_hash
    )
    hit_times_manifest_hash_matches_canonical = (
        services.hit_times_manifest_hash == canonical_hit_times_hash
    )
    target_grid_covered = all(
        bool(run["grid_evidence"]["target_grid"]["covered_by_artifact"]) for run in runs
    )
    stage_pass = {
        LOAD_HIT_TIMES_STAGE_NAME: all(
            bool(run["stage_pass"][LOAD_HIT_TIMES_STAGE_NAME]) for run in runs
        ),
        TP_SL_GRID_VALIDATION_STAGE_NAME: all(
            bool(run["stage_pass"][TP_SL_GRID_VALIDATION_STAGE_NAME]) for run in runs
        ),
    }
    stage_pass["overall"] = all(stage_pass.values())
    artifact_historical_prefix_compatible = (
        artifact_manifest_hash_matches_canonical or target_grid_covered
    )
    failure_pass = bool(validation_failure["pass"]) and bool(load_failure["pass"])
    return {
        "schema": "backtest_iteration_5_tp_sl_hit_times_loading_validation_v1",
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
        "artifact_acceptance": {
            "policy": "historical_prefix_compatible",
            "full_manifest_hash_match_required": False,
            "compatibility_evidence": (
                "full manifest hash match"
                if artifact_manifest_hash_matches_canonical
                else "target TP/SL request grid is covered by live hit_times/15m"
            ),
        },
        "scope": {
            "risk_mode": TARGET_RISK_MODE,
            "hit_times_path": HIT_TIMES_ARTIFACT_PATH_V2,
            "implemented_stages": [
                LOAD_HIT_TIMES_STAGE_NAME,
                TP_SL_GRID_VALIDATION_STAGE_NAME,
            ],
            "not_implemented_stages": [
                "tp_sl_exact_scoring",
                "heap_update",
                "top_result_assembly",
                "persistence",
                "public_api",
            ],
            "canonical_stage_comparison_only": True,
        },
        "request": {
            "risk": _target_risk_mapping(),
            "target_grid_literal": "2.0..25.0 step 0.5",
            "target_grid_cells": 47 * 47,
            "arities": list(TARGET_ARITIES),
            "direction_modes": list(TARGET_DIRECTION_MODES),
        },
        "acceptance": {
            "ratio_threshold": ACCEPTANCE_RATIO,
            "ratio_definition": "canonical_stage_seconds / service_stage_seconds",
            "stage_boundaries_compared": [
                LOAD_HIT_TIMES_STAGE_NAME,
                TP_SL_GRID_VALIDATION_STAGE_NAME,
            ],
            "service_only_cleanup_compared_to_notebook": False,
            "artifact_policy": "historical_prefix_compatible",
        },
        "warmup": dict(warmup),
        "runs": list(runs),
        "failure_evidence": {
            "validation": dict(validation_failure),
            "load": dict(load_failure),
            "pass": failure_pass,
        },
        "target_grid_covered": target_grid_covered,
        "stage_pass": stage_pass,
        "pass": bool(
            stage_pass["overall"]
            and target_grid_covered
            and failure_pass
            and artifact_historical_prefix_compatible
        ),
    }


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    stage_pass = _required_mapping(payload, "stage_pass")
    failure_evidence = _required_mapping(payload, "failure_evidence")
    lines = [
        "# Iteration 5 TP/SL hit-times loading and grid validation",
        "",
        "## Scope",
        "",
        "- Compared stages: `load_hit_times`, `tp_sl_grid_validation`.",
        "- Not implemented: `tp_sl_exact_scoring`, heap/top-K, persistence, public API.",
        f"- Runtime target path: `{payload['scope']['hit_times_path']}`.",
        "- Target grid: `2.0..25.0` inclusive, `step 0.5`.",
        f"- Acceptance ratio threshold: `{payload['acceptance']['ratio_threshold']}`.",
        "- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.",
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
        "## Results",
        "",
        "| arity | direction_mode | load_hit_times_s | canonical_load_s | load_ratio | "
        "tp_sl_grid_validation_s | canonical_validation_s | validation_ratio | pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in payload["runs"]:
        timers = run["timers"]
        targets = run["canonical_targets"]
        ratios = run["ratios"]
        lines.append(
            "| {arity} | `{direction}` | {load:.6f} | {target_load:.6f} | "
            "{load_ratio:.3f} | {validation:.6f} | {target_validation:.6f} | "
            "{validation_ratio:.3f} | `{passed}` |".format(
                arity=run["arity"],
                direction=run["direction_mode"],
                load=timers[LOAD_HIT_TIMES_STAGE_NAME],
                target_load=targets[LOAD_HIT_TIMES_STAGE_NAME],
                load_ratio=ratios[LOAD_HIT_TIMES_STAGE_NAME],
                validation=timers[TP_SL_GRID_VALIDATION_STAGE_NAME],
                target_validation=targets[TP_SL_GRID_VALIDATION_STAGE_NAME],
                validation_ratio=ratios[TP_SL_GRID_VALIDATION_STAGE_NAME],
                passed="yes" if run["pass"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Grid And Failure Evidence",
            "",
            "- Target grid covered by artifact: "
            f"`{'yes' if payload['target_grid_covered'] else 'no'}`",
            "- Missing-level failure code: "
            f"`{failure_evidence['validation']['error_code']}`",
            "- Failed-load cleanup code: "
            f"`{failure_evidence['load']['error_code']}`",
            "- Failure evidence pass: "
            f"`{'yes' if failure_evidence['pass'] else 'no'}`",
            "",
            "## Decision",
            "",
            "- `load_hit_times` pass: "
            f"`{'yes' if stage_pass[LOAD_HIT_TIMES_STAGE_NAME] else 'no'}`",
            "- `tp_sl_grid_validation` pass: "
            f"`{'yes' if stage_pass[TP_SL_GRID_VALIDATION_STAGE_NAME] else 'no'}`",
            f"- Stage pass: `{'yes' if stage_pass['overall'] else 'no'}`",
            "- Artifact historical-prefix compatible: "
            f"`{payload['artifact_historical_prefix_compatible']}`",
            f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _risk_on_canonical_runs(canonical: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runs = [
        run
        for run in _required_sequence(canonical, "runs")
        if isinstance(run, Mapping) and run.get("risk_mode") == TARGET_RISK_MODE
    ]
    runs = [
        run
        for run in runs
        if len(run.get("indicator_ids", ())) in TARGET_ARITIES
        and run.get("direction_mode") in TARGET_DIRECTION_MODES
    ]
    if len(runs) != len(TARGET_ARITIES) * len(TARGET_DIRECTION_MODES):
        raise ValueError(f"expected 14 canonical risk-on runs, got {len(runs)}")
    return runs


def _service_request(
    *,
    canonical_request: Mapping[str, Any],
    canonical_run: Mapping[str, Any],
) -> dict[str, Any]:
    request = json.loads(json.dumps(canonical_request))
    arity = len(canonical_run["indicator_ids"])
    request["indicators"] = request["indicators"][:arity]
    request["risk"] = _target_risk_mapping()
    request["execution"]["direction_mode"] = str(canonical_run["direction_mode"])
    return request


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


def _canonical_stage_targets(canonical_run: Mapping[str, Any]) -> dict[str, float]:
    timers = _required_mapping(canonical_run, "timers")
    return {
        LOAD_HIT_TIMES_STAGE_NAME: float(timers[LOAD_HIT_TIMES_STAGE_NAME]),
        TP_SL_GRID_VALIDATION_STAGE_NAME: float(timers[TP_SL_GRID_VALIDATION_STAGE_NAME]),
    }


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
