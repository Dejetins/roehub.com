from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.backtest.run_iteration_4_2_exact_scoring_benchmark import (  # noqa: E402
    ACCEPTANCE_RATIO,
    DEFAULT_CANONICAL_JSON,
    TARGET_ARITIES,
    TARGET_DIRECTION_MODES,
    TARGET_RISK_MODE,
    _build_services,
    _git_commit,
    _git_status_short,
    _limit_prepared_rows,
    _load_json,
    _ratio,
    _render_json,
    _require_mapping_value,
    _required_mapping,
    _required_sequence,
    _service_request,
)
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    NO_RISK_HEAP_UPDATE_STAGE_NAME,
)

DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_4_3_heap_update"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 4.3 no-risk heap_update benchmark against canonical "
            "notebook stage targets."
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
    sample_warmup_top_k = 1
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
        sample_warmup_top_k=sample_warmup_top_k,
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


def _run_matrix(
    *,
    canonical: Mapping[str, Any],
    services: Any,
    rows_per_indicator: int,
    warmup_rows_per_indicator: int,
) -> list[dict[str, Any]]:
    request_payload = _required_mapping(canonical, "request")
    canonical_targets = _canonical_no_risk_heap_targets(canonical)
    prepared_cache: dict[int, Any] = {}
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
                exact_result=exact_result,
                warm_result=warm_result,
                sample_warmup_s=sample_warmup_s,
                target=target,
            )
            runs.append(run)
            print(
                "arity={arity} direction={direction} heap_ratio={heap_ratio:.3f} "
                "top_identity={top_identity}".format(
                    arity=arity,
                    direction=direction_mode,
                    heap_ratio=run["ratios"][NO_RISK_HEAP_UPDATE_STAGE_NAME],
                    top_identity=run["pass"]["top_identity"],
                )
            )

    return runs


def _run_payload(
    *,
    arity: int,
    direction_mode: str,
    exact_result: Any,
    warm_result: Any,
    sample_warmup_s: float,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    stage_timings = dict(exact_result.telemetry.stage_timings)
    heap_s = float(stage_timings[NO_RISK_HEAP_UPDATE_STAGE_NAME])
    target_heap_s = float(target["heap_update_s"])
    heap_ratio = _ratio(target_heap_s, heap_s)
    service_top_identity = _service_top_identity(
        top_results=exact_result.top_results,
        indicator_ids=target["indicator_ids"],
    )
    target_top_identity = target["top_identity"]
    top_identity_match = service_top_identity == target_top_identity
    pass_payload = {
        NO_RISK_HEAP_UPDATE_STAGE_NAME: heap_ratio >= ACCEPTANCE_RATIO,
        "top_identity": top_identity_match,
    }
    pass_payload["overall"] = all(pass_payload.values())
    return {
        "risk_mode": TARGET_RISK_MODE,
        "direction_mode": direction_mode,
        "arity": arity,
        "indicator_ids": list(target["indicator_ids"]),
        "backend": exact_result.telemetry.backend_logical_name,
        "canonical_backend": target["backend"],
        "exact_candidates_evaluated": exact_result.telemetry.exact_candidates_evaluated,
        "request_top_n": exact_result.telemetry.request_top_n,
        "benchmark_top_k": exact_result.telemetry.benchmark_top_k,
        "heap_capacity": exact_result.telemetry.heap_capacity,
        "top_results_count": exact_result.telemetry.top_results_count,
        "sample_warmup_top_results_count": warm_result.telemetry.top_results_count,
        "timers": {
            "sample_warmup": sample_warmup_s,
            NO_RISK_HEAP_UPDATE_STAGE_NAME: heap_s,
        },
        "canonical_targets": {
            NO_RISK_HEAP_UPDATE_STAGE_NAME: target_heap_s,
        },
        "ratios": {
            NO_RISK_HEAP_UPDATE_STAGE_NAME: heap_ratio,
        },
        "top_result_identity": {
            "service": service_top_identity,
            "canonical": target_top_identity,
        },
        "pass": pass_payload,
    }


def _canonical_no_risk_heap_targets(
    canonical: Mapping[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    targets: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_run in _required_sequence(canonical, "runs"):
        run = _require_mapping_value(raw_run, "runs[]")
        if run.get("risk_mode") != TARGET_RISK_MODE:
            continue
        indicator_ids = tuple(str(value) for value in _required_sequence(run, "indicator_ids"))
        arity = len(indicator_ids)
        direction_mode = str(run.get("direction_mode", ""))
        if arity not in TARGET_ARITIES or direction_mode not in TARGET_DIRECTION_MODES:
            continue
        timers = _required_mapping(run, "timers")
        targets[(arity, direction_mode)] = {
            "indicator_ids": indicator_ids,
            "backend": str(run["exact_engine"]),
            "heap_update_s": float(timers[NO_RISK_HEAP_UPDATE_STAGE_NAME]),
            "top_identity": _canonical_top_identity(
                top_results=_required_sequence(run, "top_results"),
                indicator_ids=indicator_ids,
            ),
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


def _canonical_top_identity(
    *,
    top_results: Sequence[Any],
    indicator_ids: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    identity: list[tuple[int, ...]] = []
    for raw_top in top_results:
        top = _require_mapping_value(raw_top, "top_results[]")
        identity.append(
            tuple(
                int(_required_mapping(top, indicator_id)["row_id"])
                for indicator_id in indicator_ids
            )
        )
    return tuple(identity)


def _service_top_identity(
    *,
    top_results: Sequence[Any],
    indicator_ids: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(top_result.indicator_rows[indicator_id]) for indicator_id in indicator_ids)
        for top_result in top_results
    )


def _build_payload(
    *,
    canonical: Mapping[str, Any],
    canonical_json: Path,
    services: Any,
    runs: Sequence[Mapping[str, Any]],
    rows_per_indicator: int,
    warmup_rows_per_indicator: int,
    self_check_n: int,
    benchmark_top_k: int,
    sample_warmup_top_k: int,
) -> dict[str, Any]:
    canonical_request = _required_mapping(canonical, "request")
    artifact_manifest_hash_matches_canonical = (
        services.artifact_manifest_hash == str(canonical.get("artifact_manifest_hash", ""))
    )
    stage_pass = all(bool(run["pass"]["overall"]) for run in runs)
    top_results_counts = sorted({int(run["top_results_count"]) for run in runs})
    heap_capacities = sorted({int(run["heap_capacity"]) for run in runs})
    return {
        "schema": "backtest_iteration_4_3_heap_update_v1",
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
        "scope": {
            "risk_mode": TARGET_RISK_MODE,
            "stage_scope": "heap_update_only",
            "implemented_stages": [NO_RISK_HEAP_UPDATE_STAGE_NAME],
            "not_implemented_stages": [
                "top_result_proxy_fill",
                "result_hash_normalization",
                "persistence",
                "public_api_identity",
            ],
        },
        "request": {
            "top_n": int(canonical_request.get("top_n", 100)),
            "benchmark_top_k": benchmark_top_k,
            "sample_warmup_top_k": sample_warmup_top_k,
            "rows_per_indicator": rows_per_indicator,
            "warmup_rows_per_indicator": warmup_rows_per_indicator,
            "self_check_n": self_check_n,
            "top_results_count_values": top_results_counts,
            "heap_capacity_values": heap_capacities,
        },
        "acceptance": {
            "ratio_threshold": ACCEPTANCE_RATIO,
            "ratio_definition": "canonical_stage_seconds / service_stage_seconds",
            "stage_boundaries_compared": [NO_RISK_HEAP_UPDATE_STAGE_NAME],
            "requires_top_result_identity_match": True,
            "requires_artifact_manifest_hash_match": True,
        },
        "runs": list(runs),
        "stage_pass": stage_pass,
        "pass": stage_pass and artifact_manifest_hash_matches_canonical,
    }


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    request = _required_mapping(payload, "request")
    lines = [
        "# Iteration 4.3 heap_update benchmark",
        "",
        "## Scope",
        "",
        "- Compared stage: `heap_update`.",
        "- Not compared: `top_result_proxy_fill`, persistence, public API identity.",
        f"- Acceptance ratio threshold: `{payload['acceptance']['ratio_threshold']}`.",
        "- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.",
        f"- `request.top_n = {request['top_n']}`",
        f"- `benchmark_top_k = {request['benchmark_top_k']}`",
        f"- `sample_warmup_top_k = {request['sample_warmup_top_k']}`",
        f"- `top_results_count_values = {request['top_results_count_values']}`",
        f"- `heap_capacity_values = {request['heap_capacity_values']}`",
        "",
        "## Environment",
        "",
        f"- Host: `{payload['host']}`",
        f"- Git commit: `{payload['git_commit']}`",
        f"- Artifact config: `{payload['artifact_config_path']}`",
        f"- Artifact manifest hash: `{payload['artifact_manifest_hash']}`",
        "- Artifact hash matches canonical: "
        f"`{payload['artifact_manifest_hash_matches_canonical']}`",
        "",
        "## Results",
        "",
        "| arity | direction_mode | backend | heap_s | canonical_heap_s | "
        "heap_ratio | top_results | identity | pass |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for run in payload["runs"]:
        timers = _required_mapping(run, "timers")
        targets = _required_mapping(run, "canonical_targets")
        ratios = _required_mapping(run, "ratios")
        pass_payload = _required_mapping(run, "pass")
        lines.append(
            "| {arity} | `{direction}` | `{backend}` | {heap:.6f} | "
            "{target_heap:.6f} | {heap_ratio:.3f} | {top_results} | `{identity}` | "
            "`{passed}` |".format(
                arity=run["arity"],
                direction=run["direction_mode"],
                backend=run["backend"],
                heap=timers[NO_RISK_HEAP_UPDATE_STAGE_NAME],
                target_heap=targets[NO_RISK_HEAP_UPDATE_STAGE_NAME],
                heap_ratio=ratios[NO_RISK_HEAP_UPDATE_STAGE_NAME],
                top_results=run["top_results_count"],
                identity="yes" if pass_payload["top_identity"] else "no",
                passed="yes" if pass_payload["overall"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Stage pass: `{'yes' if payload['stage_pass'] else 'no'}`",
            "- Artifact hash matches canonical: "
            f"`{payload['artifact_manifest_hash_matches_canonical']}`",
            f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
