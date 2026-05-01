from __future__ import annotations

import argparse
import math
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
    DEFAULT_CANONICAL_JSON,
    TARGET_ARITIES,
    TARGET_DIRECTION_MODES,
    TARGET_RISK_MODE,
    _build_services,
    _git_commit,
    _git_status_short,
    _limit_prepared_rows,
    _load_json,
    _render_json,
    _require_mapping_value,
    _required_mapping,
    _required_sequence,
    _service_request,
)
from scripts.backtest.run_iteration_4_3_heap_update_benchmark import (  # noqa: E402
    _canonical_top_identity,
    _service_top_identity,
)
from scripts.backtest.run_iteration_4_4_top_result_proxy_fill_benchmark import (  # noqa: E402
    PROXY_SCORE_TOLERANCE,
    _canonical_proxy_metadata,
    _proxy_metadata_matches,
    _service_proxy_metadata,
)
from trading.contexts.backtest.application.dto import (  # noqa: E402
    canonical_no_risk_json_hash,
    canonical_no_risk_top_results_payload,
)
from trading.contexts.backtest.application.services.v2 import (  # noqa: E402
    NO_RISK_HEAP_UPDATE_STAGE_NAME,
    NO_RISK_METRIC_NAMES,
    NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME,
)

DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_4_5_result_shape_hash_parity"
)
SEMANTIC_FLOAT_TOLERANCE = 1e-9
SEMANTIC_RELATIVE_TOLERANCE = 1e-12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Iteration 4.5 no-risk top result shape, ordering, serialization, "
            "and hash parity checks against canonical notebook evidence."
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
        help="Write evidence and return 0 even when parity/hash acceptance fails.",
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
    canonical_targets = _canonical_no_risk_result_targets(canonical)
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
                "arity={arity} direction={direction} semantic={semantic} "
                "proxy={proxy} strict_hash={strict_hash} hash_or_waiver={hash_or_waiver}".format(
                    arity=arity,
                    direction=direction_mode,
                    semantic=run["pass"]["semantic_metrics"],
                    proxy=run["pass"]["proxy_metadata"],
                    strict_hash=run["pass"]["strict_result_hash"],
                    hash_or_waiver=run["pass"]["strict_hash_or_waiver"],
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
    stage_timings_before = dict(exact_result.telemetry.stage_timings)
    serialization_start = time.perf_counter()
    service_payload = exact_result.canonical_top_results_payload()
    service_result_hash = canonical_no_risk_json_hash(service_payload)
    serialization_s = time.perf_counter() - serialization_start
    stage_timings_after = dict(exact_result.telemetry.stage_timings)
    measured_stage_timings_unchanged = stage_timings_before == stage_timings_after

    service_top_identity = _service_top_identity(
        top_results=exact_result.top_results,
        indicator_ids=target["indicator_ids"],
    )
    target_top_identity = target["top_identity"]
    top_identity_match = service_top_identity == target_top_identity
    service_proxy_metadata = _service_proxy_metadata(exact_result.top_results)
    target_proxy_metadata = target["proxy_metadata"]
    proxy_metadata_match = _proxy_metadata_matches(
        service=service_proxy_metadata,
        canonical=target_proxy_metadata,
    )
    target_payload = target["top_results"]
    semantic_match = _semantic_metrics_match(
        service=service_payload,
        canonical=target_payload,
    )
    shape_match = _shape_matches(service=service_payload, canonical=target_payload)
    strict_hash_match = service_result_hash == target["result_hash"]
    hash_waiver = _hash_waiver(
        service=service_payload,
        canonical=target_payload,
        strict_hash_match=strict_hash_match,
        semantic_match=semantic_match,
        proxy_metadata_match=proxy_metadata_match,
        top_identity_match=top_identity_match,
        shape_match=shape_match,
    )
    strict_hash_or_waiver = strict_hash_match or bool(hash_waiver["eligible"])
    pass_payload = {
        "top_identity": top_identity_match,
        "shape": shape_match,
        "semantic_metrics": semantic_match,
        "proxy_metadata": proxy_metadata_match,
        "strict_result_hash": strict_hash_match,
        "strict_hash_or_waiver": strict_hash_or_waiver,
        "measured_stage_timings_unchanged": measured_stage_timings_unchanged,
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
            NO_RISK_HEAP_UPDATE_STAGE_NAME: float(
                stage_timings_before[NO_RISK_HEAP_UPDATE_STAGE_NAME]
            ),
            NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME: float(
                stage_timings_before[NO_RISK_TOP_RESULT_PROXY_FILL_STAGE_NAME]
            ),
            "result_serialization_normalization": serialization_s,
        },
        "top_result_identity": {
            "service": service_top_identity,
            "canonical": target_top_identity,
        },
        "top_result_proxy_metadata": {
            "service": service_proxy_metadata,
            "canonical": target_proxy_metadata,
            "proxy_score_tolerance": PROXY_SCORE_TOLERANCE,
        },
        "result_hash": {
            "service": service_result_hash,
            "canonical": target["result_hash"],
            "algorithm": "canonical_json_hash(top_results)",
            "strict result hash": strict_hash_match,
        },
        "hash_waiver": hash_waiver,
        "pass": pass_payload,
    }


def _canonical_no_risk_result_targets(
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
        top_results = canonical_no_risk_top_results_payload(
            _required_sequence(run, "top_results")
        )
        targets[(arity, direction_mode)] = {
            "indicator_ids": indicator_ids,
            "backend": str(run["exact_engine"]),
            "result_hash": str(run["result_hash"]),
            "top_results": top_results,
            "top_identity": _canonical_top_identity(
                top_results=top_results,
                indicator_ids=indicator_ids,
            ),
            "proxy_metadata": _canonical_proxy_metadata(top_results),
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


def _semantic_metrics_match(
    *,
    service: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
) -> bool:
    if len(service) != len(canonical):
        return False
    for service_row, canonical_row in zip(service, canonical):
        for metric_name in NO_RISK_METRIC_NAMES:
            if metric_name == "trade_count":
                if int(service_row[metric_name]) != int(canonical_row[metric_name]):
                    return False
                continue
            if not _float_values_match(
                service_row[metric_name],
                canonical_row[metric_name],
            ):
                return False
    return True


def _shape_matches(
    *,
    service: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
) -> bool:
    if len(service) != len(canonical):
        return False
    for service_row, canonical_row in zip(service, canonical):
        if set(service_row) != set(canonical_row):
            return False
        if any(key in service_row for key in ("rank", "score", "indicator_rows", "metrics")):
            return False
        if any(key in service_row for key in ("metadata", "_local_indices", "_proxy_pending")):
            return False
    return True


def _hash_waiver(
    *,
    service: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
    strict_hash_match: bool,
    semantic_match: bool,
    proxy_metadata_match: bool,
    top_identity_match: bool,
    shape_match: bool,
) -> dict[str, Any]:
    if strict_hash_match:
        return {
            "eligible": False,
            "reason": "strict result hash matched; waiver not used",
            "waived_fields": [],
        }
    waived_fields = _float_representation_drift_fields(service=service, canonical=canonical)
    eligible = (
        semantic_match
        and proxy_metadata_match
        and top_identity_match
        and shape_match
        and bool(waived_fields)
    )
    return {
        "eligible": eligible,
        "reason": (
            "float representation drift only"
            if eligible
            else "strict hash drift includes non-waivable shape, identity, proxy, or metric drift"
        ),
        "waived_fields": waived_fields,
    }


def _float_representation_drift_fields(
    *,
    service: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
) -> list[str]:
    fields: list[str] = []
    for row_idx, (service_row, canonical_row) in enumerate(zip(service, canonical)):
        for key in sorted(set(service_row) & set(canonical_row)):
            service_value = service_row[key]
            canonical_value = canonical_row[key]
            if isinstance(service_value, Mapping) or isinstance(canonical_value, Mapping):
                continue
            if service_value == canonical_value:
                continue
            if isinstance(service_value, (int, float)) and isinstance(
                canonical_value,
                (int, float),
            ) and _float_values_match(service_value, canonical_value):
                fields.append(f"top_results[{row_idx}].{key}")
    return fields


def _float_values_match(service_value: object, canonical_value: object) -> bool:
    service_float = float(service_value)  # type: ignore[arg-type]
    canonical_float = float(canonical_value)  # type: ignore[arg-type]
    if math.isnan(service_float) or math.isnan(canonical_float):
        return math.isnan(service_float) and math.isnan(canonical_float)
    if math.isinf(service_float) or math.isinf(canonical_float):
        return service_float == canonical_float
    tolerance = max(
        SEMANTIC_FLOAT_TOLERANCE,
        abs(canonical_float) * SEMANTIC_RELATIVE_TOLERANCE,
    )
    return abs(service_float - canonical_float) <= tolerance


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
    top_identity_pass = all(bool(run["pass"]["top_identity"]) for run in runs)
    shape_pass = all(bool(run["pass"]["shape"]) for run in runs)
    semantic_metric_parity_pass = all(
        bool(run["pass"]["semantic_metrics"]) for run in runs
    )
    proxy_metadata_parity_pass = all(bool(run["pass"]["proxy_metadata"]) for run in runs)
    strict_result_hash_pass = all(
        bool(run["pass"]["strict_result_hash"]) for run in runs
    )
    strict_hash_or_waiver_pass = all(
        bool(run["pass"]["strict_hash_or_waiver"]) for run in runs
    )
    measured_stage_boundary_clean = all(
        bool(run["pass"]["measured_stage_timings_unchanged"]) for run in runs
    )
    artifact_historical_prefix_compatible = (
        artifact_manifest_hash_matches_canonical
        or (top_identity_pass and semantic_metric_parity_pass and proxy_metadata_parity_pass)
    )
    top_results_counts = sorted({int(run["top_results_count"]) for run in runs})
    heap_capacities = sorted({int(run["heap_capacity"]) for run in runs})
    pass_count = {
        "top_identity": _pass_count(runs, "top_identity"),
        "shape": _pass_count(runs, "shape"),
        "semantic metric parity": _pass_count(runs, "semantic_metrics"),
        "proxy metadata parity": _pass_count(runs, "proxy_metadata"),
        "strict result hash": _pass_count(runs, "strict_result_hash"),
        "strict_hash_or_waiver": _pass_count(runs, "strict_hash_or_waiver"),
        "measured_stage_timings_unchanged": _pass_count(
            runs,
            "measured_stage_timings_unchanged",
        ),
    }
    return {
        "schema": "backtest_iteration_4_5_result_shape_hash_parity_v1",
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
                    "canonical request-slice top identity, semantic metrics, and "
                    "proxy metadata matched for all 14 result-shape runs"
                )
            ),
        },
        "scope": {
            "risk_mode": TARGET_RISK_MODE,
            "stage_scope": "result_shape_hash_only",
            "implemented_stages": ["result_hash_normalization"],
            "not_implemented_stages": [
                "exact_scoring",
                "heap_update",
                "top_result_proxy_fill",
                "persistence",
                "public_api_identity",
                "lazy_trades",
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
            "requires_top_result_identity_match": True,
            "requires_shape_match": True,
            "requires_semantic_metric_parity": True,
            "requires_proxy_metadata_parity": True,
            "requires_strict_result_hash_or_waiver": True,
            "semantic_float_tolerance": SEMANTIC_FLOAT_TOLERANCE,
            "semantic_relative_tolerance": SEMANTIC_RELATIVE_TOLERANCE,
            "proxy_score_tolerance": PROXY_SCORE_TOLERANCE,
            "requires_measured_stage_boundary_clean": True,
            "requires_artifact_manifest_hash_match": False,
            "requires_artifact_historical_prefix_compatibility": True,
            "artifact_policy": "historical_prefix_compatible",
        },
        "pass_count": pass_count,
        "runs": list(runs),
        "top_identity_pass": top_identity_pass,
        "shape_pass": shape_pass,
        "semantic_metric_parity_pass": semantic_metric_parity_pass,
        "proxy_metadata_parity_pass": proxy_metadata_parity_pass,
        "strict_result_hash_pass": strict_result_hash_pass,
        "strict_hash_or_waiver_pass": strict_hash_or_waiver_pass,
        "measured_stage_boundary_clean": measured_stage_boundary_clean,
        "pass": (
            top_identity_pass
            and shape_pass
            and semantic_metric_parity_pass
            and proxy_metadata_parity_pass
            and strict_hash_or_waiver_pass
            and measured_stage_boundary_clean
            and artifact_historical_prefix_compatible
        ),
    }


def _pass_count(runs: Sequence[Mapping[str, Any]], key: str) -> str:
    passed = sum(1 for run in runs if bool(run["pass"][key]))
    return f"{passed} / {len(runs)}"


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    request = _required_mapping(payload, "request")
    pass_count = _required_mapping(payload, "pass_count")
    lines = [
        "# Iteration 4.5 result shape/hash parity",
        "",
        "## Scope",
        "",
        "- Compared: top result shape, row ordering, semantic metric parity, "
        "proxy metadata parity, strict result hash.",
        "- Not compared: exact scoring, heap update, lazy trades, persistence, "
        "public API identity.",
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
        f"- Artifact policy: `{payload['artifact_acceptance']['policy']}`",
        "- Artifact historical-prefix compatible: "
        f"`{payload['artifact_historical_prefix_compatible']}`",
        "",
        "## Results",
        "",
        "| arity | direction_mode | backend | top_results | shape | semantic | proxy | "
        "strict_hash | waiver | timer_clean | pass |",
        "|---:|---|---|---:|---|---|---|---|---|---|---|",
    ]
    for run in payload["runs"]:
        pass_payload = _required_mapping(run, "pass")
        hash_waiver = _required_mapping(run, "hash_waiver")
        lines.append(
            "| {arity} | `{direction}` | `{backend}` | {top_results} | `{shape}` | "
            "`{semantic}` | `{proxy}` | `{strict_hash}` | `{waiver}` | "
            "`{timer_clean}` | `{passed}` |".format(
                arity=run["arity"],
                direction=run["direction_mode"],
                backend=run["backend"],
                top_results=run["top_results_count"],
                shape="yes" if pass_payload["shape"] else "no",
                semantic="yes" if pass_payload["semantic_metrics"] else "no",
                proxy="yes" if pass_payload["proxy_metadata"] else "no",
                strict_hash="yes" if pass_payload["strict_result_hash"] else "no",
                waiver="yes" if hash_waiver["eligible"] else "no",
                timer_clean="yes"
                if pass_payload["measured_stage_timings_unchanged"]
                else "no",
                passed="yes" if pass_payload["overall"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Top identity pass: `{pass_count['top_identity']}`",
            f"- Shape pass: `{pass_count['shape']}`",
            f"- semantic metric parity: `{pass_count['semantic metric parity']}`",
            f"- proxy metadata parity: `{pass_count['proxy metadata parity']}`",
            f"- strict result hash: `{pass_count['strict result hash']}`",
            f"- Strict hash or waiver: `{pass_count['strict_hash_or_waiver']}`",
            "- Measured stage timers unchanged by serialization: "
            f"`{pass_count['measured_stage_timings_unchanged']}`",
            "- Artifact hash matches canonical: "
            f"`{payload['artifact_manifest_hash_matches_canonical']}`",
            "- Artifact historical-prefix compatible: "
            f"`{payload['artifact_historical_prefix_compatible']}`",
            f"- Overall pass: `{'yes' if payload['pass'] else 'no'}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
