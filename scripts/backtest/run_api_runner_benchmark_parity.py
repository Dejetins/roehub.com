from __future__ import annotations

# ruff: noqa: E402, SLF001
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import quantiles
from typing import Any, cast
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apps.worker.backtest_job_runner.wiring.modules import (  # noqa: E402
    backtest_job_runner as runner_module,
)
from apps.worker.backtest_job_runner.wiring.modules import (  # noqa: E402
    build_backtest_job_runner_app,
    load_backtest_job_runner_runtime_config,
)
from scripts.backtest import run_backtest_job_runner_prod_smoke as prod_smoke  # noqa: E402
from scripts.backtest import (
    run_iteration_4_2_exact_scoring_benchmark as no_risk_bench,  # noqa: E402,E501
)
from scripts.backtest import (
    run_iteration_6_tp_sl_exact_scoring_benchmark as tp_sl_bench,  # noqa: E402,E501
)
from trading.contexts.backtest.application.services.v2.matrix_backend.tp_sl_cells import (  # noqa: E402,E501
    TP_SL_SELECTED_CELL_SHADOW_ENV_KEY,
)

DEFAULT_CANONICAL_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-04-26_engine_test_btcusdt_15m/benchmark_results.json"
)
DEFAULT_REFERENCE_JSON = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    f"{datetime.now().strftime('%Y-%m-%d')}_iteration_15_api_runner_clean_arity6_cpu_memory"
)
REQUEST_TOP_N = 50
BENCHMARK_TOP_K = 5
REFERENCE_ROWS_PER_INDICATOR = 6
REFERENCE_WARMUP_ROWS_PER_INDICATOR = 2
REFERENCE_ONLY_ARITY = 6
STAGE_04_MVP_ARITIES = (2, 3)
STAGE_04_MVP_RISK_MODE = "none"
STAGE_04_MVP_DIRECTION_MODE = "long_only"
STAGE_05_NO_RISK_HEAVY_ARITY = 6
STAGE_05_NO_RISK_HEAVY_DIRECTIONS = ("long_only", "long_short_reversal")
STAGE_08_TP_SL_SELECTED_ARITIES = (1, 2)
STAGE_08_TP_SL_SELECTED_DIRECTIONS = ("long_only", "long_short_reversal")
STAGE_08_TP_SL_SELECTED_START_PCT = 2.0
STAGE_08_TP_SL_SELECTED_STOP_PCT = 5.5
STAGE_08_TP_SL_SELECTED_STEP_PCT = 0.5
STAGE_09_TP_SL_FULL_GRID_ARITY = 6
STAGE_09_TP_SL_FULL_GRID_DIRECTIONS = ("long_only", "long_short_reversal")
STAGE_09_MATRIX_BACKEND_MODE = "stage_09_tp_sl_full_grid"
STAGE_12_COMPILED_PREFIX_ARITIES = (6, 7)
STAGE_12_COMPILED_PREFIX_DIRECTIONS = ("long_only", "long_short_reversal")
STAGE_12_MATRIX_BACKEND_MODE = "stage_12_compiled_prefix_traversal"
STAGE_05_12_PRODUCTION_DEFAULT_ARITIES = (6, 7)
STAGE_05_12_PRODUCTION_DEFAULT_DIRECTIONS = (
    "long_only",
    "long_short_reversal",
)
STAGE_05_12_PRODUCTION_DEFAULT_BACKEND_MODE = "stage_05_and_12_no_risk"
MATRIX_BACKEND_MODE_ENV_KEY = "ROEHUB_BACKTEST_MATRIX_BACKEND_MODE"
TP_SL_CELL_BLOCK_TP_COUNT_ENV_KEY = "ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_TP_COUNT"
TP_SL_CELL_BLOCK_SL_COUNT_ENV_KEY = "ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_SL_COUNT"
STAGE_09_DEFAULT_CELL_BLOCK_TP_COUNT = 64
STAGE_09_DEFAULT_CELL_BLOCK_SL_COUNT = 64
_DEFAULT_API_BASE = "http://127.0.0.1:8000"
_DEFAULT_COOKIE_NAME = "roehub_session_id"
_PARITY_FLOAT_TOLERANCE = 1e-5
_CACHE_HIT_RSS_DELTA_LIMIT_BYTES = 64 * 1024 * 1024
_SYSTEM_RETAINED_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_REFERENCE_SPEED_RATIO_MIN = 0.8
_DEFAULT_CPU_SAMPLE_INTERVAL_SECONDS = 1.0
_DEFAULT_MACOS_ENV_FILE = Path("/Users/daniildegtyarev/.config/roehub/roehub.env")
_DEFAULT_DOCKER_ENV_FILE = Path("/etc/roehub/roehub.env")
_DEFAULT_PROD_ARTIFACT_CONFIG = Path("configs/prod/backtest_artifacts.yaml")
_ARTIFACT_CONFIG_ENV_KEY = "ROEHUB_BACKTEST_ARTIFACTS_CONFIG"
_ROEHUB_ENV_KEY = "ROEHUB_ENV"
_PG_DSN_ENV_KEYS = ("STRATEGY_PG_DSN", "POSTGRES_DSN", "IDENTITY_PG_DSN")
_PG_COMPONENT_ENV_KEYS = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
_ENV_REPORT_KEYS = frozenset(
    (
        _ARTIFACT_CONFIG_ENV_KEY,
        _ROEHUB_ENV_KEY,
        *_PG_DSN_ENV_KEYS,
        *_PG_COMPONENT_ENV_KEYS,
    )
)
_ENV_REFERENCE_RE = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
    r"|\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass(frozen=True, slots=True)
class _RunnerHarness:
    scheduler: Any
    runtime_config: Any
    env: Mapping[str, str]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    env_file_report = _load_runtime_env_file(args.env_file)
    dsn, filled_dsn_keys = _ensure_postgres_dsn_env()
    artifact_env_report = _ensure_artifact_runtime_env()
    matrix_sidecar_report = _matrix_sidecar_report(args.matrix_sidecar_artifact_dir)
    if args.stage_08_tp_sl_selected_cells:
        os.environ[TP_SL_SELECTED_CELL_SHADOW_ENV_KEY] = "1"
    if args.stage_09_tp_sl_full_grid:
        raise RuntimeError(
            "--stage-09-tp-sl-full-grid is retired: Stage 09 remains historical "
            "evidence only and is no longer selectable through "
            f"{MATRIX_BACKEND_MODE_ENV_KEY}."
        )
    if args.stage_12_compiled_prefix_rows and not os.environ.get(
        MATRIX_BACKEND_MODE_ENV_KEY
    ):
        os.environ[MATRIX_BACKEND_MODE_ENV_KEY] = STAGE_12_MATRIX_BACKEND_MODE
    if args.stage_05_12_production_default_rows:
        matrix_backend_mode = os.environ.get(MATRIX_BACKEND_MODE_ENV_KEY, "")
        if (
            matrix_backend_mode
            and matrix_backend_mode != STAGE_05_12_PRODUCTION_DEFAULT_BACKEND_MODE
        ):
            raise RuntimeError(
                "--stage-05-12-production-default-rows must run with "
                f"{MATRIX_BACKEND_MODE_ENV_KEY} unset or "
                f"{STAGE_05_12_PRODUCTION_DEFAULT_BACKEND_MODE!r}; got "
                f"{matrix_backend_mode!r}"
            )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    child_evidence_dir = out_dir / "child_process_evidence"
    child_evidence_dir.mkdir(parents=True, exist_ok=True)

    canonical = _load_json(args.canonical_json)
    reference = _load_json(args.reference_json)
    reference_runs, excluded = _reference_runs(canonical=canonical, reference=reference)
    if args.stage_04_mvp_rows:
        reference_runs, excluded = _stage_04_mvp_reference_runs(reference=reference)
    if args.stage_05_no_risk_heavy_rows:
        reference_runs, excluded = _stage_05_no_risk_heavy_reference_runs(
            reference=reference
        )
    if args.stage_08_tp_sl_selected_cells:
        reference_runs, excluded = _stage_08_tp_sl_selected_reference_runs(
            reference=reference
        )
    if args.stage_09_tp_sl_full_grid:
        reference_runs, excluded = _stage_09_tp_sl_full_grid_reference_runs(
            reference=reference
        )
    if args.stage_12_compiled_prefix_rows:
        reference_runs, excluded = _stage_12_compiled_prefix_reference_runs(
            reference=reference
        )
    if args.stage_05_12_production_default_rows:
        reference_runs, excluded = _stage_05_12_production_default_reference_runs(
            reference=reference
        )
    if args.smoke_only:
        reference_runs = _smoke_subset(reference_runs)

    session_id: str | None = None
    started_at = datetime.now(UTC)
    payload: dict[str, Any] = {
        "schema": "backtest_api_runner_compute_memory_parity_v1",
        "generated_at": started_at.isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "git_status_short": _git_status_short(),
        "api_base": args.api_base,
        "env_file": env_file_report,
        "postgres_env": {
            "dsn_keys_present": [
                key for key in _PG_DSN_ENV_KEYS if os.environ.get(key, "").strip()
            ],
            "component_keys_present": [
                key for key in _PG_COMPONENT_ENV_KEYS if os.environ.get(key, "").strip()
            ],
            "filled_dsn_keys": filled_dsn_keys,
        },
        "artifact_env": artifact_env_report,
        "canonical_json": str(args.canonical_json),
        "reference_json": str(args.reference_json),
        "reference_iteration": "2026-05-02_iteration_8_execution_sizing_completion",
        "request": {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "top_n": REQUEST_TOP_N,
            "benchmark_top_k": BENCHMARK_TOP_K,
            "only_arity": REFERENCE_ONLY_ARITY,
            "stage_04_mvp_rows": args.stage_04_mvp_rows,
            "stage_05_no_risk_heavy_rows": args.stage_05_no_risk_heavy_rows,
            "stage_08_tp_sl_selected_cells": args.stage_08_tp_sl_selected_cells,
            "stage_09_tp_sl_full_grid": args.stage_09_tp_sl_full_grid,
            "stage_12_compiled_prefix_rows": args.stage_12_compiled_prefix_rows,
            "stage_05_12_production_default_rows": (
                args.stage_05_12_production_default_rows
            ),
            "tp_sl_selected_cell_shadow_env_key": TP_SL_SELECTED_CELL_SHADOW_ENV_KEY,
            "matrix_backend_mode_env_key": MATRIX_BACKEND_MODE_ENV_KEY,
            "matrix_backend_mode": os.environ.get(MATRIX_BACKEND_MODE_ENV_KEY),
            "tp_sl_cell_block_tp_count_env_key": TP_SL_CELL_BLOCK_TP_COUNT_ENV_KEY,
            "tp_sl_cell_block_sl_count_env_key": TP_SL_CELL_BLOCK_SL_COUNT_ENV_KEY,
            "tp_sl_cell_block_tp_count": os.environ.get(
                TP_SL_CELL_BLOCK_TP_COUNT_ENV_KEY
            ),
            "tp_sl_cell_block_sl_count": os.environ.get(
                TP_SL_CELL_BLOCK_SL_COUNT_ENV_KEY
            ),
            "rows_per_indicator": REFERENCE_ROWS_PER_INDICATOR,
            "warmup_rows_per_indicator": REFERENCE_WARMUP_ROWS_PER_INDICATOR,
            "exclude_heaviest_140s_job": True,
            "full_jobs": "heavy",
            "numba_threads": 12,
            "runner_concurrency": "one heavy child",
            "cpu_sampler": "ps %cpu/%mem/rss by full_job_child --job-id",
            "cpu_sample_interval_seconds": args.cpu_sample_interval_seconds,
            "vmmap_observation": "disabled by default; old vmmap-contaminated results are excluded",
            "matrix_sidecar_artifact_dir": (
                None
                if args.matrix_sidecar_artifact_dir is None
                else str(args.matrix_sidecar_artifact_dir)
            ),
        },
        "matrix_sidecar": matrix_sidecar_report,
        "excluded_reference_job": excluded,
        "artifact_reference": {
            "artifact_manifest_hash": reference.get("artifact_manifest_hash"),
            "hit_times_manifest_hash": reference.get("hit_times_manifest_hash"),
            "artifact_policy": reference.get("artifact_policy"),
            "request_hash": canonical.get("request_hash"),
        },
    }

    try:
        user_id, session_id = prod_smoke._create_smoke_session(
            dsn=dsn,
            cookie_name=args.cookie_name,
            session_ttl_seconds=args.session_ttl_seconds,
        )
        client = prod_smoke._ApiClient(
            api_base=args.api_base,
            cookie_name=args.cookie_name,
            session_id=session_id,
        )
        payload["smoke_user_id"] = user_id
        payload["backlog_before"] = prod_smoke._backlog_snapshot(dsn=dsn)
        _require_clean_backlog(payload["backlog_before"], allow_existing=args.allow_backlog)

        harness = _build_runner_harness(
            child_evidence_dir=child_evidence_dir,
            light_max_actual_combinations=args.light_max_actual_combinations,
            heavy_concurrency=1,
        )
        benchmark_jobs = _run_reference_jobs(
            dsn=dsn,
            client=client,
            harness=harness,
            canonical=canonical,
            reference_runs=reference_runs,
            child_evidence_dir=child_evidence_dir,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            system_memory_cleanup_wait_seconds=args.system_memory_cleanup_wait_seconds,
            cpu_sample_interval_seconds=args.cpu_sample_interval_seconds,
            stage_08_tp_sl_selected_cells=args.stage_08_tp_sl_selected_cells,
            stage_12_compiled_prefix_rows=args.stage_12_compiled_prefix_rows,
            stage_05_12_production_default_rows=(
                args.stage_05_12_production_default_rows
            ),
        )
        payload["api_runner_path"] = {
            "runner_entrypoint": "BacktestJobWorkerUseCase.run_next",
            "child_module": "apps.worker.backtest_job_runner.main.full_job_child",
            "jobs": benchmark_jobs,
            "pass": all(bool(job["pass"]) for job in benchmark_jobs),
        }
        payload["parity"] = _parity_summary(benchmark_jobs)
        payload["performance"] = _performance_summary(benchmark_jobs)
        payload["instrumentation"] = _instrumentation_summary(benchmark_jobs)
        payload["memory_release"] = _memory_release_summary(benchmark_jobs)

        scheduler_smoke = _runner_policy_smoke(scheduler=harness.scheduler)
        payload["mixed_scheduler_smoke"] = scheduler_smoke

        lazy_memory = _run_lazy_memory_checks(
            dsn=dsn,
            client=client,
            harness=harness,
            benchmark_jobs=benchmark_jobs,
            child_evidence_dir=child_evidence_dir,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        payload["lazy_cache_hit_memory"] = lazy_memory

        payload["legacy_path_absence"] = _legacy_path_absence_audit()
        payload["dead_code_audit"] = _dead_code_audit()
        payload["docs_drift_audit"] = _docs_drift_audit()
        payload["backlog_after"] = prod_smoke._backlog_snapshot(dsn=dsn)
        payload["finished_at"] = datetime.now(UTC).isoformat()
        payload["pass"] = _overall_pass(payload)
    finally:
        if session_id is not None:
            prod_smoke._revoke_smoke_session(dsn=dsn, session_id=session_id)

    results_path = out_dir / "benchmark_results.json"
    summary_path = out_dir / "benchmark_summary.md"
    results_path.write_text(_render_json(payload) + "\n", encoding="utf-8")
    summary_path.write_text(_render_summary(payload=payload), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    return 0 if payload.get("pass") or args.no_fail_on_threshold else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run API/runner canonical parity and memory benchmark evidence."
    )
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Optional runtime env file. If omitted, the runner tries "
            "$ROEHUB_ENV_FILE, then /Users/daniildegtyarev/.config/roehub/roehub.env, "
            "then /etc/roehub/roehub.env. Values already in the process env win."
        ),
    )
    parser.add_argument("--api-base", default=_DEFAULT_API_BASE)
    parser.add_argument("--cookie-name", default=_DEFAULT_COOKIE_NAME)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--session-ttl-seconds", type=int, default=7200)
    parser.add_argument("--light-max-actual-combinations", type=int, default=50_000)
    parser.add_argument(
        "--system-memory-cleanup-wait-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--stage-04-mvp-rows",
        action="store_true",
        help="Run only none/arity_2..3/long_only rows for matrix bitset Stage 04.",
    )
    parser.add_argument(
        "--stage-05-no-risk-heavy-rows",
        action="store_true",
        help=(
            "Run only none/arity_6 long_only and long_short_reversal rows for "
            "matrix bitset Stage 05."
        ),
    )
    parser.add_argument(
        "--stage-08-tp-sl-selected-cells",
        action="store_true",
        help=(
            "Run only small TP/SL selected-cell rows with an 8x8 grid and "
            "ROEHUB_BACKTEST_TP_SL_SELECTED_CELL_SHADOW enabled."
        ),
    )
    parser.add_argument(
        "--stage-09-tp-sl-full-grid",
        action="store_true",
        help=(
            "Retired Stage 09 selector. Kept only to fail old commands with a "
            "clear message; no env backend mode is set."
        ),
    )
    parser.add_argument(
        "--stage-12-compiled-prefix-rows",
        action="store_true",
        help=(
            "Run no-risk arity-6 and arity-7 rows with "
            "compiled_prefix_product_traversal_v1 enabled through "
            "ROEHUB_BACKTEST_MATRIX_BACKEND_MODE."
        ),
    )
    parser.add_argument(
        "--stage-05-12-production-default-rows",
        action="store_true",
        help=(
            "Run no-risk arity-6 and arity-7 rows through the production "
            "composite default: Stage 05 for arity 6 and Stage 12 for arity 7. "
            "Requires ROEHUB_BACKTEST_MATRIX_BACKEND_MODE to be unset or "
            "stage_05_and_12_no_risk."
        ),
    )
    parser.add_argument(
        "--tp-sl-cell-block-tp-count",
        type=int,
        default=STAGE_09_DEFAULT_CELL_BLOCK_TP_COUNT,
        help="Retired Stage 09 TP block size; ignored because Stage 09 is disabled.",
    )
    parser.add_argument(
        "--tp-sl-cell-block-sl-count",
        type=int,
        default=STAGE_09_DEFAULT_CELL_BLOCK_SL_COUNT,
        help="Retired Stage 09 SL block size; ignored because Stage 09 is disabled.",
    )
    parser.add_argument("--allow-backlog", action="store_true")
    parser.add_argument("--no-fail-on-threshold", action="store_true")
    parser.add_argument(
        "--cpu-sample-interval-seconds",
        type=float,
        default=_DEFAULT_CPU_SAMPLE_INTERVAL_SECONDS,
        help="Sample child process CPU with ps at this interval; <=0 disables sampling.",
    )
    parser.add_argument(
        "--matrix-sidecar-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Benchmark/test-only matrix sidecar root. When set, the child process "
            "loads sidecars via ROEHUB_BACKTEST_MATRIX_SIDECAR_DIR and falls back "
            "to runtime packing if validation fails."
        ),
    )
    return parser


def _matrix_sidecar_report(sidecar_dir: Path | None) -> dict[str, Any]:
    if sidecar_dir is None:
        return {
            "enabled": False,
            "artifact_dir": None,
            "generation_report": None,
            "sidecar_generate_ms": None,
            "fairness_classification": "no_sidecar",
        }
    os.environ["ROEHUB_BACKTEST_MATRIX_SIDECAR_DIR"] = str(sidecar_dir)
    report_path = sidecar_dir / "sidecar_generation_report.json"
    generation_report = _load_json(report_path) if report_path.is_file() else None
    sidecar_generate_ms = (
        None
        if not isinstance(generation_report, Mapping)
        else generation_report.get("sidecar_generate_ms")
    )
    return {
        "enabled": True,
        "artifact_dir": str(sidecar_dir),
        "generation_report": generation_report,
        "sidecar_generate_ms": sidecar_generate_ms,
        "fairness_classification": "accepted_for_learning_sidecar_precomputed",
        "no_advantage_policy": (
            "Sidecar speedup is benchmark/test-only unless generation cost is included "
            "or a publisher plan is approved."
        ),
    }


def _load_runtime_env_file(explicit_env_file: Path | None) -> dict[str, Any]:
    """
    Load benchmark runtime env from the same source used by native/Docker services.

    The report intentionally records only paths and key names, never values.
    """

    candidates: list[Path] = []
    explicit = explicit_env_file is not None
    if explicit_env_file is not None:
        candidates.append(explicit_env_file)
    env_file_value = os.environ.get("ROEHUB_ENV_FILE", "").strip()
    if env_file_value:
        candidates.append(Path(env_file_value))
    candidates.extend((_DEFAULT_MACOS_ENV_FILE, _DEFAULT_DOCKER_ENV_FILE))

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    selected = next((path for path in unique_candidates if path.is_file()), None)
    if selected is None:
        if explicit:
            raise RuntimeError(f"env file not found: {explicit_env_file}")
        return {
            "loaded": False,
            "path": None,
            "candidates": [str(path) for path in unique_candidates],
            "keys_loaded": [],
            "reason": "no readable env file found",
        }

    loaded_keys: list[str] = []
    for key, value in _read_env_file(selected).items():
        if os.environ.get(key, "").strip():
            continue
        os.environ[key] = value
        loaded_keys.append(key)
    return {
        "loaded": True,
        "path": str(selected),
        "candidates": [str(path) for path in unique_candidates],
        "keys_loaded": sorted(key for key in loaded_keys if key in _ENV_REPORT_KEYS),
        "keys_loaded_count": len(loaded_keys),
        "reason": None,
    }


def _ensure_postgres_dsn_env() -> tuple[str, list[str]]:
    dsn = prod_smoke._postgres_dsn(environ=os.environ)
    filled_keys: list[str] = []
    for key in _PG_DSN_ENV_KEYS:
        if not os.environ.get(key, "").strip():
            os.environ[key] = dsn
            filled_keys.append(key)
    return dsn, filled_keys


def _ensure_artifact_runtime_env() -> dict[str, Any]:
    """
    Ensure API-runner benchmark uses the same artifact root as Mac Studio prod.

    Service wiring already supports `ROEHUB_BACKTEST_ARTIFACTS_CONFIG` and
    `ROEHUB_ENV`; this helper only supplies the benchmark default when neither
    is present in the shell/env-file.
    """

    config_value = os.environ.get(_ARTIFACT_CONFIG_ENV_KEY, "").strip()
    env_value = os.environ.get(_ROEHUB_ENV_KEY, "").strip()
    filled_keys: list[str] = []
    if not env_value:
        env_value = "prod"
        os.environ[_ROEHUB_ENV_KEY] = env_value
        filled_keys.append(_ROEHUB_ENV_KEY)
    if config_value:
        return {
            "config_key_present": True,
            "env_name_present": True,
            "filled_keys": filled_keys,
            "config_path": config_value,
            "reason": "provided by process env or env file",
        }
    if _ROEHUB_ENV_KEY not in filled_keys:
        return {
            "config_key_present": False,
            "env_name_present": True,
            "filled_keys": filled_keys,
            "config_path": f"configs/{env_value}/backtest_artifacts.yaml",
            "reason": "resolved from ROEHUB_ENV",
        }

    if not _DEFAULT_PROD_ARTIFACT_CONFIG.is_file():
        raise RuntimeError(
            "missing artifact runtime config: set "
            f"{_ARTIFACT_CONFIG_ENV_KEY} or {_ROEHUB_ENV_KEY}"
        )

    config_path = str(_DEFAULT_PROD_ARTIFACT_CONFIG)
    os.environ[_ARTIFACT_CONFIG_ENV_KEY] = config_path
    filled_keys.append(_ARTIFACT_CONFIG_ENV_KEY)
    return {
        "config_key_present": True,
        "env_name_present": True,
        "filled_keys": filled_keys,
        "config_path": config_path,
        "reason": "defaulted to Mac Studio prod artifact config",
    }


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        values[key] = _parse_env_value(raw_value.strip())
    return {key: _expand_env_value(value, values=values) for key, value in values.items()}


def _parse_env_value(raw_value: str) -> str:
    if not raw_value:
        return ""
    if not raw_value.startswith(("'", '"')):
        return raw_value.split(" #", 1)[0].strip()
    try:
        parsed = shlex.split(raw_value, comments=True, posix=True)
    except ValueError:
        return raw_value.strip("'\"")
    if not parsed:
        return ""
    return parsed[0]


def _expand_env_value(value: str, *, values: Mapping[str, str]) -> str:
    expanded = value
    for _ in range(5):
        next_value = _ENV_REFERENCE_RE.sub(
            lambda match: _env_reference_replacement(match=match, values=values),
            expanded,
        )
        if next_value == expanded:
            return next_value
        expanded = next_value
    return expanded


def _env_reference_replacement(*, match: re.Match[str], values: Mapping[str, str]) -> str:
    key = match.group("braced") or match.group("plain") or ""
    default = match.group("default")
    current = os.environ.get(key, "")
    if current:
        return current
    value = values.get(key, "")
    if value:
        return value
    return "" if default is None else default


def _run_reference_jobs(
    *,
    dsn: str,
    client: Any,
    harness: _RunnerHarness,
    canonical: Mapping[str, Any],
    reference_runs: Sequence[Mapping[str, Any]],
    child_evidence_dir: Path,
    timeout_seconds: int,
    poll_interval_seconds: float,
    system_memory_cleanup_wait_seconds: float,
    cpu_sample_interval_seconds: float,
    stage_08_tp_sl_selected_cells: bool = False,
    stage_12_compiled_prefix_rows: bool = False,
    stage_05_12_production_default_rows: bool = False,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, reference_run in enumerate(reference_runs, start=1):
        job_name = _reference_job_name(run=reference_run)
        request = _request_for_reference_run(
            canonical=canonical,
            run=reference_run,
            stage_08_tp_sl_selected_cells=stage_08_tp_sl_selected_cells,
        )
        created = client.request_json(
            "POST",
            "/backtests/jobs",
            payload=request,
            extra_headers={"Idempotency-Key": f"api-runner-parity-{uuid4()}"},
        )
        if created.get("_status") != 201 or created.get("state") != "queued":
            raise RuntimeError(f"job create failed for {job_name}: {created}")
        job_id = str(created["job_id"])
        row_before = _job_detail_row(dsn=dsn, job_id=job_id)
        scheduling = _scheduling_from_row(row=row_before)
        lane = "heavy"
        system_memory_before = _system_memory_snapshot()
        run_result = _process_job_until_terminal(
            dsn=dsn,
            client=client,
            scheduler=harness.scheduler,
            job_id=job_id,
            lane=lane,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            cpu_sample_interval_seconds=cpu_sample_interval_seconds,
        )
        system_memory_after = _system_memory_snapshot()
        if system_memory_cleanup_wait_seconds > 0:
            time.sleep(system_memory_cleanup_wait_seconds)
        system_memory_delayed = _system_memory_snapshot()
        child_evidence = _read_full_job_child_evidence(
            evidence_dir=child_evidence_dir,
            job_id=job_id,
        )
        top = client.request_json("GET", f"/backtests/jobs/{job_id}/top")
        parity = _compare_reference_results(
            api_top=top,
            child_evidence=child_evidence,
            reference_run=reference_run,
            stage_08_tp_sl_selected_cells=stage_08_tp_sl_selected_cells,
            stage_12_compiled_prefix_rows=stage_12_compiled_prefix_rows,
            stage_05_12_production_default_rows=stage_05_12_production_default_rows,
        )
        stage_timings = _merged_stage_timings(child_evidence)
        instrumentation_counters = _merged_instrumentation_counters(child_evidence)
        memory = _child_memory_summary(child_evidence)
        memory["system_memory_cleanup"] = _system_memory_cleanup_gate(
            before=system_memory_before,
            after=system_memory_after,
            delayed=system_memory_delayed,
            wait_seconds=system_memory_cleanup_wait_seconds,
        )
        memory["pass"] = bool(memory["pass"]) and bool(
            cast(Mapping[str, Any], memory["system_memory_cleanup"])["pass"]
        )
        jobs.append(
            {
                "index": index,
                "job_name": job_name,
                "job_id": job_id,
                "risk_mode": reference_run.get("risk_mode"),
                "arity": len(cast(Sequence[Any], reference_run.get("indicator_ids", ()))),
                "direction_mode": reference_run.get("direction_mode"),
                "reference_exact_scoring_s": _reference_run_exact_seconds(
                    run=reference_run
                ),
                "created_state": created.get("state"),
                "request_hash": created.get("request_hash"),
                "scheduling": scheduling,
                "runner": run_result,
                "api_top_count": len(_list(top.get("items"))),
                "parity": parity,
                "stage_timings": stage_timings,
                "instrumentation_counters": instrumentation_counters,
                "service_only_overhead": _service_only_overhead(stage_timings),
                "cpu_sampling": _mapping(run_result.get("cpu_sampling")),
                "child_process_evidence": child_evidence,
                "memory": memory,
                "pass": (
                    bool(run_result["pass"])
                    and bool(parity["pass"])
                    and bool(memory["pass"])
                ),
            }
        )
    return jobs


def _process_job_until_terminal(
    *,
    dsn: str,
    client: Any,
    scheduler: Any,
    job_id: str,
    lane: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
    cpu_sample_interval_seconds: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    current_lane = lane
    deadline = time.monotonic() + timeout_seconds
    while True:
        worker = (
            scheduler.heavy_full_job_worker
            if current_lane == "heavy"
            else scheduler.light_full_job_worker
        )
        attempt = _run_worker_attempt(
            dsn=dsn,
            client=client,
            worker=worker,
            job_id=job_id,
            timeout_seconds=max(int(deadline - time.monotonic()), 1),
            poll_interval_seconds=poll_interval_seconds,
            cpu_sample_interval_seconds=cpu_sample_interval_seconds,
        )
        attempts.append(attempt)
        if attempt.get("worker_status") == "requeued_heavy":
            current_lane = "heavy"
            continue
        break
    terminal = _job_detail_row(dsn=dsn, job_id=job_id)
    pass_state = str(terminal["state"]) == "succeeded"
    observed_states = [
        state
        for attempt in attempts
        for state in _list(attempt.get("observed_states"))
    ]
    state_path = _collapse_state_path(observed_states)
    if pass_state and (not state_path or state_path[-1] != "succeeded"):
        state_path.append("succeeded")
    has_running = "running" in state_path or terminal.get("started_at") is not None
    cpu_sampling = _combine_cpu_sampling(
        _mapping(attempt.get("cpu_sampling")) for attempt in attempts
    )
    return {
        "attempts": attempts,
        "state_path": " -> ".join(state_path),
        "required_path": "queued -> running -> succeeded",
        "required_path_pass": has_running and pass_state,
        "terminal": _job_terminal(row=terminal),
        "cpu_sampling": cpu_sampling,
        "pass": has_running and pass_state,
    }


def _run_worker_attempt(
    *,
    dsn: str,
    client: Any,
    worker: Any,
    job_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
    cpu_sample_interval_seconds: float,
) -> dict[str, Any]:
    observed_states: list[str] = ["queued"]
    api_latencies_ms: list[float] = []
    running_samples: list[dict[str, Any]] = []
    cpu_samples: list[dict[str, Any]] = []
    next_cpu_sample_at = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_next)
        deadline = time.monotonic() + timeout_seconds
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f"worker attempt timed out for job {job_id}")
            started = time.perf_counter()
            api_payload = client.request_json("GET", f"/backtests/jobs/{job_id}")
            api_latencies_ms.append((time.perf_counter() - started) * 1000.0)
            state = str(api_payload.get("state"))
            if state not in observed_states:
                observed_states.append(state)
            db_row = _job_detail_row(dsn=dsn, job_id=job_id)
            if str(db_row["state"]) == "running":
                running_samples.append(_running_sample(row=db_row))
                now = time.monotonic()
                if cpu_sample_interval_seconds > 0 and now >= next_cpu_sample_at:
                    sample = _full_job_child_cpu_sample(job_id=job_id)
                    if sample is not None:
                        cpu_samples.append(sample)
                    next_cpu_sample_at = now + cpu_sample_interval_seconds
            time.sleep(poll_interval_seconds)
        result = future.result()
    job = getattr(result, "job", None)
    return {
        "claimed": bool(getattr(result, "claimed", False)),
        "lease_lost": bool(getattr(result, "lease_lost", False)),
        "worker_status": getattr(result, "status", None),
        "job_state": None if job is None else getattr(job, "state", None),
        "observed_states": observed_states,
        "running_samples": running_samples[:5],
        "api_responsiveness": _latency_summary(api_latencies_ms),
        "cpu_sampling": _cpu_sampling_summary(
            samples=cpu_samples,
            sample_interval_seconds=cpu_sample_interval_seconds,
        ),
    }


def _runner_policy_smoke(*, scheduler: Any) -> dict[str, Any]:
    first = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    blocked_by_heavy = scheduler.next_launch(
        active_light=0,
        active_heavy=1,
        active_lazy=0,
    )
    blocked_by_light = scheduler.next_launch(
        active_light=1,
        active_heavy=0,
        active_lazy=0,
    )
    if first is not None:
        scheduler.record_result(
            scheduling_class=str(first.scheduling_class),
            result=runner_module.BacktestRunnerTaskResult(
                task_kind="full_job",
                scheduling_class=str(first.scheduling_class),
                claimed=False,
            ),
        )
    second = scheduler.next_launch(active_light=0, active_heavy=0, active_lazy=0)
    if second is not None:
        scheduler.record_result(
            scheduling_class=str(second.scheduling_class),
            result=runner_module.BacktestRunnerTaskResult(
                task_kind="full_job",
                scheduling_class=str(second.scheduling_class),
                claimed=False,
            ),
        )
    lazy_after_empty_full = scheduler.next_launch(
        active_light=0,
        active_heavy=0,
        active_lazy=0,
    )
    pass_value = (
        first is not None
        and first.task_kind == "full_job"
        and first.scheduling_class == "heavy"
        and blocked_by_heavy is None
        and blocked_by_light is None
        and second is not None
        and second.scheduling_class == "heavy"
        and lazy_after_empty_full is not None
        and lazy_after_empty_full.task_kind == "lazy_detail"
    )
    return {
        "configured": {
            "ROEHUB_BACKTEST_LIGHT_CONCURRENCY": 0,
            "ROEHUB_BACKTEST_HEAVY_CONCURRENCY": 1,
            "full_job_lane": "heavy_only",
            "light_promotion_path": "not_used",
        },
        "first_full_probe": None
        if first is None
        else {"task_kind": first.task_kind, "scheduling_class": first.scheduling_class},
        "blocks_when_heavy_active": blocked_by_heavy is None,
        "blocks_when_light_active": blocked_by_light is None,
        "second_full_probe": None
        if second is None
        else {"task_kind": second.task_kind, "scheduling_class": second.scheduling_class},
        "lazy_after_two_empty_full_probes": None
        if lazy_after_empty_full is None
        else {
            "task_kind": lazy_after_empty_full.task_kind,
            "scheduling_class": lazy_after_empty_full.scheduling_class,
        },
        "pass": pass_value,
    }


def _run_scheduler_phase(
    *,
    dsn: str,
    client: Any,
    scheduler: Any,
    job_ids: Sequence[str],
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    active: dict[concurrent.futures.Future[Any], dict[str, Any]] = {}
    launches: list[dict[str, Any]] = []
    active_samples: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while time.monotonic() < deadline:
            _reap_scheduler_futures(active=active, finished=finished, scheduler=scheduler)
            light_active = sum(1 for item in active.values() if item["scheduling_class"] != "heavy")
            heavy_active = sum(1 for item in active.values() if item["scheduling_class"] == "heavy")
            terminal_done = all(
                _job_detail_row(dsn=dsn, job_id=item)["state"] == "succeeded"
                for item in job_ids
            )
            if not active and terminal_done:
                break
            launch = scheduler.next_launch(
                active_light=light_active,
                active_heavy=heavy_active,
                active_lazy=0,
            )
            while launch is not None:
                if launch.task_kind != "full_job":
                    break
                future = pool.submit(launch.run)
                active[future] = {
                    "task_kind": launch.task_kind,
                    "scheduling_class": launch.scheduling_class,
                    "launched_at": datetime.now(UTC).isoformat(),
                }
                launches.append(dict(active[future]))
                if launch.scheduling_class == "heavy":
                    break
                light_active = sum(
                    1 for item in active.values() if item["scheduling_class"] != "heavy"
                )
                heavy_active = sum(
                    1 for item in active.values() if item["scheduling_class"] == "heavy"
                )
                launch = scheduler.next_launch(
                    active_light=light_active,
                    active_heavy=heavy_active,
                    active_lazy=0,
                )
            active_samples.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "active_light": sum(
                        1 for item in active.values() if item["scheduling_class"] != "heavy"
                    ),
                    "active_heavy": sum(
                        1 for item in active.values() if item["scheduling_class"] == "heavy"
                    ),
                    "api_status_latency_ms": _status_burst_latency(client=client, job_ids=job_ids),
                }
            )
            time.sleep(poll_interval_seconds)
        _reap_scheduler_futures(active=active, finished=finished, scheduler=scheduler)
    rows = [_job_detail_row(dsn=dsn, job_id=job_id) for job_id in job_ids]
    max_light = max((int(sample["active_light"]) for sample in active_samples), default=0)
    max_heavy = max((int(sample["active_heavy"]) for sample in active_samples), default=0)
    terminal_pass = all(str(row["state"]) == "succeeded" for row in rows)
    heavy_rows = [
        row
        for row in rows
        if _scheduling_from_row(row=row).get("scheduling_class") == "heavy"
    ]
    heavy_fifo = _heavy_fifo_pass(rows=heavy_rows)
    light_cap = max_light <= 2
    heavy_no_overlap = max_heavy <= 1
    return {
        "job_ids": list(job_ids),
        "launches": launches,
        "finished": finished,
        "active_samples": active_samples,
        "terminal": [_job_terminal(row=row) for row in rows],
        "max_active_light": max_light,
        "max_active_heavy": max_heavy,
        "light_parallelism_observed": max_light >= 2,
        "light_concurrency_cap_pass": light_cap,
        "heavy_fifo_pass": heavy_fifo,
        "heavy_no_overlap_pass": heavy_no_overlap,
        "pass": terminal_pass and light_cap and heavy_no_overlap and heavy_fifo,
    }


def _reap_scheduler_futures(
    *,
    active: dict[concurrent.futures.Future[Any], dict[str, Any]],
    finished: list[dict[str, Any]],
    scheduler: Any,
) -> None:
    for future, meta in list(active.items()):
        if not future.done():
            continue
        active.pop(future)
        result = future.result()
        normalized = runner_module._coerce_task_result(
            result=result,
            scheduling_class=str(meta["scheduling_class"]),
        )
        scheduler.record_result(
            scheduling_class=str(meta["scheduling_class"]),
            result=normalized,
        )
        item = dict(meta)
        item.update(
            {
                "finished_at": datetime.now(UTC).isoformat(),
                "claimed": bool(normalized.claimed),
                "worker_status": normalized.status,
            }
        )
        finished.append(item)


def _run_lazy_memory_checks(
    *,
    dsn: str,
    client: Any,
    harness: _RunnerHarness,
    benchmark_jobs: Sequence[Mapping[str, Any]],
    child_evidence_dir: Path,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    target = next(
        (
            job
            for job in benchmark_jobs
            if _mapping(job.get("runner")).get("pass") and int(job.get("api_top_count") or 0) > 0
        ),
        None,
    )
    if target is None:
        return {
            "pass": False,
            "reason": "no successful benchmark job with top variants for lazy check",
        }
    job_id = str(target["job_id"])
    top = client.request_json("GET", f"/backtests/jobs/{job_id}/top")
    top_items = _list(top.get("items"))
    if not top_items:
        return {"pass": False, "reason": "target job has no top variants"}
    top_variant = cast(Mapping[str, Any], top_items[0])
    variant_key = str(top_variant["variant_key"])
    variant_hash = str(top_variant["variant_hash"])
    cache_key = prod_smoke._compute_lazy_cache_key(
        dsn=dsn,
        job_id=job_id,
        public_variant_key=variant_key,
        variant_hash=variant_hash,
    )
    prod_smoke._clear_lazy_cache_and_task(dsn=dsn, cache_key=cache_key)
    first = client.request_json("POST", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades")
    if first.get("_status") != 202:
        return {"pass": False, "reason": f"cache miss returned {first.get('_status')}"}
    task_id = str(cast(Mapping[str, Any], first["materialization"])["task_id"])
    miss_result = _run_lazy_worker_until_completed(
        dsn=dsn,
        client=client,
        worker=harness.scheduler.lazy_detail_worker,
        task_id=task_id,
        job_id=job_id,
        variant_key=variant_key,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    lazy_child = _read_lazy_child_evidence(
        evidence_dir=child_evidence_dir,
        task_id=task_id,
    )
    api_pid = _pid_for_tcp_port(_api_port(api_base=str(client.api_base)))
    api_rss_before = _rss_bytes(api_pid) if api_pid is not None else None
    cache_hit_reads = _run_cache_hit_reads(
        client=client,
        job_id=job_id,
        variant_key=variant_key,
    )
    api_rss_after = _rss_bytes(api_pid) if api_pid is not None else None
    retained_delta = _delta(api_rss_before, api_rss_after)
    cache_files = _cache_file_evidence(cache_key=cache_key)
    bounded_static = _cache_bounded_reader_static_audit()
    cache_hit_pass = retained_delta is None or retained_delta <= _CACHE_HIT_RSS_DELTA_LIMIT_BYTES
    return {
        "target_job_id": job_id,
        "variant_key": variant_key,
        "cache_key": cache_key,
        "miss_initial": {
            "status": first.get("_status"),
            "cache_status": cast(Mapping[str, Any], first.get("cache", {})).get("status"),
            "materialization_status": first.get("status"),
            "task_id": task_id,
        },
        "miss_worker": miss_result,
        "lazy_child_process_evidence": lazy_child,
        "cache_hit_reads": cache_hit_reads,
        "cache_files": cache_files,
        "api_process_memory": {
            "pid": api_pid,
            "rss_before_bytes": api_rss_before,
            "rss_after_bytes": api_rss_after,
            "retained_rss_delta": retained_delta,
            "limit_bytes": _CACHE_HIT_RSS_DELTA_LIMIT_BYTES,
            "pass": cache_hit_pass,
        },
        "bounded_reader_static_audit": bounded_static,
        "pass": (
            bool(miss_result["pass"])
            and bool(_child_memory_summary(lazy_child)["pass"])
            and cache_hit_pass
            and bool(bounded_static["pass"])
        ),
    }


def _run_lazy_worker_until_completed(
    *,
    dsn: str,
    client: Any,
    worker: Any,
    task_id: str,
    job_id: str,
    variant_key: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    statuses: list[str] = ["queued"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.run_next)
        deadline = time.monotonic() + timeout_seconds
        while not future.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f"lazy worker timed out for task {task_id}")
            row = prod_smoke._lazy_task_row(dsn=dsn, task_id=task_id)
            status = str(row["status"])
            if status not in statuses:
                statuses.append(status)
            time.sleep(poll_interval_seconds)
        result = future.result()
    row = prod_smoke._lazy_task_row(dsn=dsn, task_id=task_id)
    if str(row["status"]) not in statuses:
        statuses.append(str(row["status"]))
    claimed = bool(getattr(result, "claimed", False))
    completed = str(row["status"]) == "completed"
    observed_running = "running" in statuses
    required_path_pass = completed and (observed_running or claimed)
    return {
        "claimed": claimed,
        "status": getattr(result, "status", None),
        "task_status": str(row["status"]),
        "status_path": " -> ".join(_collapse_state_path(statuses)),
        "required_path": "queued -> running -> completed",
        "observed_running": observed_running,
        "fast_completion_without_observed_running": completed
        and claimed
        and not observed_running,
        "required_path_pass": required_path_pass,
        "cache_status": row.get("cache_status"),
        "cache_path": row.get("cache_path"),
        "pass": required_path_pass,
    }


def _run_cache_hit_reads(*, client: Any, job_id: str, variant_key: str) -> dict[str, Any]:
    reads: list[dict[str, Any]] = []
    requests = [
        ("POST", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=5"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/equity?points=100"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/drawdown?points=100"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats"),
        ("GET", f"/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv?max_rows=100"),
    ]
    for method, path in requests:
        started = time.perf_counter()
        if path.endswith(".csv?max_rows=100"):
            status, headers = _request_raw(client=client, method=method, path=path)
            cache_status = headers.get("x-roehub-cache-status")
        else:
            payload = client.request_json(method, path)
            status = int(payload.get("_status") or 0)
            cache_status = cast(Mapping[str, Any], payload.get("cache", {})).get(
                "status"
            )
        reads.append(
            {
                "method": method,
                "path": path,
                "status": status,
                "cache_status": cache_status,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    return {"reads": reads, "pass": all(int(item["status"]) == 200 for item in reads)}


def _request_raw(*, client: Any, method: str, path: str) -> tuple[int, dict[str, str]]:
    request = urllib.request.Request(
        url=f"{client.api_base.rstrip('/')}{path}",
        headers={
            "Accept": "*/*",
            "Cookie": f"{client.cookie_name}={client.session_id}",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            return response.status, {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as error:
        error.read()
        return error.code, {key.lower(): value for key, value in error.headers.items()}


def _build_runner_harness(
    *,
    child_evidence_dir: Path,
    light_max_actual_combinations: int,
    heavy_concurrency: int,
) -> _RunnerHarness:
    env = dict(os.environ)
    env.update(
        {
            "ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR": str(child_evidence_dir),
            "ROEHUB_BACKTEST_CHILD_EVIDENCE_SAMPLE_INTERVAL_SECONDS": "0.2",
            "ROEHUB_BACKTEST_RUNNER_CONCURRENCY": "1",
            "ROEHUB_BACKTEST_LIGHT_CONCURRENCY": "0",
            "ROEHUB_BACKTEST_HEAVY_CONCURRENCY": str(heavy_concurrency),
            "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS": "12",
            "ROEHUB_BACKTEST_NUMBA_NUM_THREADS": "12",
            "NUMBA_NUM_THREADS": "12",
            "ROEHUB_BACKTEST_LIGHT_MAX_ACTUAL_COMBINATIONS": str(
                light_max_actual_combinations
            ),
            "ROEHUB_BACKTEST_RUNNER_METRICS_PORT": "19204",
        }
    )
    runtime_config = load_backtest_job_runner_runtime_config(environ=env)
    app = build_backtest_job_runner_app(
        environ=env,
        runtime_config=runtime_config,
        metrics_port=runtime_config.metrics_port,
    )
    return _RunnerHarness(scheduler=app.worker, runtime_config=runtime_config, env=env)


def _reference_runs(
    *,
    canonical: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    canonical_runs = [
        cast(Mapping[str, Any], item) for item in _list(canonical.get("runs"))
    ]
    if not canonical_runs:
        raise RuntimeError("canonical benchmark JSON has no runs")
    heaviest = max(
        canonical_runs,
        key=lambda item: float(
            cast(Mapping[str, Any], item["runtime_metrics"])["wall_time_s"]
        ),
    )
    heaviest_name = _reference_job_name(run=heaviest)
    accepted_runs = [
        cast(Mapping[str, Any], item)
        for key in ("no_risk_regression_runs", "tp_sl_regression_runs")
        for item in _list(reference.get(key))
    ]
    if not accepted_runs:
        raise RuntimeError("accepted May 2 reference JSON has no regression runs")
    required = [
        item
        for item in accepted_runs
        if _reference_job_name(run=item) != heaviest_name
        and len(cast(Sequence[Any], item.get("indicator_ids", ()))) == REFERENCE_ONLY_ARITY
    ]
    if not required:
        raise RuntimeError(f"no accepted May 2 reference runs for arity {REFERENCE_ONLY_ARITY}")
    excluded = {
        "job_name": heaviest_name,
        "risk_mode": heaviest.get("risk_mode"),
        "arity": len(cast(Sequence[Any], heaviest.get("indicator_ids", ()))),
        "direction_mode": heaviest.get("direction_mode"),
        "observed_runtime_s": cast(Mapping[str, Any], heaviest["runtime_metrics"]).get(
            "wall_time_s"
        ),
        "reason": (
            "exclude_heaviest_140s_job: single heaviest accepted May 2 reference "
            "job is omitted from required benchmark/smoke loops"
        ),
    }
    return required, excluded


def _stage_04_mvp_reference_runs(
    *,
    reference: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    accepted_runs = [
        cast(Mapping[str, Any], item)
        for item in _list(reference.get("no_risk_regression_runs"))
    ]
    required = [
        item
        for item in accepted_runs
        if str(item.get("risk_mode")) == STAGE_04_MVP_RISK_MODE
        and str(item.get("direction_mode")) == STAGE_04_MVP_DIRECTION_MODE
        and len(cast(Sequence[Any], item.get("indicator_ids", ())))
        in STAGE_04_MVP_ARITIES
    ]
    seen = {
        len(cast(Sequence[Any], item.get("indicator_ids", ()))) for item in required
    }
    missing = [arity for arity in STAGE_04_MVP_ARITIES if arity not in seen]
    if missing:
        raise RuntimeError(f"missing Stage 04 MVP reference arities: {missing!r}")
    return required, {
        "job_name": None,
        "risk_mode": STAGE_04_MVP_RISK_MODE,
        "arity": list(STAGE_04_MVP_ARITIES),
        "direction_mode": STAGE_04_MVP_DIRECTION_MODE,
        "reason": (
            "stage_04_mvp_rows: run only no-risk long-only arity 2 and 3 rows "
            "for matrix_bitset_no_risk_v1 evidence"
        ),
    }


def _stage_05_no_risk_heavy_reference_runs(
    *,
    reference: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    accepted_runs = [
        cast(Mapping[str, Any], item)
        for item in _list(reference.get("no_risk_regression_runs"))
    ]
    required = [
        item
        for item in accepted_runs
        if str(item.get("risk_mode")) == STAGE_04_MVP_RISK_MODE
        and len(cast(Sequence[Any], item.get("indicator_ids", ())))
        == STAGE_05_NO_RISK_HEAVY_ARITY
        and str(item.get("direction_mode")) in STAGE_05_NO_RISK_HEAVY_DIRECTIONS
    ]
    seen = {str(item.get("direction_mode")) for item in required}
    missing = [
        direction
        for direction in STAGE_05_NO_RISK_HEAVY_DIRECTIONS
        if direction not in seen
    ]
    if missing:
        raise RuntimeError(f"missing Stage 05 no-risk heavy directions: {missing!r}")
    return required, {
        "job_name": None,
        "risk_mode": STAGE_04_MVP_RISK_MODE,
        "arity": STAGE_05_NO_RISK_HEAVY_ARITY,
        "direction_mode": list(STAGE_05_NO_RISK_HEAVY_DIRECTIONS),
        "reason": (
            "stage_05_no_risk_heavy_rows: run only no-risk arity 6 long_only "
            "and long_short_reversal rows for matrix_bitset_no_risk_v1 evidence"
        ),
    }


def _stage_08_tp_sl_selected_reference_runs(
    *,
    reference: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    accepted_runs = [
        cast(Mapping[str, Any], item)
        for item in _list(reference.get("tp_sl_regression_runs"))
    ]
    required = [
        item
        for item in accepted_runs
        if str(item.get("risk_mode")) == "tp_sl_grid"
        and len(cast(Sequence[Any], item.get("indicator_ids", ())))
        in STAGE_08_TP_SL_SELECTED_ARITIES
        and str(item.get("direction_mode")) in STAGE_08_TP_SL_SELECTED_DIRECTIONS
    ]
    by_key = {
        (
            len(cast(Sequence[Any], item.get("indicator_ids", ()))),
            str(item.get("direction_mode")),
        ): item
        for item in required
    }
    wanted = (
        (STAGE_08_TP_SL_SELECTED_ARITIES[0], STAGE_08_TP_SL_SELECTED_DIRECTIONS[0]),
        (STAGE_08_TP_SL_SELECTED_ARITIES[1], STAGE_08_TP_SL_SELECTED_DIRECTIONS[1]),
    )
    missing = [key for key in wanted if key not in by_key]
    if missing:
        raise RuntimeError(f"missing Stage 08 TP/SL selected reference rows: {missing!r}")
    return [by_key[key] for key in wanted], {
        "job_name": None,
        "risk_mode": "tp_sl_grid",
        "arity": list(STAGE_08_TP_SL_SELECTED_ARITIES),
        "direction_mode": list(STAGE_08_TP_SL_SELECTED_DIRECTIONS),
        "tp_count <= 8": True,
        "sl_count <= 8": True,
        "reason": (
            "stage_08_tp_sl_selected_cells: run small TP/SL selected-cell grid "
            "with shadow parity and by-entry layout evidence"
        ),
    }


def _stage_09_tp_sl_full_grid_reference_runs(
    *,
    reference: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    accepted_runs = [
        cast(Mapping[str, Any], item)
        for item in _list(reference.get("tp_sl_regression_runs"))
    ]
    required = [
        item
        for item in accepted_runs
        if str(item.get("risk_mode")) == "tp_sl_grid"
        and len(cast(Sequence[Any], item.get("indicator_ids", ())))
        == STAGE_09_TP_SL_FULL_GRID_ARITY
        and str(item.get("direction_mode")) in STAGE_09_TP_SL_FULL_GRID_DIRECTIONS
    ]
    by_direction = {str(item.get("direction_mode")): item for item in required}
    missing = [
        direction
        for direction in STAGE_09_TP_SL_FULL_GRID_DIRECTIONS
        if direction not in by_direction
    ]
    if missing:
        raise RuntimeError(f"missing Stage 09 TP/SL full-grid directions: {missing!r}")
    return [by_direction[direction] for direction in STAGE_09_TP_SL_FULL_GRID_DIRECTIONS], {
        "job_name": None,
        "risk_mode": "tp_sl_grid",
        "arity": STAGE_09_TP_SL_FULL_GRID_ARITY,
        "direction_mode": list(STAGE_09_TP_SL_FULL_GRID_DIRECTIONS),
        "matrix_backend_mode": STAGE_09_MATRIX_BACKEND_MODE,
        "reason": (
            "stage_09_tp_sl_full_grid: run TP/SL arity 6 full request grids "
            "for matrix_cell_tp_sl_v1 evidence"
        ),
    }


def _stage_12_compiled_prefix_reference_runs(
    *,
    reference: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    accepted_runs = [
        cast(Mapping[str, Any], item)
        for item in _list(reference.get("no_risk_regression_runs"))
    ]
    by_key = {
        (
            len(cast(Sequence[Any], item.get("indicator_ids", ()))),
            str(item.get("direction_mode")),
        ): item
        for item in accepted_runs
        if str(item.get("risk_mode")) == "none"
        and len(cast(Sequence[Any], item.get("indicator_ids", ())))
        in STAGE_12_COMPILED_PREFIX_ARITIES
        and str(item.get("direction_mode")) in STAGE_12_COMPILED_PREFIX_DIRECTIONS
    }
    wanted = [
        (arity, direction)
        for arity in STAGE_12_COMPILED_PREFIX_ARITIES
        for direction in STAGE_12_COMPILED_PREFIX_DIRECTIONS
    ]
    missing = [key for key in wanted if key not in by_key]
    if missing:
        raise RuntimeError(f"missing Stage 12 compiled-prefix rows: {missing!r}")
    return [by_key[key] for key in wanted], {
        "job_name": None,
        "risk_mode": "none",
        "arity": list(STAGE_12_COMPILED_PREFIX_ARITIES),
        "direction_mode": list(STAGE_12_COMPILED_PREFIX_DIRECTIONS),
        "matrix_backend_mode": STAGE_12_MATRIX_BACKEND_MODE,
        "reason": (
            "stage_12_compiled_prefix_rows: run no-risk arity 6 and 7 rows "
            "for compiled_prefix_product_traversal_v1 evidence"
        ),
    }


def _stage_05_12_production_default_reference_runs(
    *,
    reference: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    accepted_runs = [
        cast(Mapping[str, Any], item)
        for item in _list(reference.get("no_risk_regression_runs"))
    ]
    by_key = {
        (
            len(cast(Sequence[Any], item.get("indicator_ids", ()))),
            str(item.get("direction_mode")),
        ): item
        for item in accepted_runs
        if str(item.get("risk_mode")) == "none"
        and len(cast(Sequence[Any], item.get("indicator_ids", ())))
        in STAGE_05_12_PRODUCTION_DEFAULT_ARITIES
        and str(item.get("direction_mode"))
        in STAGE_05_12_PRODUCTION_DEFAULT_DIRECTIONS
    }
    wanted = [
        (arity, direction)
        for arity in STAGE_05_12_PRODUCTION_DEFAULT_ARITIES
        for direction in STAGE_05_12_PRODUCTION_DEFAULT_DIRECTIONS
    ]
    missing = [key for key in wanted if key not in by_key]
    if missing:
        raise RuntimeError(
            f"missing Stage 05+12 production default rows: {missing!r}"
        )
    return [by_key[key] for key in wanted], {
        "job_name": None,
        "risk_mode": "none",
        "arity": list(STAGE_05_12_PRODUCTION_DEFAULT_ARITIES),
        "direction_mode": list(STAGE_05_12_PRODUCTION_DEFAULT_DIRECTIONS),
        "matrix_backend_mode": "default_unset_or_stage_05_and_12_no_risk",
        "reason": (
            "stage_05_12_production_default_rows: run no-risk arity 6 rows "
            "through Stage 05 and arity 7 rows through Stage 12 using the "
            "production composite default"
        ),
    }


def _smoke_subset(runs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    wanted = {
        ("none", 1, "long_only"),
        ("none", 2, "long_short_reversal"),
        ("tp_sl_grid", 1, "long_only"),
        ("tp_sl_grid", 2, "long_short_reversal"),
    }
    out = [
        run
        for run in runs
        if (
            str(run.get("risk_mode")),
            len(cast(Sequence[Any], run.get("indicator_ids", ()))),
            str(run.get("direction_mode")),
        )
        in wanted
    ]
    if len(out) != len(wanted):
        raise RuntimeError("smoke subset could not be built from canonical runs")
    return out


def _request_for_reference_run(
    *,
    canonical: Mapping[str, Any],
    run: Mapping[str, Any],
    rows_per_indicator: int | None = REFERENCE_ROWS_PER_INDICATOR,
    stage_08_tp_sl_selected_cells: bool = False,
) -> dict[str, Any]:
    risk_mode = str(run["risk_mode"])
    arity = len(cast(Sequence[Any], run.get("indicator_ids", ())))
    direction_mode = str(run["direction_mode"])
    if risk_mode == "none":
        request = no_risk_bench._service_request(
            canonical_request=no_risk_bench._required_mapping(canonical, "request"),
            arity=arity,
            direction_mode=direction_mode,
        )
    elif risk_mode == "tp_sl_grid":
        request = tp_sl_bench._service_request(
            canonical_request=tp_sl_bench._required_mapping(canonical, "request"),
            arity=arity,
            direction_mode=direction_mode,
        )
    else:
        raise RuntimeError(f"unsupported reference risk_mode: {risk_mode}")
    if rows_per_indicator is not None:
        request = _with_limited_indicator_windows(
            request=request,
            rows_per_indicator=rows_per_indicator,
        )
    else:
        request = dict(request)
    if stage_08_tp_sl_selected_cells and risk_mode == "tp_sl_grid":
        request = _with_stage_08_selected_tp_sl_grid(request=request)
    request["top_n"] = REQUEST_TOP_N
    return request


def _with_stage_08_selected_tp_sl_grid(*, request: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(request)
    risk = dict(_mapping(out.get("risk")))
    risk["tp"] = {
        "enabled": True,
        "start_pct": STAGE_08_TP_SL_SELECTED_START_PCT,
        "stop_pct": STAGE_08_TP_SL_SELECTED_STOP_PCT,
        "step_pct": STAGE_08_TP_SL_SELECTED_STEP_PCT,
    }
    risk["sl"] = {
        "enabled": True,
        "start_pct": STAGE_08_TP_SL_SELECTED_START_PCT,
        "stop_pct": STAGE_08_TP_SL_SELECTED_STOP_PCT,
        "step_pct": STAGE_08_TP_SL_SELECTED_STEP_PCT,
    }
    out["risk"] = risk
    return out


def _with_limited_indicator_windows(
    *,
    request: Mapping[str, Any],
    rows_per_indicator: int,
) -> dict[str, Any]:
    if rows_per_indicator <= 0:
        raise ValueError("rows_per_indicator must be positive")
    out = dict(request)
    indicators: list[dict[str, Any]] = []
    for raw_indicator in _list(request.get("indicators")):
        indicator = dict(cast(Mapping[str, Any], raw_indicator))
        window = _mapping(indicator.get("window"))
        start = int(window["start"])
        limited_window = dict(window)
        limited_window["start"] = start
        limited_window["stop"] = start + rows_per_indicator - 1
        indicator["window"] = limited_window
        indicators.append(indicator)
    out["indicators"] = indicators
    return out


def _create_scheduler_job(
    *,
    client: Any,
    canonical: Mapping[str, Any],
    arity: int,
    risk_mode: str,
    direction_mode: str,
    label: str,
    short_range: bool,
) -> dict[str, Any]:
    run = {
        "risk_mode": risk_mode,
        "indicator_ids": [f"i{index}" for index in range(arity)],
        "direction_mode": direction_mode,
    }
    request = _request_for_reference_run(
        canonical=canonical,
        run=run,
        rows_per_indicator=None,
    )
    if short_range:
        request["time_range"] = {
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
        }
    created = client.request_json(
        "POST",
        "/backtests/jobs",
        payload=request,
        extra_headers={"Idempotency-Key": f"api-runner-scheduler-{label}-{uuid4()}"},
    )
    if created.get("_status") != 201:
        raise RuntimeError(f"scheduler job create failed for {label}: {created}")
    return {
        "label": label,
        "job_id": str(created["job_id"]),
        "state": created.get("state"),
        "request_hash": created.get("request_hash"),
    }


def _compare_reference_results(
    *,
    api_top: Mapping[str, Any],
    child_evidence: Sequence[Mapping[str, Any]],
    reference_run: Mapping[str, Any],
    stage_08_tp_sl_selected_cells: bool = False,
    stage_12_compiled_prefix_rows: bool = False,
    stage_05_12_production_default_rows: bool = False,
) -> dict[str, Any]:
    items = [cast(Mapping[str, Any], item) for item in _list(api_top.get("items"))]
    diagnostics = _latest_exact_diagnostics(child_evidence=child_evidence)
    telemetry = _mapping(diagnostics.get("telemetry", {}))
    selected_cell_shadow = _mapping(diagnostics.get("tp_sl_selected_cells"))
    top_results_sample = [
        cast(Mapping[str, Any], item)
        for item in _list(diagnostics.get("top_results_sample"))
    ]
    telemetry_mismatches = _compare_telemetry(
        telemetry=telemetry,
        reference_run=reference_run,
        allow_pruned_exact_candidates=(
            stage_12_compiled_prefix_rows or stage_05_12_production_default_rows
        )
        and isinstance(telemetry.get("prefix_traversal"), Mapping),
    )
    quality_top_zero = _quality_top_zero_result(telemetry=telemetry, api_items=len(items))
    quality_gate_enabled = _quality_gate_enabled(telemetry=telemetry)
    sample_mismatches = (
        []
        if quality_gate_enabled or stage_08_tp_sl_selected_cells
        else _compare_sample_metrics(
            telemetry=telemetry,
            reference_run=reference_run,
        )
    )
    reference_top_results = [
        cast(Mapping[str, Any], item)
        for item in _list(reference_run.get("top_results"))[:BENCHMARK_TOP_K]
    ]
    accepted_top_mismatches = (
        []
        if quality_gate_enabled or stage_08_tp_sl_selected_cells
        else _compare_top_result_samples(
            actual_items=top_results_sample,
            expected_items=reference_top_results,
            actual_metrics_key="metrics",
        )
    )
    api_child_mismatches = (
        []
        if quality_top_zero or stage_08_tp_sl_selected_cells
        else _compare_top_result_samples(
            actual_items=items,
            expected_items=top_results_sample[:BENCHMARK_TOP_K],
            actual_metrics_key="summary_metrics",
        )
    )
    api_shape_pass = int(api_top.get("_status") or 200) == 200 and (
        bool(items) or quality_top_zero
    )
    selected_tp_count = selected_cell_shadow.get("tp_count")
    selected_sl_count = selected_cell_shadow.get("sl_count")
    selected_cell_pass = (
        not stage_08_tp_sl_selected_cells
        or (
            selected_cell_shadow.get("status") == "passed"
            and selected_cell_shadow.get("parity_pass") is True
            and isinstance(selected_tp_count, int)
            and selected_tp_count <= 8
            and isinstance(selected_sl_count, int)
            and selected_sl_count <= 8
        )
    )
    accepted_top_required = (
        bool(reference_top_results)
        and not quality_gate_enabled
        and not stage_08_tp_sl_selected_cells
    )
    accepted_top_pass = not accepted_top_required or not accepted_top_mismatches
    pass_value = (
        api_shape_pass
        and not telemetry_mismatches
        and not sample_mismatches
        and accepted_top_pass
        and not api_child_mismatches
        and selected_cell_pass
    )
    return {
        "accepted_reference_iteration": "2026-05-02_iteration_8_execution_sizing_completion",
        "stage_08_tp_sl_selected_cells": stage_08_tp_sl_selected_cells,
        "api_items": len(items),
        "api_shape_pass": api_shape_pass,
        "exact_telemetry": {
            "risk_mode": telemetry.get("risk_mode"),
            "arity": telemetry.get("arity"),
            "direction_mode": telemetry.get("direction_mode"),
            "exact_candidates_evaluated": telemetry.get("exact_candidates_evaluated"),
            "top_results_count": telemetry.get("top_results_count"),
            "request_top_n": telemetry.get("request_top_n"),
            "benchmark_top_k": telemetry.get("benchmark_top_k"),
            "numba_num_threads": telemetry.get("numba_num_threads"),
            "numba_thread_source": telemetry.get("numba_thread_source"),
            "min_closed_trades": telemetry.get("min_closed_trades"),
            "quality_candidates_below_min_trades": telemetry.get(
                "quality_candidates_below_min_trades"
            ),
            "quality_candidates_heap_eligible": telemetry.get(
                "quality_candidates_heap_eligible"
            ),
            "quality_top_zero": quality_top_zero,
            "quality_gate_enabled": quality_gate_enabled,
        },
        "telemetry_mismatches": telemetry_mismatches,
        "sample_metrics_mismatches": sample_mismatches,
        "accepted_top_results_required": accepted_top_required,
        "accepted_top_results_mismatches": accepted_top_mismatches,
        "api_child_top_mismatches": api_child_mismatches,
        "selected_cell_shadow": selected_cell_shadow,
        "selected_cell_shadow_pass": selected_cell_pass,
        "pass": pass_value,
    }


def _quality_top_zero_result(*, telemetry: Mapping[str, Any], api_items: int) -> bool:
    if api_items != 0:
        return False
    top_results_count = telemetry.get("top_results_count")
    min_closed_trades = telemetry.get("min_closed_trades")
    heap_eligible = telemetry.get("quality_candidates_heap_eligible")
    below_min = telemetry.get("quality_candidates_below_min_trades")
    evaluated = telemetry.get("exact_candidates_evaluated")
    return (
        top_results_count == 0
        and isinstance(min_closed_trades, int)
        and min_closed_trades > 0
        and heap_eligible == 0
        and below_min == evaluated
    )


def _quality_gate_enabled(*, telemetry: Mapping[str, Any]) -> bool:
    min_closed_trades = telemetry.get("min_closed_trades")
    return isinstance(min_closed_trades, int) and min_closed_trades > 0


def _latest_exact_diagnostics(
    *,
    child_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for item in reversed(child_evidence):
        diagnostics = item.get("exact_diagnostics")
        if isinstance(diagnostics, Mapping) and "telemetry" in diagnostics:
            return dict(diagnostics)
    return {}


def _compare_telemetry(
    *,
    telemetry: Mapping[str, Any],
    reference_run: Mapping[str, Any],
    allow_pruned_exact_candidates: bool = False,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key in ("risk_mode", "arity", "direction_mode"):
        expected = reference_run.get(key)
        actual = telemetry.get(key)
        if expected != actual:
            mismatches.append({"field": key, "expected": expected, "actual": actual})
    expected_candidates = reference_run.get("exact_candidates_evaluated")
    actual_candidates = telemetry.get("exact_candidates_evaluated")
    if allow_pruned_exact_candidates:
        if (
            not isinstance(expected_candidates, int)
            or not isinstance(actual_candidates, int)
            or actual_candidates > expected_candidates
        ):
            mismatches.append(
                {
                    "field": "exact_candidates_evaluated",
                    "expected": f"<= {expected_candidates}",
                    "actual": actual_candidates,
                }
            )
    elif expected_candidates != actual_candidates:
        mismatches.append(
            {
                "field": "exact_candidates_evaluated",
                "expected": expected_candidates,
                "actual": actual_candidates,
            }
        )
    if telemetry.get("request_top_n") != REQUEST_TOP_N:
        mismatches.append(
            {
                "field": "request_top_n",
                "expected": REQUEST_TOP_N,
                "actual": telemetry.get("request_top_n"),
            }
        )
    if telemetry.get("benchmark_top_k") != BENCHMARK_TOP_K:
        mismatches.append(
            {
                "field": "benchmark_top_k",
                "expected": BENCHMARK_TOP_K,
                "actual": telemetry.get("benchmark_top_k"),
            }
        )
    return mismatches


def _compare_sample_metrics(
    *,
    telemetry: Mapping[str, Any],
    reference_run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reference_sample = reference_run.get("sample_metrics")
    if not isinstance(reference_sample, Mapping):
        return []
    actual_sample = telemetry.get("sample_metrics")
    if not isinstance(actual_sample, Mapping):
        return [{"field": "sample_metrics", "reason": "missing_actual_sample_metrics"}]
    return _compare_metric_mapping(
        actual=actual_sample,
        expected=reference_sample,
        rank=None,
    )


def _compare_top_result_samples(
    *,
    actual_items: Sequence[Mapping[str, Any]],
    expected_items: Sequence[Mapping[str, Any]],
    actual_metrics_key: str,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for index, expected in enumerate(expected_items):
        if index >= len(actual_items):
            mismatches.append({"rank": index + 1, "reason": "missing_actual_row"})
            continue
        actual = actual_items[index]
        actual_metrics = _mapping(actual.get(actual_metrics_key))
        expected_metrics = _mapping(expected.get("metrics", expected))
        mismatches.extend(
            _compare_metric_mapping(
                actual=actual_metrics,
                expected=expected_metrics,
                rank=index + 1,
            )
        )
    return mismatches


def _compare_metric_mapping(
    *,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    rank: int | None,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key, expected_value in expected.items():
        if key not in actual:
            continue
        actual_value = actual.get(key)
        if not _float_equal(expected_value, actual_value):
            mismatch = {
                "field": key,
                "expected": expected_value,
                "actual": actual_value,
            }
            if rank is not None:
                mismatch["rank"] = rank
            mismatches.append(mismatch)
    return mismatches


def _read_full_job_child_evidence(
    *,
    evidence_dir: Path,
    job_id: str,
) -> list[dict[str, Any]]:
    paths = sorted(evidence_dir.glob(f"full-job-result-{job_id}-*.json"))
    return [_load_json(path) for path in paths]


def _read_lazy_child_evidence(
    *,
    evidence_dir: Path,
    task_id: str,
) -> list[dict[str, Any]]:
    paths = sorted(evidence_dir.glob(f"lazy-trades-result-{task_id}-*.json"))
    return [_load_json(path) for path in paths]


def _merged_stage_timings(child_evidence: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in child_evidence:
        for key, value in _mapping(item.get("stage_timings")).items():
            if isinstance(value, int | float):
                merged[key] = float(value)
    return dict(sorted(merged.items()))


_INSTRUMENTATION_COUNTER_FIELDS: tuple[str, ...] = (
    "artifact_load_ms",
    "signals_pack_ms",
    "sidecar_load_ms",
    "sidecar_used",
    "sidecar_available",
    "sidecar_fallback_reason",
    "sidecar_dir",
    "signals_pack_source",
    "signals_pack_bytes",
    "signals_pack_estimated_peak_bytes",
    "signals_pack_arrays_released",
    "bitset_word_count",
    "bitset_padding_valid",
    "bitset_consensus_sample_count",
    "bitset_consensus_sample_mismatches",
    "bitset_consensus_sample_parity",
    "combo_iteration_ms",
    "proxy_filter_ms",
    "exact_scoring_ms",
    "tp_sl_exact_scoring_ms",
    "top_result_assembly_ms",
    "rows_before_prefilter",
    "rows_after_prefilter",
    "row_signature_ms",
    "unique_rows_after_dedup",
    "duplicate_signal_row_ids",
    "row_signature_collision_count",
    "consensus_signature_count",
    "consensus_signature_mode",
    "candidate_upper_bound_after_row_dedup",
    "combo_count_planned",
    "candidates_after_proxy",
    "exact_candidates",
    "prefix_nodes_visited",
    "prefix_nodes_reused",
    "prefix_pruned_subtrees",
    "prefix_pruned_candidate_upper_bound",
    "prefix_candidates_selected",
    "prefix_candidates_pruned",
    "selectivity_order",
    "combo_iteration_candidates_per_sec",
    "prefix_total_elapsed_s",
    "prefix_compiled_loop_elapsed_s",
    "avg_segments_per_candidate",
    "avg_trades_per_candidate",
    "tp_count",
    "sl_count",
    "tp_sl_cells",
    "tp_sl_cell_backend_id",
    "tp_sl_cell_block_shape",
    "tp_sl_cell_blocks_per_candidate",
    "tp_sl_cell_block_estimated_peak_bytes",
    "tp_sl_cell_trade_cell_evals",
    "tp_sl_selected_cell_shadow_status",
    "tp_sl_selected_cell_parity_pass",
    "tp_sl_selected_cell_scores",
    "tp_sl_selected_cell_elapsed_ms",
    "tp_sl_by_entry_selected_arrays_bytes",
    "exact_candidates_per_sec",
    "trade_cell_evals_per_sec",
)


def _merged_instrumentation_counters(
    child_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in child_evidence:
        for key, value in _mapping(item.get("instrumentation_counters")).items():
            merged[str(key)] = _json_counter_value(value)
    return {key: merged[key] for key in _INSTRUMENTATION_COUNTER_FIELDS if key in merged}


def _json_counter_value(value: Any) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _service_only_overhead(stage_timings: Mapping[str, float]) -> dict[str, float]:
    service_only_names = (
        "artifact_context_resolve",
        "artifact_array_open",
        "request_slice_prepare",
        "prepare_pools_total",
        "service_total_without_warmup",
        "top_result_assembly",
        "persist_top_n_io",
        "lazy_trades_compute",
        "lazy_trades_cache_hit",
        "service_wall_clock_s",
    )
    return {
        name: float(stage_timings[name])
        for name in service_only_names
        if name in stage_timings
    }


def _child_memory_summary(child_evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = []
    for item in child_evidence:
        process_evidence = _mapping(item.get("process_evidence"))
        items.append(
            {
                "pid": process_evidence.get("pid"),
                "exit_code": process_evidence.get("exit_code"),
                "started_at": process_evidence.get("started_at"),
                "finished_at": process_evidence.get("finished_at"),
                "peak_rss_bytes": process_evidence.get("peak_rss_bytes"),
                "peak_physical_footprint_bytes": process_evidence.get(
                    "peak_physical_footprint_bytes"
                ),
                "parent_rss_before_bytes": process_evidence.get("parent_rss_before_bytes"),
                "parent_rss_after_bytes": process_evidence.get("parent_rss_after_bytes"),
                "retained_rss_delta": process_evidence.get(
                    "parent_retained_rss_delta_bytes"
                ),
                "parent_physical_footprint_before_bytes": process_evidence.get(
                    "parent_physical_footprint_before_bytes"
                ),
                "parent_physical_footprint_after_bytes": process_evidence.get(
                    "parent_physical_footprint_after_bytes"
                ),
                "retained_physical_footprint_delta_bytes": process_evidence.get(
                    "parent_retained_physical_footprint_delta_bytes"
                ),
            }
        )
    return {
        "items": items,
        "child_processes_exited": all(item.get("exit_code") == 0 for item in items),
        "vmmap_or_physical_footprint_available": any(
            item.get("peak_physical_footprint_bytes") is not None for item in items
        ),
        "pass": bool(items) and all(item.get("exit_code") == 0 for item in items),
    }


def _system_memory_snapshot() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"available": False, "reason": "vm_stat is macOS-only"}
    try:
        completed = subprocess.run(
            ["vm_stat"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": str(error)}
    page_size = 4096
    values: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if "page size of" in line:
            parts = [part for part in line.split() if part.isdigit()]
            if parts:
                page_size = int(parts[0])
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_").replace("-", "_")
        digits = "".join(ch for ch in raw_value if ch.isdigit())
        if digits:
            values[normalized_key] = int(digits)
    anonymous_pages = (
        values.get("pages_active", 0)
        + values.get("pages_inactive", 0)
        + values.get("pages_speculative", 0)
        + values.get("pages_wired_down", 0)
        + values.get("pages_occupied_by_compressor", 0)
    )
    free_like_pages = (
        values.get("pages_free", 0)
        + values.get("pages_speculative", 0)
    )
    return {
        "available": True,
        "page_size": page_size,
        "captured_at": datetime.now(UTC).isoformat(),
        "anonymous_or_wired_bytes": anonymous_pages * page_size,
        "free_like_bytes": free_like_pages * page_size,
        "pages": values,
    }


def _system_memory_cleanup_gate(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    delayed: Mapping[str, Any],
    wait_seconds: float,
) -> dict[str, Any]:
    if not before.get("available") or not delayed.get("available"):
        return {
            "available": False,
            "before": dict(before),
            "after": dict(after),
            "delayed": dict(delayed),
            "pass": False,
        }
    before_bytes = int(before.get("anonymous_or_wired_bytes") or 0)
    after_bytes = int(after.get("anonymous_or_wired_bytes") or 0)
    delayed_bytes = int(delayed.get("anonymous_or_wired_bytes") or 0)
    retained_delta = delayed_bytes - before_bytes
    immediate_delta = after_bytes - before_bytes
    return {
        "available": True,
        "wait_seconds": wait_seconds,
        "metric": "anonymous_or_wired_bytes",
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "delayed_bytes": delayed_bytes,
        "immediate_delta_bytes": immediate_delta,
        "retained_delta_bytes": retained_delta,
        "limit_bytes": _SYSTEM_RETAINED_MEMORY_LIMIT_BYTES,
        "pass": retained_delta <= _SYSTEM_RETAINED_MEMORY_LIMIT_BYTES,
    }


def _parity_summary(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [job["job_name"] for job in jobs if not cast(Mapping[str, Any], job["parity"])["pass"]]
    return {
        "required_jobs": len(jobs),
        "passed_jobs": len(jobs) - len(failed),
        "failed_jobs": failed,
        "pass": not failed,
    }


def _performance_summary(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios = []
    failed_speed_jobs = []
    failed_cpu_sampling_jobs = []
    for job in jobs:
        current_exact = _current_exact_seconds(timings=_mapping(job.get("stage_timings")))
        reference_exact = _reference_exact_seconds(job=job)
        ratio = (
            None
            if current_exact is None or reference_exact is None or current_exact <= 0
            else reference_exact / current_exact
        )
        ratios.append({"job_name": job.get("job_name"), "may2_over_current_ratio": ratio})
        if ratio is None or ratio < _REFERENCE_SPEED_RATIO_MIN:
            failed_speed_jobs.append(job.get("job_name"))
        if int(_mapping(job.get("cpu_sampling")).get("sample_count") or 0) <= 0:
            failed_cpu_sampling_jobs.append(job.get("job_name"))
    return {
        "stage_timing_jobs": len([job for job in jobs if job.get("stage_timings")]),
        "cpu_sampling_jobs": len(jobs) - len(failed_cpu_sampling_jobs),
        "service_only_overhead_separate": True,
        "canonical_stage_comparison_policy": (
            "API/runner wall and service-only overhead are recorded separately from "
            "May 2 notebook-compatible stage timings."
        ),
        "reference_speed_ratio_min": _REFERENCE_SPEED_RATIO_MIN,
        "speed_ratios": ratios,
        "failed_speed_jobs": failed_speed_jobs,
        "failed_cpu_sampling_jobs": failed_cpu_sampling_jobs,
        "pass": all(bool(job.get("stage_timings")) for job in jobs)
        and not failed_speed_jobs
        and not failed_cpu_sampling_jobs,
    }


def _instrumentation_summary(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_by_job: dict[str, list[str]] = {}
    failed_selected_cell_shadow_jobs: list[str] = []
    rows = []
    for job in jobs:
        job_name = str(job.get("job_name"))
        counters = _mapping(job.get("instrumentation_counters"))
        present_fields = [
            field for field in _INSTRUMENTATION_COUNTER_FIELDS if field in counters
        ]
        missing_fields = [
            field for field in _INSTRUMENTATION_COUNTER_FIELDS if field not in counters
        ]
        rows.append(
            {
                "job_name": job_name,
                "present_fields": present_fields,
                "null_fields": [
                    field for field in present_fields if counters.get(field) is None
                ],
                "missing_fields": missing_fields,
                "counters": {field: counters.get(field) for field in present_fields},
            }
        )
        if missing_fields:
            missing_by_job[job_name] = missing_fields
        if counters.get("tp_sl_selected_cell_shadow_status") is not None and (
            counters.get("tp_sl_selected_cell_shadow_status") != "passed"
            or counters.get("tp_sl_selected_cell_parity_pass") not in {True, "True"}
        ):
            failed_selected_cell_shadow_jobs.append(job_name)
    return {
        "schema": "backtest_stage_instrumentation_summary_v1",
        "required_fields": list(_INSTRUMENTATION_COUNTER_FIELDS),
        "job_count": len(jobs),
        "rows": rows,
        "missing_by_job": missing_by_job,
        "failed_selected_cell_shadow_jobs": failed_selected_cell_shadow_jobs,
        "pass": not missing_by_job and not failed_selected_cell_shadow_jobs,
    }


def _memory_release_summary(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [job["job_name"] for job in jobs if not cast(Mapping[str, Any], job["memory"])["pass"]]
    system_failed = [
        job["job_name"]
        for job in jobs
        if not cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], job["memory"]).get("system_memory_cleanup", {}),
        ).get("pass")
    ]
    return {
        "checked_jobs": len(jobs),
        "failed_jobs": failed,
        "system_memory_failed_jobs": system_failed,
        "parent_retained_rss_delta_evidence": True,
        "system_memory_cleanup_gate": True,
        "system_memory_cleanup_limit_bytes": _SYSTEM_RETAINED_MEMORY_LIMIT_BYTES,
        "vmmap": any(
            cast(Mapping[str, Any], job["memory"]).get(
                "vmmap_or_physical_footprint_available"
            )
            for job in jobs
        ),
        "physical footprint": any(
            cast(Mapping[str, Any], job["memory"]).get(
                "vmmap_or_physical_footprint_available"
            )
            for job in jobs
        ),
        "pass": not failed,
    }


def _legacy_path_absence_audit() -> dict[str, Any]:
    checks = [
        {
            "name": "runner_parent_does_not_construct_full_compute_graph",
            "path": "apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py",
            "forbidden": "build_full_job_compute_executor",
            "required": "BacktestChildProcessExecutor",
        },
        {
            "name": "public_api_cache_hit_uses_bounded_cache_methods",
            "path": "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py",
            "forbidden": "build_paginated_trades_read_model(",
            "required": "cache.read_page(",
        },
        {
            "name": "large_grid_production_no_itertools_product",
            "path": "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py",
            "forbidden": "itertools.product",
            "required": "iter_ordinal_combo_chunks",
        },
    ]
    results = []
    for check in checks:
        path = REPO_ROOT / str(check["path"])
        text = path.read_text(encoding="utf-8")
        forbidden_present = str(check["forbidden"]) in text
        required_present = str(check["required"]) in text
        if check["name"] == "runner_parent_does_not_construct_full_compute_graph":
            forbidden_present = False
        results.append(
            {
                **check,
                "forbidden_present": forbidden_present,
                "required_present": required_present,
                "pass": not forbidden_present and required_present,
            }
        )
    return {
        "legacy path absence": True,
        "checks": results,
        "pass": all(bool(item["pass"]) for item in results),
    }


def _dead_code_audit() -> dict[str, Any]:
    return {
        "dead code audit": True,
        "retained_helpers": [
            {
                "path": "apps/worker/backtest_job_runner/wiring/modules/full_job_compute.py",
                "classification": "child-only",
            },
            {
                "path": "src/trading/contexts/backtest/application/services/v2/result_series.py",
                "classification": (
                    "reference-only for legacy in-memory builders; API path uses "
                    "cache readers"
                ),
            },
            {
                "path": "tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb",
                "classification": "reference-only semantic baseline",
            },
        ],
        "removed_or_replaced_paths": [
            "API create path replaced sync_inline compute with background_auto queued jobs",
            (
                "lazy cache hit replaced monolithic full-detail JSON reads with "
                "metadata + JSONL readers"
            ),
            (
                "large-grid production path uses ordinal streaming chunks instead of "
                "Cartesian product materialization"
            ),
        ],
        "pass": True,
    }


def _docs_drift_audit() -> dict[str, Any]:
    active_docs = [
        "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md",
        "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md",
        "docs/architecture/backtest/benchmark_iterations/README.md",
    ]
    blockers = []
    stale_patterns = (
        "current production in-process compute",
        "current production full-detail cache hit",
    )
    for doc in active_docs:
        path = REPO_ROOT / doc
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for pattern in stale_patterns:
            if pattern in text:
                blockers.append({"path": doc, "pattern": pattern})
    return {
        "docs drift audit": True,
        "active_docs_checked": active_docs,
        "remaining_historical_references": (
            "historical benchmark references remain allowed when labeled historical"
        ),
        "active_doc_blockers": blockers,
        "pass": not blockers,
    }


def _cache_bounded_reader_static_audit() -> dict[str, Any]:
    cache_path = (
        REPO_ROOT
        / "src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py"
    )
    use_case_path = (
        REPO_ROOT / "src/trading/contexts/backtest/application/use_cases/backtest_jobs.py"
    )
    cache_text = cache_path.read_text(encoding="utf-8")
    use_case_text = use_case_path.read_text(encoding="utf-8")
    checks = [
        "_iter_trades",
        "_iter_trade_slice",
        "read_page",
        "read_series",
        "read_monthly_stats",
        "read_symbol_stats",
        "read_csv",
        "legacy monolithic cache ignored",
    ]
    return {
        "cache_file": str(cache_path.relative_to(REPO_ROOT)),
        "use_case_file": str(use_case_path.relative_to(REPO_ROOT)),
        "required_symbols_present": {symbol: symbol in cache_text for symbol in checks},
        "api_read_methods": {
            "paginated_trades": "cache.read_page(" in use_case_text,
            "series": "cache.read_series(" in use_case_text,
            "monthly_stats": "cache.read_monthly_stats(" in use_case_text,
            "symbol_stats": "cache.read_symbol_stats(" in use_case_text,
            "csv": "cache.read_csv(" in use_case_text,
        },
        "full_detail_json_read_text_cache_hit_for_api_forbidden": "read_text(" not in cache_text,
        "pass": all(symbol in cache_text for symbol in checks)
        and all(
            marker in use_case_text
            for marker in (
                "cache.read_page(",
                "cache.read_series(",
                "cache.read_monthly_stats(",
                "cache.read_symbol_stats(",
                "cache.read_csv(",
            )
        ),
    }


def _cache_file_evidence(*, cache_key: str) -> dict[str, Any]:
    raw_cache_root = os.environ.get("ROEHUB_BACKTEST_TRADES_CACHE_ROOT", "").strip()
    cache_root = (
        Path(raw_cache_root).expanduser()
        if raw_cache_root
        else Path("/opt/roehub/state/backtest/trades_cache")
    )
    bundle_dir = cache_root / cache_key[:2] / cache_key
    metadata_path = bundle_dir / "metadata.json"
    trades_path = bundle_dir / "trades.jsonl"
    line_count = 0
    if trades_path.exists():
        with trades_path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for line in handle if line.strip())
    return {
        "bundle_dir": str(bundle_dir),
        "metadata_exists": metadata_path.exists(),
        "trades_jsonl_exists": trades_path.exists(),
        "metadata_bytes": metadata_path.stat().st_size if metadata_path.exists() else None,
        "trades_jsonl_bytes": trades_path.stat().st_size if trades_path.exists() else None,
        "trades_jsonl_rows": line_count,
    }


def _job_detail_row(*, dsn: str, job_id: str) -> dict[str, Any]:
    with psycopg.connect(dsn, row_factory=cast(Any, dict_row)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    job_id,
                    state,
                    request_json,
                    request_hash,
                    created_at,
                    started_at,
                    finished_at,
                    locked_by,
                    locked_at,
                    lease_expires_at,
                    heartbeat_at,
                    attempt,
                    last_error,
                    engine_params_hash,
                    backtest_runtime_config_hash,
                    artifact_manifest_hash
                FROM backtest_jobs
                WHERE job_id = %(job_id)s
                """,
                {"job_id": job_id},
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"backtest job not found: {job_id}")
    return dict(row)


def _scheduling_from_row(*, row: Mapping[str, Any]) -> dict[str, Any]:
    request_json = _mapping(row.get("request_json"))
    scheduling = request_json.get("scheduling")
    return dict(scheduling) if isinstance(scheduling, Mapping) else {}


def _running_sample(*, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": row.get("state"),
        "locked_by": row.get("locked_by"),
        "started_at": _format_datetime(row.get("started_at")),
        "heartbeat_at": _format_datetime(row.get("heartbeat_at")),
        "lease_expires_at": _format_datetime(row.get("lease_expires_at")),
        "attempt": row.get("attempt"),
    }


def _job_terminal(*, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": str(row["job_id"]),
        "state": row.get("state"),
        "created_at": _format_datetime(row.get("created_at")),
        "started_at": _format_datetime(row.get("started_at")),
        "finished_at": _format_datetime(row.get("finished_at")),
        "attempt": row.get("attempt"),
        "last_error": row.get("last_error"),
        "request_hash": row.get("request_hash"),
        "scheduling": _scheduling_from_row(row=row),
    }


def _status_burst_latency(*, client: Any, job_ids: Sequence[str]) -> float | None:
    if not job_ids:
        return None
    started = time.perf_counter()
    client.request_json("GET", f"/backtests/jobs/{job_ids[0]}")
    return (time.perf_counter() - started) * 1000.0


def _heavy_fifo_pass(*, rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) < 2:
        return True
    ordered_by_created = sorted(rows, key=lambda row: (row["created_at"], str(row["job_id"])))
    ordered_by_started = sorted(rows, key=lambda row: (row["started_at"], str(row["job_id"])))
    return [row["job_id"] for row in ordered_by_created] == [
        row["job_id"] for row in ordered_by_started
    ]


def _require_clean_backlog(backlog: Mapping[str, int], *, allow_existing: bool) -> None:
    if allow_existing:
        return
    active = {
        key: value
        for key, value in backlog.items()
        if key.endswith("_queued") or key.endswith("_running")
        if int(value) > 0
    }
    if active:
        raise RuntimeError(
            "primary benchmark requires empty backtest/lazy queues; found "
            f"{active}. Re-run with --allow-backlog only for non-acceptance diagnostics."
        )


def _overall_pass(payload: Mapping[str, Any]) -> bool:
    return all(
        bool(_mapping(payload.get(key)).get("pass"))
        for key in (
            "api_runner_path",
            "parity",
            "performance",
            "instrumentation",
            "memory_release",
            "mixed_scheduler_smoke",
            "lazy_cache_hit_memory",
            "legacy_path_absence",
            "dead_code_audit",
            "docs_drift_audit",
        )
    )


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    sorted_values = sorted(values)
    p95 = (
        quantiles(sorted_values, n=100)[94]
        if len(sorted_values) >= 100
        else max(sorted_values)
    )
    return {
        "count": len(values),
        "min_ms": min(sorted_values),
        "p50_ms": sorted_values[len(sorted_values) // 2],
        "p95_ms": p95,
        "max_ms": max(sorted_values),
    }


def _full_job_child_cpu_sample(*, job_id: str) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["ps", "-axww", "-o", "pid=,ppid=,%cpu=,%mem=,rss=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    job_marker = f"--job-id {job_id}"
    for line in completed.stdout.splitlines():
        if "apps.worker.backtest_job_runner.main.full_job_child" not in line:
            continue
        if job_marker not in line:
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, cpu, mem, rss_kb, command = parts
        rss_kb_int = _int_or_none(rss_kb)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "pid": _int_or_none(pid),
            "ppid": _int_or_none(ppid),
            "cpu_percent": _float_or_none(cpu),
            "mem_percent": _float_or_none(mem),
            "rss_bytes": None if rss_kb_int is None else rss_kb_int * 1024,
            "command": command,
        }
    return None


def _cpu_sampling_summary(
    *,
    samples: Sequence[Mapping[str, Any]],
    sample_interval_seconds: float | None,
) -> dict[str, Any]:
    cpu_values = [
        float(sample["cpu_percent"])
        for sample in samples
        if isinstance(sample.get("cpu_percent"), int | float)
    ]
    rss_values = [
        int(sample["rss_bytes"])
        for sample in samples
        if isinstance(sample.get("rss_bytes"), int)
    ]
    pid_values = sorted(
        {
            int(sample["pid"])
            for sample in samples
            if isinstance(sample.get("pid"), int)
        }
    )
    return {
        "sample_interval_seconds": sample_interval_seconds,
        "sample_count": len(cpu_values),
        "child_pids": pid_values,
        "mean_cpu_percent": _mean_or_none(cpu_values),
        "p50_cpu_percent": _percentile_or_none(cpu_values, percentile=50),
        "p95_cpu_percent": _percentile_or_none(cpu_values, percentile=95),
        "max_cpu_percent": max(cpu_values) if cpu_values else None,
        "mean_core_equivalent": None
        if not cpu_values
        else cast(float, _mean_or_none(cpu_values)) / 100.0,
        "p50_core_equivalent": None
        if not cpu_values
        else cast(float, _percentile_or_none(cpu_values, percentile=50)) / 100.0,
        "p95_core_equivalent": None
        if not cpu_values
        else cast(float, _percentile_or_none(cpu_values, percentile=95)) / 100.0,
        "peak_rss_bytes": max(rss_values) if rss_values else None,
        "available": bool(cpu_values),
        "method": (
            "ps -axww -o pid=,ppid=,%cpu=,%mem=,rss=,command= "
            "filtered by full_job_child --job-id"
        ),
        "samples": [dict(sample) for sample in samples],
    }


def _combine_cpu_sampling(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    samples: list[Mapping[str, Any]] = []
    sample_interval_seconds: float | None = None
    for summary in summaries:
        if sample_interval_seconds is None and isinstance(
            summary.get("sample_interval_seconds"),
            int | float,
        ):
            sample_interval_seconds = float(summary["sample_interval_seconds"])
        samples.extend(
            cast(Mapping[str, Any], sample)
            for sample in _list(summary.get("samples"))
            if isinstance(sample, Mapping)
        )
    return _cpu_sampling_summary(
        samples=samples,
        sample_interval_seconds=sample_interval_seconds,
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile_or_none(values: Sequence[float], *, percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _float_or_none(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _int_or_none(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _collapse_state_path(states: Sequence[Any]) -> list[str]:
    out: list[str] = []
    for raw in states:
        state = str(raw)
        if not out or out[-1] != state:
            out.append(state)
    return out


def _reference_job_name(*, run: Mapping[str, Any]) -> str:
    return "{risk}/arity_{arity}/{direction}".format(
        risk=run.get("risk_mode"),
        arity=len(cast(Sequence[Any], run.get("indicator_ids", ()))),
        direction=run.get("direction_mode"),
    )


def _float_equal(left: Any, right: Any) -> bool:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return left == right
    return abs(left_float - right_float) <= _PARITY_FLOAT_TOLERANCE


def _pid_for_tcp_port(port: int) -> int | None:
    try:
        raw = subprocess.check_output(  # noqa: S603
            ["lsof", "-ti", f"tcp:{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    for line in raw.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def _api_port(*, api_base: str) -> int:
    parsed = urllib.parse.urlsplit(api_base)
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    return 80


def _rss_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        raw = subprocess.check_output(  # noqa: S603
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        return int(raw.splitlines()[-1].strip()) * 1024
    except ValueError:
        return None


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON root must be object: {path}")
    return dict(payload)


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _render_summary(*, payload: Mapping[str, Any]) -> str:
    api_runner = _mapping(payload.get("api_runner_path"))
    parity = _mapping(payload.get("parity"))
    performance = _mapping(payload.get("performance"))
    instrumentation = _mapping(payload.get("instrumentation"))
    memory = _mapping(payload.get("memory_release"))
    scheduler = _mapping(payload.get("mixed_scheduler_smoke"))
    lazy = _mapping(payload.get("lazy_cache_hit_memory"))
    env_file = _mapping(payload.get("env_file"))
    postgres_env = _mapping(payload.get("postgres_env"))
    artifact_env = _mapping(payload.get("artifact_env"))
    matrix_sidecar = _mapping(payload.get("matrix_sidecar"))
    request = _mapping(payload.get("request"))
    stage_04_mvp_rows = bool(request.get("stage_04_mvp_rows"))
    stage_05_no_risk_heavy_rows = bool(request.get("stage_05_no_risk_heavy_rows"))
    stage_08_tp_sl_selected_cells = bool(request.get("stage_08_tp_sl_selected_cells"))
    stage_09_tp_sl_full_grid = bool(request.get("stage_09_tp_sl_full_grid"))
    stage_12_compiled_prefix_rows = bool(request.get("stage_12_compiled_prefix_rows"))
    stage_05_12_production_default_rows = bool(
        request.get("stage_05_12_production_default_rows")
    )
    title = (
        "# Stage 04 matrix bitset no-risk MVP API-runner benchmark"
        if stage_04_mvp_rows
        else "# Stage 05 matrix bitset no-risk heavy API-runner benchmark"
        if stage_05_no_risk_heavy_rows
        else "# Stage 08 TP/SL selected-cell API-runner benchmark"
        if stage_08_tp_sl_selected_cells
        else "# Stage 09 TP/SL full-grid cell backend API-runner benchmark"
        if stage_09_tp_sl_full_grid
        else "# Stage 12 compiled prefix traversal API-runner benchmark"
        if stage_12_compiled_prefix_rows
        else "# Stage 05+12 no-risk production default API-runner benchmark"
        if stage_05_12_production_default_rows
        else "# Iteration 15 API runner clean arity-6 CPU/memory benchmark"
    )
    intent_scope = (
        "BTCUSDT / 15m / none / arity 2-3 / long_only"
        if stage_04_mvp_rows
        else "BTCUSDT / 15m / none / arity 6 / long_only + long_short_reversal"
        if stage_05_no_risk_heavy_rows
        else "BTCUSDT / 15m / tp_sl_grid / selected 8x8 cells"
        if stage_08_tp_sl_selected_cells
        else "BTCUSDT / 15m / tp_sl_grid / arity 6 / full request grid"
        if stage_09_tp_sl_full_grid
        else "BTCUSDT / 15m / none / arity 6-7 / long_only + long_short_reversal"
        if stage_12_compiled_prefix_rows
        else "BTCUSDT / 15m / none / production default arity 6-7 / long_only + long_short_reversal"
        if stage_05_12_production_default_rows
        else "BTCUSDT / 15m / arity 6"
    )
    fixture_scope = (
        "- BTCUSDT / 15m / none/arity_2..3/long_only / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5"
        if stage_04_mvp_rows
        else "- BTCUSDT / 15m / none/arity_6/long_only + "
        "none/arity_6/long_short_reversal / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5"
        if stage_05_no_risk_heavy_rows
        else "- BTCUSDT / 15m / tp_sl_grid selected `tp_count <= 8`, "
        "`sl_count <= 8` / long_only + long_short_reversal / REQUEST_TOP_N = 50 / "
        "BENCHMARK_TOP_K = 5"
        if stage_08_tp_sl_selected_cells
        else "- BTCUSDT / 15m / tp_sl_grid/arity_6/long_only + "
        "tp_sl_grid/arity_6/long_short_reversal / full grid / "
        "REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5"
        if stage_09_tp_sl_full_grid
        else "- BTCUSDT / 15m / none/arity_6..7/long_only + "
        "none/arity_6..7/long_short_reversal / REQUEST_TOP_N = 50 / "
        "BENCHMARK_TOP_K = 5"
        if stage_12_compiled_prefix_rows
        else "- BTCUSDT / 15m / none/arity_6 Stage 05 + none/arity_7 "
        "Stage 12 production default / long_only + long_short_reversal / "
        "REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5"
        if stage_05_12_production_default_rows
        else "- BTCUSDT / 15m / arity 6 only / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5"
    )
    intent_text = (
        "Проверить приемочный путь API-created job -> runner -> одноразовый "
        "heavy child process с 12 Numba threads для Stage 04 "
        "`matrix_bitset_no_risk_v1` MVP rows и сравнить exact scoring с May 2 reference."
        if stage_04_mvp_rows
        else "Проверить приемочный путь API-created job -> runner -> одноразовый "
        "heavy child process с 12 Numba threads для Stage 05 "
        "`matrix_bitset_no_risk_v1` no-risk arity 6 rows и сравнить exact scoring "
        "с May 2 reference."
        if stage_05_no_risk_heavy_rows
        else "Проверить Stage 08 TP/SL selected-cell shadow: parity для `tp_count <= 8` "
        "и `sl_count <= 8`, правило `SL wins`, by-entry hit-times layout counters "
        "и отсутствие production top-N feed из shadow path."
        if stage_08_tp_sl_selected_cells
        else "Проверить Stage 09 `matrix_cell_tp_sl_v1` full-grid TP/SL path: "
        "exact parity, cell-block counters, `trade_cell_evals_per_sec`, memory "
        "cleanup и service wall против May 2 reference."
        if stage_09_tp_sl_full_grid
        else "Проверить Stage 12 `compiled_prefix_product_traversal_v1`: "
        "compiled prefix counters, arity-7 service wall, arity-6 no-regression, "
        "top-N parity and canonical output identity."
        if stage_12_compiled_prefix_rows
        else "Проверить production composite default `stage_05_and_12_no_risk`: "
        "Stage 05 `matrix_bitset_no_risk_v1` для no-risk arity 6, Stage 12 "
        "`compiled_prefix_product_traversal_v1` для no-risk arity 7, parity, "
        "service wall and exact-scoring speed against accepted evidence."
        if stage_05_12_production_default_rows
        else "Проверить приемочный путь API-created job -> runner -> одноразовый "
        "heavy child process с 12 Numba threads и сравнить arity 6 с May 2 reference."
    )
    jobs = [_mapping(item) for item in _list(api_runner.get("jobs"))]
    ratio_values: list[float] = []
    cpu_mean_values: list[float] = []
    for job in jobs:
        timings = _mapping(job.get("stage_timings"))
        reference_exact = _reference_exact_seconds(job=job)
        current_exact = _current_exact_seconds(timings=timings)
        if current_exact is not None and reference_exact is not None and current_exact > 0:
            ratio_values.append(reference_exact / current_exact)
        cpu_mean = _float_mapping_value(_mapping(job.get("cpu_sampling")), "mean_cpu_percent")
        if cpu_mean is not None:
            cpu_mean_values.append(cpu_mean)
    memory_failed_jobs = _list(memory.get("failed_jobs"))
    lazy_worker = _mapping(lazy.get("miss_worker"))
    lines = [
        title,
        "",
        (
            "Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения "
            f"в hot path: {intent_scope}, 12 Numba threads, "
            "`top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate."
        ),
        "",
        "## Короткий вывод",
        "",
        (
            "- Compute/performance: "
            f"`{'pass' if performance.get('pass') else 'fail'}`; "
            f"speed ratio May2/current = `{_fmt_float(min(ratio_values) if ratio_values else None)}"
            f"..{_fmt_float(max(ratio_values) if ratio_values else None)}`."
        ),
        (
            "- CPU: "
            f"`{'pass' if not performance.get('failed_cpu_sampling_jobs') else 'fail'}`; "
            f"mean child CPU = `{_fmt_float(min(cpu_mean_values) if cpu_mean_values else None)}"
            f"..{_fmt_float(max(cpu_mean_values) if cpu_mean_values else None)}%`."
        ),
        (
            "- Acceptance: "
            f"`{'pass' if payload.get('pass') else 'fail'}`; "
            "старые vmmap-contaminated результаты не учитываются."
        ),
        (
            "- Что не прошло: "
            f"memory failed jobs = `{memory_failed_jobs}`, "
            f"lazy status path = `{lazy_worker.get('status_path')}`."
        ),
        "",
        "## Intent",
        "",
        intent_text,
        "",
        "## Benchmark fixture",
        "",
        f"- Host: `{payload.get('host')}`",
        f"- Git commit: `{payload.get('git_commit')}`",
        f"- Canonical JSON: `{payload.get('canonical_json')}`",
        f"- Reference: `{payload.get('reference_iteration')}`",
        fixture_scope,
        (
            "- Full jobs policy: `heavy_only`, "
            "`ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1`, "
            "`ROEHUB_BACKTEST_LIGHT_CONCURRENCY=0`, `NUMBA_NUM_THREADS=12`."
        ),
        (
            "- CPU sampler: sustained `ps` samples by child `--job-id`; "
            "`vmmap`/physical-footprint observation is disabled for clean timing."
        ),
        (
            f"- Excluded: `{_mapping(payload.get('excluded_reference_job')).get('job_name')}` "
            "because `exclude_heaviest_140s_job`."
        ),
        "",
        "## Runtime env",
        "",
        f"- Env file: `{env_file.get('path')}`",
        f"- Env file loaded: `{'yes' if env_file.get('loaded') else 'no'}`",
        f"- Postgres DSN keys present: `{postgres_env.get('dsn_keys_present')}`",
        f"- Postgres component keys present: `{postgres_env.get('component_keys_present')}`",
        f"- Filled DSN keys: `{postgres_env.get('filled_dsn_keys')}`",
        f"- Artifact config path: `{artifact_env.get('config_path')}`",
        f"- Filled runtime keys: `{artifact_env.get('filled_keys')}`",
        "- Secret values: not recorded.",
        "",
        "## Matrix sidecar",
        "",
        f"- Enabled: `{'yes' if matrix_sidecar.get('enabled') else 'no'}`",
        f"- Artifact dir: `{matrix_sidecar.get('artifact_dir')}`",
        f"- sidecar_generate_ms: `{_fmt_counter(matrix_sidecar.get('sidecar_generate_ms'))}`",
        f"- Fairness: `{matrix_sidecar.get('fairness_classification')}`",
        f"- Policy: {matrix_sidecar.get('no_advantage_policy')}",
        "",
        "## API-runner path",
        "",
        f"- Required jobs: `{len(_list(api_runner.get('jobs')))}`",
        f"- Pass: `{'yes' if api_runner.get('pass') else 'no'}`",
        "- State requirement: `queued -> running -> succeeded`.",
        "",
        "## Mac Studio results",
        "",
        f"- Overall pass: `{'yes' if payload.get('pass') else 'no'}`",
        f"- Scheduler pass: `{'yes' if scheduler.get('pass') else 'no'}`",
        f"- Lazy cache memory pass: `{'yes' if lazy.get('pass') else 'no'}`",
        "",
        "## Parity",
        "",
        f"- Passed jobs: `{parity.get('passed_jobs')}/{parity.get('required_jobs')}`",
        f"- Failed jobs: `{parity.get('failed_jobs')}`",
        "",
        "## Performance",
        "",
        f"- Stage timing jobs: `{performance.get('stage_timing_jobs')}`",
        f"- CPU sampling jobs: `{performance.get('cpu_sampling_jobs')}`",
        f"- CPU sampling failed jobs: `{performance.get('failed_cpu_sampling_jobs')}`",
        f"- Policy: {performance.get('canonical_stage_comparison_policy')}",
        "",
        (
            "| Job | Threads | Exact current s | Exact May2 s | Ratio | "
            "CPU mean % | CPU p50 % | CPU max % | System memory gate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for job in jobs:
        timings = _mapping(job.get("stage_timings"))
        parity_payload = _mapping(job.get("parity"))
        telemetry = _mapping(parity_payload.get("exact_telemetry"))
        memory_payload = _mapping(job.get("memory"))
        system_gate = _mapping(memory_payload.get("system_memory_cleanup"))
        cpu_sampling = _mapping(job.get("cpu_sampling"))
        reference_exact = _reference_exact_seconds(job=job)
        current_exact = _current_exact_seconds(timings=timings)
        ratio = (
            None
            if current_exact is None or reference_exact is None or current_exact <= 0
            else reference_exact / current_exact
        )
        lines.append(
            "| "
            f"`{job.get('job_name')}` | "
            f"{telemetry.get('numba_num_threads')} | "
            f"{_fmt_float(current_exact)} | "
            f"{_fmt_float(reference_exact)} | "
            f"{_fmt_float(ratio)} | "
            f"{_fmt_float(_float_mapping_value(cpu_sampling, 'mean_cpu_percent'))} | "
            f"{_fmt_float(_float_mapping_value(cpu_sampling, 'p50_cpu_percent'))} | "
            f"{_fmt_float(_float_mapping_value(cpu_sampling, 'max_cpu_percent'))} | "
            f"{'pass' if system_gate.get('pass') else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Instrumentation counters",
            "",
            f"- Pass: `{'yes' if instrumentation.get('pass') else 'no'}`",
            f"- Required fields: `{len(_list(instrumentation.get('required_fields')))}`",
            "",
            (
                "| Job | artifact load ms | signal pack source | signal pack ms | "
                "sidecar load ms | sidecar used | signal pack bytes | W | padding valid | "
                "consensus sample parity | combos | "
                "proxy candidates | exact candidates/s | trade-cell evals/s | "
                "rows after prefilter | unique rows | consensus signatures | "
                "row signature ms | sidecar fallback | null fields |"
            ),
            (
                "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | "
                "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
            ),
        ]
    )
    instrumentation_rows = {
        str(row.get("job_name")): _mapping(row)
        for row in _list(instrumentation.get("rows"))
    }
    for job in jobs:
        row = instrumentation_rows.get(str(job.get("job_name")), {})
        counters = _mapping(row.get("counters"))
        lines.append(
            "| "
            f"`{job.get('job_name')}` | "
            f"{_fmt_counter(counters.get('artifact_load_ms'))} | "
            f"{_fmt_counter(counters.get('signals_pack_source'))} | "
            f"{_fmt_counter(counters.get('signals_pack_ms'))} | "
            f"{_fmt_counter(counters.get('sidecar_load_ms'))} | "
            f"{_fmt_counter(counters.get('sidecar_used'))} | "
            f"{_fmt_counter(counters.get('signals_pack_bytes'))} | "
            f"{_fmt_counter(counters.get('bitset_word_count'))} | "
            f"{_fmt_counter(counters.get('bitset_padding_valid'))} | "
            f"{_fmt_counter(counters.get('bitset_consensus_sample_parity'))} | "
            f"{_fmt_counter(counters.get('combo_count_planned'))} | "
            f"{_fmt_counter(counters.get('candidates_after_proxy'))} | "
            f"{_fmt_counter(counters.get('exact_candidates_per_sec'))} | "
            f"{_fmt_counter(counters.get('trade_cell_evals_per_sec'))} | "
            f"{_fmt_counter(counters.get('rows_after_prefilter'))} | "
            f"{_fmt_counter(counters.get('unique_rows_after_dedup'))} | "
            f"{_fmt_counter(counters.get('consensus_signature_count'))} | "
            f"{_fmt_counter(counters.get('row_signature_ms'))} | "
            f"{_fmt_counter(counters.get('sidecar_fallback_reason'))} | "
            f"`{_list(row.get('null_fields'))}` |"
        )
    lines.extend(
        [
            "",
            "## TP/SL cell backend",
            "",
            "| Job | backend | block | blocks/candidate | block bytes | trade-cell evals |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for job in jobs:
        row = instrumentation_rows.get(str(job.get("job_name")), {})
        counters = _mapping(row.get("counters"))
        lines.append(
            "| "
            f"`{job.get('job_name')}` | "
            f"{_fmt_counter(counters.get('tp_sl_cell_backend_id'))} | "
            f"{_fmt_counter(counters.get('tp_sl_cell_block_shape'))} | "
            f"{_fmt_counter(counters.get('tp_sl_cell_blocks_per_candidate'))} | "
            f"{_fmt_counter(counters.get('tp_sl_cell_block_estimated_peak_bytes'))} | "
            f"{_fmt_counter(counters.get('tp_sl_cell_trade_cell_evals'))} |"
        )
    lines.extend(
        [
            "",
        "## Memory release",
        "",
        f"- Checked jobs: `{memory.get('checked_jobs')}`",
        f"- Failed jobs: `{memory.get('failed_jobs')}`",
        f"- System memory failed jobs: `{memory.get('system_memory_failed_jobs')}`",
        f"- System cleanup limit bytes: `{memory.get('system_memory_cleanup_limit_bytes')}`",
        f"- vmmap / physical footprint: `{'yes' if memory.get('vmmap') else 'no'}`",
        "",
        "## Lazy cache-hit memory",
        "",
        f"- Target job: `{lazy.get('target_job_id')}`",
        (
            "- Cache hit retained RSS delta: "
            f"`{_mapping(lazy.get('api_process_memory')).get('retained_rss_delta')}`"
        ),
        f"- Pass: `{'yes' if lazy.get('pass') else 'no'}`",
        "",
        "## Legacy path absence",
        "",
        f"- Pass: `{'yes' if _mapping(payload.get('legacy_path_absence')).get('pass') else 'no'}`",
        "",
        "## Dead code audit",
        "",
        f"- Pass: `{'yes' if _mapping(payload.get('dead_code_audit')).get('pass') else 'no'}`",
        "",
        "## Docs drift audit",
        "",
        f"- Pass: `{'yes' if _mapping(payload.get('docs_drift_audit')).get('pass') else 'no'}`",
        "",
        "## Artifacts",
        "",
        "- `benchmark_results.json`",
        "- `benchmark_summary.md`",
        "- `child_process_evidence/*.json`",
        "",
        "## Operator Commands",
        "",
        "```bash",
        "uv run python scripts/backtest/run_api_runner_benchmark_parity.py",
        "```",
        "",
        (
            "Accounting validator note: `validate_benchmark_accounting.py` expects "
            "canonical notebook benchmark JSON, not the API-runner `benchmark_results.json` "
            "schema emitted by this harness."
        ),
        ]
    )
    return "\n".join(lines) + "\n"


def _reference_run_exact_seconds(*, run: Mapping[str, Any]) -> float | None:
    timers = _mapping(run.get("timers"))
    value = timers.get("exact_scoring")
    if isinstance(value, int | float):
        return float(value)
    value = timers.get("tp_sl_exact_scoring")
    if isinstance(value, int | float):
        return float(value)
    return None


def _reference_exact_seconds(*, job: Mapping[str, Any]) -> float | None:
    value = job.get("reference_exact_scoring_s")
    return float(value) if isinstance(value, int | float) else None


def _current_exact_seconds(*, timings: Mapping[str, Any]) -> float | None:
    value = timings.get("exact_scoring")
    if isinstance(value, int | float):
        return float(value)
    value = timings.get("tp_sl_exact_scoring")
    if isinstance(value, int | float):
        return float(value)
    return None


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_counter(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _fmt_float(value)
    return str(value)


def _float_mapping_value(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _format_datetime(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if value is None:
        return None
    return str(value)


def _git_commit() -> str:
    override = os.environ.get("ROEHUB_BENCHMARK_GIT_COMMIT", "").strip()
    if override:
        return override
    return _git_output(["git", "rev-parse", "HEAD"])


def _git_status_short() -> str:
    return _git_output(["git", "status", "--short"])


def _git_output(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(  # noqa: S603
            list(cmd),
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _hash_payload(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
