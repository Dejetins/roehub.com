from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestShard:
    name: str
    target: str


def _target(*paths: str) -> str:
    return " ".join(paths)


TEST_SHARDS: dict[str, TestShard] = {
    "web-api": TestShard(
        name="web-api",
        target=_target(
            "tests/unit/apps/web "
            "tests/unit/apps/api/test_ui_account_routes.py "
            "tests/unit/apps/api/test_ui_backtests_routes.py "
            "tests/unit/apps/api/test_ui_dashboard_routes.py "
            "tests/unit/apps/api/test_ui_strategy_dashboard_routes.py"
        ),
    ),
    "apps-platform": TestShard(
        name="apps-platform",
        target=_target(
            "tests/test_smoke.py tests/unit/apps tests/unit/infra "
            "tests/unit/platform tests/unit/shared_kernel tests/unit/tools"
        ),
    ),
    "backtest-artifacts": TestShard(
        name="backtest-artifacts",
        target=_target(
            "tests/unit/contexts/backtest/adapters/outbound/artifacts_fs",
            "tests/unit/contexts/backtest/adapters/test_backtest_artifacts_runtime_config.py",
            "tests/unit/contexts/backtest/adapters/test_indicators_yaml_defaults_provider.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_artifact_manifest_validator_v2.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_artifact_precompute_runner_v2.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_artifact_slot_publisher_v2.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_yaml_backtest_artifact_loader_v2.py",
            "tests/unit/contexts/backtest/application/use_cases/"
            "test_publish_backtest_artifacts_v2.py",
            "tests/unit/apps/cli/test_backtest_artifact_publish_cli.py",
            "tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py",
            "tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py",
        ),
    ),
    "backtest-scoring": TestShard(
        name="backtest-scoring",
        target=_target(
            "tests/unit/contexts/backtest/application/services/"
            "test_signals_from_indicators_v1.py",
            "tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py",
            "tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py",
            "tests/unit/contexts/backtest/application/services/v2/test_hit_times_compute_v2.py",
            "tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_lazy_trades_detail_service.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_no_risk_exact_scoring_service.py",
            "tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py",
            "tests/unit/contexts/backtest/application/services/v2/test_signal_rules_engine_v2.py",
            "tests/unit/contexts/backtest/application/services/v2/test_top_result_assembly.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_tp_sl_exact_scoring_service.py",
            "tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py",
        ),
    ),
    "backtest-use-cases": TestShard(
        name="backtest-use-cases",
        target=_target(
            "tests/unit/contexts/backtest/application/use_cases/"
            "test_backtest_job_worker_use_case.py",
            "tests/unit/contexts/backtest/application/use_cases/test_backtest_jobs_use_case.py",
            "tests/unit/contexts/backtest/application/use_cases/"
            "test_lazy_trades_materialization_worker_use_case.py",
            "tests/unit/apps/api/test_backtests_routes.py",
            "tests/unit/apps/api/test_ui_backtests_routes.py",
            "tests/unit/apps/worker/backtest_job_runner",
            "tests/unit/apps/worker/test_backtest_job_runner.py",
        ),
    ),
    "backtest-domain": TestShard(
        name="backtest-domain",
        target=_target(
            "tests/unit/contexts/backtest/application/test_backtest_errors.py",
            "tests/unit/contexts/backtest/application/services/v2/test_admission.py",
            "tests/unit/contexts/backtest/application/services/v2/"
            "test_backtest_preflight_service.py",
            "tests/unit/contexts/backtest/domain",
        ),
    ),
    "indicators-engine": TestShard(
        name="indicators-engine",
        target="tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_engine.py",
    ),
    "indicators-kernels-a": TestShard(
        name="indicators-kernels-a",
        target=_target(
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_common_kernels.py "
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_ma_kernels.py "
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_trend_kernels.py"
        ),
    ),
    "indicators-kernels-b": TestShard(
        name="indicators-kernels-b",
        target=_target(
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/"
            "test_momentum_kernels.py",
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/"
            "test_runtime_wiring.py",
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/"
            "test_structure_kernels.py",
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/"
            "test_volatility_kernels.py",
            "tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volume_kernels.py",
        ),
    ),
    "indicators-api-domain": TestShard(
        name="indicators-api-domain",
        target=_target(
            "tests/unit/contexts/indicators/adapters/outbound/compute_numpy "
            "tests/unit/contexts/indicators/adapters/outbound/config "
            "tests/unit/contexts/indicators/adapters/outbound/feeds "
            "tests/unit/contexts/indicators/adapters/outbound/registry "
            "tests/unit/contexts/indicators/api "
            "tests/unit/contexts/indicators/application "
            "tests/unit/contexts/indicators/domain"
        ),
    ),
    "market-identity-strategy": TestShard(
        name="market-identity-strategy",
        target=_target(
            "tests/unit/contexts/identity",
            "tests/unit/contexts/market_data",
            "tests/unit/contexts/strategy",
            "tests/unit/identity",
        ),
    ),
}

BACKTEST_SHARDS = (
    "backtest-artifacts",
    "backtest-scoring",
    "backtest-use-cases",
    "backtest-domain",
)
INDICATOR_SHARDS = (
    "indicators-engine",
    "indicators-kernels-a",
    "indicators-kernels-b",
    "indicators-api-domain",
)
ALL_SHARDS = (
    "apps-platform",
    *BACKTEST_SHARDS,
    *INDICATOR_SHARDS,
    "market-identity-strategy",
)
MATRIX_ORDER = ("web-api", *ALL_SHARDS)


def _has_prefix(path: str, prefixes: Iterable[str]) -> bool:
    return any(path.startswith(prefix) for prefix in prefixes)


def _is_docs_path(path: str) -> bool:
    return path in {
        "README.md",
        "AGENTS.md",
        ".codex/AGENTS.md",
    } or path.startswith("docs/")


def _is_code_path(path: str) -> bool:
    if path in {
        ".python-version",
        "pyproject.toml",
        "uv.lock",
        "alembic.ini",
        "pyrightconfig.json",
        "Dockerfile.api",
        "infra/docker/Dockerfile.market_data",
        "infra/docker/docker-compose.web.prod.yml",
        "infra/caddy/Caddyfile.vps",
    }:
        return True
    return _has_prefix(
        path,
        (
            ".github/workflows/",
            ".codex/hooks/",
            "apps/",
            "src/",
            "tests/",
            "tools/",
            "scripts/",
            "configs/",
            "alembic/",
            "migrations/",
            "infra/macos/",
            "infra/scripts/",
        ),
    )


def _is_backtest_api_path(path: str) -> bool:
    return path in {
        "apps/api/routes/backtests.py",
        "apps/api/routes/ui_backtests.py",
        "apps/api/dto/backtests.py",
        "apps/api/dto/ui_backtests.py",
        "apps/api/wiring/modules/backtest.py",
        "apps/api/wiring/modules/ui_backtests.py",
        "tests/unit/apps/api/test_backtests_routes.py",
        "tests/unit/apps/api/test_ui_backtests_routes.py",
    }


def _is_web_path(path: str) -> bool:
    return (
        path.startswith("apps/web/")
        or path.startswith("tests/unit/apps/web/")
        or path
        in {
            "apps/api/routes/ui_backtests.py",
            "apps/api/dto/ui_backtests.py",
            "apps/api/wiring/modules/ui_backtests.py",
            "tests/unit/apps/api/test_ui_account_routes.py",
            "tests/unit/apps/api/test_ui_backtests_routes.py",
            "tests/unit/apps/api/test_ui_dashboard_routes.py",
            "tests/unit/apps/api/test_ui_strategy_dashboard_routes.py",
        }
    )


def _is_backtest_path(path: str) -> bool:
    return (
        path.startswith("src/trading/contexts/backtest/")
        or path.startswith("src/trading/contexts/backtest_artifacts/")
        or path.startswith("tests/unit/contexts/backtest/")
        or path.startswith("scripts/backtest/")
        or path.startswith("apps/worker/backtest_job_runner/")
        or path.startswith("tests/unit/apps/worker/backtest_job_runner/")
        or path
        in {
            "tests/unit/apps/worker/test_backtest_job_runner.py",
            "apps/cli/commands/backtest_artifact_publish.py",
            "tests/unit/apps/cli/test_backtest_artifact_publish_cli.py",
            "apps/scheduler/backtest_artifact_publisher/main/main.py",
            "apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py",
            "tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py",
            "tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py",
        }
        or _is_backtest_api_path(path)
    )


def _is_backtest_config_path(path: str) -> bool:
    return path.startswith("configs/") and ("backtest" in path or path.endswith("/indicators.yaml"))


def _is_indicator_path(path: str) -> bool:
    return (
        path.startswith("src/trading/contexts/indicators/")
        or path.startswith("tests/unit/contexts/indicators/")
        or path.startswith("tests/perf_smoke/contexts/indicators/")
        or path in {"apps/api/routes/indicators.py", "apps/api/dto/indicators.py"}
    )


def _is_market_identity_strategy_path(path: str) -> bool:
    return _has_prefix(
        path,
        (
            "src/trading/contexts/identity/",
            "src/trading/contexts/market_data/",
            "src/trading/contexts/strategy/",
            "tests/unit/contexts/identity/",
            "tests/unit/contexts/market_data/",
            "tests/unit/contexts/strategy/",
            "tests/unit/identity/",
        ),
    )


def _needs_migration_check(path: str) -> bool:
    if path in {
        ".python-version",
        "pyproject.toml",
        "uv.lock",
        "alembic.ini",
    }:
        return True
    return _has_prefix(
        path,
        (
            "apps/api/",
            "apps/cli/",
            "apps/migrations/",
            "apps/monitoring/",
            "apps/scheduler/",
            "apps/worker/",
            "src/",
            "configs/",
            "alembic/",
            "migrations/",
            "scripts/backtest/",
            "scripts/macos/",
            "infra/macos/",
            "infra/scripts/",
        ),
    )


def _is_web_image_path(path: str) -> bool:
    return path.startswith("apps/web/") or path in {
        ".python-version",
        "pyproject.toml",
        "uv.lock",
        "infra/docker/Dockerfile.market_data",
        ".github/workflows/publish-app-image.yml",
    }


def _matrix(shard_names: Iterable[str]) -> str:
    selected = set(shard_names)
    include = [
        {"name": TEST_SHARDS[name].name, "target": TEST_SHARDS[name].target}
        for name in MATRIX_ORDER
        if name in selected
    ]
    return json.dumps({"include": include}, separators=(",", ":"))


def classify_ci(paths: Iterable[str], *, all_changes: bool = False) -> dict[str, str]:
    path_list = [path for path in paths if path]
    if all_changes:
        return {
            "code": "true",
            "docs": "true",
            "run_migrations": "true",
            "has_tests": "true",
            "test_matrix": _matrix(ALL_SHARDS),
        }

    code = any(_is_code_path(path) for path in path_list)
    docs = any(_is_docs_path(path) for path in path_list)
    shards: set[str] = set()
    run_migrations = False
    run_all = False

    for path in path_list:
        if path.startswith(".github/workflows/") or path in {
            ".python-version",
            "pyproject.toml",
            "uv.lock",
            "pyrightconfig.json",
        }:
            run_all = True
            run_migrations = True
            continue

        if _is_web_path(path):
            shards.add("web-api")

        if _is_backtest_path(path) or _is_backtest_config_path(path):
            shards.update(BACKTEST_SHARDS)

        if _is_indicator_path(path):
            shards.update(INDICATOR_SHARDS)

        if _is_market_identity_strategy_path(path):
            shards.add("market-identity-strategy")

        if _needs_migration_check(path):
            run_migrations = True

        if (
            (
                path.startswith("apps/")
                or path.startswith("tests/unit/apps/")
                or path.startswith("tests/unit/infra/")
                or path.startswith("tests/unit/platform/")
                or path.startswith("tests/unit/shared_kernel/")
                or path.startswith("tests/unit/tools/")
                or path.startswith("tools/")
                or path.startswith("infra/")
                or path.startswith("scripts/macos/")
            )
            and not _is_web_path(path)
            and not _is_backtest_path(path)
        ):
            shards.add("apps-platform")

        if path.startswith("src/trading/shared_kernel/") or path.startswith(
            "tests/unit/shared_kernel/"
        ):
            run_all = True

        if path.startswith("configs/") and not (
            _is_backtest_config_path(path) or "market_data" in path
        ):
            run_all = True

    if run_all:
        shards.update(ALL_SHARDS)

    if code and not shards:
        shards.update(ALL_SHARDS)
        run_migrations = True

    return {
        "code": "true" if code else "false",
        "docs": "true" if docs else "false",
        "run_migrations": "true" if run_migrations else "false",
        "has_tests": "true" if shards else "false",
        "test_matrix": _matrix(shards),
    }


def classify_web_image(paths: Iterable[str], *, all_changes: bool = False) -> bool:
    return all_changes or any(_is_web_image_path(path) for path in paths if path)


def _read_paths(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _print_outputs(outputs: dict[str, str]) -> None:
    for key, value in outputs.items():
        print(f"{key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify changed paths for GitHub Actions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("ci", "web-image"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--changed-files", type=Path, required=True)
        subparser.add_argument("--all", action="store_true", dest="all_changes")

    args = parser.parse_args()
    paths = _read_paths(args.changed_files)

    if args.command == "ci":
        _print_outputs(classify_ci(paths, all_changes=args.all_changes))
    elif args.command == "web-image":
        _print_outputs(
            {
                "web_image_changed": (
                    "true" if classify_web_image(paths, all_changes=args.all_changes) else "false"
                )
            }
        )


if __name__ == "__main__":
    main()
