from __future__ import annotations

import json

from tools.ci.route_changes import (
    classify_ci,
    classify_web_image,
)


def _matrix_names(outputs: dict[str, str]) -> set[str]:
    return {item["name"] for item in json.loads(outputs["test_matrix"])["include"]}


def test_web_only_change_runs_web_tests_without_backtest_or_migrations() -> None:
    paths = [
        "apps/web/dist/css/pages/backtests.css",
        "tests/unit/apps/web/test_app_routes.py",
    ]

    outputs = classify_ci(paths)

    assert outputs["code"] == "true"
    assert outputs["run_migrations"] == "false"
    assert _matrix_names(outputs) == {"web-api"}
    assert classify_web_image(paths)


def test_backtest_context_change_runs_backtest_subshards() -> None:
    outputs = classify_ci(
        ["src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"]
    )

    assert _matrix_names(outputs) == {
        "backtest-artifacts",
        "backtest-scoring",
        "backtest-use-cases",
        "backtest-domain",
    }
    assert outputs["run_migrations"] == "true"


def test_backtest_api_change_runs_backtest_and_web_api_tests() -> None:
    outputs = classify_ci(["apps/api/routes/ui_backtests.py"])

    assert _matrix_names(outputs) == {
        "web-api",
        "backtest-artifacts",
        "backtest-scoring",
        "backtest-use-cases",
        "backtest-domain",
    }


def test_workflow_or_lockfile_change_runs_full_ci() -> None:
    outputs = classify_ci(["uv.lock"])

    assert outputs["run_migrations"] == "true"
    assert _matrix_names(outputs) == {
        "apps-platform",
        "backtest-artifacts",
        "backtest-scoring",
        "backtest-use-cases",
        "backtest-domain",
        "indicators-engine",
        "indicators-kernels-a",
        "indicators-kernels-b",
        "indicators-api-domain",
        "market-identity-strategy",
    }


def test_web_image_routing_is_explicit() -> None:
    assert classify_web_image(["apps/web/main/app.py"])


def test_codex_hook_change_runs_code_gates() -> None:
    outputs = classify_ci([".codex/hooks/validators/common.py"])

    assert outputs["code"] == "true"
    assert outputs["has_tests"] == "true"
