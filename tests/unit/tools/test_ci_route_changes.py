from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ci.route_changes import (
    ALL_SHARDS,
    TEST_SHARDS,
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
    assert _matrix_names(outputs) == set(ALL_SHARDS)


def test_web_image_routing_is_explicit() -> None:
    assert classify_web_image(["apps/web/main/app.py"])


def test_codex_hook_change_runs_code_gates() -> None:
    outputs = classify_ci([".codex/hooks/validators/common.py"])

    assert outputs["code"] == "true"
    assert outputs["has_tests"] == "true"


_REPO_ROOT = Path(__file__).resolve().parents[3]
_UNIT_TESTS = sorted(
    path.relative_to(_REPO_ROOT).as_posix()
    for path in (_REPO_ROOT / "tests/unit").rglob("test_*.py")
)


def _covers(target: str, test_path: str) -> bool:
    candidate = _REPO_ROOT / target
    test = _REPO_ROOT / test_path
    return test == candidate or (candidate.is_dir() and test.is_relative_to(candidate))


@pytest.mark.parametrize("test_path", _UNIT_TESTS)
def test_every_unit_test_is_in_full_and_changed_file_matrices(test_path: str) -> None:
    full_targets = [target for name in ALL_SHARDS for target in TEST_SHARDS[name].target.split()]
    assert any(_covers(target, test_path) for target in full_targets), test_path

    outputs = classify_ci([test_path])
    selected_targets = [
        target
        for item in json.loads(outputs["test_matrix"])["include"]
        for target in item["target"].split()
    ]
    assert any(_covers(target, test_path) for target in selected_targets), test_path


def test_full_matrix_targets_exist() -> None:
    for name in ALL_SHARDS:
        for target in TEST_SHARDS[name].target.split():
            assert (_REPO_ROOT / target).exists(), (name, target)


@pytest.mark.parametrize(
    "path",
    [
        "schemas/ops/runbook.schema.json",
        "sdk/typescript/src/index.ts",
        "sdk/python/roehub_plugin_sdk/v1alpha1.py",
    ],
)
def test_schema_and_sdk_changes_run_contract_tests(path: str) -> None:
    outputs = classify_ci([path])
    assert outputs["code"] == "true"
    assert outputs["has_tests"] == "true"
    assert "platform-contracts" in _matrix_names(outputs)


def test_local_authorization_kernel_tests_are_selected() -> None:
    outputs = classify_ci(
        [
            "tests/unit/identity/authorization/test_capability_authorization.py",
        ]
    )
    assert "platform-contracts" in _matrix_names(outputs)


@pytest.mark.parametrize(
    "path",
    [
        "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
        "apps/api/wiring/modules/extensions.py",
        "src/trading/contexts/identity/domain/user.py",
        "configs/prod/backtest.yaml",
        "alembic/versions/new_migration.py",
        "migrations/postgres/new_migration.sql",
        "alembic.ini",
    ],
)
def test_app_image_rebuilds_for_copied_runtime_inputs(path: str) -> None:
    assert classify_web_image([path])


@pytest.mark.parametrize(
    "path",
    [
        "schemas/release/release-manifest.schema.json",
        "configs/installation/runtime-service-manifest.json",
        "configs/installation/generated/base/compose.yaml",
    ],
)
def test_app_image_ignores_digest_bound_outputs_excluded_from_context(path: str) -> None:
    assert not classify_web_image([path])
