from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.docs.generate_runbooks import (
    CANONICAL_DIR,
    CAPABILITIES,
    INDEX_PATH,
    RU_LOCALE_DIR,
    RunbookError,
    _load_json,
    _load_yaml,
    _validate_capabilities,
    _validate_locale_coverage,
    _validate_no_secret_or_shell_fields,
    expected_outputs,
    run_generator,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_runbook_generator_is_deterministic_and_indexed() -> None:
    assert run_generator(REPO_ROOT, check=True) == 0

    outputs = expected_outputs(REPO_ROOT)
    index = json.loads(outputs[INDEX_PATH])
    assert len(index["runbooks"]) == 9
    assert len(index["problem_index"]) == 20
    assert len(index["legacy_unmigrated"]) == 23
    assert all(
        problem in index["problem_index"]
        for runbook in index["runbooks"]
        for problem in runbook["problems"]
    )


def test_russian_render_preserves_safety_and_typed_actions() -> None:
    outputs = expected_outputs(REPO_ROOT)
    for canonical_path in sorted((REPO_ROOT / CANONICAL_DIR).glob("*.yaml")):
        runbook = _load_yaml(canonical_path)
        runbook_id = runbook["metadata"]["id"]
        locale = _load_yaml(REPO_ROOT / RU_LOCALE_DIR / f"{runbook_id}.yaml")
        rendered = outputs[Path(f"docs/runbooks/generated/ru/{runbook_id}.md")].decode()
        for warning in runbook["spec"]["safety"]["warnings"]:
            assert locale["translations"]["warnings"][warning["id"]] in rendered
        for action in runbook["spec"]["allowed_actions"]:
            assert f"`{action['capability']}`" in rendered
            assert f"`{action['approval']}`" in rendered


def test_arbitrary_operations_and_secret_shaped_values_are_rejected() -> None:
    with pytest.raises(RunbookError, match="arbitrary operation field"):
        _validate_no_secret_or_shell_fields({"command": "restart everything"}, Path("bad.yaml"))
    with pytest.raises(RunbookError, match="secret-shaped value"):
        _validate_no_secret_or_shell_fields(
            {"text": "Bearer abcdefghijklmnopqrstuvwxyz"}, Path("bad.yaml")
        )


def test_missing_russian_safety_translation_is_rejected() -> None:
    path = REPO_ROOT / CANONICAL_DIR / "execution.provider-state-unknown.yaml"
    runbook = _load_yaml(path)
    locale = _load_yaml(
        REPO_ROOT / RU_LOCALE_DIR / "execution.provider-state-unknown.yaml"
    )
    broken = copy.deepcopy(locale)
    broken["translations"]["warnings"].pop("no_blind_retry")

    with pytest.raises(RunbookError, match="locale coverage mismatch"):
        _validate_locale_coverage(runbook, broken)


def test_action_cannot_weaken_catalog_approval() -> None:
    path = REPO_ROOT / CANONICAL_DIR / "execution.provider-state-unknown.yaml"
    runbook = _load_yaml(path)
    broken = copy.deepcopy(runbook)
    broken["spec"]["allowed_actions"][0]["approval"] = "none"
    catalog = _load_json(REPO_ROOT / CAPABILITIES)

    with pytest.raises(RunbookError, match="approval is weaker"):
        _validate_capabilities(broken, catalog)
