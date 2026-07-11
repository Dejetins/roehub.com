from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.codex_quality_benchmark.models import (
    BenchmarkError,
    EvalCase,
    Manifest,
    RubricDimension,
    TargetRecord,
    VersionRecord,
)

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_VERSION_RE = re.compile(r"^v(?P<iteration>\d+)$")
_SKILL_TYPES = {
    "workflow_skill",
    "research_skill",
    "coding_skill",
    "review_skill",
    "artifact_skill",
    "plugin_tool_skill",
}


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid JSON in {path}: {exc}") from exc


def load_manifest(path: Path) -> Manifest:
    data = load_json_file(path)
    if not isinstance(data, dict):
        raise BenchmarkError("manifest root must be an object")

    run_id = _required_str(data, "run_id", "manifest")
    rubric = _parse_rubric(_required_list(data, "rubric", "manifest"))
    eval_cases = _parse_eval_cases(_required_list(data, "eval_cases", "manifest"))
    targets = _parse_targets(_required_list(data, "targets", "manifest"))
    manifest = Manifest(run_id=run_id, rubric=rubric, eval_cases=eval_cases, targets=targets)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Manifest) -> None:
    if manifest.rubric_total != 100:
        raise BenchmarkError(f"rubric dimensions must sum to 100, got {manifest.rubric_total}")
    if not manifest.targets:
        raise BenchmarkError("manifest must include at least one target")
    if len({target.target_id for target in manifest.targets}) != len(manifest.targets):
        raise BenchmarkError("target_id values must be unique")
    if len({case.case_id for case in manifest.eval_cases}) != len(manifest.eval_cases):
        raise BenchmarkError("eval case IDs must be unique")

    for target in manifest.targets:
        if target.skill_type not in _SKILL_TYPES:
            raise BenchmarkError(
                f"unsupported skill_type for {target.target_id}: {target.skill_type}"
            )
        if not target.versions:
            raise BenchmarkError(f"target {target.target_id} must include at least one version")
        if not manifest.expected_case_ids_for(target.skill_type):
            raise BenchmarkError(f"target {target.target_id} has no applicable eval cases")
        seen_versions: set[str] = set()
        for version in target.versions:
            if version.version_id in seen_versions:
                raise BenchmarkError(
                    f"target {target.target_id} has duplicate version {version.version_id}"
                )
            seen_versions.add(version.version_id)
            if not _SHA256_RE.fullmatch(version.sha256):
                raise BenchmarkError(
                    f"target {target.target_id} version {version.version_id} has invalid sha256"
                )


def _parse_rubric(items: list[Any]) -> tuple[RubricDimension, ...]:
    rubric: list[RubricDimension] = []
    for index, item in enumerate(items):
        context = f"rubric[{index}]"
        if not isinstance(item, dict):
            raise BenchmarkError(f"{context} must be an object")
        name = _required_str(item, "dimension", context)
        points = _required_int(item, "points", context)
        if points <= 0:
            raise BenchmarkError(f"{context}.points must be positive")
        rubric.append(RubricDimension(name=name, points=points))
    return tuple(rubric)


def _parse_eval_cases(items: list[Any]) -> tuple[EvalCase, ...]:
    cases: list[EvalCase] = []
    for index, item in enumerate(items):
        context = f"eval_cases[{index}]"
        if not isinstance(item, dict):
            raise BenchmarkError(f"{context} must be an object")
        cases.append(
            EvalCase(
                case_id=_required_str(item, "case_id", context),
                applies_to=_required_str(item, "applies_to", context),
            )
        )
    return tuple(cases)


def _parse_targets(items: list[Any]) -> tuple[TargetRecord, ...]:
    targets: list[TargetRecord] = []
    for index, item in enumerate(items):
        context = f"targets[{index}]"
        if not isinstance(item, dict):
            raise BenchmarkError(f"{context} must be an object")
        target_id = _required_str(item, "target_id", context)
        versions = _parse_versions(_required_list(item, "versions", context))
        targets.append(
            TargetRecord(
                target_id=target_id,
                target_path=_required_str(item, "target_path", context),
                skill_type=_required_str(item, "skill_type", context),
                versions=versions,
            )
        )
    return tuple(targets)


def _parse_versions(items: list[Any]) -> tuple[VersionRecord, ...]:
    versions: list[VersionRecord] = []
    for index, item in enumerate(items):
        context = f"versions[{index}]"
        if not isinstance(item, dict):
            raise BenchmarkError(f"{context} must be an object")
        version_id = _required_str(item, "version_id", context)
        version_match = _VERSION_RE.fullmatch(version_id)
        default_iteration = int(version_match.group("iteration")) if version_match else 0
        versions.append(
            VersionRecord(
                version_id=version_id,
                sha256=_required_str(item, "sha256", context),
                iteration=_optional_int(item, "iteration", default_iteration, context),
                approach_label=str(item.get("approach_label", "baseline")),
                attempt_status=str(item.get("attempt_status", "candidate")),
            )
        )
    return tuple(versions)


def _required_str(data: dict[str, Any], field: str, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{context}.{field} must be a non-empty string")
    return value


def _required_int(data: dict[str, Any], field: str, context: str) -> int:
    value = data.get(field)
    if not isinstance(value, int):
        raise BenchmarkError(f"{context}.{field} must be an integer")
    return value


def _optional_int(data: dict[str, Any], field: str, default: int, context: str) -> int:
    value = data.get(field, default)
    if not isinstance(value, int):
        raise BenchmarkError(f"{context}.{field} must be an integer")
    return value


def _required_list(data: dict[str, Any], field: str, context: str) -> list[Any]:
    value = data.get(field)
    if not isinstance(value, list):
        raise BenchmarkError(f"{context}.{field} must be a list")
    return value
