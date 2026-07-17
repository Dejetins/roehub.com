from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.codex_quality_benchmark.skill_contract import (
    load_schema,
    validate_instance,
    validate_skill_result,
    validate_skill_spec,
)


def test_all_skill_system_schemas_are_valid() -> None:
    for name in (
        "skill-spec-v1.schema.json",
        "skill-result-v1.schema.json",
        "skill-contract-case-result-v1.schema.json",
        "skill-catalog-v1.schema.json",
    ):
        assert load_schema(name)["$schema"].endswith("2020-12/schema")


def test_skill_spec_accepts_machine_readable_metadata(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(
        """---
name: example-skill
description: "Use when inspecting an example. Do not use for mutation."
metadata:
  short-description: "Inspect example"
  skill-spec: "v1"
  role: "workflow"
  visibility: "public"
  owner: "user"
  mutability: "source"
  side-effect-class: "read-only"
  primary-output: "report"
  companions: []
  conflicts: []
---

# Example
""",
        encoding="utf-8",
    )
    assert validate_skill_spec(path).valid


def test_skill_spec_accepts_standard_minimal_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(
        "---\nname: example-skill\ndescription: Use when inspecting an example.\n---\n",
        encoding="utf-8",
    )
    assert validate_skill_spec(path).valid


def test_skill_result_accepts_generic_runtime_proof_label() -> None:
    result = _result()
    result["skill_run"]["proof_boundary"] = {
        "surface": "runtime",
        "profile": "example-project",
        "label": "generic-runtime",
    }
    assert validate_skill_result(result).valid


def test_fixture_result_schema_is_closed() -> None:
    value = {
        "spec": "skill-contract-case-result/v1",
        "run_id": "fixture",
        "results": [],
        "summary": {"total": 0, "passed": 0, "failed": 0},
        "unexpected": True,
    }
    assert not validate_instance(value, "skill-contract-case-result-v1.schema.json").valid


def _result() -> dict[str, Any]:
    return {
        "skill_run": {
            "spec": "skill-result/v1",
            "skill": "example-skill",
            "status": "completed",
            "mode": "inspect",
            "inputs": [],
            "assumptions": [],
            "actions": [],
            "evidence": [],
            "side_effects": [],
            "contract_impact": {"classification": "none", "dimensions": []},
            "files": {
                "created": [],
                "modified": [],
                "deleted": [],
                "outside_expected_paths": [],
            },
            "redaction": "not-needed",
            "proof_boundary": {"surface": "none", "profile": "generic", "label": "none"},
            "residual_risks": [],
            "next_action": "none",
        }
    }
