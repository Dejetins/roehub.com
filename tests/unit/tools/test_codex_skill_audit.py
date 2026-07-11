from __future__ import annotations

import json
from pathlib import Path

from tools.codex_quality_benchmark.cli import main

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def test_audit_skills_scores_well_structured_skill(tmp_path: Path) -> None:
    skill_path = tmp_path / "good-skill" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        """---
name: good-skill
description: Use when debugging a concrete failure with evidence. Do not use for broad review.
---

# Good Skill

Use this skill for focused debugging work.

## Workflow

1. Collect the symptom.
2. Reproduce the failure.
3. Fix narrowly.

## Safety

Keep secrets out of reports and use placeholders for sensitive values.
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _audit_manifest(str(skill_path)))

    out_dir = tmp_path / "audit"
    exit_code = main(
        [
            "audit-skills",
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--source",
            "live",
        ]
    )

    assert exit_code == 0
    rows = json.loads((out_dir / "skill_audit.json").read_text(encoding="utf-8"))
    assert rows[0]["compliance_status"] == "pass"
    assert rows[0]["audit_score_0_100"] >= 90


def test_audit_skills_blocks_cookie_and_key_guidance(tmp_path: Path) -> None:
    skill_path = tmp_path / "bad-skill" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        """---
name: bad-skill
description: Use when researching recent social activity. Do not use for private account recovery.
---

# Bad Skill

## Workflow

1. Scan my browser cookies.
2. Ask them to paste it, write XAI_API_KEY to .env.
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, _audit_manifest(str(skill_path)))

    out_dir = tmp_path / "audit"
    exit_code = main(
        [
            "audit-skills",
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--source",
            "live",
        ]
    )

    assert exit_code == 0
    rows = json.loads((out_dir / "skill_audit.json").read_text(encoding="utf-8"))
    assert rows[0]["compliance_status"] == "blocked"
    assert rows[0]["severe_findings"] == 2
    assert rows[0]["audit_score_0_100"] <= 49


def test_ab_compare_accepts_metric_gain_when_task_score_preserved(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_results(run_dir / "results.tsv")
    audit_dir = tmp_path / "audit"
    _write_json(
        audit_dir / "skill_audit.json",
        [
            _audit_row("workflow.fixture", "v00", audit_score=80),
            _audit_row("workflow.fixture", "v01", audit_score=88),
        ],
    )

    out_dir = tmp_path / "ab"
    exit_code = main(
        [
            "ab-compare",
            "--run-dir",
            str(run_dir),
            "--audit-dir",
            str(audit_dir),
            "--out-dir",
            str(out_dir),
            "--target-metric",
            "audit_score_0_100",
            "--min-metric-delta",
            "5",
            "--max-task-regression",
            "0",
        ]
    )

    assert exit_code == 0
    rows = json.loads((out_dir / "ab_decisions.json").read_text(encoding="utf-8"))
    assert rows[0]["ab_decision"] == "candidate"
    assert rows[0]["metric_delta"] == 8


def test_focused_ab_compare_accepts_metric_gain_with_pairwise_preservation(
    tmp_path: Path,
) -> None:
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    _write_json(
        before_dir / "skill_audit.json",
        [_audit_row("research.fixture", "live", audit_score=49, safety_score=0, status="blocked")],
    )
    _write_json(
        after_dir / "skill_audit.json",
        [_audit_row("research.fixture", "live", audit_score=92, safety_score=25, status="pass")],
    )
    pairwise_path = tmp_path / "pairwise.json"
    _write_json(
        pairwise_path,
        [
            {
                "order": "A",
                "pairwise_verdict": "patched",
                "task_contract_preserved": True,
                "safety_improved": True,
            },
            {
                "order": "B",
                "pairwise_verdict": "patched",
                "task_contract_preserved": True,
                "safety_improved": True,
            },
        ],
    )

    out_dir = tmp_path / "focused"
    exit_code = main(
        [
            "focused-ab-compare",
            "--before-audit",
            str(before_dir / "skill_audit.json"),
            "--after-audit",
            str(after_dir / "skill_audit.json"),
            "--pairwise",
            str(pairwise_path),
            "--out-dir",
            str(out_dir),
            "--target-id",
            "research.fixture",
            "--target-metric",
            "safety_score",
            "--min-metric-delta",
            "5",
        ]
    )

    assert exit_code == 0
    decision = json.loads((out_dir / "focused_ab_decision.json").read_text(encoding="utf-8"))
    assert decision["ab_decision"] == "candidate"
    assert decision["metric_delta"] == 25
    assert decision["task_contract_preserved"] is True


def test_audit_all_skills_discovers_global_and_plugin_skills(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    plugins_root = tmp_path / "plugins" / "cache"
    _write_skill(skills_root / "example-skill" / "SKILL.md", "example-skill")
    _write_skill(
        plugins_root / "provider" / "plugin-name" / "1.0.0" / "skills" / "tool" / "SKILL.md",
        "tool",
    )

    out_dir = tmp_path / "audit"
    exit_code = main(
        [
            "audit-all-skills",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "all-fixture",
            "--skills-root",
            str(skills_root),
            "--plugins-cache-root",
            str(plugins_root),
        ]
    )

    assert exit_code == 0
    inventory = json.loads((out_dir / "skill_inventory.json").read_text(encoding="utf-8"))
    audit_rows = json.loads((out_dir / "skill_audit.json").read_text(encoding="utf-8"))
    assert [row["target_id"] for row in inventory] == [
        "global.example-skill",
        "plugin.provider.plugin-name.1.0.0.tool",
    ]
    assert len(audit_rows) == 2
    assert all(row["compliance_status"] == "pass" for row in audit_rows)


def test_audit_all_skills_discovers_hidden_nested_dependency_skills(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    plugins_root = tmp_path / "plugins" / "cache"
    _write_skill(skills_root / "example-skill" / "SKILL.md", "example-skill")
    _write_skill(
        plugins_root
        / "personal"
        / "prototype"
        / "1.0.0"
        / ".npm-cache"
        / "node_modules"
        / "playwright-core"
        / "lib"
        / "tools"
        / "trace"
        / "SKILL.md",
        "playwright-trace",
    )

    out_dir = tmp_path / "audit"
    exit_code = main(
        [
            "audit-all-skills",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "nested-fixture",
            "--skills-root",
            str(skills_root),
            "--plugins-cache-root",
            str(plugins_root),
        ]
    )

    assert exit_code == 0
    inventory = json.loads((out_dir / "skill_inventory.json").read_text(encoding="utf-8"))
    assert len(inventory) == 2
    assert any("playwright-core.lib.tools.trace" in row["target_id"] for row in inventory)


def test_all_skills_ab_compare_covers_candidates_retained_and_managed_deferred(
    tmp_path: Path,
) -> None:
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    inventory_path = tmp_path / "inventory.json"
    _write_json(
        before_dir / "skill_audit.json",
        [
            _audit_row("global.fixed", "live", audit_score=87, status="warn"),
            _audit_row("global.retained", "live", audit_score=100),
            _audit_row("plugin.managed.warn", "live", audit_score=82, status="warn"),
        ],
    )
    _write_json(
        after_dir / "skill_audit.json",
        [
            _audit_row("global.fixed", "live", audit_score=92),
            _audit_row("global.retained", "live", audit_score=100),
            _audit_row("plugin.managed.warn", "live", audit_score=82, status="warn"),
        ],
    )
    _write_json(
        inventory_path,
        [
            _inventory_row("global.fixed", managed_cache=False),
            _inventory_row("global.retained", managed_cache=False),
            _inventory_row("plugin.managed.warn", managed_cache=True),
        ],
    )

    out_dir = tmp_path / "ab"
    exit_code = main(
        [
            "all-skills-ab-compare",
            "--before-audit",
            str(before_dir / "skill_audit.json"),
            "--after-audit",
            str(after_dir / "skill_audit.json"),
            "--inventory",
            str(inventory_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    rows = {
        row["target_id"]: row
        for row in json.loads((out_dir / "all_skills_ab_decisions.json").read_text())
    }
    assert rows["global.fixed"]["ab_decision"] == "candidate"
    assert rows["global.retained"]["ab_decision"] == "baseline_retained"
    assert rows["plugin.managed.warn"]["ab_decision"] == "deferred_managed_cache"


def _audit_manifest(skill_path: str) -> dict[str, object]:
    return {
        "run_id": "audit-fixture",
        "rubric": [
            {"dimension": "Routing precision", "points": 15},
            {"dimension": "Context economy", "points": 10},
            {"dimension": "Task execution clarity", "points": 15},
            {"dimension": "Safety and locality", "points": 15},
            {"dimension": "Verification depth", "points": 15},
            {"dimension": "Clean-context robustness", "points": 10},
            {"dimension": "Failure behavior", "points": 10},
            {"dimension": "Output/report quality", "points": 10},
        ],
        "eval_cases": [
            {"case_id": "generic.activation_boundary", "applies_to": "all targets"},
        ],
        "targets": [
            {
                "target_id": "workflow.fixture",
                "target_path": skill_path,
                "skill_type": "workflow_skill",
                "versions": [
                    {
                        "version_id": "v00",
                        "sha256": _HASH_A,
                        "iteration": 0,
                        "approach_label": "baseline",
                        "attempt_status": "baseline",
                    }
                ],
            }
        ],
    }


def _audit_row(
    target_id: str,
    version_id: str,
    *,
    audit_score: int,
    safety_score: int = 25,
    status: str = "pass",
) -> dict[str, object]:
    return {
        "run_id": "run-fixture",
        "target_id": target_id,
        "version_id": version_id,
        "skill_type": "workflow_skill",
        "source_path": "/tmp/SKILL.md",
        "audit_score_0_100": audit_score,
        "format_score": 25,
        "description_score": 20,
        "structure_score": 20,
        "safety_score": safety_score,
        "compliance_status": status,
        "severe_findings": 0,
        "findings_count": 0,
        "line_count": 20,
        "description_length": 80,
        "findings": [],
    }


def _write_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\t".join(
            [
                "run_id",
                "target_id",
                "target_path",
                "skill_type",
                "iteration",
                "version_id",
                "sha256",
                "approach_label",
                "score_0_100",
                "dimension_scores_json",
                "pairwise_verdict",
                "candidate_vs_champion",
                "eval_cases_total",
                "eval_cases_passed",
                "contract_violations",
                "locality_violations",
                "secret_redaction_violations",
                "decision_reason",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "run-fixture",
                "workflow.fixture",
                "/tmp/SKILL.md",
                "workflow_skill",
                "0",
                "v00",
                _HASH_A,
                "baseline",
                "90",
                "{}",
                "not_run",
                "not_run",
                "1",
                "1",
                "0",
                "0",
                "0",
                "baseline",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "run-fixture",
                "workflow.fixture",
                "/tmp/SKILL.md",
                "workflow_skill",
                "1",
                "v01",
                _HASH_B,
                "candidate",
                "90",
                "{}",
                "tie",
                "1-1",
                "1",
                "1",
                "0",
                "0",
                "0",
                "candidate",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_skill(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
name: {name}
description: Use when testing skill discovery and audit behavior. Do not use for production tasks.
---

# {name}

Use this skill only in tests.

## Workflow

1. Read the task.
2. Keep secrets out of reports.
""",
        encoding="utf-8",
    )


def _inventory_row(target_id: str, *, managed_cache: bool) -> dict[str, object]:
    return {
        "run_id": "inventory-fixture",
        "target_id": target_id,
        "scope": "plugin_skill" if managed_cache else "global_skill",
        "source_path": "/tmp/SKILL.md",
        "managed_cache": managed_cache,
    }
