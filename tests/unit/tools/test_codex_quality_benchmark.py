from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.codex_quality_benchmark.cli import main
from tools.codex_quality_benchmark.manifest import load_manifest
from tools.codex_quality_benchmark.models import BenchmarkError
from tools.codex_quality_benchmark.pairwise import PairwiseRecord, decide_pairwise
from tools.codex_quality_benchmark.scoring import aggregate_run

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def test_validate_manifest_blocks_missing_version_hash(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    data = _manifest()
    del data["targets"][0]["versions"][0]["sha256"]
    _write_json(manifest_path, data)

    with pytest.raises(BenchmarkError, match="sha256"):
        load_manifest(manifest_path)


def test_pairwise_requires_strict_two_zero_for_candidate_keep() -> None:
    record = PairwiseRecord(
        run_id="run-fixture",
        target_id="workflow.fixture",
        iteration=1,
        candidate_version_id="v01",
        champion_version_id="v00",
        candidate_wins_order_a=True,
        candidate_wins_order_b=False,
        decision_reason="split decision",
        keep_candidate=True,
    )

    with pytest.raises(BenchmarkError, match="2-0"):
        decide_pairwise(record, severe_violation=False)


def test_aggregate_writes_results_events_and_summary(tmp_path: Path) -> None:
    run_dir = _sample_run(tmp_path)

    exit_code = main(["validate-manifest", "--manifest", str(run_dir / "manifest.json")])
    assert exit_code == 0

    aggregate_exit = main(["aggregate", "--run-dir", str(run_dir)])
    assert aggregate_exit == 0

    summary_exit = main(["summarize", "--run-dir", str(run_dir)])
    assert summary_exit == 0

    results = (run_dir / "results.tsv").read_text(encoding="utf-8")
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")

    assert "workflow.fixture\t" in results
    assert "\tv01\t" in results
    assert "\tcandidate\t2-0\t" in results
    assert '"event": "aggregation_result"' in events
    assert "| workflow.fixture | v01 | 91 | candidate | 2-0 | 2/2 |" in summary


def test_aggregate_records_secret_violation_as_blocked_row(tmp_path: Path) -> None:
    run_dir = _sample_run(tmp_path)
    evaluation_path = run_dir / "evaluations" / "workflow.fixture" / "v01" / "generic.json"
    data = json.loads(evaluation_path.read_text(encoding="utf-8"))
    data["secret_redaction_violations"] = ["severe: token echoed"]
    _write_json(evaluation_path, data)

    _, rows = aggregate_run(run_dir)

    blocked = next(row for row in rows if row.version_id == "v01")
    assert blocked.score_0_100 == 49
    assert blocked.pairwise_verdict == "blocked"
    assert blocked.candidate_vs_champion == "not_run"
    assert blocked.secret_redaction_violations == 1


def test_aggregate_blocks_declared_candidate_without_two_zero(tmp_path: Path) -> None:
    run_dir = _sample_run(tmp_path)
    pairwise_path = run_dir / "pairwise" / "workflow.fixture" / "v01.json"
    data = json.loads(pairwise_path.read_text(encoding="utf-8"))
    data["candidate_wins_order_b"] = False
    data["pairwise_verdict"] = "candidate"
    data["candidate_vs_champion"] = "1-1"
    data["keep_candidate"] = False
    _write_json(pairwise_path, data)

    with pytest.raises(BenchmarkError, match="declared pairwise_verdict"):
        aggregate_run(run_dir)


def _sample_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run-fixture"
    _write_json(run_dir / "manifest.json", _manifest())
    for version_id, scores in {
        "v00": {"Routing precision": 12, "Context economy": 8},
        "v01": {"Routing precision": 14, "Context economy": 9},
    }.items():
        _write_json(
            run_dir / "evaluations" / "workflow.fixture" / version_id / "generic.json",
            _evaluation(version_id, "generic.activation_boundary", scores),
        )
        _write_json(
            run_dir / "evaluations" / "workflow.fixture" / version_id / "stage_gate.json",
            _evaluation(version_id, "workflow.stage_gate", scores),
        )
    _write_json(
        run_dir / "pairwise" / "workflow.fixture" / "v01.json",
        {
            "run_id": "run-fixture",
            "target_id": "workflow.fixture",
            "iteration": 1,
            "candidate_version_id": "v01",
            "champion_version_id": "v00",
            "candidate_wins_order_a": True,
            "candidate_wins_order_b": True,
            "pairwise_verdict": "candidate",
            "candidate_vs_champion": "2-0",
            "keep_candidate": True,
            "decision_reason": "candidate wins both orderings",
        },
    )
    return run_dir


def _manifest() -> dict[str, Any]:
    return {
        "run_id": "run-fixture",
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
            {"case_id": "workflow.stage_gate", "applies_to": "workflow_skill"},
        ],
        "targets": [
            {
                "target_id": "workflow.fixture",
                "target_path": "/tmp/workflow-fixture/SKILL.md",
                "skill_type": "workflow_skill",
                "versions": [
                    {
                        "version_id": "v00",
                        "sha256": _HASH_A,
                        "iteration": 0,
                        "approach_label": "baseline",
                        "attempt_status": "baseline",
                    },
                    {
                        "version_id": "v01",
                        "sha256": _HASH_B,
                        "iteration": 1,
                        "approach_label": "routing_precision",
                        "attempt_status": "candidate",
                    },
                ],
            }
        ],
    }


def _evaluation(version_id: str, case_id: str, overrides: dict[str, int]) -> dict[str, Any]:
    scores = {
        "Routing precision": 13,
        "Context economy": 8,
        "Task execution clarity": 14,
        "Safety and locality": 14,
        "Verification depth": 13,
        "Clean-context robustness": 9,
        "Failure behavior": 9,
        "Output/report quality": 9,
    }
    scores.update(overrides)
    return {
        "run_id": "run-fixture",
        "target_id": "workflow.fixture",
        "version_id": version_id,
        "case_id": case_id,
        "dimension_scores_json": scores,
        "eval_case_passed": True,
        "contract_violations": [],
        "locality_violations": [],
        "secret_redaction_violations": [],
        "decision_reason": f"{version_id} {case_id}",
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
