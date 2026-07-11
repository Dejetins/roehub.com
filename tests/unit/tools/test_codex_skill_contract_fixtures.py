from __future__ import annotations

from tools.codex_quality_benchmark.skill_contract_fixtures import (
    evaluate_case,
    run_fixture_manifest,
)


def _catalog() -> dict[str, object]:
    return {
        "skills": [
            {
                "skill_id": "S056",
                "logical_name": "Presentations",
                "canonical_name": "presentations",
                "aliases": ["Presentations"],
                "implementation_channel": "overlay",
            }
        ]
    }


def test_paid_job_without_budget_blocks() -> None:
    status, reasons = evaluate_case(
        {
            "side_effect": "paid-job",
            "mode": "execute",
            "intent": "execute",
            "authority": True,
            "target": "job-a",
        },
        _catalog(),
    )
    assert status == "blocked"
    assert reasons == ["missing-budget"]


def test_alias_resolution_completes() -> None:
    status, reasons = evaluate_case(
        {
            "side_effect": "read-only",
            "mode": "inspect",
            "intent": "inspect",
            "alias": "Presentations",
            "expected_skill_id": "S056",
        },
        _catalog(),
    )
    assert status == "completed"
    assert reasons == []


def test_fixture_manifest_reports_expected_results() -> None:
    result = run_fixture_manifest(
        {
            "spec": "skill-contract-cases/v1",
            "run_id": "fixture",
            "cases": [
                {
                    "case_id": "read-only.safe",
                    "side_effect": "read-only",
                    "mode": "inspect",
                    "intent": "read-only",
                    "expected_status": "completed",
                },
                {
                    "case_id": "read-only.mutation",
                    "side_effect": "local-write",
                    "mode": "execute",
                    "intent": "read-only",
                    "expected_status": "blocked",
                },
            ],
        },
        _catalog(),
    )
    assert result["summary"] == {"total": 2, "passed": 2, "failed": 0}
