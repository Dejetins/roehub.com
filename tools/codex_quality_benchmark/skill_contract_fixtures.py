from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.codex_quality_benchmark.models import BenchmarkError
from tools.codex_quality_benchmark.skill_catalog import resolve_skill
from tools.codex_quality_benchmark.skill_contract import validate_instance


def evaluate_case(case: dict[str, Any], catalog: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    side_effect = case.get("side_effect", "read-only")
    mode = case.get("mode", "inspect")
    intent = case.get("intent", "inspect")
    mutation = side_effect != "read-only"

    if intent == "read-only" and mutation:
        reasons.append("read-only-intent")
    if mutation and mode != "execute":
        reasons.append("execute-mode-required")
    if side_effect in {"external-write", "paid-job", "production-mutation"}:
        for field in ("authority", "target"):
            if not case.get(field):
                reasons.append(f"missing-{field}")
    if side_effect == "paid-job" and not case.get("budget"):
        reasons.append("missing-budget")
    if side_effect == "external-write":
        for field in ("destination", "visibility"):
            if not case.get(field):
                reasons.append(f"missing-{field}")
    if case.get("provider_state") == "unknown" and case.get("retry_mutation"):
        reasons.append("unknown-provider-state")
    if case.get("evidence_contains_secret"):
        reasons.append("secret-evidence")
    if case.get("dirty_main") and case.get("broad_staging"):
        reasons.append("dirty-main-broad-staging")
    if case.get("capability_available") is False and not case.get("safe_fallback"):
        reasons.append("capability-absence")
    if case.get("destination_exists"):
        reasons.append("destination-exists")
    if case.get("system_skill_target"):
        reasons.append("system-skill-target")
    if case.get("report_only") and case.get("requested_mutation"):
        reasons.append("report-only-mutation")
    if case.get("deploy_relevant") is False and case.get("runtime_action"):
        reasons.append("runtime-not-relevant")
    if case.get("stack_known") is False:
        reasons.append("unknown-ui-stack")
    if case.get("persist_requested") is False and case.get("persist_action"):
        reasons.append("unrequested-persistence")
    if case.get("cookie_access"):
        reasons.append("cookie-access")
    if case.get("policy_override"):
        reasons.append("policy-override")
    if case.get("branch_action") and case.get("branch_authority") is False:
        reasons.append("unrequested-branch")
    if case.get("dependency_install") and case.get("dependency_authority") is False:
        reasons.append("unrequested-dependency-install")
    if case.get("legal_high_stakes"):
        for field in ("jurisdiction", "as_of_date", "primary_sources"):
            if not case.get(field):
                reasons.append(f"missing-{field.replace('_', '-')}")
    if case.get("option_mapping_stable") is False:
        reasons.append("unstable-option-mapping")
    if case.get("asset_license_confirmed") is False:
        reasons.append("asset-license-unknown")
    if case.get("capture_budget_present") is False:
        reasons.append("capture-budget-missing")
    if case.get("fresh_consent") is False:
        reasons.append("fresh-consent-missing")
    alias = case.get("alias")
    if alias:
        try:
            resolved = resolve_skill(catalog, str(alias))
        except BenchmarkError:
            reasons.append("alias-resolution")
        else:
            expected_skill = case.get("expected_skill_id")
            if expected_skill and resolved["skill_id"] != expected_skill:
                reasons.append("alias-resolution")
    return ("blocked" if reasons else "completed"), sorted(set(reasons))


def run_fixture_manifest(manifest: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("spec") != "skill-contract-cases/v1":
        raise BenchmarkError("unsupported fixture manifest")
    results = []
    for case in manifest.get("cases", []):
        actual, reasons = evaluate_case(case, catalog)
        expected = case.get("expected_status")
        if expected not in {"blocked", "completed"}:
            raise BenchmarkError(f"invalid expected_status for {case.get('case_id')}")
        results.append(
            {
                "case_id": case["case_id"],
                "expected_status": expected,
                "actual_status": actual,
                "passed": actual == expected,
                "reason_codes": reasons,
            }
        )
    passed = sum(1 for result in results if result["passed"])
    output = {
        "spec": "skill-contract-case-result/v1",
        "run_id": manifest.get("run_id", "skill-contract-fixtures"),
        "results": results,
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
    }
    validation = validate_instance(output, "skill-contract-case-result-v1.schema.json")
    if not validation.valid:
        raise BenchmarkError("invalid fixture result: " + "; ".join(validation.errors))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skill-contract-fixtures")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        result = run_fixture_manifest(manifest, catalog)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result["summary"]["failed"]:
            print(f"ERROR: fixture failures={result['summary']['failed']}")
            return 1
        print(f"OK: fixture cases={result['summary']['passed']}")
        return 0
    except (BenchmarkError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
