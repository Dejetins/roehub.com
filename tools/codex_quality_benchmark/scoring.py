from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.codex_quality_benchmark.manifest import load_json_file, load_manifest
from tools.codex_quality_benchmark.models import (
    BenchmarkError,
    EvaluationRecord,
    Manifest,
    ResultRow,
    TargetRecord,
    VersionRecord,
)
from tools.codex_quality_benchmark.pairwise import decide_pairwise, load_pairwise_records

_NO_PAIRWISE_STATUSES = {"baseline", "no_op", "blocked"}


def aggregate_run(run_dir: Path) -> tuple[Manifest, list[ResultRow]]:
    manifest = load_manifest(run_dir / "manifest.json")
    evaluations = load_evaluation_records(run_dir)
    pairwise = load_pairwise_records(run_dir)
    rows = aggregate_records(manifest, evaluations, pairwise)
    return manifest, rows


def load_evaluation_records(run_dir: Path) -> list[EvaluationRecord]:
    evaluation_paths = sorted((run_dir / "evaluations").glob("**/*.json"))
    if not evaluation_paths:
        raise BenchmarkError(f"no evaluator JSON files found under {run_dir / 'evaluations'}")
    return [parse_evaluation_record(load_json_file(path), path) for path in evaluation_paths]


def parse_evaluation_record(data: Any, path: Path | None = None) -> EvaluationRecord:
    context = str(path) if path is not None else "evaluation record"
    if not isinstance(data, dict):
        raise BenchmarkError(f"{context} must be an object")

    dimension_scores = data.get("dimension_scores_json")
    if not isinstance(dimension_scores, dict) or not dimension_scores:
        raise BenchmarkError(f"{context}.dimension_scores_json must be a non-empty object")

    return EvaluationRecord(
        run_id=_required_str(data, "run_id", context),
        target_id=_required_str(data, "target_id", context),
        version_id=_required_str(data, "version_id", context),
        case_id=_required_str(data, "case_id", context),
        dimension_scores={
            str(key): _number(value, context, str(key))
            for key, value in dimension_scores.items()
        },
        eval_case_passed=_required_bool(data, "eval_case_passed", context),
        contract_violations=_optional_list(data, "contract_violations", context),
        locality_violations=_optional_list(data, "locality_violations", context),
        secret_redaction_violations=_optional_list(data, "secret_redaction_violations", context),
        decision_reason=str(data.get("decision_reason", "")),
    )


def aggregate_records(
    manifest: Manifest,
    evaluations: list[EvaluationRecord],
    pairwise_records: dict[tuple[str, str], Any],
) -> list[ResultRow]:
    by_target = manifest.target_by_id()
    grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
    for evaluation in evaluations:
        if evaluation.run_id != manifest.run_id:
            raise BenchmarkError(f"evaluation run_id mismatch for {evaluation.target_id}")
        target = by_target.get(evaluation.target_id)
        if target is None:
            raise BenchmarkError(f"evaluation references unknown target {evaluation.target_id}")
        if evaluation.version_id not in {version.version_id for version in target.versions}:
            raise BenchmarkError(
                "evaluation references unknown version "
                f"{evaluation.target_id} {evaluation.version_id}"
            )
        _validate_dimension_scores(evaluation, manifest)
        grouped[(evaluation.target_id, evaluation.version_id)].append(evaluation)

    rows: list[ResultRow] = []
    for target in manifest.targets:
        for version in target.versions:
            version_evaluations = grouped.get((target.target_id, version.version_id), [])
            if not version_evaluations:
                raise BenchmarkError(
                    f"missing evaluator JSON for {target.target_id} {version.version_id}"
                )
            _validate_expected_cases(manifest, target, version, version_evaluations)
            rows.append(
                _build_result_row(
                    manifest,
                    target,
                    version,
                    version_evaluations,
                    pairwise_records,
                )
            )
    return rows


def _validate_dimension_scores(evaluation: EvaluationRecord, manifest: Manifest) -> None:
    rubric = manifest.rubric_by_name()
    score_names = set(evaluation.dimension_scores)
    rubric_names = set(rubric)
    if score_names != rubric_names:
        missing = sorted(rubric_names - score_names)
        extra = sorted(score_names - rubric_names)
        raise BenchmarkError(
            f"dimension scores for {evaluation.target_id} {evaluation.version_id} "
            f"must match rubric; missing={missing}, extra={extra}"
        )
    for name, value in evaluation.dimension_scores.items():
        if value < 0 or value > rubric[name]:
            raise BenchmarkError(
                f"dimension score {name} for {evaluation.target_id} {evaluation.version_id} "
                f"must be between 0 and {rubric[name]}"
            )


def _validate_expected_cases(
    manifest: Manifest,
    target: TargetRecord,
    version: VersionRecord,
    evaluations: list[EvaluationRecord],
) -> None:
    expected = manifest.expected_case_ids_for(target.skill_type)
    observed = {evaluation.case_id for evaluation in evaluations}
    if observed != expected:
        raise BenchmarkError(
            f"case coverage for {target.target_id} {version.version_id} must match expected cases; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _build_result_row(
    manifest: Manifest,
    target: TargetRecord,
    version: VersionRecord,
    evaluations: list[EvaluationRecord],
    pairwise_records: dict[tuple[str, str], Any],
) -> ResultRow:
    severe_violation = any(_has_severe_violation(evaluation) for evaluation in evaluations)

    dimension_averages: dict[str, float] = {}
    for dimension in manifest.rubric:
        total = sum(evaluation.dimension_scores[dimension.name] for evaluation in evaluations)
        dimension_averages[dimension.name] = round(total / len(evaluations), 4)
    score = round(sum(dimension_averages.values()), 4)
    if severe_violation:
        score = min(score, 49)

    if severe_violation:
        pairwise_verdict = "blocked"
        candidate_vs_champion = "not_run"
    elif version.iteration == 0:
        pairwise_verdict = "not_run"
        candidate_vs_champion = "not_run"
    else:
        pairwise_record = pairwise_records.get((target.target_id, version.version_id))
        if pairwise_record is None:
            if version.attempt_status not in _NO_PAIRWISE_STATUSES:
                raise BenchmarkError(
                    f"missing pairwise record for {target.target_id} {version.version_id}"
                )
            pairwise_verdict = "not_run"
            candidate_vs_champion = "not_run"
        else:
            pairwise_verdict, candidate_vs_champion = decide_pairwise(
                pairwise_record, severe_violation=False
            )

    return ResultRow(
        run_id=manifest.run_id,
        target_id=target.target_id,
        target_path=target.target_path,
        skill_type=target.skill_type,
        iteration=version.iteration,
        version_id=version.version_id,
        sha256=version.sha256,
        approach_label=version.approach_label,
        score_0_100=score,
        dimension_scores_json=json.dumps(dimension_averages, ensure_ascii=False, sort_keys=True),
        pairwise_verdict=pairwise_verdict,
        candidate_vs_champion=candidate_vs_champion,
        eval_cases_total=len(evaluations),
        eval_cases_passed=sum(1 for evaluation in evaluations if evaluation.eval_case_passed),
        contract_violations=sum(len(evaluation.contract_violations) for evaluation in evaluations),
        locality_violations=sum(len(evaluation.locality_violations) for evaluation in evaluations),
        secret_redaction_violations=sum(
            len(evaluation.secret_redaction_violations) for evaluation in evaluations
        ),
        decision_reason=_decision_reason(evaluations, pairwise_verdict),
    )


def _has_severe_violation(evaluation: EvaluationRecord) -> bool:
    if evaluation.secret_redaction_violations:
        return True
    for violation in [*evaluation.locality_violations, *evaluation.contract_violations]:
        if isinstance(violation, dict) and str(violation.get("severity", "")).lower() in {
            "severe",
            "critical",
        }:
            return True
        if isinstance(violation, str) and violation.lower().startswith(("severe:", "critical:")):
            return True
    return False


def _decision_reason(evaluations: list[EvaluationRecord], pairwise_verdict: str) -> str:
    reasons = [
        evaluation.decision_reason
        for evaluation in evaluations
        if evaluation.decision_reason
    ]
    if reasons:
        return " | ".join(reasons)
    return f"aggregated evaluator scores; pairwise={pairwise_verdict}"


def _required_str(data: dict[str, Any], field: str, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{context}.{field} must be a non-empty string")
    return value


def _required_bool(data: dict[str, Any], field: str, context: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise BenchmarkError(f"{context}.{field} must be a boolean")
    return value


def _optional_list(data: dict[str, Any], field: str, context: str) -> list[Any]:
    value = data.get(field, [])
    if not isinstance(value, list):
        raise BenchmarkError(f"{context}.{field} must be a list when present")
    return value


def _number(value: Any, context: str, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BenchmarkError(f"{context}.dimension_scores_json[{field}] must be numeric")
    return float(value)
