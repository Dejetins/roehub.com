from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.codex_quality_benchmark.manifest import load_json_file
from tools.codex_quality_benchmark.models import BenchmarkError, PairwiseRecord


def load_pairwise_records(run_dir: Path) -> dict[tuple[str, str], PairwiseRecord]:
    records: dict[tuple[str, str], PairwiseRecord] = {}
    for path in sorted((run_dir / "pairwise").glob("**/*.json")):
        record = parse_pairwise_record(load_json_file(path), path)
        key = (record.target_id, record.candidate_version_id)
        if key in records:
            raise BenchmarkError(f"duplicate pairwise record for {key[0]} {key[1]}")
        records[key] = record
    return records


def parse_pairwise_record(data: Any, path: Path | None = None) -> PairwiseRecord:
    context = str(path) if path is not None else "pairwise record"
    if not isinstance(data, dict):
        raise BenchmarkError(f"{context} must be an object")
    record = PairwiseRecord(
        run_id=_required_str(data, "run_id", context),
        target_id=_required_str(data, "target_id", context),
        iteration=_required_int(data, "iteration", context),
        candidate_version_id=_required_str(data, "candidate_version_id", context),
        champion_version_id=_required_str(data, "champion_version_id", context),
        candidate_wins_order_a=_required_bool(data, "candidate_wins_order_a", context),
        candidate_wins_order_b=_required_bool(data, "candidate_wins_order_b", context),
        decision_reason=str(data.get("decision_reason", "")),
        declared_pairwise_verdict=_optional_str(data, "pairwise_verdict"),
        declared_candidate_vs_champion=_optional_str(data, "candidate_vs_champion"),
        keep_candidate=_optional_bool(data, "keep_candidate", context),
    )
    decide_pairwise(record, severe_violation=False)
    return record


def decide_pairwise(record: PairwiseRecord, *, severe_violation: bool) -> tuple[str, str]:
    if severe_violation:
        verdict = "blocked"
        candidate_vs_champion = "not_run"
    elif record.candidate_wins_order_a and record.candidate_wins_order_b:
        verdict = "candidate"
        candidate_vs_champion = "2-0"
    elif record.candidate_wins_order_a or record.candidate_wins_order_b:
        verdict = "champion"
        candidate_vs_champion = "1-1"
    else:
        verdict = "champion"
        candidate_vs_champion = "0-2"

    if record.keep_candidate and candidate_vs_champion != "2-0":
        raise BenchmarkError(
            "candidate keep is invalid without strict 2-0 pairwise win "
            f"for {record.target_id} {record.candidate_version_id}"
        )
    if record.declared_pairwise_verdict and record.declared_pairwise_verdict != verdict:
        raise BenchmarkError(
            "declared pairwise_verdict does not match computed verdict "
            f"for {record.target_id} {record.candidate_version_id}"
        )
    if (
        record.declared_candidate_vs_champion
        and record.declared_candidate_vs_champion != candidate_vs_champion
    ):
        raise BenchmarkError(
            "declared candidate_vs_champion does not match computed result "
            f"for {record.target_id} {record.candidate_version_id}"
        )
    return verdict, candidate_vs_champion


def _required_str(data: dict[str, Any], field: str, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{context}.{field} must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{field} must be a non-empty string when present")
    return value


def _required_int(data: dict[str, Any], field: str, context: str) -> int:
    value = data.get(field)
    if not isinstance(value, int):
        raise BenchmarkError(f"{context}.{field} must be an integer")
    return value


def _required_bool(data: dict[str, Any], field: str, context: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise BenchmarkError(f"{context}.{field} must be a boolean")
    return value


def _optional_bool(data: dict[str, Any], field: str, context: str) -> bool | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise BenchmarkError(f"{context}.{field} must be a boolean when present")
    return value
