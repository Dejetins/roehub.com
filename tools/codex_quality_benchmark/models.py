from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BenchmarkError(ValueError):
    """Raised when benchmark input is incomplete, unsafe, or inconsistent."""


@dataclass(frozen=True)
class RubricDimension:
    name: str
    points: int


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    applies_to: str


@dataclass(frozen=True)
class VersionRecord:
    version_id: str
    sha256: str
    iteration: int
    approach_label: str
    attempt_status: str


@dataclass(frozen=True)
class TargetRecord:
    target_id: str
    target_path: str
    skill_type: str
    versions: tuple[VersionRecord, ...]


@dataclass(frozen=True)
class Manifest:
    run_id: str
    rubric: tuple[RubricDimension, ...]
    eval_cases: tuple[EvalCase, ...]
    targets: tuple[TargetRecord, ...]

    @property
    def rubric_total(self) -> int:
        return sum(dimension.points for dimension in self.rubric)

    def rubric_by_name(self) -> dict[str, int]:
        return {dimension.name: dimension.points for dimension in self.rubric}

    def target_by_id(self) -> dict[str, TargetRecord]:
        return {target.target_id: target for target in self.targets}

    def expected_case_ids_for(self, skill_type: str) -> set[str]:
        expected: set[str] = set()
        for eval_case in self.eval_cases:
            applies_to = eval_case.applies_to
            if applies_to in {"all", "all targets", "*"} or applies_to == skill_type:
                expected.add(eval_case.case_id)
        return expected


@dataclass(frozen=True)
class EvaluationRecord:
    run_id: str
    target_id: str
    version_id: str
    case_id: str
    dimension_scores: dict[str, float]
    eval_case_passed: bool
    contract_violations: list[Any]
    locality_violations: list[Any]
    secret_redaction_violations: list[Any]
    decision_reason: str


@dataclass(frozen=True)
class PairwiseRecord:
    run_id: str
    target_id: str
    iteration: int
    candidate_version_id: str
    champion_version_id: str
    candidate_wins_order_a: bool
    candidate_wins_order_b: bool
    decision_reason: str
    declared_pairwise_verdict: str | None = None
    declared_candidate_vs_champion: str | None = None
    keep_candidate: bool | None = None


@dataclass(frozen=True)
class ResultRow:
    run_id: str
    target_id: str
    target_path: str
    skill_type: str
    iteration: int
    version_id: str
    sha256: str
    approach_label: str
    score_0_100: float
    dimension_scores_json: str
    pairwise_verdict: str
    candidate_vs_champion: str
    eval_cases_total: int
    eval_cases_passed: int
    contract_violations: int
    locality_violations: int
    secret_redaction_violations: int
    decision_reason: str

    def as_tsv_row(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "target_id": self.target_id,
            "target_path": self.target_path,
            "skill_type": self.skill_type,
            "iteration": str(self.iteration),
            "version_id": self.version_id,
            "sha256": self.sha256,
            "approach_label": self.approach_label,
            "score_0_100": _format_score(self.score_0_100),
            "dimension_scores_json": self.dimension_scores_json,
            "pairwise_verdict": self.pairwise_verdict,
            "candidate_vs_champion": self.candidate_vs_champion,
            "eval_cases_total": str(self.eval_cases_total),
            "eval_cases_passed": str(self.eval_cases_passed),
            "contract_violations": str(self.contract_violations),
            "locality_violations": str(self.locality_violations),
            "secret_redaction_violations": str(self.secret_redaction_violations),
            "decision_reason": self.decision_reason,
        }


def _format_score(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")
