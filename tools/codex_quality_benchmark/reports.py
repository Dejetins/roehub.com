from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.codex_quality_benchmark.models import ResultRow

RESULT_FIELDS = [
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


def write_results_tsv(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_tsv_row())


def write_events_jsonl(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            event = {"event": "aggregation_result", **row.as_tsv_row()}
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Codex Quality Benchmark Summary",
        "",
        "Local deterministic summary generated from saved evaluator JSON and pairwise records.",
        "",
        "| target_id | version_id | score_0_100 | pairwise_verdict | "
        "candidate_vs_champion | eval_cases | decision_reason |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        eval_cases = f"{row.eval_cases_passed}/{row.eval_cases_total}"
        lines.append(
            "| "
            f"{row.target_id} | {row.version_id} | {row.score_0_100:g} | "
            f"{row.pairwise_verdict} | {row.candidate_vs_champion} | {eval_cases} | "
            f"{_escape_table(row.decision_reason)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_artifacts(run_dir: Path, rows: list[ResultRow], *, include_summary: bool) -> None:
    write_results_tsv(run_dir / "results.tsv", rows)
    write_events_jsonl(run_dir / "events.jsonl", rows)
    if include_summary:
        write_summary_md(run_dir / "summary.md", rows)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
