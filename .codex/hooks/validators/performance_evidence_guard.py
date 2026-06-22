"""Require comparable evidence for performance claims in artifacts."""

from __future__ import annotations

import re
from typing import Any

from validators.common import (
    CONTINUE_BEFORE_FINAL,
    WARN_WITH_CONTEXT,
    Finding,
    assistant_text,
    has_any,
    is_post_tool,
    is_stop,
    read_text_if_exists,
    resolve_repo_path,
    touched_paths,
)

PERFORMANCE_TERMS = [
    "performance",
    "benchmark",
    "speedup",
    "faster",
    "slower",
    "latency",
    "throughput",
    "memory",
    "cpu",
    "overhead",
    "regression",
    "perf",
    "производитель",
    "бенчмарк",
    "ускор",
    "медлен",
    "памят",
]
CLAIM_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x|%|ms|s|sec|secs|seconds|rows/s|rps|qps|mb|gb)\b",
    re.IGNORECASE,
)
CLAIM_WORDS = [
    "speedup",
    "faster",
    "slower",
    "regression",
    "overhead",
    "ускор",
    "быстр",
    "медлен",
]

BASELINE_ANCHORS = ["baseline", "before", "current baseline", "базов", "до изменения"]
CANDIDATE_ANCHORS = [
    "candidate",
    "after",
    "current measurement",
    "candidate current",
    "после",
    "кандидат",
]
EVIDENCE_ANCHORS = [
    "benchmark_results",
    "benchmark_summary",
    "command",
    "artifact",
    "evidence",
    "script",
    "команд",
    "артефакт",
    "доказатель",
]
COMPARABILITY_ANCHORS = [
    "same",
    "comparable",
    "environment",
    "hardware",
    "mac studio",
    "sample",
    "median",
    "p95",
    "warmup",
    "одинаков",
    "сопостав",
]


def _is_reviewed_markdown(path: str) -> bool:
    return path.endswith(".md") and (
        path.startswith("docs/architecture/") or path.startswith(".codex/agents/generated/")
    )


def _looks_like_performance_claim(text: str) -> bool:
    return has_any(text, PERFORMANCE_TERMS) and (
        bool(CLAIM_PATTERN.search(text)) or has_any(text, CLAIM_WORDS)
    )


def _missing_anchors(text: str) -> list[str]:
    missing: list[str] = []
    if not has_any(text, BASELINE_ANCHORS):
        missing.append("baseline/before measurement")
    if not has_any(text, CANDIDATE_ANCHORS):
        missing.append("candidate/current measurement")
    if not has_any(text, EVIDENCE_ANCHORS):
        missing.append("benchmark command or evidence artifact")
    if not has_any(text, COMPARABILITY_ANCHORS):
        missing.append("environment/comparability note")
    return missing


def validate(payload: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if is_post_tool(payload):
        for path_text in touched_paths(payload):
            if not _is_reviewed_markdown(path_text):
                continue
            text = read_text_if_exists(resolve_repo_path(payload, path_text))
            if not text or not _looks_like_performance_claim(text):
                continue
            missing = _missing_anchors(text)
            if missing:
                findings.append(
                    Finding(
                        severity=CONTINUE_BEFORE_FINAL,
                        title="Performance claim lacks comparable evidence",
                        message=(
                            "Performance/benchmark claims must include comparable evidence. "
                            "Add or mark N/A: " + ", ".join(missing) + "."
                        ),
                        validator="performance_evidence_guard",
                        target=path_text,
                    )
                )
    if is_stop(payload):
        text = assistant_text(payload)
        if _looks_like_performance_claim(text):
            missing = _missing_anchors(text)
            if missing:
                findings.append(
                    Finding(
                        severity=WARN_WITH_CONTEXT,
                        title="Final performance claim lacks comparable evidence",
                        message=(
                            "If reporting performance impact, include baseline, candidate, "
                            "benchmark command/artifact, and comparability context."
                        ),
                        validator="performance_evidence_guard",
                        target="last_assistant_message",
                    )
                )
    return findings
