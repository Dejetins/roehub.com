"""Require cold-head review traces before finalizing architecture/prompt artifacts."""

from __future__ import annotations

import re
from typing import Any

from validators.common import CONTINUE_BEFORE_FINAL, Finding, assistant_text, has_any, is_stop

ARTIFACT_TERMS = [
    "architecture",
    "архитектур",
    "adr",
    "design note",
    "rollout plan",
    "migration plan",
    "implementation plan",
    "prompt pack",
    "executor prompt",
    ".codex/agents/generated",
]

ARTIFACT_PATH_PATTERNS = [
    re.compile(r"(?:^|\s)`?\.codex/agents/generated/[^`\s]*", re.IGNORECASE),
    re.compile(r"(?:^|\s)`?docs/architecture/[^`\s]*", re.IGNORECASE),
    re.compile(r"(?:^|\s)`?\.codex/AGENTS\.md`?", re.IGNORECASE),
    re.compile(r"(?:^|\s)`?[^`\s]*/?SKILL\.md`?", re.IGNORECASE),
]

ARTIFACT_WORK_TERMS = [
    "artifact",
    "артефакт",
    "document",
    "документ",
    "plan",
    "план",
    "prompt",
    "промт",
    "pack",
    "пак",
    "skill",
    "инструкц",
]

DISCUSSION_ONLY_TERMS = [
    "обсудили",
    "обсужда",
    "question",
    "вопрос",
    "why",
    "почему",
    "not claiming",
    "не заявляю",
]

READY_TERMS = [
    "готов",
    "ready",
    "final",
    "заверш",
    "подготов",
    "доработ",
    "created",
    "updated",
]

RECEIPT_PATTERNS = [
    re.compile(r"(?im)^Cold-head review:\s*completed\s*$"),
    re.compile(r"(?im)^Mode:\s*(independent subagent|cold self-review fallback)\s*$"),
    re.compile(r"(?im)^Review scope:\s*\S.*$"),
    re.compile(
        r"(?im)^Review instructions:\s*architecture-review/references/cold-head-plan-prompt-pack-review\.md\s*$"
    ),
    re.compile(r"(?im)^Verdict:\s*(Release|Release after fixes|Block)\s*$"),
    re.compile(r"(?im)^Blockers fixed:\s*\S.*$"),
    re.compile(r"(?im)^Local follow-up check:\s*(completed|not needed|blocked)\s*$"),
    re.compile(r"(?im)^Residual risks:\s*\S.*$"),
]


def _has_structured_receipt(text: str) -> bool:
    return all(pattern.search(text) for pattern in RECEIPT_PATTERNS)


def _has_artifact_path(text: str) -> bool:
    return any(pattern.search(text) for pattern in ARTIFACT_PATH_PATTERNS)


def _looks_like_artifact_completion(text: str) -> bool:
    if not has_any(text, READY_TERMS):
        return False
    if _has_artifact_path(text):
        return True
    if has_any(text, DISCUSSION_ONLY_TERMS):
        return False
    return has_any(text, ARTIFACT_TERMS) and has_any(text, ARTIFACT_WORK_TERMS)


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_stop(payload):
        return []
    text = assistant_text(payload)
    if not text:
        return []
    if _looks_like_artifact_completion(text) and not _has_structured_receipt(text):
        message = (
            "Architecture and prompt-manager artifacts must pass one cold-head "
            "review gate before being reported as ready. Add the structured "
            "receipt block: Cold-head review, Mode, Review scope, Review "
            "instructions, Verdict, Blockers fixed, Local follow-up check, "
            "and Residual risks."
        )
        return [
            Finding(
                severity=CONTINUE_BEFORE_FINAL,
                title="Cold-head artifact gate not reported",
                message=message,
                validator="cold_head_gate",
                target="last_assistant_message",
            )
        ]
    return []
