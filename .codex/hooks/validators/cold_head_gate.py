"""Require cold-head review traces before finalizing architecture/prompt artifacts."""

from __future__ import annotations

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

COLD_HEAD_TERMS = [
    "cold-head",
    "cold head",
    "cold self-review",
    "cold self review",
    "холод",
    "read-only reviewer",
]


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_stop(payload):
        return []
    text = assistant_text(payload)
    if not text:
        return []
    if (
        has_any(text, ARTIFACT_TERMS)
        and has_any(text, READY_TERMS)
        and not has_any(text, COLD_HEAD_TERMS)
    ):
        message = (
            "Architecture and prompt-manager artifacts must pass one cold-head "
            "review gate before being reported as ready. Run it or state the "
            "cold self-review fallback result."
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
