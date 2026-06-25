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
    "report",
    "отчет",
    "ledger",
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
    "выполн",
    "подготов",
    "доработ",
    "измен",
    "добавлен",
    "blocked",
    "заблок",
    "created",
    "updated",
]

RECEIPT_PATTERNS = [
    re.compile(r"(?im)^Cold-head review:\s*completed\s*$"),
    re.compile(r"(?im)^Mode:\s*(independent subagent|cold self-review fallback)\s*$"),
    re.compile(r"(?im)^Review scope:\s*\S.*$"),
    re.compile(
        r"(?im)^Review instructions:\s*`?architecture-review/references/"
        r"cold-head-plan-prompt-pack-review\.md`?\s*$"
    ),
    re.compile(r"(?im)^Verdict:\s*(Release|Release after fixes|Block)\s*$"),
    re.compile(r"(?im)^Blockers fixed:\s*\S.*$"),
    re.compile(r"(?im)^Local follow-up check:\s*(completed|not needed|blocked)\s*$"),
    re.compile(r"(?im)^Residual risks:\s*\S.*$"),
]

HUMAN_RECEIPT_PATTERNS = [
    re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?Проверка перед финалом(?:\*\*)?:?\s*$"),
    re.compile(
        r"(?im)^\s*-\s*Статус проверки:\s*"
        r"(выполнена|проведена|completed|заблокирована|blocked)\b.*$"
    ),
    re.compile(
        r"(?im)^\s*-\s*Режим:\s*"
        r"(independent subagent|cold self-review fallback|независимая проверка|"
        r"холодная самопроверка|самопроверка)\b.*$"
    ),
    re.compile(r"(?im)^\s*-\s*Что проверено:\s*\S.*$"),
    re.compile(
        r"(?im)^\s*-\s*Итог:\s*"
        r"(Release|Release after fixes|Block|можно продолжать|нужны исправления|"
        r"заблокировано|заблокирован)\b.*$"
    ),
    re.compile(
        r"(?im)^\s*-\s*(Что исправлено/добавлено|Что исправлено|Что добавлено|"
        r"Исправлено/добавлено):\s*\S.*$"
    ),
    re.compile(r"(?im)^\s*-\s*Остаточные риски:\s*\S.*$"),
    re.compile(
        r"(?im)^\s*-\s*(Что это значит для следующего шага|Следующий шаг):\s*\S.*$"
    ),
]


def _has_structured_receipt(text: str) -> bool:
    return all(pattern.search(text) for pattern in RECEIPT_PATTERNS) or all(
        pattern.search(text) for pattern in HUMAN_RECEIPT_PATTERNS
    )


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
            "Перед финальным ответом по architecture/prompt artifacts нужен "
            "понятный cold-head блок. Верните исходный ответ полностью и ниже "
            "добавьте раздел `Проверка перед финалом` со статусом, режимом, "
            "объемом проверки, итогом, исправлениями, остаточными рисками и "
            "смыслом для следующего шага."
        )
        return [
            Finding(
                severity=CONTINUE_BEFORE_FINAL,
                title="Cold-head финальный блок не найден",
                message=message,
                validator="cold_head_gate",
                target="last_assistant_message",
            )
        ]
    return []
