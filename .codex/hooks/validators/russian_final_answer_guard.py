"""Require Russian user-facing final answers for Roehub work."""

from __future__ import annotations

import re
from typing import Any

from validators.common import (
    CONTINUE_BEFORE_FINAL,
    Finding,
    assistant_text,
    count_cyrillic,
    is_stop,
)

MEMORY_CITATION_RE = re.compile(r"<oai-mem-citation>.*?</oai-mem-citation>", re.DOTALL)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
URL_RE = re.compile(r"https?://\S+")
ABSOLUTE_PATH_RE = re.compile(
    r"(?<!\w)/(?:Users|opt|var|tmp|private|Volumes)/\S+"
)
RELATIVE_PATH_RE = re.compile(
    r"(?<!\w)(?:\.?[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
)
HOOK_MARKER_RE = re.compile(r"ROEHUB_HOOK_REASON:[0-9a-f]+")
LATIN_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9'/-]*\b")

ALLOWED_TECHNICAL_WORDS = {
    "adr",
    "api",
    "app",
    "apps",
    "after",
    "bash",
    "ci",
    "cli",
    "cold",
    "cpu",
    "csv",
    "depth",
    "docker",
    "env",
    "fallback",
    "fixes",
    "gpu",
    "git",
    "hf",
    "hook",
    "html",
    "http",
    "https",
    "independent",
    "json",
    "jwt",
    "keycloak",
    "ledger",
    "markdown",
    "mac",
    "mcp",
    "manifest",
    "mps",
    "openapi",
    "pdf",
    "pid",
    "pnl",
    "pack",
    "perf",
    "pr",
    "prompt",
    "python",
    "redis",
    "release",
    "rl",
    "runtime",
    "self-review",
    "sha",
    "sha256",
    "skill",
    "skills",
    "smoke",
    "sql",
    "ssh",
    "stage",
    "studio",
    "subagent",
    "toml",
    "ui",
    "unit",
    "url",
    "uv",
    "validation",
    "yaml",
}

USER_FACING_ENGLISH_WORDS = {
    "acceptance",
    "accounting",
    "after",
    "also",
    "argmax",
    "backtest",
    "baseline",
    "before",
    "best",
    "blocker",
    "boundary",
    "candidate",
    "committed",
    "completed",
    "costs",
    "created",
    "deployed",
    "diagnostic",
    "done",
    "evaluator",
    "evidence",
    "failed",
    "files",
    "filtered",
    "final",
    "focused",
    "future",
    "green",
    "hash",
    "implemented",
    "language",
    "ledger",
    "main",
    "manifest",
    "metrics",
    "native",
    "passed",
    "proof",
    "ratio",
    "report",
    "requires",
    "runtime",
    "sanity",
    "session",
    "simulator",
    "smoke",
    "snapshot",
    "staged",
    "surface",
    "test",
    "tests",
    "tightened",
    "unit",
    "updated",
    "verification",
    "were",
    "with",
    "wording",
}

ENGLISH_PROSE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*(verification|key evidence|main blocker|metrics acceptance surface)\s*:",
        r"\bno files were (staged|committed)\b",
        r"\bi also\b",
        r"\bwas explicitly\b",
        r"\bfuture production proof requires\b",
        r"\bpassed\b",
        r"\bfailed\b",
        r"\bafter final report wording\b",
    ]
]


def _without_non_user_text(text: str) -> str:
    text = MEMORY_CITATION_RE.sub("", text)
    text = FENCED_CODE_RE.sub("", text)
    text = HOOK_MARKER_RE.sub("", text)
    return text


def _visible_line(line: str) -> str:
    line = MARKDOWN_LINK_RE.sub(r"\1", line)
    line = INLINE_CODE_RE.sub(" ", line)
    line = URL_RE.sub(" ", line)
    line = ABSOLUTE_PATH_RE.sub(" ", line)
    line = RELATIVE_PATH_RE.sub(" ", line)
    return line.strip()


def _is_allowed_word(word: str) -> bool:
    lowered = word.lower().strip("-/")
    if not lowered:
        return True
    if lowered in ALLOWED_TECHNICAL_WORDS:
        return True
    if word.isupper() and 2 <= len(word) <= 8:
        return True
    return False


def _problem_words(line: str) -> list[str]:
    words = LATIN_WORD_RE.findall(line)
    return [word for word in words if not _is_allowed_word(word)]


def _line_has_english_prose(line: str) -> bool:
    if not line:
        return False
    if any(pattern.search(line) for pattern in ENGLISH_PROSE_PATTERNS):
        return True
    words = _problem_words(line)
    if not words:
        return False
    lowered_words = {word.lower().strip("-/") for word in words}
    if lowered_words & USER_FACING_ENGLISH_WORDS:
        return True
    cyrillic_count = count_cyrillic(line)
    if cyrillic_count == 0 and len(words) >= 2:
        return True
    return cyrillic_count > 0 and len(words) >= 2


def _offending_lines(text: str, *, limit: int = 5) -> list[str]:
    checked = _without_non_user_text(text)
    offenders: list[str] = []
    for raw_line in checked.splitlines():
        line = _visible_line(raw_line)
        if _line_has_english_prose(line):
            offenders.append(line[:180])
            if len(offenders) >= limit:
                break
    return offenders


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_stop(payload):
        return []
    text = assistant_text(payload)
    if not text:
        return []
    offenders = _offending_lines(text)
    if not offenders:
        return []
    message = (
        "Финальный ответ содержит англоязычный пользовательский текст. "
        "Перепишите ответ на русском языке: заголовки, пояснения, статусы "
        "проверок и итоговые комментарии должны быть по-русски. Технические "
        "идентификаторы, команды, пути, хеши и значения в backticks можно "
        "оставлять без перевода."
    )
    return [
        Finding(
            severity=CONTINUE_BEFORE_FINAL,
            title="Финальный ответ не на русском",
            message=message,
            validator="russian_final_answer_guard",
            target="last_assistant_message",
        )
    ]
    "cold",
    "depth",
    "fallback",
    "fixes",
    "independent",
    "ledger",
    "prompt",
    "self-review",
    "skill",
    "skills",
    "smoke",
