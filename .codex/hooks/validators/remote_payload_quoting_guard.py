"""Guard SSH commands against brittle inline SQL/JSON payload quoting."""

from __future__ import annotations

import re
from typing import Any

from validators.common import (
    FATAL_BLOCK,
    WARN_WITH_CONTEXT,
    Finding,
    command_text,
    is_bash_pre_tool,
    is_codex_execpolicy_check,
)

SSH_MACSTUDIO = re.compile(r"\bssh\b[^\n;|&]*\bmacstudio\b", re.IGNORECASE)
CLICKHOUSE_INLINE_QUERY = re.compile(
    r"\bclickhouse(?:\s+client)?\b[^\n;|&]*\s--query(?:=|\s+)", re.IGNORECASE
)
SQL_VERB = re.compile(r"\b(?:SELECT|WITH|INSERT|ALTER|CREATE|DROP|OPTIMIZE|SYSTEM)\b", re.IGNORECASE)
INLINE_JSON_CURL = re.compile(
    r"\bcurl\b[^\n;|&]*(?:--data(?:-raw|-binary)?|--json|-d)\s+['\"]?\{",
    re.IGNORECASE,
)
HEREDOC_OR_STDIN = [
    re.compile(r"<<-?\s*['\"]?(?:SQL|JSON|EOF|PY|SH|BASH)\b", re.IGNORECASE),
    re.compile(r"\b--queries-file\s+/dev/stdin\b", re.IGNORECASE),
    re.compile(r"\bquery=@-\b", re.IGNORECASE),
    re.compile(r"\b(?:--data(?:-raw|-binary)?|--json|-d)\s+@-\b", re.IGNORECASE),
]


def _uses_heredoc_or_stdin(command: str) -> bool:
    return any(pattern.search(command) for pattern in HEREDOC_OR_STDIN)


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_bash_pre_tool(payload):
        return []
    command = command_text(payload)
    if not command or is_codex_execpolicy_check(command):
        return []
    if not SSH_MACSTUDIO.search(command) or _uses_heredoc_or_stdin(command):
        return []

    findings: list[Finding] = []
    if CLICKHOUSE_INLINE_QUERY.search(command) and SQL_VERB.search(command):
        findings.append(
            Finding(
                severity=FATAL_BLOCK,
                title="SSH SQL payload uses inline nested quoting",
                message=(
                    "For SSH + ClickHouse SQL, do not put SQL in an inline --query string. "
                    "Pass SQL via a quoted heredoc/stdin, for example "
                    "`ssh macstudio '... clickhouse client --queries-file /dev/stdin' <<'SQL'`."
                ),
                validator="remote_payload_quoting_guard",
                target="Bash.command",
            )
        )

    if INLINE_JSON_CURL.search(command):
        findings.append(
            Finding(
                severity=WARN_WITH_CONTEXT,
                title="SSH JSON payload appears inline quoted",
                message=(
                    "For SSH + JSON/curl payloads, prefer stdin or a quoted heredoc "
                    "instead of nested inline shell quoting. Do not create temp files "
                    "only to work around quoting."
                ),
                validator="remote_payload_quoting_guard",
                target="Bash.command",
            )
        )
    return findings
