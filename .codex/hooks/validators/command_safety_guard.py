"""Block deterministic destructive command patterns before execution."""

from __future__ import annotations

import re
from typing import Any

from validators.common import (
    FATAL_BLOCK,
    Finding,
    command_text,
    is_bash_pre_tool,
    is_codex_execpolicy_check,
)

PATTERNS = [
    (
        "Recursive root removal",
        re.compile(r"\brm\s+-[^\n;|&]*r[^\n;|&]*f[^\n;|&]*\s+/(?:\s|$|[;|&*])"),
    ),
    ("Git hard reset", re.compile(r"\bgit\s+reset\s+--hard\b")),
    ("Git clean destructive", re.compile(r"\bgit\s+clean\s+-[^\n;|&]*[xfd][^\n;|&]*\b")),
    ("Force push", re.compile(r"\bgit\s+push\b[^\n;|&]*(?:--force|-f)\b")),
]


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_bash_pre_tool(payload):
        return []
    command = command_text(payload)
    if not command or is_codex_execpolicy_check(command):
        return []
    findings: list[Finding] = []
    message = (
        "This destructive command is blocked by Roehub hook policy. Ask the user "
        "explicitly and use a narrow recovery command instead."
    )
    for title, pattern in PATTERNS:
        if pattern.search(command):
            findings.append(
                Finding(
                    severity=FATAL_BLOCK,
                    title=title,
                    message=message,
                    validator="command_safety_guard",
                    target="Bash.command",
                )
            )
    return findings
