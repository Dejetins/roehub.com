"""Require the pinned Roehub Playwright CLI wrapper/version."""

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

PINNED = "@playwright/cli@0.1.14"


FLOATING_PATTERNS = [
    re.compile(r"\bnpx\b[^\n;|&]*\bplaywright\b"),
    re.compile(r"\bnpx\b[^\n;|&]*(?:--package|-p)\s+@playwright/cli(?:\s|$|[;|&])"),
    re.compile(r"\bnpx\b[^\n;|&]*(?:--package=|-p=)@playwright/cli(?:\s|$|[;|&])"),
    re.compile(r"\bnpx\b[^\n;|&]*@playwright/cli@latest\b"),
    re.compile(r"\bnpm\s+exec\b[^\n;|&]*\bplaywright\b"),
    re.compile(r"\bpnpm\s+(?:exec|dlx)\b[^\n;|&]*\bplaywright\b"),
    re.compile(r"\byarn\s+(?:exec|dlx)?\b[^\n;|&]*\bplaywright\b"),
    re.compile(r"\bbunx\b[^\n;|&]*\bplaywright\b"),
]


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_bash_pre_tool(payload):
        return []
    command = command_text(payload)
    if not command or PINNED in command or is_codex_execpolicy_check(command):
        return []
    if not any(pattern.search(command) for pattern in FLOATING_PATTERNS):
        return []
    return [
        Finding(
            severity=FATAL_BLOCK,
            title="Floating Playwright CLI invocation",
            message=(
                f"Use the Roehub/global Playwright wrapper pinned to {PINNED}. "
                "Do not use floating npx/npm exec Playwright commands unless "
                "intentionally refreshing the matching browser revision."
            ),
            validator="playwright_wrapper_guard",
            target="Bash.command",
        )
    ]
