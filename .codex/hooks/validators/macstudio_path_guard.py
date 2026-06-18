"""Protect the Mac Studio checkout/runtime path contract."""

from __future__ import annotations

import re
from typing import Any

from validators.common import (
    FATAL_BLOCK,
    REPO_CHECKOUT,
    RUNTIME_TREE,
    Finding,
    command_text,
    is_bash_pre_tool,
    is_codex_execpolicy_check,
)

GIT_IN_RUNTIME_PATTERNS = [
    re.compile(rf"\bgit\s+-C\s+['\"]?{re.escape(RUNTIME_TREE)}(?:['\"]|\b)"),
    re.compile(rf"\bcd\s+['\"]?{re.escape(RUNTIME_TREE)}(?:['\"]|\b)[^\n]*(?:&&|;|\n)\s*git\b"),
    re.compile(rf"\bGIT_DIR=['\"]?{re.escape(RUNTIME_TREE)}/\.git(?:['\"]|\b)"),
]


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_bash_pre_tool(payload):
        return []
    command = command_text(payload)
    if not command or is_codex_execpolicy_check(command):
        return []
    if not any(pattern.search(command) for pattern in GIT_IN_RUNTIME_PATTERNS):
        return []
    return [
        Finding(
            severity=FATAL_BLOCK,
            title="Mac Studio runtime path used as git checkout",
            message=(
                f"{RUNTIME_TREE} is runtime rsync state, not the authoritative git checkout. "
                f"Run git commands under {REPO_CHECKOUT}; deploy to {RUNTIME_TREE} "
                "only through the deploy/rsync workflow."
            ),
            validator="macstudio_path_guard",
            target="Bash.command",
        )
    ]
