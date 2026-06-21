"""Enforce Roehub prompt-pack branch creation discipline."""

from __future__ import annotations

import re
from typing import Any

from validators.common import FATAL_BLOCK, Finding, command_text, is_bash_pre_tool, is_codex_execpolicy_check

APPROVAL_MARKER = "ROEHUB_PROMPT_PACK_BRANCH_APPROVED=1"
STAGE_BRANCH = re.compile(r"(?:^|[/_.-])stage[/_.-]?\d+[a-z]?(?:$|[/_.-])", re.IGNORECASE)

BRANCH_CREATE_PATTERNS = [
    re.compile(r"\bgit\s+switch\s+(?:--create|-c|-C)\s+(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\s+(?:-b|-B)\s+(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(r"\bgit\s+branch\s+(?!-)(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(r"\bgit\s+worktree\s+add\b[^\n;|&]*\s(?:-b|-B)\s+(?P<branch>[^\s;&|]+)", re.IGNORECASE),
]

BRANCH_USE_PATTERNS = [
    re.compile(r"\bgit\s+switch\s+(?!-)(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\s+(?!-)(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(r"\bgit\s+worktree\s+add\b(?![^\n;|&]*\s(?:-b|-B)\s)[^\n;|&]*\s(?P<branch>codex/[^\s;&|]+)", re.IGNORECASE),
]


def _branch_names(patterns: list[re.Pattern[str]], command: str) -> list[str]:
    names: list[str] = []
    for pattern in patterns:
        names.extend(match.group("branch").strip("'\"`") for match in pattern.finditer(command))
    return names


def _is_stage_branch(name: str) -> bool:
    return bool(STAGE_BRANCH.search(name))


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_bash_pre_tool(payload):
        return []
    command = command_text(payload)
    if not command or is_codex_execpolicy_check(command):
        return []

    findings: list[Finding] = []
    created = _branch_names(BRANCH_CREATE_PATTERNS, command)
    used = _branch_names(BRANCH_USE_PATTERNS, command)
    stage_names = [name for name in created + used if _is_stage_branch(name)]
    if stage_names:
        findings.append(
            Finding(
                severity=FATAL_BLOCK,
                title="Stage-specific prompt-pack branch is forbidden",
                message=(
                    "Do not create or switch to per-stage branches for prompt-pack work. "
                    "Use main by default, or one user-approved branch for the whole prompt pack."
                ),
                validator="branch_workflow_guard",
                target=", ".join(stage_names),
            )
        )
    if created and APPROVAL_MARKER not in command:
        findings.append(
            Finding(
                severity=FATAL_BLOCK,
                title="Branch creation requires explicit prompt-pack approval marker",
                message=(
                    "Roehub prompt-pack work defaults to main. Create a branch only when the "
                    "user explicitly requested one branch for the whole pack, and include "
                    f"`{APPROVAL_MARKER}` in the command."
                ),
                validator="branch_workflow_guard",
                target=", ".join(created),
            )
        )
    return findings
