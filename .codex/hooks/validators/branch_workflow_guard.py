"""Enforce Roehub prompt-pack branch creation discipline."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from validators.common import FATAL_BLOCK, Finding, command_text, cwd_path, is_bash_pre_tool, is_codex_execpolicy_check

APPROVAL_MARKER = "ROEHUB_PROMPT_PACK_BRANCH_APPROVED=1"
PRIMARY_CHECKOUT = Path("/Users/daniildegtyarev/Projects/roehub.com")
WORKTREE_ROOT = Path("/Users/daniildegtyarev/Projects/roehub-worktrees")
GIT = r"\bgit(?:\s+-C\s+[^\s;&|]+)?"
STAGE_BRANCH = re.compile(r"(?:^|[/_.-])stage[/_.-]?\d+[a-z]?(?:$|[/_.-])", re.IGNORECASE)
GIT_C_PATH = re.compile(r"\bgit\s+-C\s+(?P<path>[^\s;&|]+)\s+", re.IGNORECASE)
WORKTREE_ADD = re.compile(rf"{GIT}\s+worktree\s+add\b[^\n;&|]*", re.IGNORECASE)

DIRECT_BRANCH_CREATE_PATTERNS = [
    re.compile(rf"{GIT}\s+switch\s+(?:--create|-c|-C)\s+(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(rf"{GIT}\s+checkout\s+(?:-b|-B)\s+(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(rf"{GIT}\s+branch\s+(?!-)(?P<branch>[^\s;&|]+)", re.IGNORECASE),
]

WORKTREE_BRANCH_CREATE_PATTERNS = [
    re.compile(rf"{GIT}\s+worktree\s+add\b[^\n;|&]*\s(?:-b|-B)\s+(?P<branch>[^\s;&|]+)", re.IGNORECASE),
]

DIRECT_BRANCH_USE_PATTERNS = [
    re.compile(rf"{GIT}\s+switch\s+(?!-)(?P<branch>[^\s;&|]+)", re.IGNORECASE),
    re.compile(rf"{GIT}\s+checkout\s+(?!-)(?P<branch>[^\s;&|]+)", re.IGNORECASE),
]

WORKTREE_BRANCH_USE_PATTERNS = [
    re.compile(rf"{GIT}\s+worktree\s+add\b(?![^\n;|&]*\s(?:-b|-B)\s)[^\n;|&]*\s(?P<branch>codex/[^\s;&|]+)", re.IGNORECASE),
]


def _branch_names(patterns: list[re.Pattern[str]], command: str) -> list[str]:
    names: list[str] = []
    for pattern in patterns:
        names.extend(match.group("branch").strip("'\"`") for match in pattern.finditer(command))
    return names


def _is_stage_branch(name: str) -> bool:
    return bool(STAGE_BRANCH.search(name))


def _inside_primary_checkout(path: Path) -> bool:
    try:
        path.resolve().relative_to(PRIMARY_CHECKOUT)
    except ValueError:
        return False
    return True


def _command_targets_primary_checkout(command: str) -> bool:
    for match in GIT_C_PATH.finditer(command):
        path = Path(match.group("path").strip("'\"`")).expanduser()
        if _inside_primary_checkout(path):
            return True
    return False


def _inside_worktree_root(path_text: str) -> bool:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        path.resolve().relative_to(WORKTREE_ROOT)
    except ValueError:
        return False
    return True


def _worktree_add_targets(command: str) -> list[tuple[str | None, str | None]]:
    targets: list[tuple[str | None, str | None]] = []
    for match in WORKTREE_ADD.finditer(command):
        try:
            parts = shlex.split(match.group(0))
        except ValueError:
            continue
        try:
            worktree_index = parts.index("worktree")
            add_index = parts.index("add", worktree_index + 1)
        except ValueError:
            continue
        args = parts[add_index + 1 :]
        branch: str | None = None
        path: str | None = None
        index = 0
        while index < len(args):
            arg = args[index]
            if arg in {"-b", "-B"}:
                if index + 1 < len(args):
                    branch = args[index + 1]
                index += 2
                continue
            if arg.startswith("-"):
                index += 1
                continue
            path = arg
            if branch is None and index + 1 < len(args) and args[index + 1].startswith("codex/"):
                branch = args[index + 1]
            break
        targets.append((branch, path))
    return targets


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_bash_pre_tool(payload):
        return []
    command = command_text(payload)
    if not command or is_codex_execpolicy_check(command):
        return []

    findings: list[Finding] = []
    direct_created = _branch_names(DIRECT_BRANCH_CREATE_PATTERNS, command)
    worktree_created = _branch_names(WORKTREE_BRANCH_CREATE_PATTERNS, command)
    created = direct_created + worktree_created
    direct_used = _branch_names(DIRECT_BRANCH_USE_PATTERNS, command)
    worktree_used = _branch_names(WORKTREE_BRANCH_USE_PATTERNS, command)
    used = direct_used + worktree_used
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
    if direct_created:
        findings.append(
            Finding(
                severity=FATAL_BLOCK,
                title="Prompt-pack branch work must use a dedicated worktree",
                message=(
                    "Do not create prompt-pack branches in the primary checkout with git switch, "
                    "git checkout, or git branch. Use `git worktree add` under "
                    f"`{WORKTREE_ROOT}` and keep the primary checkout on main."
                ),
                validator="branch_workflow_guard",
                target=", ".join(direct_created),
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
    if direct_used and (_inside_primary_checkout(cwd_path(payload)) or _command_targets_primary_checkout(command)):
        non_main = [name for name in direct_used if name not in {"main", "origin/main"}]
        if non_main:
            findings.append(
                Finding(
                    severity=FATAL_BLOCK,
                    title="Primary checkout must stay on main",
                    message=(
                        "Do not switch `/Users/daniildegtyarev/Projects/roehub.com` to a "
                        "prompt-pack branch. Use the branch's dedicated worktree under "
                        f"`{WORKTREE_ROOT}`."
                    ),
                    validator="branch_workflow_guard",
                    target=", ".join(non_main),
                )
            )
    bad_worktrees = [
        path for _, path in _worktree_add_targets(command) if path and not _inside_worktree_root(path)
    ]
    if bad_worktrees:
        findings.append(
            Finding(
                severity=FATAL_BLOCK,
                title="Branch worktree path must use the Roehub worktree root",
                message=(
                    "Create branch worktrees only under "
                    f"`{WORKTREE_ROOT}` using a branch-specific folder. "
                    "Do not create prompt-pack worktrees inside the primary checkout or ad hoc temp paths."
                ),
                validator="branch_workflow_guard",
                target=", ".join(bad_worktrees),
            )
        )
    return findings
