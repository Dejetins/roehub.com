"""Block broad Git staging/commit commands in Roehub's shared main checkout."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from validators.common import (
    FATAL_BLOCK,
    Finding,
    command_text,
    is_bash_pre_tool,
    is_codex_execpolicy_check,
)

MESSAGE = (
    "Roehub uses a shared main checkout. Stage only owned files or hunks with "
    "explicit paths, then inspect `git diff --cached --name-status` before commit/push."
)
SCOPED_REVIEW_MARKER = "ROEHUB_SCOPED_STAGING_REVIEWED=1"


def _split_segments(command: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            current.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            index += 1
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            index += 1
            continue
        if char == "\n" or char == ";":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 2
            continue
        if char == "|":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments


def _git_command_args(parts: list[str]) -> tuple[str, list[str]] | None:
    index = 0
    while index < len(parts):
        item = parts[index]
        name = Path(item).name
        if name in {"sudo", "doas", "command"}:
            index += 1
            while index < len(parts) and parts[index].startswith("-"):
                index += 1
            continue
        if name == "env":
            index += 1
            while index < len(parts):
                env_item = parts[index]
                if env_item.startswith("-") or "=" in env_item:
                    index += 1
                    continue
                break
            continue
        if "=" in item and not item.startswith("-"):
            index += 1
            continue
        break
    if index >= len(parts) or Path(parts[index]).name != "git":
        return None
    index += 1
    while index < len(parts):
        item = parts[index]
        if item in {"-C", "-c", "--git-dir", "--work-tree"}:
            index += 2
            continue
        if item.startswith("--git-dir=") or item.startswith("--work-tree="):
            index += 1
            continue
        break
    if index >= len(parts):
        return None
    return parts[index], parts[index + 1 :]


def _shell_script_args(parts: list[str]) -> list[str]:
    if not parts or Path(parts[0]).name not in {"bash", "zsh", "sh"}:
        return []
    for index, item in enumerate(parts[1:], start=1):
        if item in {"-c", "-lc"} and index + 1 < len(parts):
            return [parts[index + 1]]
        if item.startswith("-") and "c" in item.lstrip("-") and index + 1 < len(parts):
            return [parts[index + 1]]
    return []


def _classify_git(command_name: str, args: list[str]) -> str | None:
    if command_name == "add":
        if any(arg in {"-A", "--all"} for arg in args):
            return "Broad git add option"
        if args in (["-u"], ["--update"]):
            return "Broad git update option"
        normalized = [arg for arg in args if arg != "--"]
        if any(arg in {".", ":/", "*"} for arg in normalized):
            return "Broad git add pathspec"
    if command_name == "restore" and "--staged" in args:
        normalized = [arg for arg in args if arg != "--"]
        if any(arg in {".", ":/", "*"} for arg in normalized):
            return "Broad git unstage pathspec"
    if command_name == "reset":
        normalized = [arg for arg in args if arg != "--"]
        if any(arg in {".", ":/", "*"} for arg in normalized):
            return "Broad git reset pathspec"
    if command_name == "commit":
        for arg in args:
            if arg == "--all":
                return "Commit with implicit staging"
            if arg.startswith("-") and not arg.startswith("--") and "a" in arg:
                return "Commit with implicit staging"
        if any(arg in {".", ":/", "*"} for arg in args):
            return "Commit with broad pathspec"
    return None


def _missing_review_marker(command_name: str, segment: str) -> str | None:
    if command_name not in {"commit", "push"}:
        return None
    if SCOPED_REVIEW_MARKER in segment:
        return None
    return "Commit/push without scoped staging review marker"


def _indirect_git_command(parts: list[str]) -> str | None:
    if not parts or Path(parts[0]).name != "xargs":
        return None
    joined = " ".join(parts[1:])
    if "git add" in joined or "git commit" in joined or "git push" in joined:
        return "Indirect git staging command"
    return None


def _find_broad_git_command(command: str) -> str | None:
    for segment in _split_segments(command):
        try:
            parts = shlex.split(segment)
        except ValueError:
            continue
        indirect = _indirect_git_command(parts)
        if indirect:
            return indirect
        for script in _shell_script_args(parts):
            nested = _find_broad_git_command(script)
            if nested:
                return nested
        parsed = _git_command_args(parts)
        if not parsed:
            continue
        command_name, args = parsed
        title = _classify_git(command_name, args) or _missing_review_marker(
            command_name,
            segment,
        )
        if title:
            return title
    return None


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_bash_pre_tool(payload):
        return []
    command = command_text(payload)
    if not command or is_codex_execpolicy_check(command):
        return []

    title = _find_broad_git_command(command)
    if not title:
        return []
    return [
        Finding(
            severity=FATAL_BLOCK,
            title=title,
            message=MESSAGE,
            validator="scoped_git_staging_guard",
            target="Bash.command",
        )
    ]
