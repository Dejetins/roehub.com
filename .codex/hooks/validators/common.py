"""Shared primitives for Roehub Codex hook validators."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_CHECKOUT = "/Users/daniildegtyarev/Projects/roehub.com"
RUNTIME_TREE = "/opt/roehub/app"

FATAL_BLOCK = "FATAL_BLOCK"
CONTINUE_BEFORE_FINAL = "CONTINUE_BEFORE_FINAL"
WARN_WITH_CONTEXT = "WARN_WITH_CONTEXT"
OBSERVE = "OBSERVE"


@dataclass(frozen=True)
class Finding:
    severity: str
    title: str
    message: str
    validator: str
    target: str | None = None

    def line(self) -> str:
        target = f" [{self.target}]" if self.target else ""
        return f"{self.severity}: {self.title}{target} - {self.message}"


def hook_event(payload: dict[str, Any]) -> str:
    return str(payload.get("hook_event_name") or payload.get("hookEventName") or "")


def tool_name(payload: dict[str, Any]) -> str:
    return str(payload.get("tool_name") or payload.get("toolName") or "")


def tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("tool_input") or payload.get("toolInput") or {}
    return value if isinstance(value, dict) else {}


def command_text(payload: dict[str, Any]) -> str:
    data = tool_input(payload)
    value = data.get("command")
    if isinstance(value, str):
        return value
    value = data.get("cmd")
    return value if isinstance(value, str) else ""


def is_bash_tool(payload: dict[str, Any]) -> bool:
    return tool_name(payload) == "Bash"


def is_bash_pre_tool(payload: dict[str, Any]) -> bool:
    return is_pre_tool(payload) and is_bash_tool(payload)


def is_codex_execpolicy_check(command: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if len(parts) < 3:
        return False
    return Path(parts[0]).name == "codex" and parts[1:3] == [
        "execpolicy",
        "check",
    ]


def prompt_text(payload: dict[str, Any]) -> str:
    value = payload.get("prompt")
    return value if isinstance(value, str) else ""


def assistant_text(payload: dict[str, Any]) -> str:
    value = payload.get("last_assistant_message") or payload.get("lastAssistantMessage")
    return value if isinstance(value, str) else ""


def cwd_path(payload: dict[str, Any]) -> Path:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        return Path(cwd)
    return Path.cwd()


def is_pre_tool(payload: dict[str, Any]) -> bool:
    return hook_event(payload) in {"PreToolUse", "PermissionRequest"}


def is_post_tool(payload: dict[str, Any]) -> bool:
    return hook_event(payload) == "PostToolUse"


def is_stop(payload: dict[str, Any]) -> bool:
    return hook_event(payload) == "Stop"


def is_user_prompt(payload: dict[str, Any]) -> bool:
    return hook_event(payload) == "UserPromptSubmit"


def stringify(value: Any, *, limit: int = 20000) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = repr(value)
    if len(text) > limit:
        return text[:limit] + "\n...[truncated by roehub hook validator]"
    return text


def iter_text_surfaces(payload: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for key in ("prompt", "last_assistant_message", "lastAssistantMessage"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            yield key, value
    data = tool_input(payload)
    if data:
        yield "tool_input", stringify(data)
    response = payload.get("tool_response") or payload.get("toolResponse")
    if response:
        yield "tool_response", stringify(response)


def extract_patch_paths(command: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", command, re.MULTILINE):
        paths.append(match.group(1).strip())
    return paths


def touched_paths(payload: dict[str, Any]) -> list[str]:
    paths = extract_patch_paths(command_text(payload))
    data = tool_input(payload)
    for key in ("path", "file", "file_path", "target_file"):
        value = data.get(key)
        if isinstance(value, str):
            paths.append(value)
    return dedupe(paths)


def resolve_repo_path(payload: dict[str, Any], path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return cwd_path(payload) / path


def read_text_if_exists(path: Path) -> str:
    try:
        if path.is_file() and path.stat().st_size <= 1_000_000:
            return path.read_text(encoding="utf-8")
    except OSError:
        return ""
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def has_any(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def count_cyrillic(text: str) -> int:
    return sum(1 for char in text if "\u0400" <= char <= "\u04ff")


def format_findings(findings: Iterable[Finding]) -> str:
    return "\n".join(f"- {finding.line()}" for finding in findings)
