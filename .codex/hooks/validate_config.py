"""Validate the deliberately narrow repository hook registration contract."""
from __future__ import annotations

import json
from pathlib import Path

COMMAND = '/usr/bin/python3 "$(git rev-parse --show-toplevel)/.codex/hooks/roehub_hook_router.py"'
MATCHER = "^(Bash|apply_patch|Edit|Write)$"


def validate_config(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"hooks"}:
        raise ValueError("Expected the repository hooks object")
    events = data["hooks"]
    if not isinstance(events, dict) or set(events) != {"PreToolUse", "PostToolUse"}:
        raise ValueError("Only pre/post-tool guards are registered; no Stop inference")
    for event, groups in events.items():
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError(f"{event}: expected one guard registration")
        group = groups[0]
        if not isinstance(group, dict) or group.get("matcher") != MATCHER:
            raise ValueError(f"{event}: invalid native tool matcher")
        commands = group.get("hooks")
        if not isinstance(commands, list) or len(commands) != 1:
            raise ValueError(f"{event}: expected one router command")
        command = commands[0]
        if not isinstance(command, dict) or command.get("type") != "command":
            raise ValueError(f"{event}: expected command hook")
        if command.get("command") != COMMAND:
            raise ValueError(f"{event}: router command does not match the supported entrypoint")
        timeout = command.get("timeout")
        if type(timeout) is not int or not 0 < timeout <= 60:
            raise ValueError(f"{event}: invalid timeout")
    if not (path.parent / "hooks/roehub_hook_router.py").is_file():
        raise ValueError("Missing hook router")
