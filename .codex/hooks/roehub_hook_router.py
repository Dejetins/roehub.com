#!/usr/bin/env python3
"""Route Codex hook events through Roehub repository policy validators.

This script intentionally uses only the Python standard library. It does not
persist hook payloads by default because hook inputs may contain secrets.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

HOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HOOK_DIR))

from validators.common import (  # noqa: E402
    CONTINUE_BEFORE_FINAL,
    FATAL_BLOCK,
    WARN_WITH_CONTEXT,
    Finding,
    format_findings,
    hook_event,
)

VALIDATOR_MODULES = [
    "validators.secret_redaction_guard",
    "validators.command_safety_guard",
    "validators.scoped_git_staging_guard",
]


def load_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        emit_json({"systemMessage": f"Roehub hook ignored malformed JSON input: {exc}"})
        raise SystemExit(0)
    return data if isinstance(data, dict) else {}


def run_validators(payload: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for module_name in VALIDATOR_MODULES:
        module = importlib.import_module(module_name)
        validate = getattr(module, "validate")
        findings.extend(validate(payload))
    return findings


def finding_groups(findings: list[Finding]) -> dict[str, list[Finding]]:
    return {
        FATAL_BLOCK: [f for f in findings if f.severity == FATAL_BLOCK],
        CONTINUE_BEFORE_FINAL: [f for f in findings if f.severity == CONTINUE_BEFORE_FINAL],
        WARN_WITH_CONTEXT: [f for f in findings if f.severity == WARN_WITH_CONTEXT],
    }


def emit_json(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
    sys.stdout.write("\n")


def compact_reason(findings: list[Finding]) -> str:
    return "Roehub hook policy requires attention:\n" + format_findings(findings)


def emit_for_event(event: str, groups: dict[str, list[Finding]]) -> None:
    fatal = groups[FATAL_BLOCK]
    cont = groups[CONTINUE_BEFORE_FINAL]
    warn = groups[WARN_WITH_CONTEXT]

    if event == "PreToolUse":
        if fatal:
            reason = compact_reason(fatal)
            emit_json(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            )
            return
        if warn:
            emit_json(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "additionalContext": compact_reason(warn),
                    }
                }
            )
            return

    if event == "PermissionRequest":
        if fatal:
            emit_json(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PermissionRequest",
                        "decision": {
                            "behavior": "deny",
                            "message": compact_reason(fatal),
                        },
                    }
                }
            )
            return

    if event == "PostToolUse":
        blocking = fatal + cont
        if blocking:
            reason = compact_reason(blocking)
            emit_json(
                {
                    "decision": "block",
                    "reason": reason,
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": reason,
                    },
                }
            )
            return
        if warn:
            emit_json(
                {
                    "systemMessage": compact_reason(warn),
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": compact_reason(warn),
                    },
                }
            )
            return

    if event == "UserPromptSubmit":
        if fatal:
            emit_json({"decision": "block", "reason": compact_reason(fatal)})
            return
        if warn:
            emit_json(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": compact_reason(warn),
                    }
                }
            )
            return


def main() -> int:
    payload = load_payload()
    findings = run_validators(payload)
    emit_for_event(hook_event(payload), finding_groups(findings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
