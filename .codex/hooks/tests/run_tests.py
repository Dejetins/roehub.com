#!/usr/bin/env python3
"""Focused regression tests for the active Roehub hook invariants."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEST_DIR = Path(__file__).resolve().parent
HOOK_DIR = TEST_DIR.parent
SYNTHETIC_SECRET = "Smoke" + "E2E!" + "9999"
SYNTHETIC_JWT = "eyJ" + ("a" * 16) + "." + "eyJ" + ("b" * 16) + "." + "eyJ" + ("c" * 16)
RUSSIAN_FINAL_RECEIPT = "\n".join(
    (
        "Изменения завершены.",
        "",
        "**Проверка перед финалом**",
        "- Статус проверки: выполнена",
        "- Режим: холодная самопроверка",
        "- Что проверено: итоговый текст и измененные инструкции.",
        "- Итог: можно продолжать",
        "- Что исправлено/добавлено: убраны устаревшие правила.",
        "- Остаточные риски: новая среда поставки еще не выбрана.",
        "- Что это значит для следующего шага: выбрать отдельный runtime ticket.",
    )
)


@dataclass(frozen=True)
class Case:
    name: str
    payload: dict[str, Any]
    expected_empty: bool = False
    expected_path: str | None = None
    expected_value: str | None = None
    expected_text: str | None = None


CASES = (
    Case(
        "raw secret is denied before Bash",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": f"printf '{SYNTHETIC_SECRET}'"},
        },
        expected_path="hookSpecificOutput.permissionDecision",
        expected_value="deny",
        expected_text="password-like literal",
    ),
    Case(
        "destructive root removal is denied",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
        },
        expected_path="hookSpecificOutput.permissionDecision",
        expected_value="deny",
        expected_text="Recursive root removal",
    ),
    Case(
        "environment secret reference is allowed",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "printf '%s' \"$ROEHUB_SMOKE_E2E_PASSWORD\""},
        },
        expected_empty=True,
    ),
    Case(
        "broad git staging is denied",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "git add ."},
        },
        expected_path="hookSpecificOutput.permissionDecision",
        expected_value="deny",
        expected_text="Broad git add pathspec",
    ),
    Case(
        "explicit git staging is allowed",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {
                "command": "git add .codex/AGENTS.md && git diff --cached --name-status"
            },
        },
        expected_empty=True,
    ),
    Case(
        "commit does not need a magic review marker",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "git commit -m scoped"},
        },
        expected_empty=True,
    ),
    Case(
        "raw jwt in tool output blocks continuation",
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "printf token"},
            "tool_response": {"stdout": f"provider returned {SYNTHETIC_JWT}"},
        },
        expected_path="decision",
        expected_value="block",
        expected_text="JWT",
    ),
    Case(
        "mixed-language final report is continued",
        {
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "Изменения готовы. Verification: passed.",
        },
        expected_path="decision",
        expected_value="block",
        expected_text="Качественно переведи",
    ),
    Case(
        "policy completion requires a review receipt",
        {
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "Обновил `.codex/AGENTS.md` и завершил документ.",
        },
        expected_path="decision",
        expected_value="block",
        expected_text="Проверка перед финалом",
    ),
    Case(
        "russian final with a review receipt is allowed",
        {
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": RUSSIAN_FINAL_RECEIPT,
        },
        expected_empty=True,
    ),
    Case(
        "technical Mac Studio name is allowed in russian final",
        {
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": RUSSIAN_FINAL_RECEIPT.replace(
                "новая среда поставки еще не выбрана",
                "Mac Studio выведен из эксплуатации",
            ),
        },
        expected_empty=True,
    ),
)


def _lookup(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        current = current[part]
    return current


def run_case(case: Case) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="roehub-hook-test-") as raw_tmp:
        tmpdir = Path(raw_tmp)
        shutil.copytree(HOOK_DIR, tmpdir / ".codex" / "hooks")
        payload = {**case.payload, "cwd": str(tmpdir)}
        proc = subprocess.run(
            [sys.executable, str(tmpdir / ".codex" / "hooks" / "roehub_hook_router.py")],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    stdout = proc.stdout.strip()
    if proc.returncode:
        return False, f"{case.name}: router returned {proc.returncode}: {proc.stderr!r}"
    if case.expected_empty:
        if stdout:
            return False, f"{case.name}: expected empty stdout, got {stdout!r}"
        return True, case.name
    if case.expected_text and case.expected_text not in stdout:
        return False, f"{case.name}: missing {case.expected_text!r}: {stdout!r}"
    if case.expected_path:
        try:
            value = _lookup(json.loads(stdout), case.expected_path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return False, f"{case.name}: invalid hook output: {exc}: {stdout!r}"
        if value != case.expected_value:
            return False, f"{case.name}: {case.expected_path}={value!r}"
    return True, case.name


def main() -> int:
    failures: list[str] = []
    for case in CASES:
        ok, message = run_case(case)
        print(("ok " if ok else "not ok ") + message)
        if not ok:
            failures.append(message)
    if failures:
        print(f"\n{len(failures)} active hook regression(s) failed")
        return 1
    print("\nall active hook regressions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
