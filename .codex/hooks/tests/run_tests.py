#!/usr/bin/env python3
"""Fixture tests for Roehub Codex hook router."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TEST_DIR = Path(__file__).resolve().parent
HOOK_DIR = TEST_DIR.parent
ROUTER = HOOK_DIR / "roehub_hook_router.py"
FIXTURES = TEST_DIR / "fixtures"
SYNTHETIC_SMOKE_SECRET = "Smoke" + "E2E!" + "9999"
SYNTHETIC_SECRET_ASSIGNMENT = "service " + "password: " + "abcdefgh1234"
SYNTHETIC_JWT = "eyJ" + ("a" * 16) + "." + "eyJ" + ("b" * 16) + "." + "eyJ" + ("c" * 16)


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_fixture(fixture: dict[str, Any], tmpdir: Path) -> dict[str, Any]:
    payload = fixture["payload"]
    payload_text = json.dumps(payload)
    payload_text = payload_text.replace("{tmp}", str(tmpdir))
    payload_text = payload_text.replace(
        "__ROEHUB_SYNTHETIC_SMOKE_SECRET__",
        SYNTHETIC_SMOKE_SECRET,
    )
    payload_text = payload_text.replace(
        "__ROEHUB_SYNTHETIC_SECRET_ASSIGNMENT__",
        SYNTHETIC_SECRET_ASSIGNMENT,
    )
    payload_text = payload_text.replace(
        "__ROEHUB_SYNTHETIC_JWT__",
        SYNTHETIC_JWT,
    )
    rendered = json.loads(payload_text)
    rendered["cwd"] = str(tmpdir)

    for relative, content in fixture.get("setup_files", {}).items():
        path = tmpdir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return rendered


def run_fixture(path: Path) -> tuple[bool, str]:
    fixture = load_fixture(path)
    with tempfile.TemporaryDirectory(prefix="roehub-hook-test-") as raw_tmp:
        tmpdir = Path(raw_tmp)
        shutil.copytree(HOOK_DIR, tmpdir / ".codex" / "hooks")
        payload = materialize_fixture(fixture, tmpdir)
        proc = subprocess.run(
            [sys.executable, str(tmpdir / ".codex" / "hooks" / "roehub_hook_router.py")],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    expected = fixture["expect"]
    if proc.returncode != expected.get("returncode", 0):
        expected_returncode = expected.get("returncode", 0)
        return (
            False,
            f"{path.name}: returncode {proc.returncode}, "
            f"expected {expected_returncode}; stderr={proc.stderr!r}",
        )
    stdout = proc.stdout.strip()
    if expected.get("stdout_empty"):
        if stdout:
            return False, f"{path.name}: expected empty stdout, got {stdout!r}"
        return True, path.name
    if "stdout_contains" in expected and expected["stdout_contains"] not in stdout:
        return (
            False,
            f"{path.name}: stdout missing {expected['stdout_contains']!r}; " f"got {stdout!r}",
        )
    if "json_path" in expected:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return False, f"{path.name}: stdout is not JSON: {exc}: {stdout!r}"
        current: Any = data
        for part in expected["json_path"].split("."):
            current = current[part]
        if current != expected["json_value"]:
            return (
                False,
                f"{path.name}: {expected['json_path']}={current!r}, "
                f"expected {expected['json_value']!r}",
            )
    return True, path.name


def main() -> int:
    failures: list[str] = []
    for path in sorted(FIXTURES.glob("*.json")):
        ok, message = run_fixture(path)
        if ok:
            print(f"ok {message}")
        else:
            print(f"not ok {message}")
            failures.append(message)
    if failures:
        print(f"\n{len(failures)} fixture(s) failed")
        return 1
    print("\nall hook fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
