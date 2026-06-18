"""Detect tests-only validation wording on non-trivial runtime surfaces."""

from __future__ import annotations

from typing import Any

from validators.common import (
    CONTINUE_BEFORE_FINAL,
    WARN_WITH_CONTEXT,
    Finding,
    assistant_text,
    has_any,
    is_post_tool,
    is_stop,
    read_text_if_exists,
    resolve_repo_path,
    touched_paths,
)

RUNTIME_TERMS = [
    "browser",
    "playwright",
    "runtime",
    "Mac Studio",
    "deploy",
    "exchange",
    "provider",
    "worker",
    "queue",
    "performance",
    "benchmark",
    "external",
]

REAL_BOUNDARY_TERMS = [
    "playwright",
    "screenshot",
    "trace",
    "network check",
    "console check",
    "smoke",
    "browser smoke",
    "real browser",
    "runtime smoke",
    "deploy smoke",
    "mac studio smoke",
    "ssh macstudio",
    "scripts/macos/smoke_prod.sh",
    "benchmark",
    "real adapter",
    "real-boundary",
    "end-to-end",
    "e2e",
]

TEST_ONLY_TERMS = ["pytest", "unit test", "ruff", "pyright", "mypy"]


def _looks_tests_only(text: str) -> bool:
    return (
        has_any(text, RUNTIME_TERMS)
        and has_any(text, TEST_ONLY_TERMS)
        and not has_any(text, REAL_BOUNDARY_TERMS)
    )


def validate(payload: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    post_tool_message = (
        "This artifact mentions runtime/integration risk but validation appears "
        "limited to tests. Add real-boundary evidence or explain why it is not "
        "applicable."
    )
    stop_message = (
        "If the task touched runtime/browser/deploy/provider/performance behavior, "
        "add real-boundary evidence or state why tests-only verification is sufficient."
    )
    if is_post_tool(payload):
        for path_text in touched_paths(payload):
            if not path_text.endswith(".md"):
                continue
            text = read_text_if_exists(resolve_repo_path(payload, path_text))
            if _looks_tests_only(text):
                findings.append(
                    Finding(
                        severity=CONTINUE_BEFORE_FINAL,
                        title="Validation depth appears tests-only for runtime stage",
                        message=post_tool_message,
                        validator="validation_depth_linter",
                        target=path_text,
                    )
                )
    if is_stop(payload):
        text = assistant_text(payload)
        if _looks_tests_only(text):
            findings.append(
                Finding(
                    severity=WARN_WITH_CONTEXT,
                    title="Final validation summary appears tests-only",
                    message=stop_message,
                    validator="validation_depth_linter",
                    target="last_assistant_message",
                )
            )
    return findings
