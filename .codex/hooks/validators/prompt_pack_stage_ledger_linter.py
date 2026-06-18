"""Check generated prompt-pack files for stage ledger and manifest anchors."""

from __future__ import annotations

from typing import Any

from validators.common import (
    CONTINUE_BEFORE_FINAL,
    Finding,
    has_any,
    is_post_tool,
    read_text_if_exists,
    resolve_repo_path,
    touched_paths,
)


def _is_generated_prompt(path: str) -> bool:
    return path.startswith(".codex/agents/generated/") and path.endswith(".md")


def _missing_browser_auth_anchors(text: str) -> list[str]:
    if not (
        has_any(text, ["browser", "playwright", "ui", "frontend", "screenshot"])
        and has_any(text, ["auth", "authenticated", "login", "keycloak", "session"])
    ):
        return []

    missing: list[str] = []
    if "smoke_e2e_keycloak" not in text:
        missing.append("Roehub smoke Keycloak username")
    if "ROEHUB_SMOKE_E2E_PASSWORD" not in text:
        missing.append("host-local smoke password env var source")
    if not has_any(text, ["redaction", "redact", "raw password", "credentials", "secrets"]):
        missing.append("credential redaction rule")
    return missing


def _missing_prompt_anchors(text: str) -> list[str]:
    missing: list[str] = []
    if "ledger" not in text.lower():
        missing.append("stage ledger path/update rule")
    if (
        "expected_primary_touches" in text
        and "ledger" not in text.lower().split("expected_primary_touches", 1)[-1][:1000]
    ):
        missing.append("ledger in expected_primary_touches")
    if not has_any(
        text,
        [
            "file manifest",
            "created/modified/deleted",
            "created files",
            "modified files",
            "deleted files",
        ],
    ):
        missing.append("file manifest created/modified/deleted")
    if not has_any(
        text,
        ["previous required stage", "previous stage", "prerequisite", "accepted in the ledger"],
    ):
        missing.append("previous-stage ledger gate")
    missing.extend(_missing_browser_auth_anchors(text))
    return missing


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_post_tool(payload):
        return []
    findings: list[Finding] = []
    for path_text in touched_paths(payload):
        if not _is_generated_prompt(path_text):
            continue
        path = resolve_repo_path(payload, path_text)
        text = read_text_if_exists(path)
        if not text:
            continue
        missing = _missing_prompt_anchors(text)
        if missing:
            message = (
                "Generated executor prompt is missing: "
                + ", ".join(missing)
                + ". Add explicit ledger, manifest, and stage-gate instructions "
                "or mark why they are not applicable."
            )
            findings.append(
                Finding(
                    severity=CONTINUE_BEFORE_FINAL,
                    title="Prompt-pack stage readiness anchors missing",
                    message=message,
                    validator="prompt_pack_stage_ledger_linter",
                    target=path_text,
                )
            )
    return findings
