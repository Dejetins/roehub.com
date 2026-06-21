"""Warn when architecture docs appear to miss Roehub documentation anchors."""

from __future__ import annotations

from typing import Any

from validators.common import (
    WARN_WITH_CONTEXT,
    Finding,
    count_cyrillic,
    has_any,
    is_post_tool,
    read_text_if_exists,
    resolve_repo_path,
    touched_paths,
)

RISK_TERMS = [
    "exchange",
    "provider",
    "service",
    "worker",
    "runtime",
    "secret",
    "token",
    "order",
    "deploy",
    "Mac Studio",
    "OpenBao",
]


def _is_arch_doc(path: str) -> bool:
    return (
        path.startswith("docs/architecture/")
        and path.endswith(".md")
        and path != "docs/architecture/README.md"
    )


def _missing_sections(text: str) -> list[str]:
    missing: list[str] = []
    lowered = text.lower()
    if count_cyrillic(text) < 80:
        missing.append("Russian narrative/business-readable explanation")
    if "бизнес" not in lowered and "business" not in lowered:
        missing.append("business impact layer")
    risky = has_any(text, RISK_TERMS)
    if risky and "сервис" not in lowered and "service call" not in lowered:
        missing.append("conditional service-call coverage or explicit N/A")
    if risky and "redaction" not in lowered and "секрет" not in lowered and "лог" not in lowered:
        missing.append("logging/redaction coverage or explicit N/A")
    if risky and "alert" not in lowered and "monitor" not in lowered and "runbook" not in lowered:
        missing.append("alerts/monitoring/runbook coverage or explicit N/A")
    return missing


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_post_tool(payload):
        return []
    findings: list[Finding] = []
    message_suffix = (
        ". Keep sections conditional; use explicit N/A when the risk surface is absent."
    )
    for path_text in touched_paths(payload):
        if not _is_arch_doc(path_text):
            continue
        text = read_text_if_exists(resolve_repo_path(payload, path_text))
        if not text:
            continue
        missing = _missing_sections(text)
        if missing:
            findings.append(
                Finding(
                    severity=WARN_WITH_CONTEXT,
                    title="Architecture doc may miss required Roehub context",
                    message="Review and add if applicable: " + ", ".join(missing) + message_suffix,
                    validator="architecture_doc_linter",
                    target=path_text,
                )
            )
    return findings
