"""Block obvious raw secrets and warn on softer secret-looking output."""

from __future__ import annotations

import re
from typing import Any

from validators.common import FATAL_BLOCK, WARN_WITH_CONTEXT, Finding, iter_text_surfaces

RAW_SECRET_PATTERNS = [
    ("Roehub smoke E2E password-like literal", re.compile(r"SmokeE2E!\d{4}")),
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{30,}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
]

SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:password|passwd|token|secret|api[_-]?key|access[_-]?key|refresh[_-]?token)"
    r"\b\s*[:=]\s*['\"]?([^'\"\s]{8,})"
)

SAFE_VALUES = {
    "redacted",
    "[redacted]",
    "***",
    "****",
    "xxxxx",
    "<redacted>",
    "<secret>",
    "$ROEHUB_SMOKE_E2E_PASSWORD",
    "ROEHUB_SMOKE_E2E_PASSWORD",
}


def _unsafe_assignment_value(value: str) -> bool:
    stripped = value.strip()
    if stripped in SAFE_VALUES:
        return False
    if stripped.startswith("${") and stripped.endswith("}"):
        return False
    if stripped.startswith("$"):
        return False
    return True


def validate(payload: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    raw_secret_message = (
        "Raw credential-like material is forbidden in prompts, commands, logs, "
        "traces, reports, ledgers, and generated artifacts. Use env var names "
        "or redacted placeholders only."
    )
    assignment_message = (
        "A password/token/secret assignment appears to include a raw value. "
        "Replace it with a host-local env var reference or redacted placeholder."
    )
    for surface, text in iter_text_surfaces(payload):
        for title, pattern in RAW_SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(
                    Finding(
                        severity=FATAL_BLOCK,
                        title=title,
                        message=raw_secret_message,
                        validator="secret_redaction_guard",
                        target=surface,
                    )
                )
        for match in SECRET_ASSIGNMENT.finditer(text):
            if _unsafe_assignment_value(match.group(1)):
                severity = WARN_WITH_CONTEXT if surface == "tool_response" else FATAL_BLOCK
                findings.append(
                    Finding(
                        severity=severity,
                        title="Secret-looking assignment",
                        message=assignment_message,
                        validator="secret_redaction_guard",
                        target=surface,
                    )
                )
    return findings
