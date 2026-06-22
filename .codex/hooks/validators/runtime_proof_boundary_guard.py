"""Lint Mac Studio runtime proof wording in docs and generated prompts."""

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

MAC_STUDIO_TERMS = [
    "mac studio",
    "macstudio",
    "/opt/roehub/app",
]

AMBIGUOUS_PROOF_TERMS = [
    "target-runtime proof",
    "target runtime proof",
    "target-runtime validation",
    "production runtime proof",
    "runtime proof for changed code",
    "changed-code runtime proof",
]

BOUNDARY_LABELS = [
    "target_host_readiness_pre_main",
    "read_only_existing_runtime_smoke",
    "post_main_production_runtime_proof",
]


def _is_relevant_markdown(path_text: str) -> bool:
    return path_text.endswith(".md") and (
        path_text.startswith(".codex/agents/generated/")
        or path_text.startswith("docs/architecture/")
        or path_text.startswith("docs/runbooks/")
    )


def _post_main_terms_present(text: str) -> bool:
    lowered = text.lower()
    has_main = "main" in lowered
    has_green = "green ci" in lowered or "github actions" in lowered
    has_delivery = "deploy" in lowered or "sync" in lowered
    return has_main and has_green and has_delivery


def _missing_runtime_boundary(text: str) -> list[str]:
    lowered = text.lower()
    if not has_any(lowered, MAC_STUDIO_TERMS):
        return []

    missing: list[str] = []
    if has_any(lowered, AMBIGUOUS_PROOF_TERMS) and not has_any(lowered, BOUNDARY_LABELS):
        missing.append("replace ambiguous target-runtime proof wording with explicit proof_boundary labels")
    if "post_main_production_runtime_proof" in lowered and not _post_main_terms_present(lowered):
        missing.append("post_main_production_runtime_proof must require main, green CI/GitHub Actions, and deploy/sync")
    if (
        ("pre-main" in lowered or "before main" in lowered or "before merge" in lowered)
        and "changed code" in lowered
        and ("production proof" in lowered or "production runtime proof" in lowered)
    ):
        missing.append("do not claim pre-main Mac Studio evidence as changed-code production proof")
    return missing


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_post_tool(payload):
        return []

    findings: list[Finding] = []
    for path_text in touched_paths(payload):
        if not _is_relevant_markdown(path_text):
            continue
        text = read_text_if_exists(resolve_repo_path(payload, path_text))
        missing = _missing_runtime_boundary(text)
        if missing:
            findings.append(
                Finding(
                    severity=CONTINUE_BEFORE_FINAL,
                    title="Mac Studio runtime proof boundary is ambiguous",
                    message=(
                        "Separate pre-main host/read-only checks from post-main changed-code "
                        "production runtime proof. Fix: " + ", ".join(missing) + "."
                    ),
                    validator="runtime_proof_boundary_guard",
                    target=path_text,
                )
            )
    return findings
