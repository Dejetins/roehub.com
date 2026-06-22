"""Lint generated prompt packs for branch policy drift."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from validators.common import (
    CONTINUE_BEFORE_FINAL,
    Finding,
    is_post_tool,
    read_text_if_exists,
    resolve_repo_path,
    touched_paths,
)

GENERATED_ROOT = ".codex/agents/generated/"
BRANCH_MENTION = [
    "branch_policy",
    "single_allowed_branch",
    "git switch",
    "git checkout -b",
    "git branch",
    "git worktree add",
]
BRANCH_CONTEXT = re.compile(
    r"\b(?:branch|single_allowed_branch|git\s+switch|git\s+checkout|git\s+branch|git\s+worktree)\b"
    r"[^\n]{0,200}(?<!\.)\bcodex/",
    re.IGNORECASE,
)
BRANCH_NAME = re.compile(r"(?<!\.)\bcodex/[A-Za-z0-9._/-]+")
STAGE_BRANCH_SEGMENT = re.compile(r"(?:^|[/_.-])stage[/_.-]?\d+[a-z]?(?:$|[/_.-])", re.IGNORECASE)


def _is_generated_prompt(path: str) -> bool:
    return path.startswith(GENERATED_ROOT) and path.endswith(".md")


def _pack_dir(path: Path) -> Path:
    marker = path.parts.index("generated")
    return Path(*path.parts[: marker + 2])


def _has_branch_mention(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in BRANCH_MENTION) or bool(
        BRANCH_CONTEXT.search(text)
    )


def _branch_names(text: str) -> set[str]:
    return {match.group(0).rstrip("`'\"),.;:") for match in BRANCH_NAME.finditer(text)}


def _has_stage_branch(text: str) -> bool:
    return any(STAGE_BRANCH_SEGMENT.search(name) for name in _branch_names(text))


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_post_tool(payload):
        return []

    findings: list[Finding] = []
    pack_dirs: set[Path] = set()
    for path_text in touched_paths(payload):
        if _is_generated_prompt(path_text):
            try:
                pack_dirs.add(_pack_dir(Path(path_text)))
            except ValueError:
                continue

    for pack_rel in sorted(pack_dirs):
        pack_abs = resolve_repo_path(payload, str(pack_rel))
        texts: list[tuple[str, str]] = []
        if pack_abs.is_dir():
            for path in sorted(pack_abs.glob("*.md")):
                rel = str(pack_rel / path.name)
                texts.append((rel, read_text_if_exists(path)))
        combined = "\n".join(text for _, text in texts)
        if not combined or not _has_branch_mention(combined):
            continue

        missing: list[str] = []
        if "branch_policy" not in combined:
            missing.append("shared branch_policy block")
        if "default_branch" not in combined or "main" not in combined:
            missing.append("default_branch: main")
        if "stage_specific_branches_forbidden" not in combined:
            missing.append("stage_specific_branches_forbidden=true")
        if _has_stage_branch(combined):
            missing.append("remove stage-specific branch names")
        names = _branch_names(combined)
        if len(names) > 1:
            missing.append("use at most one codex/* branch name across the prompt pack")
        if (
            ("git switch -c" in combined or "git checkout -b" in combined or "git worktree add -b" in combined)
            and "ROEHUB_PROMPT_PACK_BRANCH_APPROVED=1" not in combined
        ):
            missing.append(
                "branch creation command must carry ROEHUB_PROMPT_PACK_BRANCH_APPROVED=1"
            )
        if (
            ("git worktree add" in combined or "worktree_path" in combined or "worktree_root" in combined)
            and "ROEHUB_WORKTREE_APPROVED=1" not in combined
        ):
            missing.append(
                "remove worktree/folder instructions unless the user explicitly requested them"
            )

        if missing:
            findings.append(
                Finding(
                    severity=CONTINUE_BEFORE_FINAL,
                    title="Prompt pack branch policy is incomplete",
                    message=(
                        "Generated prompt packs default to main, and may use only one "
                        "user-requested branch for the whole pack. Fix: " + ", ".join(missing) + "."
                    ),
                    validator="prompt_pack_branch_policy_guard",
                    target=str(pack_rel),
                )
            )
    return findings
