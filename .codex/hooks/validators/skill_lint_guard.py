"""Warn on mechanical SKILL.md frontmatter omissions."""

from __future__ import annotations

import re
from typing import Any

from validators.common import (
    WARN_WITH_CONTEXT,
    Finding,
    is_post_tool,
    read_text_if_exists,
    resolve_repo_path,
    touched_paths,
)

FRONTMATTER = re.compile(r"^---\n(?P<body>.*?)\n---", re.DOTALL)


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_post_tool(payload):
        return []
    findings: list[Finding] = []
    for path_text in touched_paths(payload):
        if not path_text.endswith("SKILL.md"):
            continue
        text = read_text_if_exists(resolve_repo_path(payload, path_text))
        match = FRONTMATTER.search(text)
        missing: list[str] = []
        if not match:
            missing.append("YAML frontmatter")
        else:
            frontmatter = match.group("body")
            if not re.search(r"^name:\s*\S+", frontmatter, re.MULTILINE):
                missing.append("name")
            if not re.search(r"^description:\s*.+", frontmatter, re.MULTILINE):
                missing.append("description")
        if missing:
            findings.append(
                Finding(
                    severity=WARN_WITH_CONTEXT,
                    title="Skill frontmatter may be incomplete",
                    message="SKILL.md is missing mechanical metadata: "
                    + ", ".join(missing)
                    + ". Run skill validation before treating it as ready.",
                    validator="skill_lint_guard",
                    target=path_text,
                )
            )
    return findings
