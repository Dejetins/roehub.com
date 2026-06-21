"""Warn when documentation edits can drift from generated docs indexes."""

from __future__ import annotations

from typing import Any

from validators.common import WARN_WITH_CONTEXT, Finding, is_post_tool, touched_paths

DOC_INDEX_PATHS = {
    "docs/INDEX.md",
    "docs/architecture/README.md",
}


def _is_markdown_doc(path: str) -> bool:
    return path.startswith("docs/") and path.endswith(".md")


def validate(payload: dict[str, Any]) -> list[Finding]:
    if not is_post_tool(payload):
        return []
    paths = touched_paths(payload)
    if not any(_is_markdown_doc(path) for path in paths):
        return []
    if any(path in DOC_INDEX_PATHS for path in paths):
        return []
    return [
        Finding(
            severity=WARN_WITH_CONTEXT,
            title="Docs index may need refresh",
            message=(
                "Markdown docs changed without touching docs/architecture/README.md "
                "or docs/INDEX.md in the same edit. Before final handoff, run "
                "`uv run python -m tools.docs.generate_docs_index --check` or "
                "regenerate the index and report the result."
            ),
            validator="docs_index_drift_guard",
            target="docs/*.md",
        )
    ]
