from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BEGIN_MARKER = "<!-- BEGIN GENERATED DOCS INDEX -->"
END_MARKER = "<!-- END GENERATED DOCS INDEX -->"

KNOWN_GROUP_ORDER: tuple[str, ...] = ("architecture", "runbooks", "api", "decisions", "other")
GROUP_TITLES: dict[str, str] = {
    "architecture": "Архитектура",
    "runbooks": "Ранбуки",
    "api": "API",
    "decisions": "Решения",
    "other": "Прочее",
}


@dataclass(frozen=True, slots=True)
class DocumentEntry:
    """
    Store one markdown document line item for docs index rendering.

    Parameters:
    - path: repository-relative path (POSIX style), e.g. `docs/runbooks/help_commands.md`.
    - title: human-readable title extracted from `# H1` or fallback from filename.
    - description: first meaningful text line after `# H1`, may be empty.
    - group_key: stable group key used for deterministic section ordering.

    Returns:
    - Immutable data object used by renderer.

    Key assumptions / invariants:
    - `path` always starts with `docs/`.
    - `group_key` is one of keys used by `GROUP_TITLES`.

    Errors / exceptions:
    - No runtime errors are raised by the dataclass itself.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    path: str
    title: str
    description: str
    group_key: str


def _is_ignored_component(name: str) -> bool:
    """
    Check whether a path component should be ignored during docs scan.

    Parameters:
    - name: one path component from a path under `docs/`.

    Returns:
    - `True` when component is hidden (`.name`) or pycache-like (`__pycache__*`), else `False`.

    Key assumptions / invariants:
    - Called for individual path components from normalized `Path.parts`.

    Errors / exceptions:
    - No exceptions are expected for string input.

    Side effects:
    - None.

    Related docs and files:
    - `docs/repository_three.md`
    - `tools/docs/generate_docs_index.py`
    """

    return name.startswith(".") or name.startswith("__pycache__")


def _fallback_title(path: Path) -> str:
    """
    Build fallback human-readable title from file stem.

    Parameters:
    - path: markdown file path relative to repository root.

    Returns:
    - Normalized title from filename stem with dashes/underscores converted to spaces.

    Key assumptions / invariants:
    - File stem is non-empty for regular markdown files.

    Errors / exceptions:
    - No exceptions are expected for regular paths.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    normalized = path.stem.replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if normalized else path.as_posix()


def collect_markdown_files(docs_root: Path) -> list[Path]:
    """
    Collect all markdown files under docs tree with deterministic filtering and ordering.

    Parameters:
    - docs_root: repository path to docs root (`<repo>/docs`).

    Returns:
    - Sorted list of markdown file paths. Sorting key is POSIX relative path to `docs_root`.

    Key assumptions / invariants:
    - Only `*.md` files are included.
    - Hidden files/directories and `__pycache__*` components are ignored.
    - Output order is deterministic across OSes.

    Errors / exceptions:
    - Raises `FileNotFoundError` when `docs_root` does not exist.

    Side effects:
    - Reads filesystem tree under `docs_root`.

    Related docs and files:
    - `docs/repository_three.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    if not docs_root.exists():
        raise FileNotFoundError(f"Docs root does not exist: {docs_root}")

    collected: list[Path] = []
    for file_path in docs_root.rglob("*.md"):
        relative_to_docs = file_path.relative_to(docs_root)
        if any(_is_ignored_component(component) for component in relative_to_docs.parts):
            continue
        collected.append(file_path)

    collected.sort(key=lambda item: item.relative_to(docs_root).as_posix())
    return collected


def extract_title_and_description(markdown_text: str, fallback_title: str) -> tuple[str, str]:
    """
    Extract document title and one-line description from markdown content.

    Parameters:
    - markdown_text: full markdown file content.
    - fallback_title: title used when first `# H1` line is absent.

    Returns:
    - Tuple `(title, description)`:
      - `title`: first `# H1` value or fallback.
      - `description`: first non-empty non-heading text line after H1, otherwise empty string.

    Key assumptions / invariants:
    - Only the first `# H1` is treated as canonical title.
    - Description is a single line and keeps original wording.

    Errors / exceptions:
    - No exceptions are expected for string input.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    lines = markdown_text.splitlines()
    title = fallback_title
    h1_index: int | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# "):
            candidate_title = stripped[2:].strip()
            if candidate_title:
                title = candidate_title
            h1_index = index
            break

    if h1_index is None:
        return title, ""

    for line in lines[h1_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("```"):
            continue
        return title, stripped

    return title, ""


def _resolve_group_key(relative_to_docs: Path) -> str:
    """
    Map docs relative path to deterministic high-level group key.

    Parameters:
    - relative_to_docs: markdown path relative to `docs/`.

    Returns:
    - Group key: one of `architecture`, `runbooks`, `api`, `decisions`, `other`.

    Key assumptions / invariants:
    - Top-level folder under `docs/` defines the section for known groups.
    - Top-level markdown files under `docs/` are grouped into `other`.

    Errors / exceptions:
    - No exceptions are expected for regular relative paths.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tools/docs/generate_docs_index.py`
    """

    if len(relative_to_docs.parts) >= 2:
        top_level = relative_to_docs.parts[0]
        if top_level in {"architecture", "runbooks", "api", "decisions"}:
            return top_level
    return "other"


def build_document_entries(repo_root: Path) -> list[DocumentEntry]:
    """
    Build index entries for all markdown docs under `docs/`.

    Parameters:
    - repo_root: repository root path containing `docs/`.

    Returns:
    - Deterministically sorted list of `DocumentEntry`.

    Key assumptions / invariants:
    - Files are sorted lexicographically by POSIX repo-relative path.
    - Each entry has stable `group_key` from `_resolve_group_key`.

    Errors / exceptions:
    - Raises `FileNotFoundError` if `docs/` is missing.
    - Raises `OSError` if file read fails.

    Side effects:
    - Reads all markdown files from `docs/`.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    docs_root = repo_root / "docs"
    entries: list[DocumentEntry] = []

    for markdown_file in collect_markdown_files(docs_root):
        relative_repo_path = markdown_file.relative_to(repo_root)
        relative_to_docs = markdown_file.relative_to(docs_root)
        fallback_title = _fallback_title(relative_repo_path)
        text = markdown_file.read_text(encoding="utf-8")
        title, description = extract_title_and_description(text, fallback_title)

        entries.append(
            DocumentEntry(
                path=relative_repo_path.as_posix(),
                title=title,
                description=description,
                group_key=_resolve_group_key(relative_to_docs),
            )
        )

    entries.sort(key=lambda entry: entry.path)
    return entries


def _group_entries(entries: Sequence[DocumentEntry]) -> dict[str, list[DocumentEntry]]:
    """
    Group document entries by stable section keys.

    Parameters:
    - entries: flat sequence of document index entries.

    Returns:
    - Dictionary `{group_key: [entries...]}` where each list preserves deterministic input order.

    Key assumptions / invariants:
    - Input entries are already sorted by path.
    - All unknown groups are treated as `other`.

    Errors / exceptions:
    - No exceptions are expected for valid entries.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tools/docs/generate_docs_index.py`
    """

    grouped: dict[str, list[DocumentEntry]] = {key: [] for key in KNOWN_GROUP_ORDER}
    for entry in entries:
        group_key = entry.group_key if entry.group_key in grouped else "other"
        grouped[group_key].append(entry)
    return grouped


def render_generated_index(entries: Sequence[DocumentEntry]) -> str:
    """
    Render generated markdown index body grouped by canonical sections.

    Parameters:
    - entries: sorted docs entries to render.

    Returns:
    - Markdown string without marker lines.

    Key assumptions / invariants:
    - Rendering is deterministic for the same entries.
    - Group order always follows `KNOWN_GROUP_ORDER`.

    Errors / exceptions:
    - No exceptions are expected for valid entries.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    grouped = _group_entries(entries)
    rendered_lines: list[str] = []

    for group_key in KNOWN_GROUP_ORDER:
        group_entries = grouped[group_key]
        rendered_lines.append(f"### {GROUP_TITLES[group_key]}")
        rendered_lines.append("")
        if not group_entries:
            rendered_lines.append("- (пока нет документов)")
            rendered_lines.append("")
            continue

        for entry in group_entries:
            line = f"- [{entry.title}]({entry.path}) — `{entry.path}`"
            if entry.description:
                line = f"{line} — {entry.description}"
            rendered_lines.append(line)
        rendered_lines.append("")

    return "\n".join(rendered_lines).rstrip()


def default_manual_header() -> str:
    """
    Build default manual section for architecture docs index README.

    Parameters:
    - None.

    Returns:
    - Russian markdown header containing purpose, update commands, and index markers.

    Key assumptions / invariants:
    - Header is manually maintained and preserved by generator outside marker block.
    - Contains required literals for onboarding and CI docs check.

    Errors / exceptions:
    - No exceptions are expected.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `docs/runbooks/help_commands.md`
    """

    return (
        "# Архитектурная документация\n\n"
        "`docs/architecture/README.md` — каноническая точка входа для навигации по всем документам в `docs/**`.\n\n"  # noqa: E501
        "## Ключевые документы\n\n"
        "- [Indicators Architecture](docs/architecture/indicators/README.md)\n"
        "- [Market Data — Application Ports (Walking Skeleton v1)](docs/architecture/market_data/market-data-application-ports.md)\n"  # noqa: E501
        "- [Strategy — Milestone 3 Epics]"
        "(docs/architecture/strategy/strategy-milestone-3-epics-v1.md)\n"
        "- [Shared Kernel — Primitives](docs/architecture/shared-kernel-primitives.md)\n"
        "- [Runbook: Help commands](docs/runbooks/help_commands.md)\n\n"
        "## Как обновлять индекс\n\n"
        "- Обновить файл: `python -m tools.docs.generate_docs_index`\n"
        "- Проверить актуальность: `python -m tools.docs.generate_docs_index --check`\n\n"
        "## Содержание\n\n"
        f"{BEGIN_MARKER}\n"
        f"{END_MARKER}\n"
    )


def _replace_generated_block(readme_text: str, generated_block: str) -> str:
    """
    Replace generated markdown section between sentinel markers.

    Parameters:
    - readme_text: full README content.
    - generated_block: rendered markdown body that must be inserted between markers.

    Returns:
    - README content with replaced generated section and preserved manual sections.

    Key assumptions / invariants:
    - `BEGIN_MARKER` and `END_MARKER` both exist and are ordered.

    Errors / exceptions:
    - Raises `ValueError` when markers are missing or malformed.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tools/docs/generate_docs_index.py`
    """

    begin_index = readme_text.find(BEGIN_MARKER)
    end_index = readme_text.find(END_MARKER)
    if begin_index == -1 or end_index == -1 or begin_index > end_index:
        raise ValueError("README markers are missing or malformed")

    begin_index += len(BEGIN_MARKER)
    before = readme_text[:begin_index]
    after = readme_text[end_index:]
    return f"{before}\n{generated_block}\n{after}"


def build_readme_content(repo_root: Path, existing_readme: str | None) -> str:
    """
    Construct final README content from manual section and generated index block.

    Parameters:
    - repo_root: repository root used for docs scan.
    - existing_readme: current README content; if absent, default manual header is used.

    Returns:
    - Full deterministic README markdown with normalized `\\n` newlines.

    Key assumptions / invariants:
    - Generated section is fully deterministic for the same docs tree.
    - Manual header above markers is preserved when markers exist.

    Errors / exceptions:
    - Raises `ValueError` when existing README exists but markers are malformed.
    - Propagates filesystem errors from docs scanning.

    Side effects:
    - None.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    base_text = existing_readme if existing_readme is not None else default_manual_header()
    if BEGIN_MARKER not in base_text or END_MARKER not in base_text:
        base_text = default_manual_header()

    entries = build_document_entries(repo_root)
    generated_block = render_generated_index(entries)
    replaced = _replace_generated_block(base_text, generated_block)
    normalized = replaced.replace("\r\n", "\n").replace("\r", "\n")
    return f"{normalized.rstrip()}\n"


def run_generator(repo_root: Path, check: bool) -> int:
    """
    Execute docs index generation in write mode or check-only mode.

    Parameters:
    - repo_root: repository root path containing target README and docs tree.
    - check: if `True`, verify drift only; if `False`, write updated README.

    Returns:
    - Process-like exit code (`0` success, `1` when drift detected in check mode).

    Key assumptions / invariants:
    - Target path is `docs/architecture/README.md`.
    - Check mode must be fail-fast for CI drift detection.

    Errors / exceptions:
    - Propagates `OSError` on read/write failures.
    - Propagates `ValueError` for malformed marker layout in existing README.

    Side effects:
    - Reads entire docs tree.
    - Writes `docs/architecture/README.md` in write mode when content changed.
    - Prints status lines for CI logs.

    Related docs and files:
    - `docs/architecture/README.md`
    - `.github/workflows/ci.yml`
    """

    readme_path = repo_root / "docs" / "architecture" / "README.md"
    existing_readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    updated_readme = build_readme_content(repo_root, existing_readme)

    if check:
        if existing_readme != updated_readme:
            print(f"ERROR: {readme_path.as_posix()} is out-of-date.")
            return 1
        print(f"OK: {readme_path.as_posix()} is up-to-date.")
        return 0

    if existing_readme != updated_readme:
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        with readme_path.open("w", encoding="utf-8", newline="\n") as readme_file:
            readme_file.write(updated_readme)
        print(f"Updated: {readme_path.as_posix()}")
    else:
        print(f"Unchanged: {readme_path.as_posix()}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI options for docs index generator module.

    Parameters:
    - argv: optional argument list for tests; `None` uses `sys.argv`.

    Returns:
    - Parsed argparse namespace with `check` and `repo_root`.

    Key assumptions / invariants:
    - CLI contract stays compatible with `python -m tools.docs.generate_docs_index`.

    Errors / exceptions:
    - `SystemExit` may be raised by argparse on invalid CLI usage.

    Side effects:
    - Reads process argument list when `argv` is `None`.

    Related docs and files:
    - `docs/architecture/README.md`
    - `tests/unit/tools/test_generate_docs_index.py`
    """

    parser = argparse.ArgumentParser(
        description="Generate deterministic docs index for docs/architecture/README.md."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that docs/architecture/README.md is up-to-date and exit non-zero on drift.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run CLI entrypoint for docs index generation and drift checks.

    Parameters:
    - argv: optional CLI arguments list for tests.

    Returns:
    - Integer exit code suitable for `sys.exit`.

    Key assumptions / invariants:
    - Default repository root is inferred from script location.

    Errors / exceptions:
    - Propagates unexpected runtime errors to caller for fail-fast behavior.

    Side effects:
    - Delegates filesystem reads/writes and console output to `run_generator`.

    Related docs and files:
    - `docs/architecture/README.md`
    - `.github/workflows/ci.yml`
    """

    args = parse_args(argv)
    return run_generator(repo_root=args.repo_root.resolve(), check=args.check)


if __name__ == "__main__":
    sys.exit(main())
