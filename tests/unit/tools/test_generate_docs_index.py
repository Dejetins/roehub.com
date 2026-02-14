from __future__ import annotations

from pathlib import Path

import pytest

from tools.docs.generate_docs_index import (
    BEGIN_MARKER,
    END_MARKER,
    collect_markdown_files,
    extract_title_and_description,
    run_generator,
)


def _write(path: Path, content: str) -> None:
    """
    Write UTF-8 text file with parent directory creation for test fixtures.

    Parameters:
    - path: target file path inside synthetic tmp repository.
    - content: markdown or text content to write.

    Returns:
    - None.

    Key assumptions / invariants:
    - Tests operate only inside `tmp_path`.
    - Newlines are normalized to `\\n` for deterministic assertions.

    Errors / exceptions:
    - Propagates `OSError` if filesystem write fails.

    Side effects:
    - Creates parent directories and writes file content.

    Related docs and files:
    - `tools/docs/generate_docs_index.py`
    - `docs/architecture/README.md`
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.replace("\r\n", "\n"), encoding="utf-8")


def test_collect_markdown_files_stable_order_and_ignore_rules(tmp_path: Path) -> None:
    """
    Validate deterministic markdown ordering and ignore filtering for hidden/pycache paths.

    Parameters:
    - tmp_path: pytest temporary directory used as synthetic repository root.

    Returns:
    - None.

    Key assumptions / invariants:
    - Scanner includes only `*.md` under `docs/`.
    - Scanner ignores hidden components and `__pycache__*` artifacts.

    Errors / exceptions:
    - Test fails with assertion errors on unexpected ordering or filtering.

    Side effects:
    - Creates temporary files under `tmp_path`.

    Related docs and files:
    - `tools/docs/generate_docs_index.py`
    - `docs/repository_three.md`
    """

    docs_root = tmp_path / "docs"
    _write(docs_root / "runbooks" / "z-last.md", "# Z")
    _write(docs_root / "runbooks" / "a-first.md", "# A")
    _write(docs_root / "architecture" / "sub" / "b-mid.md", "# B")
    _write(docs_root / ".hidden" / "skip.md", "# Hidden")
    _write(docs_root / "api" / "__pycache__" / "skip.md", "# Cache")
    _write(docs_root / "api" / "note.txt", "not markdown")

    files = collect_markdown_files(docs_root)
    actual = [file_path.relative_to(docs_root).as_posix() for file_path in files]

    assert actual == [
        "architecture/sub/b-mid.md",
        "runbooks/a-first.md",
        "runbooks/z-last.md",
    ]


def test_extract_title_and_description_from_h1_and_first_text_line() -> None:
    """
    Verify metadata extraction from first H1 and first non-empty plain line after H1.

    Parameters:
    - None.

    Returns:
    - None.

    Key assumptions / invariants:
    - First `# H1` defines title.
    - Description is the first non-heading line after H1.

    Errors / exceptions:
    - Test fails with assertion errors on metadata mismatch.

    Side effects:
    - None.

    Related docs and files:
    - `tools/docs/generate_docs_index.py`
    - `docs/architecture/README.md`
    """

    markdown = (
        "Intro before heading\n"
        "# Canonical Title\n"
        "\n"
        "Краткое описание в первой строке.\n"
        "Следующая строка абзаца.\n"
    )
    title, description = extract_title_and_description(markdown, fallback_title="Fallback")

    assert title == "Canonical Title"
    assert description == "Краткое описание в первой строке."


def test_run_generator_check_detects_drift_and_reports_up_to_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    Ensure check mode fails on README drift and succeeds after regeneration.

    Parameters:
    - tmp_path: pytest temporary directory used as synthetic repository root.
    - capsys: pytest capture fixture for CLI-like output validation.

    Returns:
    - None.

    Key assumptions / invariants:
    - `run_generator(..., check=True)` returns non-zero when README differs from expected.
    - Success output contains both `OK` and `up-to-date` for CI assertions.

    Errors / exceptions:
    - Test fails with assertion errors when drift behavior is inconsistent.

    Side effects:
    - Writes synthetic docs files and generated README under `tmp_path`.

    Related docs and files:
    - `tools/docs/generate_docs_index.py`
    - `.github/workflows/ci.yml`
    """

    docs_root = tmp_path / "docs"
    _write(docs_root / "runbooks" / "help_commands.md", "# Help\n\nКоманды.")
    _write(docs_root / "architecture" / "market_data.md", "# Market Data\n\nОписание.")
    _write(
        docs_root / "architecture" / "README.md",
        (
            "# Ручной заголовок\n\n"
            "## Содержание\n\n"
            f"{BEGIN_MARKER}\n"
            "- stale entry\n"
            f"{END_MARKER}\n"
        ),
    )

    drift_exit_code = run_generator(tmp_path, check=True)
    drift_output = capsys.readouterr().out
    assert drift_exit_code == 1
    assert "out-of-date" in drift_output

    write_exit_code = run_generator(tmp_path, check=False)
    assert write_exit_code == 0

    check_exit_code = run_generator(tmp_path, check=True)
    check_output = capsys.readouterr().out
    assert check_exit_code == 0
    assert "OK" in check_output
    assert "up-to-date" in check_output
