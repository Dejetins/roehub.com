from __future__ import annotations

from pathlib import Path

from tools.release.generate_installation_config import run


def test_cli_writes_and_checks_all_profiles(tmp_path: Path) -> None:
    assert run(["--output", str(tmp_path), "--write"]) == 0
    assert run(["--output", str(tmp_path), "--check"]) == 0

    for profile in ("base", "trading", "ml"):
        assert (tmp_path / profile / "compose.yaml").is_file()
        assert (tmp_path / profile / "generation-manifest.json").is_file()


def test_cli_detects_stale_generated_output(tmp_path: Path) -> None:
    assert run(["--output", str(tmp_path), "--profile", "base", "--write"]) == 0
    (tmp_path / "base" / "service-config.json").write_text("{}\n", encoding="utf-8")

    assert run(["--output", str(tmp_path), "--profile", "base", "--check"]) == 1
