from __future__ import annotations

import json
from pathlib import Path

from tools.codex_quality_benchmark.skill_discovery import discover_skill_paths


def test_resource_only_plugin_contracts_are_not_loader_candidates(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    plugin = root / "personal" / "resource-only" / "1.0.0"
    _manifest(plugin, {})
    resource = plugin / "resources" / "skills" / "S001" / "SKILL.md"
    resource.parent.mkdir(parents=True)
    resource.write_text("resource contract", encoding="utf-8")

    assert discover_skill_paths(root) == []


def test_explicit_resource_skills_path_is_loader_candidate(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    plugin = root / "personal" / "declared" / "1.0.0"
    _manifest(plugin, {"skills": "./resources/skills"})
    skill = plugin / "resources" / "skills" / "example" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("declared skill", encoding="utf-8")

    assert discover_skill_paths(root) == [skill.resolve()]


def test_hidden_nested_dependency_remains_discoverable(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    plugin = root / "personal" / "prototype" / "1.0.0"
    _manifest(plugin, {})
    nested = plugin / ".npm-cache" / "node_modules" / "tool" / "SKILL.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("nested dependency", encoding="utf-8")

    assert discover_skill_paths(root) == [nested.resolve()]


def _manifest(plugin: Path, extra: dict[str, object]) -> None:
    path = plugin / ".codex-plugin" / "plugin.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"name": plugin.parent.name, **extra}),
        encoding="utf-8",
    )
