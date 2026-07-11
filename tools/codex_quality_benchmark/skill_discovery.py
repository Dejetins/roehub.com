from __future__ import annotations

import json
from pathlib import Path


def discover_skill_paths(root: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.rglob("SKILL.md")
        if not _is_undeclared_resource_contract(path.resolve(), root.resolve())
    )


def _is_undeclared_resource_contract(path: Path, root: Path) -> bool:
    plugin_root = _nearest_plugin_root(path, root)
    if plugin_root is None:
        return False
    relative = path.relative_to(plugin_root)
    if relative.parts[:2] != ("resources", "skills"):
        return False
    manifest_path = plugin_root / ".codex-plugin" / "plugin.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    declared = manifest.get("skills")
    if not isinstance(declared, str):
        return True
    declared_root = (plugin_root / declared).resolve()
    return not path.is_relative_to(declared_root)


def _nearest_plugin_root(path: Path, root: Path) -> Path | None:
    for candidate in path.parents:
        if not candidate.is_relative_to(root):
            break
        if (candidate / ".codex-plugin" / "plugin.json").is_file():
            return candidate
    return None
