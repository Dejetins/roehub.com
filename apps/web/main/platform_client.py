"""Load the local Vite build; protected HTML remains owned by Web's session gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PLATFORM_DIST = Path(__file__).resolve().parents[2] / "platform-web" / "dist"


def load_platform_assets(dist: Path) -> dict[str, Any]:
    """Fail startup on an incomplete opted-in build instead of serving a blank client."""
    manifest_path = dist / ".vite" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["src/main.tsx"]
        styles: list[str] = []
        seen: set[str] = set()

        def asset_url(name: str) -> str:
            path = (dist / name).resolve()
            if not path.is_relative_to((dist / "assets").resolve()) or not path.is_file():
                raise ValueError("Platform client asset missing or outside assets directory")
            return f"/platform-assets/{name}"

        def visit(key: str) -> None:
            if key in seen:
                return
            seen.add(key)
            chunk = manifest[key]
            asset_url(chunk["file"])
            for style in chunk.get("css", []):
                url = asset_url(style)
                if url not in styles:
                    styles.append(url)
            for imported in chunk.get("imports", []):
                visit(imported)

        visit("src/main.tsx")
        return {"script": asset_url(entry["file"]), "styles": styles}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            "WEB_BACKTESTS_CLIENT_ENABLED requires a complete platform-web build; "
            "run pnpm --filter @roehub/platform-web build or disable the setting"
        ) from error
