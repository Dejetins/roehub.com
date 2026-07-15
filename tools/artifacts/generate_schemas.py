from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from trading.integration import ArtifactBackupCatalog, ArtifactManifest, ArtifactStoreDescriptor

_SCHEMAS: tuple[tuple[str, Type[BaseModel], str], ...] = (
    (
        "artifact-manifest-v1.schema.json",
        ArtifactManifest,
        "https://schemas.roehub.io/artifacts/artifact-manifest-v1.schema.json",
    ),
    (
        "artifact-store-v1.schema.json",
        ArtifactStoreDescriptor,
        "https://schemas.roehub.io/artifacts/artifact-store-v1.schema.json",
    ),
    (
        "artifact-backup-v1.schema.json",
        ArtifactBackupCatalog,
        "https://schemas.roehub.io/artifacts/artifact-backup-v1.schema.json",
    ),
)


def generate(*, output_root: Path, check: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for name, model, schema_id in _SCHEMAS:
        payload = model.model_json_schema(by_alias=True)
        payload["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        payload["$id"] = schema_id
        content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        target = output_root / name
        if check:
            if not target.exists() or target.read_bytes() != content:
                stale.append(name)
        elif not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
    if stale:
        raise SystemExit(f"artifact schemas are stale: {', '.join(stale)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("schemas/artifacts"),
    )
    args = parser.parse_args()
    generate(output_root=args.output_root, check=args.check)


if __name__ == "__main__":
    main()
