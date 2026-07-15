"""Generate deterministic JSON Schemas for installation recovery contracts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from trading.contexts.operations import (
    BackupManifestSignature,
    BackupStateOwner,
    InstallationBackupManifest,
    InstallationBackupPolicy,
    InstallationCaptureRecord,
    InstallationReleasePolicy,
)
from trading.contexts.operations.backup_contracts import REQUIRED_CONSISTENCY_MODES

_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT = _ROOT / "schemas/backup"
_SCHEMAS = (
    (
        "installation-backup-manifest.schema.json",
        InstallationBackupManifest,
        "https://schemas.roehub.io/backup/installation-backup-manifest.schema.json",
    ),
    (
        "installation-backup-signature.schema.json",
        BackupManifestSignature,
        "https://schemas.roehub.io/backup/installation-backup-signature.schema.json",
    ),
    (
        "installation-backup-policy.schema.json",
        InstallationBackupPolicy,
        "https://schemas.roehub.io/backup/installation-backup-policy.schema.json",
    ),
    (
        "installation-capture-record.schema.json",
        InstallationCaptureRecord,
        "https://schemas.roehub.io/backup/installation-capture-record.schema.json",
    ),
    (
        "installation-release-policy.schema.json",
        InstallationReleasePolicy,
        "https://schemas.roehub.io/backup/installation-release-policy.schema.json",
    ),
)


def _render(model: type[object], schema_id: str) -> bytes:
    schema = model.model_json_schema(by_alias=True)  # type: ignore[attr-defined]
    if model in {
        InstallationBackupManifest,
        InstallationBackupPolicy,
        InstallationCaptureRecord,
    }:
        field = "entries" if model is not InstallationBackupPolicy else "sources"
        collection = schema["properties"][field]
        collection["minItems"] = len(BackupStateOwner)
        collection["maxItems"] = len(BackupStateOwner)
        collection["uniqueItems"] = True
        constraints: list[dict[str, object]] = []
        for owner in BackupStateOwner:
            properties: dict[str, object] = {"owner": {"const": owner.value}}
            required = ["owner"]
            if model is not InstallationCaptureRecord:
                properties["consistency_mode"] = {
                    "const": REQUIRED_CONSISTENCY_MODES[owner]
                }
                required.append("consistency_mode")
            constraints.append(
                {
                    "contains": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                    "minContains": 1,
                    "maxContains": 1,
                }
            )
        collection["allOf"] = constraints
    schema["$id"] = schema_id
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()


def generate(*, check: bool) -> None:
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    drift: list[str] = []
    for filename, model, schema_id in _SCHEMAS:
        target = _OUTPUT / filename
        expected = _render(model, schema_id)
        if check:
            if not target.is_file() or target.read_bytes() != expected:
                drift.append(str(target.relative_to(_ROOT)))
        else:
            target.write_bytes(expected)
    if drift:
        raise SystemExit("backup schema drift: " + ", ".join(drift))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    generate(check=args.check)
    print(f"backup schemas passed: mode={'check' if args.check else 'write'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
