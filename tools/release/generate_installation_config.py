#!/usr/bin/env python3
"""Validate roehub.yaml and render deterministic installation profile inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from trading.platform.config.installation import (
    InstallationConfigError,
    check_outputs,
    load_json_bytes,
    load_yaml_bytes,
    render_profile,
    validate_installation,
    write_outputs,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "installation" / "roehub.yaml"
DEFAULT_RELEASE_MANIFEST = ROOT / "tools" / "release" / "release-metadata.json"
INSTALLATION_SCHEMA = ROOT / "schemas" / "config" / "roehub.schema.json"
RELEASE_SCHEMA = ROOT / "schemas" / "config" / "release-manifest.schema.json"


def _load_schema(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InstallationConfigError(f"schema root must be an object: {path}")
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        choices=("base", "trading", "ml"),
        help="profile to render; repeat for multiple profiles; defaults to roehub.yaml profiles",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_source = args.config.read_bytes()
        manifest_source = args.release_manifest.read_bytes()
        config = load_yaml_bytes(config_source, source=str(args.config))
        release_manifest = load_json_bytes(
            manifest_source,
            source=str(args.release_manifest),
        )
        validate_installation(
            config,
            release_manifest,
            _load_schema(INSTALLATION_SCHEMA),
            _load_schema(RELEASE_SCHEMA),
        )
        profiles = tuple(args.profiles or config["profiles"])
        for profile in profiles:
            outputs = render_profile(
                config,
                release_manifest,
                profile,
                config_source=config_source,
                manifest_source=manifest_source,
            )
            if args.write:
                write_outputs(args.output, profile, outputs)
            else:
                check_outputs(args.output, profile, outputs)
    except (
        InstallationConfigError,
        KeyError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(f"installation config generation failed: {error}", file=sys.stderr)
        return 1
    print(
        "installation config generation passed: "
        f"mode={'write' if args.write else 'check'}, profiles={','.join(profiles)}"
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
