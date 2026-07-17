#!/usr/bin/env python3
"""Generate and validate complete base/trading/ml runtime topology."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from trading.platform.config.installation import (
    InstallationConfigError,
    load_json_bytes,
    load_yaml_bytes,
    validate_installation,
)
from trading.platform.config.runtime_topology import (
    check_runtime_outputs,
    load_json_object,
    render_runtime_profile,
    validate_runtime_service_manifest,
    write_runtime_outputs,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/installation/roehub.yaml"
DEFAULT_RELEASE = ROOT / "tools/release/release-metadata.json"
DEFAULT_SERVICE_MANIFEST = ROOT / "configs/installation/runtime-service-manifest.json"
DEFAULT_OUTPUT = ROOT / "configs/installation/generated"
INSTALLATION_SCHEMA = ROOT / "schemas/config/roehub.schema.json"
RELEASE_SCHEMA = ROOT / "schemas/config/release-manifest.schema.json"
SERVICE_SCHEMA = ROOT / "schemas/config/runtime-service-manifest.schema.json"
PROJECT_MAP = ROOT / "docs/architecture/project-map/project-map.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--service-manifest", type=Path, default=DEFAULT_SERVICE_MANIFEST)
    parser.add_argument("--project-map", type=Path, default=PROJECT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        choices=("base", "trading", "ml"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def _templates() -> dict[str, bytes]:
    return {
        "notifications.yaml": (ROOT / "configs/prod/notifications.yaml").read_bytes(),
        "market-data.yaml": (ROOT / "configs/prod/market_data.yaml").read_bytes(),
        "strategy.yaml": (ROOT / "configs/prod/strategy.yaml").read_bytes(),
        "exchange-execution.yaml": (
            ROOT / "configs/prod/exchange_execution.yaml"
        ).read_bytes(),
        "rl-runtime.yaml": (ROOT / "configs/test/rl_trading_ml_runtime.yaml").read_bytes(),
        "backtest-artifacts.yaml": (
            ROOT / "configs/prod/backtest_artifacts.yaml"
        ).read_bytes(),
        "indicators.yaml": (ROOT / "configs/prod/indicators.yaml").read_bytes(),
    }


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config_source = args.config.read_bytes()
        release_source = args.release_manifest.read_bytes()
        service_source = args.service_manifest.read_bytes()
        config = load_yaml_bytes(config_source, source=str(args.config))
        release = load_json_bytes(release_source, source=str(args.release_manifest))
        service_manifest = load_json_bytes(service_source, source=str(args.service_manifest))
        validate_installation(
            config,
            release,
            load_json_object(INSTALLATION_SCHEMA),
            load_json_object(RELEASE_SCHEMA),
        )
        validate_runtime_service_manifest(
            service_manifest,
            load_json_object(SERVICE_SCHEMA),
            load_json_object(args.project_map),
            repo_root=ROOT,
        )
        profiles = tuple(args.profiles or config["profiles"])
        templates = _templates()
        for profile in profiles:
            outputs = render_runtime_profile(
                config=config,
                release_manifest=release,
                service_manifest=service_manifest,
                profile=profile,
                config_source=config_source,
                release_source=release_source,
                service_manifest_source=service_source,
                config_templates=templates,
            )
            if args.write:
                write_runtime_outputs(args.output, profile, outputs)
            else:
                check_runtime_outputs(args.output, profile, outputs)
    except (InstallationConfigError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"runtime topology generation failed: {error}", file=sys.stderr)
        return 1
    print(
        "runtime topology generation passed: "
        f"mode={'write' if args.write else 'check'}, profiles={','.join(profiles)}"
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
