from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from trading.contexts.extensions.application import (
    PluginBundleValidationError,
    PluginBundleValidator,
    load_publisher_key_file,
)


def validate_bundle(
    *,
    bundle_path: Path,
    publisher_key_path: Path | None,
    allow_unsigned_development: bool = False,
    trading_mode: str = "paper",
):  # type: ignore[no-untyped-def]
    repo_root = Path(__file__).resolve().parents[2]
    publisher_keys = (
        load_publisher_key_file(publisher_key_path) if publisher_key_path is not None else {}
    )
    validator = PluginBundleValidator(
        schema_path=repo_root
        / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
        trusted_publisher_keys=publisher_keys,
        roehub_version="0.1.0",
        supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
        allow_unsigned_development=allow_unsigned_development,
        trading_mode=trading_mode,
    )
    return validator.validate(bundle_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roehubctl plugins validate")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--publisher-keys", type=Path)
    parser.add_argument("--allow-unsigned-development", action="store_true")
    parser.add_argument(
        "--trading-mode", choices=("paper", "testnet", "mainnet"), default="paper"
    )
    args = parser.parse_args(argv)
    try:
        validated = validate_bundle(
            bundle_path=args.bundle,
            publisher_key_path=args.publisher_keys,
            allow_unsigned_development=args.allow_unsigned_development,
            trading_mode=args.trading_mode,
        )
    except (PluginBundleValidationError, ValueError) as error:
        code = getattr(error, "code", "plugin.validation_failed")
        print(
            json.dumps(
                {"contract": "PluginValidation/v1alpha1", "status": "failed", "code": code}
            )
        )
        return 2
    manifest = validated.manifest
    print(
        json.dumps(
            {
                "contract": "PluginValidation/v1alpha1",
                "status": "passed",
                "plugin_id": manifest.plugin_id,
                "version": manifest.version,
                "package_digest": manifest.package_digest,
                "image_digest": manifest.image_digest,
                "signed": manifest.signed,
                "permissions": list(manifest.permissions),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
