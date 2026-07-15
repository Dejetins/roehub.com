from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import httpx
import yaml

from tools.plugins.validate import validate_bundle
from trading.contexts.extensions.application import PluginBundleValidationError


class PluginsCli:
    def __init__(self, *, environ: Mapping[str, str] | None = None) -> None:
        self._environ = os.environ if environ is None else environ

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(prog="roehubctl plugins")
        subparsers = parser.add_subparsers(dest="action", required=True)
        initialize = subparsers.add_parser("init")
        initialize.add_argument("directory", type=Path)
        validate = subparsers.add_parser("validate")
        _add_bundle_validation_arguments(validate)
        install = subparsers.add_parser("install")
        _add_bundle_validation_arguments(install)
        _add_management_arguments(install)
        install.add_argument("--instance-name", required=True)
        install.add_argument("--permission", action="append", default=[])
        install.add_argument("--config-json", default="{}")
        update = subparsers.add_parser("update")
        update.add_argument("plugin_id")
        update.add_argument("--version", required=True)
        _add_bundle_validation_arguments(update)
        _add_management_arguments(update)
        update.add_argument("--instance-name", required=True)
        update.add_argument("--permission", action="append", default=[])
        update.add_argument("--config-json", default="{}")
        rollback = subparsers.add_parser("rollback")
        rollback.add_argument("plugin_id")
        _add_management_arguments(rollback)
        doctor = subparsers.add_parser("doctor")
        doctor.add_argument("plugin_id")
        _add_bundle_validation_arguments(doctor)
        args = parser.parse_args(argv)

        if args.action == "init":
            return _initialize_bundle(directory=args.directory, parser=parser)
        if args.action in {"validate", "install", "update", "doctor"}:
            try:
                validated = validate_bundle(
                    bundle_path=args.bundle,
                    publisher_key_path=args.publisher_keys,
                    allow_unsigned_development=args.allow_unsigned_development,
                    trading_mode=args.trading_mode,
                )
            except (PluginBundleValidationError, ValueError) as error:
                print(
                    json.dumps(
                        {
                            "contract": "PluginValidation/v1alpha1",
                            "status": "failed",
                            "code": getattr(error, "code", "plugin.validation_failed"),
                        },
                        sort_keys=True,
                    )
                )
                return 2
            if args.action == "validate":
                return _print_validated(validated.manifest)
            if args.action == "doctor":
                if validated.manifest.plugin_id != args.plugin_id:
                    parser.error("doctor plugin id does not match the signed manifest")
                return _print_doctor(validated.manifest)
            if args.action == "update" and (
                validated.manifest.plugin_id != args.plugin_id
                or validated.manifest.version != args.version
            ):
                parser.error("update id/version does not match the signed manifest")
            config = _config_object(value=args.config_json, parser=parser)
            payload = {
                "bundle_id": args.bundle.resolve().name,
                "instance_name": args.instance_name,
                "permissions": args.permission,
                "config": config,
            }
            return _management_request(
                method="POST",
                path=f"/api/v1/organizations/{args.organization_id}/plugins/installations",
                args=args,
                payload=payload,
                parser=parser,
            )
        return _management_request(
            method="POST",
            path=(
                f"/api/v1/organizations/{args.organization_id}/plugins/"
                f"installations/{args.plugin_id}:rollback"
            ),
            args=args,
            payload=None,
            parser=parser,
        )


def _add_bundle_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--publisher-keys", type=Path)
    parser.add_argument("--allow-unsigned-development", action="store_true")
    parser.add_argument(
        "--trading-mode", choices=("paper", "testnet", "mainnet"), default="paper"
    )


def _add_management_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--organization-id", type=UUID, required=True)
    parser.add_argument("--session-file", type=Path, required=True)
    parser.add_argument("--idempotency-key", required=True)


def _management_request(
    *,
    method: str,
    path: str,
    args: argparse.Namespace,
    payload: Mapping[str, object] | None,
    parser: argparse.ArgumentParser,
) -> int:
    session_path = args.session_file.expanduser().resolve()
    try:
        mode = stat.S_IMODE(session_path.stat().st_mode)
        if not session_path.is_file() or mode & 0o077:
            parser.error("--session-file must be a regular file with mode 0600")
        session_value = session_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        parser.error(f"--session-file cannot be read: {error}")
    if not session_value:
        parser.error("--session-file is empty")
    try:
        response = httpx.request(
            method,
            args.api_url.rstrip("/") + path,
            headers={"Idempotency-Key": args.idempotency_key},
            cookies={"roehub_session_id": session_value},
            json=dict(payload) if payload is not None else None,
            timeout=10.0,
        )
    except httpx.HTTPError:
        print(json.dumps({"contract": "PluginOperation/v1alpha1", "status": "unavailable"}))
        return 3
    try:
        response_payload = response.json()
    except ValueError:
        response_payload = {"status": "invalid_response"}
    print(json.dumps(response_payload, sort_keys=True))
    return 0 if response.status_code < 400 else 2


def _initialize_bundle(*, directory: Path, parser: argparse.ArgumentParser) -> int:
    root = directory.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("plugin directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    config_schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "replace-me",
        "documentNamespace": "https://example.invalid/replace-me",
        "creationInfo": {"created": "1970-01-01T00:00:00Z", "creators": ["Tool: roehubctl"]},
        "packages": [],
    }
    license_text = "Replace this file with the declared SPDX license text.\n"
    (root / "config.schema.json").write_text(
        json.dumps(config_schema, indent=2) + "\n", encoding="utf-8"
    )
    (root / "sbom.spdx.json").write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
    (root / "LICENSE").write_text(license_text, encoding="utf-8")
    manifest = {
        "apiVersion": "roehub.io/v1alpha1",
        "kind": "Plugin",
        "metadata": {
            "id": "replace-me.plugin",
            "version": "0.1.0",
            "publisher": "replace-me",
            "developmentMode": True,
        },
        "spec": {
            "type": "data-source",
            "pluginApi": "v1alpha1",
            "rpc": {"version": "roehub.plugin.rpc/v1alpha1", "port": 8080},
            "image": {
                "reference": "replace-me:0.1.0",
                "digest": "sha256:" + "0" * 64,
                "architectures": ["linux/amd64", "linux/arm64"],
            },
            "compatibility": {"roehubMin": "0.1.0", "roehubMaxExclusive": "0.2.0"},
            "permissions": [{"capability": "data.read"}],
            "configSchema": _artifact(root / "config.schema.json"),
            "license": {**_artifact(root / "LICENSE"), "spdx": "Apache-2.0"},
            "sbom": _artifact(root / "sbom.spdx.json"),
            "runtime": {
                "nonRootUid": 10001,
                "readOnlyRootFilesystem": True,
                "noNewPrivileges": True,
                "resources": {"cpus": 0.5, "memoryMb": 128, "pids": 64},
                "egress": [],
            },
        },
    }
    (root / "roehub.plugin.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({"contract": "PluginInit/v1alpha1", "status": "created", "path": str(root)}))
    return 0


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _config_object(*, value: str, parser: argparse.ArgumentParser) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        parser.error(f"--config-json is invalid: {error}")
    if not isinstance(payload, dict):
        parser.error("--config-json must be an object")
    return payload


def _print_validated(manifest: Any) -> int:
    print(
        json.dumps(
            {
                "contract": "PluginValidation/v1alpha1",
                "status": "passed",
                "plugin_id": manifest.plugin_id,
                "version": manifest.version,
                "package_digest": manifest.package_digest,
                "signed": manifest.signed,
            },
            sort_keys=True,
        )
    )
    return 0


def _print_doctor(manifest: Any) -> int:
    print(
        json.dumps(
            {
                "contract": "PluginDoctor/v1alpha1",
                "status": "ready",
                "plugin_id": manifest.plugin_id,
                "manifest": "passed",
                "signature": "passed" if manifest.signed else "development-only",
                "runtime_boundary": "not_started",
            },
            sort_keys=True,
        )
    )
    return 0
