from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trading.contexts.extensions.application import (
    PluginBundleValidationError,
    PluginBundleValidator,
    canonical_package_digest,
    load_publisher_key_file,
    sign_package_digest,
)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _bundle(tmp_path: Path, *, signed: bool = True) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.schema.json").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    (bundle / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    (bundle / "sbom.spdx.json").write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "SPDXID": "SPDXRef-DOCUMENT",
                "dataLicense": "CC0-1.0",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "apiVersion": "roehub.io/v1alpha1",
        "kind": "Plugin",
        "metadata": {
            "id": "fixture.data",
            "version": "0.1.0",
            "publisher": "fixture.publisher",
            "developmentMode": not signed,
        },
        "spec": {
            "type": "data-source",
            "pluginApi": "v1alpha1",
            "rpc": {"version": "roehub.plugin.rpc/v1alpha1", "port": 8080},
            "image": {
                "reference": "fixture/plugin:0.1.0",
                "digest": "sha256:" + "1" * 64,
                "architectures": ["linux/amd64", "linux/arm64"],
            },
            "compatibility": {"roehubMin": "0.1.0", "roehubMaxExclusive": "0.2.0"},
            "permissions": [{"capability": "data.read"}],
            "configSchema": _artifact(bundle / "config.schema.json"),
            "license": {**_artifact(bundle / "LICENSE"), "spdx": "Apache-2.0"},
            "sbom": _artifact(bundle / "sbom.spdx.json"),
            "runtime": {
                "nonRootUid": 10001,
                "readOnlyRootFilesystem": True,
                "noNewPrivileges": True,
                "resources": {"cpus": 0.5, "memoryMb": 128, "pids": 64},
                "egress": [],
            },
        },
    }
    signing_key = Ed25519PrivateKey.generate()
    if signed:
        manifest["signature"] = {
            "algorithm": "Ed25519",
            "keyId": "fixture.publisher-key",
            "value": "placeholder",
        }
        manifest["signature"]["value"] = sign_package_digest(
            private_key=signing_key,
            package_digest=canonical_package_digest(manifest),
        )
    (bundle / "roehub.plugin.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    public_bytes = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    publisher_keys = tmp_path / "publisher-keys.json"
    publisher_keys.write_text(
        json.dumps(
            {
                "contract": "PluginPublisherKeys/v1alpha1",
                "keys": {
                    "fixture.publisher-key": base64.b64encode(public_bytes).decode("ascii")
                },
            }
        ),
        encoding="utf-8",
    )
    return bundle, publisher_keys


def _validator(
    *, publisher_keys: Path, unsigned: bool = False, trading_mode: str = "paper"
) -> PluginBundleValidator:
    repo_root = Path(__file__).resolve().parents[4]
    return PluginBundleValidator(
        schema_path=repo_root
        / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
        trusted_publisher_keys=load_publisher_key_file(publisher_keys),
        roehub_version="0.1.0",
        supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
        allow_unsigned_development=unsigned,
        trading_mode=trading_mode,
    )


def test_validates_signed_bundle_and_rejects_tampered_artifact(tmp_path: Path) -> None:
    bundle, publisher_keys = _bundle(tmp_path)
    validated = _validator(publisher_keys=publisher_keys).validate(bundle)

    assert validated.manifest.signed is True
    assert validated.manifest.plugin_id == "fixture.data"
    assert validated.manifest.permissions == ("data.read",)

    (bundle / "sbom.spdx.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(PluginBundleValidationError) as error:
        _validator(publisher_keys=publisher_keys).validate(bundle)
    assert error.value.code == "plugin.artifact_digest_mismatch"


def test_unsigned_development_is_explicit_and_unavailable_to_mainnet(tmp_path: Path) -> None:
    bundle, publisher_keys = _bundle(tmp_path, signed=False)

    with pytest.raises(PluginBundleValidationError) as error:
        _validator(publisher_keys=publisher_keys).validate(bundle)
    assert error.value.code == "plugin.signature_required"

    validated = _validator(publisher_keys=publisher_keys, unsigned=True).validate(bundle)
    assert validated.manifest.signed is False

    with pytest.raises(ValueError, match="unavailable to mainnet"):
        _validator(publisher_keys=publisher_keys, unsigned=True, trading_mode="mainnet")
