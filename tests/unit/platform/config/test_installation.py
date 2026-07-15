from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from trading.platform.config.installation import (
    InstallationConfigError,
    load_json_bytes,
    load_yaml_bytes,
    render_profile,
    sha256_bytes,
    validate_installation,
    yaml_bytes,
)

ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "configs" / "installation" / "roehub.yaml"
MANIFEST_PATH = ROOT / "tools" / "release" / "release-metadata.json"
INSTALLATION_SCHEMA_PATH = ROOT / "schemas" / "config" / "roehub.schema.json"
RELEASE_SCHEMA_PATH = ROOT / "schemas" / "config" / "release-manifest.schema.json"
GOLDEN_PATH = ROOT / "tests" / "golden" / "installation" / "profile-output-sha256.json"


def _inputs() -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    config_source = CONFIG_PATH.read_bytes()
    manifest_source = MANIFEST_PATH.read_bytes()
    return (
        config_source,
        manifest_source,
        load_yaml_bytes(config_source, source=str(CONFIG_PATH)),
        load_json_bytes(manifest_source, source=str(MANIFEST_PATH)),
    )


def _schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(INSTALLATION_SCHEMA_PATH.read_text(encoding="utf-8")),
        json.loads(RELEASE_SCHEMA_PATH.read_text(encoding="utf-8")),
    )


def test_profile_matrix_is_schema_valid_deterministic_and_golden() -> None:
    config_source, manifest_source, config, manifest = _inputs()
    installation_schema, release_schema = _schemas()
    jsonschema.Draft202012Validator.check_schema(installation_schema)
    jsonschema.Draft202012Validator.check_schema(release_schema)
    validate_installation(config, manifest, installation_schema, release_schema)
    expected_hashes = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    actual_hashes: dict[str, dict[str, str]] = {}
    for profile in ("base", "trading", "ml"):
        first = render_profile(
            config,
            manifest,
            profile,
            config_source=config_source,
            manifest_source=manifest_source,
        )
        second = render_profile(
            config,
            manifest,
            profile,
            config_source=config_source,
            manifest_source=manifest_source,
        )
        assert first == second
        assert set(first) == {
            "compose.yaml",
            "effective-config.redacted.json",
            "generation-manifest.json",
            "oidc.json",
            "openbao.json",
            "prometheus.yml",
            "service-config.json",
        }
        compose = first["compose.yaml"].decode()
        assert "network_mode: none" in compose
        assert "@sha256:" in compose
        assert "latest" not in compose.lower()
        assert "mainnet" not in compose.lower()
        actual_hashes[profile] = {
            name: sha256_bytes(content) for name, content in first.items()
        }

    assert actual_hashes == expected_hashes


def test_effective_config_redacts_references_without_losing_runtime_refs() -> None:
    config_source, manifest_source, config, manifest = _inputs()
    outputs = render_profile(
        config,
        manifest,
        "base",
        config_source=config_source,
        manifest_source=manifest_source,
    )

    effective = outputs["effective-config.redacted.json"].decode()
    runtime = outputs["service-config.json"].decode()
    assert "openbao://" not in effective
    assert "<secret-reference:redacted>" in effective
    assert "openbao://" in runtime


@pytest.mark.parametrize("architecture", ["linux/amd64", "linux/arm64"])
def test_supported_release_architectures_render_explicit_platform(architecture: str) -> None:
    config_source, manifest_source, config, manifest = _inputs()
    installation_schema, release_schema = _schemas()
    config["architecture"] = architecture
    validate_installation(config, manifest, installation_schema, release_schema)

    outputs = render_profile(
        config,
        manifest,
        "base",
        config_source=yaml_bytes(config),
        manifest_source=manifest_source,
    )
    assert f"platform: {architecture}" in outputs["compose.yaml"].decode()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"architecture": "darwin/arm64"}), "schema validation"),
        (lambda value: value["trading"].update({"mode": "mainnet"}), "mainnet value"),
        (lambda value: value["ports"].update({"https": value["ports"]["http"]}), "unique"),
        (
            lambda value: value.update({"domain": "example.com"}),
            "TLS may be disabled only",
        ),
        (
            lambda value: value["stores"]["redis"].update(
                {"credentials_ref": "not-a-secret-reference"}
            ),
            "schema validation",
        ),
        (
            lambda value: value.update({"password": "placeholder"}),
            "raw secret-shaped installation key",
        ),
        (
            lambda value: value.update({"command": "placeholder"}),
            "dangerous installation key",
        ),
        (
            lambda value: value["stores"]["redis"].update(
                {
                    "credentials_ref": (
                        "openbao://kv/another-root/storage/redis#credentials"
                    )
                }
            ),
            "outside configured OpenBao root",
        ),
    ],
)
def test_invalid_installation_properties_fail_closed(
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _, _, config, manifest = _inputs()
    installation_schema, release_schema = _schemas()
    broken = copy.deepcopy(config)
    mutation(broken)

    with pytest.raises(InstallationConfigError, match=message):
        validate_installation(broken, manifest, installation_schema, release_schema)


def test_unpinned_release_image_is_rejected() -> None:
    _, _, config, manifest = _inputs()
    installation_schema, release_schema = _schemas()
    broken = copy.deepcopy(manifest)
    broken["images"]["config_consumer"]["reference"] = "alpine:latest"

    with pytest.raises(InstallationConfigError, match="release manifest schema validation"):
        validate_installation(config, broken, installation_schema, release_schema)


def test_duplicate_yaml_key_is_rejected() -> None:
    with pytest.raises(InstallationConfigError, match="duplicate YAML key"):
        load_yaml_bytes(
            b"schema: io.roehub.installation/v1alpha1\nschema: duplicate\n",
            source="duplicate.yaml",
        )
