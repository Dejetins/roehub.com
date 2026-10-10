from __future__ import annotations

import pytest

import tools.release.oss_metadata as oss_metadata
from tools.release.oss_metadata import (
    Component,
    PolicyError,
    _discover_container_images,
    _expected_outputs,
    _normalize_name,
    _release_image_components,
    _validate_statuses,
)


def test_container_inventory_ignores_multistage_aliases(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(
        "FROM python:3.12 AS builder\n"
        "FROM builder AS packaged\n"
        "FROM alpine:3.22 AS runtime\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(oss_metadata, "ROOT", tmp_path)

    assert _discover_container_images([dockerfile.name]) == {
        "alpine:3.22",
        "python:3.12",
    }


def test_container_inventory_excludes_test_fixtures(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "tests" / "fixtures" / "runtime" / "compose.yaml"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("services:\n  probe:\n    image: fixture-only:1\n", encoding="utf-8")
    monkeypatch.setattr(oss_metadata, "ROOT", tmp_path)

    assert _discover_container_images(["tests/fixtures/runtime/compose.yaml"]) == set()


def test_committed_release_metadata_matches_policy() -> None:
    outputs = _expected_outputs()

    assert outputs
    for path, expected in outputs.items():
        assert path.read_bytes() == expected


def test_dependency_names_follow_python_normalization() -> None:
    assert _normalize_name("types_PyYAML") == "types-pyyaml"
    assert _normalize_name("SQLAlchemy") == "sqlalchemy"


def test_bundled_asset_outside_legacy_vendor_tree_is_not_first_party(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    asset = tmp_path / "brand.svg"
    asset.write_text("<svg/>")
    policy = {
        "bundled_assets": {
            asset.name: {
                "component": "external-brand", "version": "1",
                "license": "LicenseRef-Brand", "status": "conditional",
                "source": "https://example.invalid/brand",
                "obligation": "Preserve attribution",
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }
        },
        "first_party_assets": {}, "excluded_path_prefixes": [],
    }
    monkeypatch.setattr(oss_metadata, "ROOT", tmp_path)
    components = oss_metadata._bundled_components(policy, [asset.name])
    assert components[0].name == "external-brand"
    oss_metadata._validate_first_party_assets(policy, [asset.name])
    with pytest.raises(PolicyError, match="stale"):
        oss_metadata._bundled_components(policy, [])
    asset.write_text("changed")
    with pytest.raises(PolicyError, match="hash changed"):
        oss_metadata._bundled_components(policy, [asset.name])


def test_conditional_component_requires_explicit_obligation() -> None:
    component = Component(
        kind="container",
        name="example:latest",
        version="latest",
        license_expression="Apache-2.0",
        status="conditional",
        source="pkg:docker/example:latest",
    )

    with pytest.raises(PolicyError, match="lacks obligation"):
        _validate_statuses([component])


def test_release_images_require_digest_and_complete_platform_matrix() -> None:
    policy = {
        "release_supported_architectures": ["linux/amd64", "linux/arm64"],
        "release_images": {
            "consumer": {
                "reference": "alpine:latest",
                "platforms": ["linux/amd64", "linux/arm64"],
                "license": "LicenseRef-Alpine-Base-Image",
                "status": "conditional",
                "obligation": "audit layers",
            }
        },
    }

    with pytest.raises(PolicyError, match="not digest-pinned"):
        _release_image_components(policy)
