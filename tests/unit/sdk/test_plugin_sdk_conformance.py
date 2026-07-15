from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_python_typescript_openapi_and_fixture_share_v1alpha1_contract() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    python_sdk = (
        repo_root / "sdk/python/roehub_plugin_sdk/v1alpha1.py"
    ).read_text(encoding="utf-8")
    typescript_sdk = (repo_root / "sdk/typescript/src/index.ts").read_text(encoding="utf-8")
    openapi = yaml.safe_load(
        (repo_root / "schemas/plugins/plugin-rpc-v1alpha1.openapi.yaml").read_text(
            encoding="utf-8"
        )
    )
    fixture = json.loads(
        (repo_root / "schemas/plugins/conformance/plugin-response.json").read_text(
            encoding="utf-8"
        )
    )

    assert "roehub.plugin.rpc/v1alpha1" in python_sdk
    assert "roehub.plugin.rpc/v1alpha1" in typescript_sdk
    assert openapi["components"]["parameters"]["Protocol"]["schema"]["const"] == (
        "roehub.plugin.rpc/v1alpha1"
    )
    capabilities = openapi["components"]["schemas"]["PluginCapability"]["enum"]
    assert capabilities == [
        "app.action",
        "data.read",
        "notification.send",
        "panel.describe",
    ]
    assert all(capability in python_sdk for capability in capabilities)
    assert all(capability in typescript_sdk for capability in capabilities)
    for python_field, typescript_field in (
        ("organization_id", "organizationId"),
        ("instance_id", "instanceId"),
        ("package_digest", "packageDigest"),
        ("package_version", "packageVersion"),
        ("capability", "capability"),
    ):
        assert python_field in python_sdk
        assert typescript_field in typescript_sdk
    assert fixture["contract"] == "PluginResponse/v1alpha1"
    assert "/execute" not in "\n".join(openapi["paths"])
