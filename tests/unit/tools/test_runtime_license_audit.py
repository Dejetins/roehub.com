from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.release.runtime_license_audit import (
    RuntimeLicenseAuditError,
    build_runtime_license_audit,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    license_concluded: str,
    external_refs: list[dict[str, str]] | None = None,
) -> tuple[Path, dict[str, dict[str, object]], Path]:
    policy = tmp_path / "tools/release/runtime-license-policy.json"
    _write_json(
        policy,
        {
            "audited_images": ["runtime", "ml_runtime"],
            "explicit_components": [],
            "false_positive_records": [],
            "first_party_image": {
                "evidence_path": "/opt/roehub/LICENSE",
                "evidence_sha256": "a" * 64,
                "license": "Apache-2.0",
                "source": "https://github.com/Dejetins/roehub.com",
            },
            "schema": "io.roehub.runtime-license-policy/v1alpha1",
        },
    )
    sbom = {
        "hasExtractedLicensingInfos": [],
        "packages": [
            {
                "SPDXID": "SPDXRef-DocumentRoot-Image-fixture",
                "licenseConcluded": "Apache-2.0",
                "licenseDeclared": "Apache-2.0",
                "name": "fixture",
                "sourceInfo": "fixture image",
                "versionInfo": "0.1.0",
            },
            {
                "SPDXID": "SPDXRef-Package-mystery",
                "externalRefs": external_refs or [],
                "licenseConcluded": license_concluded,
                "licenseDeclared": "NOASSERTION",
                "name": "mystery",
                "sourceInfo": "fixture package",
                "versionInfo": "1.0.0",
            },
        ],
        "spdxVersion": "SPDX-2.3",
    }
    image_records: dict[str, dict[str, object]] = {}
    for image in ("runtime", "ml_runtime"):
        sboms: dict[str, str] = {}
        for platform in ("linux/amd64", "linux/arm64"):
            relative = f"sbom/{image}-{platform.replace('/', '-')}.json"
            _write_json(tmp_path / relative, sbom)
            sboms[platform] = relative
        image_records[image] = {
            "index_digest": "sha256:" + "b" * 64,
            "sboms": sboms,
        }
    return tmp_path, image_records, policy


def test_runtime_license_audit_rejects_unresolved_noassertion(tmp_path: Path) -> None:
    bundle, image_records, policy = _fixture(
        tmp_path,
        license_concluded="NOASSERTION",
    )

    with pytest.raises(RuntimeLicenseAuditError, match="unresolved runtime SPDX"):
        build_runtime_license_audit(
            bundle_root=bundle,
            image_records=image_records,
            policy_path=policy,
        )


def test_runtime_license_audit_accepts_scanner_conclusion_with_purl(tmp_path: Path) -> None:
    bundle, image_records, policy = _fixture(
        tmp_path,
        license_concluded="MIT",
        external_refs=[
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceLocator": "pkg:pypi/mystery@1.0.0",
                "referenceType": "purl",
            }
        ],
    )

    result = build_runtime_license_audit(
        bundle_root=bundle,
        image_records=image_records,
        policy_path=policy,
    )

    assert result["classification_counts"] == {"scanner-concluded": 4}
    assert result["raw_noassertion_count"] == 4
    assert result["unresolved_count"] == 0
