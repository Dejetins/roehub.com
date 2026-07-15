from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import jsonschema
import pytest
from pydantic import ValidationError

from tools.artifacts.generate_schemas import generate
from trading.integration import (
    ArtifactBackupBlob,
    ArtifactBackupCatalog,
    ArtifactManifest,
    ArtifactStoreDescriptor,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_demo_manifest_has_known_digest() -> None:
    fixture = _REPO_ROOT / "tests/fixtures/artifacts/demo_bundle"
    manifest = ArtifactManifest.model_validate_json((fixture / "artifact.bundle.json").read_bytes())
    expected = json.loads((fixture / "expected-digests.json").read_text())

    assert manifest.manifest_digest == expected["manifest"]
    assert {entry.path: entry.blob.digest for entry in manifest.entries} == expected["payload"]
    expected_signed = (fixture / "expected-signed-payload.json").read_text().rstrip("\n").encode()
    assert manifest.signed_bytes() == expected_signed
    packaged = _REPO_ROOT / "src/trading/resources/artifacts/demo_bundle"
    for relative in (
        "artifact.bundle.json",
        "publisher-keys.json",
        "payload/demo/hello.json",
        "payload/demo/model-card.txt",
    ):
        assert (packaged / relative).read_bytes() == (fixture / relative).read_bytes()


def test_manifest_rejects_path_escape_and_secret_metadata() -> None:
    payload = json.loads(
        (_REPO_ROOT / "tests/fixtures/artifacts/demo_bundle/artifact.bundle.json").read_text()
    )
    payload["entries"][0]["path"] = "../secret"
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate(payload)

    payload = json.loads(
        (_REPO_ROOT / "tests/fixtures/artifacts/demo_bundle/artifact.bundle.json").read_text()
    )
    payload["metadata"]["api_token"] = "not-allowed"
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate(payload)


def test_generated_artifact_schemas_are_current_and_executable() -> None:
    output_root = _REPO_ROOT / "schemas/artifacts"
    generate(output_root=output_root, check=True)
    manifest_schema = json.loads((output_root / "artifact-manifest-v1.schema.json").read_text())
    store_schema = json.loads((output_root / "artifact-store-v1.schema.json").read_text())
    backup_schema = json.loads((output_root / "artifact-backup-v1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(manifest_schema)
    jsonschema.Draft202012Validator.check_schema(store_schema)
    jsonschema.Draft202012Validator.check_schema(backup_schema)
    manifest = json.loads(
        (_REPO_ROOT / "tests/fixtures/artifacts/demo_bundle/artifact.bundle.json").read_text()
    )
    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    assert manifest_schema["properties"]["entries"]["x-roehub-unique-by"] == "path"
    assert manifest_schema["properties"]["metadata"]["maxProperties"] == 32
    escaped_manifest = json.loads(json.dumps(manifest))
    escaped_manifest["entries"][0]["path"] = "../secret"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(manifest_schema).validate(escaped_manifest)
    secret_manifest = json.loads(json.dumps(manifest))
    secret_manifest["metadata"]["api_token"] = "not-allowed"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(manifest_schema).validate(secret_manifest)
    jsonschema.Draft202012Validator(store_schema).validate(
        ArtifactStoreDescriptor(schema="ArtifactStore/v1", backend="local_cas").model_dump(
            mode="json", by_alias=True
        )
    )
    validated_manifest = ArtifactManifest.model_validate(manifest)
    backup = ArtifactBackupCatalog(
        schema="ArtifactBackup/v1",
        source_organization_id=UUID("00000000-0000-4000-8000-000000000001"),
        quota_bytes=4096,
        manifests=(validated_manifest,),
        pinned_digests=(validated_manifest.entries[0].blob.digest,),
        blobs=tuple(
            ArtifactBackupBlob(
                digest=entry.blob.digest,
                size_bytes=entry.blob.size_bytes,
            )
            for entry in validated_manifest.entries
        ),
    )
    jsonschema.Draft202012Validator(backup_schema).validate(
        backup.model_dump(mode="json", by_alias=True)
    )
