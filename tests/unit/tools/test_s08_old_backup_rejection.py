"""Old eight-owner bundles must fail before creating any restore target."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.unit.tools.test_installation_backup_bundle import _create
from tools.backup import restore_backup


def test_old_backup_schema_rejected_before_restore_write(tmp_path: Path):
    bundle, identity, verification, _ = _create(tmp_path)
    path = bundle / "backup-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"] = "io.roehub.installation-backup/v1alpha1"
    manifest.pop("compatibility")
    legacy = {**manifest["entries"][0], "owner": "clickhouse", "artifact_name": "clickhouse.age"}
    manifest["entries"].append(legacy)
    path.write_text(json.dumps(manifest))
    target = tmp_path / "old-format-target"
    calls = []

    def apply(root, data):
        calls.append(root)
        return {}

    with pytest.raises(ValidationError):
        restore_backup(
            bundle=bundle,
            restore_root=target,
            age_identity_file=identity,
            verification_public_key_file=verification,
            apply_restored_state=apply,
        )
    assert not target.exists()
    assert calls == []
