from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.cli.commands import artifacts as module
from apps.cli.commands.artifacts import ArtifactsCli


def test_artifacts_install_uses_private_dsn_and_reports_digest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dsn_file = tmp_path / "catalog.dsn"
    dsn_file.write_text("postgresql://fixture\n")
    dsn_file.chmod(0o600)
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"publisher": "public"}))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    calls: dict[str, object] = {}

    class FakeService:
        def __init__(self, *, blobs, catalog, trusted_public_keys) -> None:
            calls["blobs"] = blobs
            calls["catalog"] = catalog
            calls["trusted_public_keys"] = trusted_public_keys

        def set_quota(self, *, organization_id, max_bytes: int) -> None:
            calls["quota"] = (organization_id, max_bytes)

        def install_bundle(self, **kwargs):
            calls["install"] = kwargs
            return SimpleNamespace(
                bundle_id="roehub.demo.bundle",
                version="0.1.0",
                manifest_digest="sha256:" + "a" * 64,
            )

    monkeypatch.setattr(module, "LocalCasBlobStore", lambda **kwargs: ("blobs", kwargs))
    monkeypatch.setattr(
        module, "PostgresArtifactCatalogRepository", lambda **kwargs: ("catalog", kwargs)
    )
    monkeypatch.setattr(module, "ArtifactStoreService", FakeService)
    organization_id = str(uuid4())

    result = ArtifactsCli().run(
        [
            "install",
            str(bundle),
            "--organization-id",
            organization_id,
            "--catalog-dsn-file",
            str(dsn_file),
            "--cas-root",
            str(tmp_path / "cas"),
            "--publisher-keys",
            str(keys_file),
            "--quota-bytes",
            "4096",
        ]
    )

    assert result == 0
    assert calls["catalog"] == ("catalog", {"dsn": "postgresql://fixture"})
    quota_call = calls["quota"]
    assert isinstance(quota_call, tuple)
    assert quota_call[1] == 4096
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "bundle_id": "roehub.demo.bundle",
        "manifest_digest": "sha256:" + "a" * 64,
        "schema": "io.roehub.artifact-install/v1",
        "status": "installed",
        "version": "0.1.0",
    }


def test_artifacts_install_rejects_permissive_dsn_file(tmp_path: Path) -> None:
    dsn_file = tmp_path / "catalog.dsn"
    dsn_file.write_text("postgresql://fixture\n")
    dsn_file.chmod(0o644)
    keys_file = tmp_path / "keys.json"
    keys_file.write_text("{}")

    try:
        ArtifactsCli().run(
            [
                "install",
                str(tmp_path),
                "--organization-id",
                str(uuid4()),
                "--catalog-dsn-file",
                str(dsn_file),
                "--cas-root",
                str(tmp_path / "cas"),
                "--publisher-keys",
                str(keys_file),
            ]
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("permissive DSN file was accepted")


def test_artifacts_install_rejects_dsn_symlink(tmp_path: Path) -> None:
    target = tmp_path / "catalog-target.dsn"
    target.write_text("postgresql://fixture\n")
    target.chmod(0o600)
    dsn_file = tmp_path / "catalog.dsn"
    dsn_file.symlink_to(target)
    keys_file = tmp_path / "keys.json"
    keys_file.write_text("{}")

    with pytest.raises(SystemExit) as exc_info:
        ArtifactsCli().run(
            [
                "install",
                str(tmp_path),
                "--organization-id",
                str(uuid4()),
                "--catalog-dsn-file",
                str(dsn_file),
                "--cas-root",
                str(tmp_path / "cas"),
                "--publisher-keys",
                str(keys_file),
            ]
        )
    assert exc_info.value.code == 2
