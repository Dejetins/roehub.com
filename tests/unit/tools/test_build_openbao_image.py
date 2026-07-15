from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from tools.release.build_openbao_image import (
    OpenBaoBuildError,
    combine_platform_archives,
)
from tools.release.offline_bundle import inspect_oci_archive


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _single_platform_archive(path: Path, platform: str) -> None:
    operating_system, architecture = platform.split("/", 1)
    config = _json_bytes(
        {
            "architecture": architecture,
            "config": {},
            "os": operating_system,
            "rootfs": {"diff_ids": [], "type": "layers"},
        }
    )
    config_digest = _digest(config)
    manifest = _json_bytes(
        {
            "config": {
                "digest": config_digest,
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "size": len(config),
            },
            "layers": [],
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "schemaVersion": 2,
        }
    )
    manifest_digest = _digest(manifest)
    index = _json_bytes(
        {
            "manifests": [
                {
                    "digest": manifest_digest,
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "platform": {"architecture": architecture, "os": operating_system},
                    "size": len(manifest),
                }
            ],
            "schemaVersion": 2,
        }
    )
    members = {
        "blobs/sha256/" + config_digest.removeprefix("sha256:"): config,
        "blobs/sha256/" + manifest_digest.removeprefix("sha256:"): manifest,
        "index.json": index,
        "oci-layout": _json_bytes({"imageLayoutVersion": "1.0.0"}),
    }
    with tarfile.open(path, "w") as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            info.mtime = 0
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_combine_platform_archives_is_deterministic(tmp_path: Path) -> None:
    amd64 = tmp_path / "amd64.oci.tar"
    arm64 = tmp_path / "arm64.oci.tar"
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    _single_platform_archive(amd64, "linux/amd64")
    _single_platform_archive(arm64, "linux/arm64")

    image_reference = "roehub/runtime:0.1.0"
    first_digest = combine_platform_archives(
        [arm64, amd64], first, image_reference=image_reference
    )
    second_digest = combine_platform_archives(
        [amd64, arm64], second, image_reference=image_reference
    )

    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()
    assert inspect_oci_archive(first).platforms == ("linux/amd64", "linux/arm64")
    with tarfile.open(first, "r") as archive:
        index_stream = archive.extractfile("index.json")
        assert index_stream is not None
        index = json.load(index_stream)
    assert index["manifests"][0]["annotations"] == {
        "org.opencontainers.image.ref.name": image_reference
    }


def test_combine_platform_archives_rejects_duplicate_platform(tmp_path: Path) -> None:
    first = tmp_path / "first-amd64.oci.tar"
    second = tmp_path / "second-amd64.oci.tar"
    _single_platform_archive(first, "linux/amd64")
    _single_platform_archive(second, "linux/amd64")

    with pytest.raises(OpenBaoBuildError, match="must be exactly"):
        combine_platform_archives([first, second], tmp_path / "combined.oci.tar")


def test_combine_platform_archives_rejects_tampered_blob(tmp_path: Path) -> None:
    source = tmp_path / "amd64.oci.tar"
    tampered = tmp_path / "tampered.oci.tar"
    _single_platform_archive(source, "linux/amd64")
    with tarfile.open(source, "r") as input_archive, tarfile.open(tampered, "w") as output_archive:
        for member in input_archive.getmembers():
            stream = input_archive.extractfile(member)
            assert stream is not None
            payload = stream.read()
            if member.name.startswith("blobs/sha256/") and b'"architecture"' in payload:
                payload = payload.replace(b'"amd64"', b'"arm64"')
            output_archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(OpenBaoBuildError, match="digest mismatch"):
        combine_platform_archives([tampered], tmp_path / "combined.oci.tar")
