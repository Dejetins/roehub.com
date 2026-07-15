from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tools.release.offline_bundle import (
    OfflineBundleError,
    _api_version_tuple,
    _verify_corresponding_source_archive,
    create_bundle,
    inspect_oci_archive,
    verify_bundle,
)
from tools.release.verify_offline_bundle_runtime import (
    RuntimeProofError,
    _digest_only_reference,
    _registry_index_copy_format,
)

ROOT = Path(__file__).resolve().parents[3]
IMAGE_NAMES = (
    "alertmanager",
    "blackbox",
    "clickhouse",
    "config_consumer",
    "grafana",
    "loki",
    "ml_runtime",
    "openbao",
    "postgresql",
    "prometheus",
    "redis",
    "runtime",
    "secret_init",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _oci_archive(
    path: Path,
    *,
    platforms: tuple[str, ...],
    include_attestations: bool = False,
) -> str:
    blobs: dict[str, bytes] = {}
    children: list[dict[str, object]] = []
    for platform in platforms:
        operating_system, architecture = platform.split("/", 1)
        config_payload = _json_bytes(
            {
                "architecture": architecture,
                "config": {},
                "os": operating_system,
                "rootfs": {"diff_ids": [], "type": "layers"},
            }
        )
        config_digest = _digest(config_payload)
        blobs[config_digest] = config_payload
        payload = _json_bytes(
            {
                "config": {
                    "digest": config_digest,
                    "mediaType": "application/vnd.oci.image.config.v1+json",
                    "size": len(config_payload),
                },
                "layers": [],
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "schemaVersion": 2,
            }
        )
        digest = _digest(payload)
        blobs[digest] = payload
        children.append(
            {
                "digest": digest,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "platform": {"architecture": architecture, "os": operating_system},
                "size": len(payload),
            }
        )
    if include_attestations:
        children.extend(
            {
                "digest": next(iter(blobs)),
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "platform": {"architecture": "unknown", "os": "unknown"},
                "size": len(next(iter(blobs.values()))),
            }
            for _ in range(2)
        )
    image_index = _json_bytes(
        {
            "manifests": children,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    index_digest = _digest(image_index)
    blobs[index_digest] = image_index
    layout_index = _json_bytes(
        {
            "manifests": [
                {
                    "digest": index_digest,
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "size": len(image_index),
                }
            ],
            "schemaVersion": 2,
        }
    )
    members = {
        "oci-layout": _json_bytes({"imageLayoutVersion": "1.0.0"}),
        "index.json": layout_index,
        **{
            f"blobs/sha256/{digest.removeprefix('sha256:')}": payload
            for digest, payload in blobs.items()
        },
    }
    with tarfile.open(path, "w") as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.mode = 0o644
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return index_digest


def _write_key_pair(root: Path) -> tuple[Path, Path]:
    private_path = root / "release.key"
    public_path = root / "release.key.pub"
    subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "unit-test",
            "-f",
            str(private_path),
        ],
        check=True,
        capture_output=True,
    )
    return private_path, public_path


def _source_archive(path: Path, *, root: str, license_files: tuple[str, ...]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for relative in license_files:
            payload = f"{root}/{relative}\n".encode()
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.mtime = 0
            info.mode = 0o644
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _source_metadata(
    *, grafana_source: Path, loki_source: Path, openbao_source: Path
) -> dict[str, dict[str, object]]:
    return {
        "grafana": {
            "archive_root": "grafana-12.0.2",
            "expected_sha256": hashlib.sha256(grafana_source.read_bytes()).hexdigest(),
            "filename": grafana_source.name,
            "image": "grafana",
            "license_files": ["LICENSE", "LICENSING.md", "NOTICE.md"],
            "revision": "a" * 40,
            "tag": "v12.0.2",
            "url": "https://github.com/grafana/grafana/archive/refs/tags/v12.0.2.tar.gz",
        },
        "loki": {
            "archive_root": "loki-3.5.1",
            "expected_sha256": hashlib.sha256(loki_source.read_bytes()).hexdigest(),
            "filename": loki_source.name,
            "image": "loki",
            "license_files": ["LICENSE", "LICENSING.md"],
            "revision": "b" * 40,
            "tag": "v3.5.1",
            "url": "https://github.com/grafana/loki/archive/refs/tags/v3.5.1.tar.gz",
        },
        "openbao": {
            "archive_root": "openbao-2.5.4-roehub-licensed-qr.1",
            "expected_sha256": hashlib.sha256(openbao_source.read_bytes()).hexdigest(),
            "filename": openbao_source.name,
            "image": "openbao",
            "license_files": [
                "LICENSE",
                "roehub/openbao-2.5.4-licensed-qr.NOTICE",
                "roehub/openbao-2.5.4-licensed-qr.patch",
                "third_party/skip2-go-qrcode/LICENSE",
            ],
            "revision": "a" * 40 + "+roehub-licensed-qr.1",
            "tag": "2.5.4-roehub-licensed-qr.1",
            "url": (
                "https://github.com/Dejetins/roehub.com/releases/download/v0.1.0/"
                "openbao-v2.5.4-roehub-licensed-qr.1.tar.gz"
            ),
        },
    }


def _source_archive_with_link(
    path: Path,
    *,
    root: str,
    link_name: str,
    link_target: str,
) -> dict[str, object]:
    with tarfile.open(path, "w:gz") as archive:
        payload = b"fixture license\n"
        license_info = tarfile.TarInfo(f"{root}/LICENSE")
        license_info.mtime = 0
        license_info.mode = 0o644
        license_info.size = len(payload)
        archive.addfile(license_info, io.BytesIO(payload))
        link_info = tarfile.TarInfo(f"{root}/{link_name}")
        link_info.mtime = 0
        link_info.mode = 0o777
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = link_target
        archive.addfile(link_info)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "archive_root": root,
        "expected_sha256": digest,
        "license_files": ["LICENSE"],
        "sha256": digest,
    }


def _release_manifest(path: Path, *, digest: str) -> None:
    path.write_text(
        json.dumps(
            {
                "images": {
                    name: {
                        "platforms": ["linux/amd64", "linux/arm64"],
                        "reference": f"registry.example/roehub/{name}@{digest}",
                    }
                    for name in IMAGE_NAMES
                },
                "license": "Apache-2.0",
                "schema": "io.roehub.release/v1alpha1",
                "supported_architectures": ["linux/amd64", "linux/arm64"],
                "version": "0.1.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _fake_sbom(**values: object) -> None:
    output = values["output"]
    assert isinstance(output, Path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "SPDXID": "SPDXRef-DOCUMENT",
                "creationInfo": {"created": "1970-01-01T00:00:00Z", "creators": ["Tool: test"]},
                "dataLicense": "CC0-1.0",
                "documentNamespace": "https://sbom.roehub.io/test",
                "name": "test",
                "packages": [
                    {
                        "SPDXID": "SPDXRef-DocumentRoot-Image-fixture",
                        "licenseConcluded": "Apache-2.0",
                        "licenseDeclared": "Apache-2.0",
                        "name": "fixture",
                        "primaryPackagePurpose": "CONTAINER",
                        "sourceInfo": "fixture image",
                        "versionInfo": "0.1.0",
                    }
                ],
                "spdxVersion": "SPDX-2.3",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_inspect_oci_archive_requires_both_release_platforms(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.oci.tar"
    digest = _oci_archive(
        archive,
        platforms=("linux/amd64", "linux/arm64"),
        include_attestations=True,
    )

    descriptor = inspect_oci_archive(archive)

    assert descriptor.digest == digest
    assert descriptor.platforms == ("linux/amd64", "linux/arm64")

    incomplete = tmp_path / "incomplete.oci.tar"
    _oci_archive(incomplete, platforms=("linux/amd64",))
    with pytest.raises(OfflineBundleError, match="lacks required platforms"):
        inspect_oci_archive(incomplete)


def test_corresponding_source_allows_internal_parent_link(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    record = _source_archive_with_link(
        archive,
        root="source-1.0",
        link_name="testdata/plugin",
        link_target="../fixtures/plugin",
    )

    _verify_corresponding_source_archive(
        path=archive,
        record=record,
        name="fixture",
    )


def test_corresponding_source_rejects_link_outside_root(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    record = _source_archive_with_link(
        archive,
        root="source-1.0",
        link_name="testdata/plugin",
        link_target="../../outside",
    )

    with pytest.raises(OfflineBundleError, match="link target is unsafe"):
        _verify_corresponding_source_archive(
            path=archive,
            record=record,
            name="fixture",
        )


def test_signed_bundle_verifies_exact_inventory_and_rejects_tampering(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.oci.tar"
    digest = _oci_archive(archive, platforms=("linux/amd64", "linux/arm64"))
    release = tmp_path / "release.json"
    _release_manifest(release, digest=digest)
    private_key, public_key = _write_key_pair(tmp_path)
    wheel = tmp_path / "roehub-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fixture-wheel")
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"status":"passed"}\n', encoding="utf-8")
    grafana_source = tmp_path / "grafana-12.0.2.tar.gz"
    loki_source = tmp_path / "loki-3.5.1.tar.gz"
    openbao_source = tmp_path / "openbao-2.5.4-roehub-licensed-qr.1.tar.gz"
    _source_archive(
        grafana_source,
        root="grafana-12.0.2",
        license_files=("LICENSE", "LICENSING.md", "NOTICE.md"),
    )
    _source_archive(
        loki_source,
        root="loki-3.5.1",
        license_files=("LICENSE", "LICENSING.md"),
    )
    _source_archive(
        openbao_source,
        root="openbao-2.5.4-roehub-licensed-qr.1",
        license_files=(
            "LICENSE",
            "roehub/openbao-2.5.4-licensed-qr.NOTICE",
            "roehub/openbao-2.5.4-licensed-qr.patch",
            "third_party/skip2-go-qrcode/LICENSE",
        ),
    )
    bundle = tmp_path / "bundle"

    create_bundle(
        output=bundle,
        release_manifest=release,
        signing_key=private_key,
        wheel=wheel,
        image_inputs={name: archive for name in IMAGE_NAMES},
        source_inputs={
            "grafana": grafana_source,
            "loki": loki_source,
            "openbao": openbao_source,
        },
        source_metadata=_source_metadata(
            grafana_source=grafana_source,
            loki_source=loki_source,
            openbao_source=openbao_source,
        ),
        provenance=provenance,
        repo_root=ROOT,
        sbom_generator=_fake_sbom,
    )

    result = verify_bundle(bundle=bundle, trusted_public_key=public_key)
    assert result["signature_verified"] is True
    assert result["image_count"] == len(IMAGE_NAMES)
    assert result["runtime_license_audit"] == {
        "raw_noassertion_count": 0,
        "status": "passed",
        "unresolved_count": 0,
    }
    assert (bundle / "sbom/runtime-license-audit.json").is_file()
    assert (bundle / "infra/openbao/config/openbao.hcl").is_file()
    assert list((bundle / "images").iterdir()) == [
        bundle / "images" / f"{digest.replace(':', '-')}.oci.tar"
    ]
    standalone = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools/release/offline_bundle.py"),
            "verify",
            "--bundle",
            str(bundle),
            "--trusted-public-key",
            str(public_key),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert standalone.returncode == 0, standalone.stderr

    (bundle / "NOTICE").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(OfflineBundleError, match="(?:size|digest) mismatch"):
        verify_bundle(bundle=bundle, trusted_public_key=public_key)


def test_create_bundle_rejects_incomplete_image_inventory(tmp_path: Path) -> None:
    archive = tmp_path / "runtime.oci.tar"
    digest = _oci_archive(archive, platforms=("linux/amd64", "linux/arm64"))
    release = tmp_path / "release.json"
    _release_manifest(release, digest=digest)
    private_key, _ = _write_key_pair(tmp_path)
    wheel = tmp_path / "roehub.whl"
    wheel.write_bytes(b"wheel")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    grafana_source = tmp_path / "grafana-12.0.2.tar.gz"
    loki_source = tmp_path / "loki-3.5.1.tar.gz"
    openbao_source = tmp_path / "openbao-2.5.4-roehub-licensed-qr.1.tar.gz"
    _source_archive(
        grafana_source,
        root="grafana-12.0.2",
        license_files=("LICENSE", "LICENSING.md", "NOTICE.md"),
    )
    _source_archive(
        loki_source,
        root="loki-3.5.1",
        license_files=("LICENSE", "LICENSING.md"),
    )
    _source_archive(
        openbao_source,
        root="openbao-2.5.4-roehub-licensed-qr.1",
        license_files=(
            "LICENSE",
            "roehub/openbao-2.5.4-licensed-qr.NOTICE",
            "roehub/openbao-2.5.4-licensed-qr.patch",
            "third_party/skip2-go-qrcode/LICENSE",
        ),
    )

    with pytest.raises(OfflineBundleError, match="image input mismatch"):
        create_bundle(
            output=tmp_path / "bundle",
            release_manifest=release,
            signing_key=private_key,
            wheel=wheel,
            image_inputs={"runtime": archive},
            source_inputs={
                "grafana": grafana_source,
                "loki": loki_source,
                "openbao": openbao_source,
            },
            source_metadata=_source_metadata(
                grafana_source=grafana_source,
                loki_source=loki_source,
                openbao_source=openbao_source,
            ),
            provenance=provenance,
            repo_root=ROOT,
            sbom_generator=_fake_sbom,
        )


def test_digest_only_reference_removes_tag_and_requires_digest() -> None:
    digest = "sha256:" + "a" * 64

    assert (
        _digest_only_reference(f"docker.io/library/alpine:3.22@{digest}")
        == f"docker.io/library/alpine@{digest}"
    )
    assert (
        _digest_only_reference(f"registry.example:5000/team/image@{digest}")
        == f"registry.example:5000/team/image@{digest}"
    )
    with pytest.raises(RuntimeProofError, match="not digest-pinned"):
        _digest_only_reference("docker.io/library/alpine:3.22")


def test_docker_api_version_comparison_is_numeric_and_fail_closed() -> None:
    assert _api_version_tuple("1.44", label="minimum") < _api_version_tuple(
        "1.54", label="server"
    )
    with pytest.raises(OfflineBundleError, match="not a valid Docker API version"):
        _api_version_tuple("1.44-rc1", label="server")


@pytest.mark.parametrize(
    ("media_type", "expected_format"),
    [
        ("application/vnd.docker.distribution.manifest.list.v2+json", "v2s2"),
        ("application/vnd.oci.image.index.v1+json", "oci"),
    ],
)
def test_registry_index_copy_format_preserves_source_digest(
    media_type: str,
    expected_format: str,
) -> None:
    raw_manifest = json.dumps(
        {"manifests": [], "mediaType": media_type, "schemaVersion": 2},
        separators=(",", ":"),
    )
    digest = _digest(raw_manifest.encode())

    assert (
        _registry_index_copy_format(
            raw_manifest=raw_manifest,
            expected_digest=digest,
        )
        == expected_format
    )
    with pytest.raises(RuntimeProofError, match="registry manifest digest mismatch"):
        _registry_index_copy_format(
            raw_manifest=raw_manifest,
            expected_digest="sha256:" + "0" * 64,
        )
