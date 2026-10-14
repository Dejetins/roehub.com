import json

import pytest
import yaml

from apps.control_agent.docker_backend import DockerComposeControlBackend
from tools.installation.local_platform import WORKERS, render
from trading.contexts.operations import ControlOperationError


def configuration(tmp_path):
    return dict(
        schema="io.roehub.local-installation/v2alpha1",
        root=str(tmp_path),
        runtime_image="local-runtime:test",
        runtime_user="501:20",
        control_image="sha256:" + "b" * 64,
        artifact_root="/owned/artifacts",
        network="local-platform",
        postgres_volume="existing-pg",
        postgres_budget_volume="existing-budget",
        postgres_role_contract="separated-v1",
        socket_volume="control-socket",
    )


def release_manifest():
    return dict(
        schema="io.roehub.release/v1alpha1",
        version="0.1.0",
        images={
            key: {"reference": key + "@sha256:" + "a" * 64} for key in ("runtime", "postgresql")
        },
    )


def test_adoption_bundle_preserves_storage_and_binds_controller_policy(tmp_path):
    config, release = configuration(tmp_path), release_manifest()
    source, release_source = json.dumps(config).encode(), json.dumps(release).encode()
    outputs = render(config, release, source=source, release_source=release_source)
    for name, value in outputs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (tmp_path / "release.json").write_bytes(release_source)
    DockerComposeControlBackend(
        profile_root=tmp_path,
        project="local-test",
        trusted_release_manifest=tmp_path / "release.json",
    )
    controller = yaml.safe_load(outputs["control-compose.yaml"])["services"]["control-agent"]
    assert controller["user"] == "501:20"
    assert controller["image"] == config["control_image"]
    assert "/var/run/docker.sock:/var/run/docker.sock" in controller["volumes"]
    assert controller["healthcheck"]
    compose = yaml.safe_load(outputs["compose.yaml"])
    assert "clickhouse" not in compose["services"]
    assert "max_connections=48" in compose["services"]["postgresql"]["command"]
    assert compose["volumes"]["pg-budget"] == {"external": True, "name": "existing-budget"}
    assert compose["volumes"]["postgres-data"] == {"external": True, "name": "existing-pg"}
    assert "control-agent" not in compose["services"]
    assert "--requests-only" in compose["services"]["market-data-scheduler"]["command"]
    for name, service in compose["services"].items():
        assert not any("docker.sock" in mount for mount in service["volumes"])
        if name != "postgresql":
            assert service["deploy"]["resources"]["limits"]["memory"]
        if name in WORKERS:
            assert service["stop_grace_period"] == "180s"
            assert "/owned/artifacts:/owned/artifacts" in service["volumes"]
            assert not any("control-socket" in mount for mount in service["volumes"])
    # A post-generation override cannot silently broaden runtime authority.
    compose["services"]["api"]["volumes"].append("/var/run/docker.sock:/var/run/docker.sock")
    (tmp_path / "compose.yaml").write_text(yaml.safe_dump(compose))
    with pytest.raises(ControlOperationError):
        DockerComposeControlBackend(
            profile_root=tmp_path,
            project="local-test",
            trusted_release_manifest=tmp_path / "release.json",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema": "io.roehub.local-installation/v1alpha1"},
        {"clickhouse_volume": "old-store"},
        {"postgres_budget_volume": ""},
        {"postgres_role_contract": "shared-admin"},
    ],
)
def test_legacy_or_unseparated_adoption_is_rejected(tmp_path, mutation):
    config = configuration(tmp_path) | mutation
    release = release_manifest()
    with pytest.raises(ValueError):
        render(
            config,
            release,
            source=json.dumps(config).encode(),
            release_source=json.dumps(release).encode(),
        )
    assert list(tmp_path.iterdir()) == []
