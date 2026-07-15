"""Real Docker proof for a signed and isolated Plugin API v1alpha1 fixture."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from apps.migrations.verify_storage_runtime import (
    _assert_versions,
    _compose,
    _generate_config,
    _proof_environment,
    _run,
    _run_storage,
)
from trading.contexts.extensions.application import (
    PluginBundleValidator,
    canonical_package_digest,
    load_publisher_key_file,
    sign_package_digest,
)
from trading.contexts.extensions.domain import PluginRuntimePolicy


class PluginRuntimeProofError(RuntimeError):
    """Raised when one real plugin boundary assertion fails."""


def run_runtime_proof(repo_root: Path) -> dict[str, object]:
    if shutil.which("docker") is None:
        raise PluginRuntimeProofError("docker executable is unavailable")
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = secrets.token_hex(4)
    project = f"roehub-stage12-{suffix}"
    plugin_network = f"{project}-plugin-internal"
    plugin_container = f"{project}-plugin"
    probe_container = f"{project}-probe"
    plugin_image = f"roehub-stage12-fixture:{suffix}"
    compose_path = repo_root / "infra/docker/storage-embedded.compose.yml"
    fixture_root = repo_root / "tests/fixtures/plugins/signed_data_source"
    organization_id = uuid4()
    foreign_organization_id = uuid4()
    installation_id = uuid4()
    user_id = uuid4()
    instance_id = uuid4()
    cleanup_complete = False
    proof_payload: dict[str, object] = {}
    plugin_image_id = ""
    storage_image_id = ""
    environ: Mapping[str, str] = os.environ

    with tempfile.TemporaryDirectory(prefix="roehub-stage12-", dir=cache_root) as temp:
        temp_root = Path(temp)
        generated_config = _generate_config(repo_root, temp_root / "generated", config=None)
        environ = _proof_environment(generated_config)
        image_context = temp_root / "plugin-image"
        image_context.mkdir()
        shutil.copy2(fixture_root / "Dockerfile", image_context / "Dockerfile")
        shutil.copy2(fixture_root / "server.py", image_context / "server.py")
        gateway_signing_key = Ed25519PrivateKey.generate()
        gateway_public_bytes = gateway_signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        (image_context / "fixture-config.json").write_text(
            json.dumps(
                {
                    "organization_id": str(organization_id),
                    "instance_id": str(instance_id),
                    "allowed_capabilities": ["data.read"],
                    "public_key_b64": base64.b64encode(gateway_public_bytes).decode("ascii"),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        signing_key_path = temp_root / "gateway-signing.pem"
        signing_key_path.write_bytes(
            gateway_signing_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        signing_key_path.chmod(0o600)
        publisher_signing_key = Ed25519PrivateKey.generate()
        publisher_public_bytes = publisher_signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        publisher_keys_path = temp_root / "publisher-keys.json"
        publisher_keys_path.write_text(
            json.dumps(
                {
                    "contract": "PluginPublisherKeys/v1alpha1",
                    "keys": {
                        "stage12-publisher": base64.b64encode(
                            publisher_public_bytes
                        ).decode("ascii")
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        shutil.copy2(
            repo_root / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
            temp_root / "plugin-manifest.schema.json",
        )
        try:
            _run(
                ["docker", "build", "--tag", plugin_image, str(image_context)],
                cwd=repo_root,
                environ=environ,
            )
            image_inspect = _docker_json(
                ["docker", "image", "inspect", plugin_image],
                cwd=repo_root,
                environ=environ,
            )[0]
            plugin_image_id = str(image_inspect["Id"])
            architecture = "linux/" + str(image_inspect["Architecture"])
            bundle_v1 = _build_bundle(
                root=temp_root / "bundle-v1",
                version="0.1.0",
                image_reference=plugin_image,
                image_digest=plugin_image_id,
                architecture=architecture,
                signing_key=publisher_signing_key,
            )
            bundle_v2 = _build_bundle(
                root=temp_root / "bundle-v2",
                version="0.1.1",
                image_reference=plugin_image,
                image_digest=plugin_image_id,
                architecture=architecture,
                signing_key=publisher_signing_key,
            )
            validator = PluginBundleValidator(
                schema_path=repo_root
                / "schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json",
                trusted_publisher_keys=load_publisher_key_file(publisher_keys_path),
                roehub_version="0.1.0",
                supported_architectures=frozenset({"linux/amd64", "linux/arm64"}),
                trading_mode="testnet",
            )
            validated_v1 = validator.validate(bundle_v1)
            validator.validate(bundle_v2)
            runtime_spec = PluginRuntimePolicy.from_manifest(
                validated_v1.manifest
            ).to_oci_spec(
                manifest=validated_v1.manifest,
                internal_network=plugin_network,
            )

            _run(
                _compose(project, compose_path, "config", "--quiet"),
                cwd=repo_root,
                environ=environ,
            )
            _run(
                _compose(project, compose_path, "build", "storage-migrations"),
                cwd=repo_root,
                environ=environ,
            )
            _run(
                _compose(
                    project,
                    compose_path,
                    "up",
                    "-d",
                    "postgresql",
                    "clickhouse",
                    "redis",
                ),
                cwd=repo_root,
                environ=environ,
            )
            storage_status = _run_storage(project, compose_path, repo_root, environ)
            _assert_versions(storage_status, mode="embedded")
            storage_image_id = f"{project}-storage-migrations"
            _run(
                ["docker", "image", "inspect", storage_image_id],
                cwd=repo_root,
                environ=environ,
            )
            _run(
                ["docker", "network", "create", "--internal", plugin_network],
                cwd=repo_root,
                environ=environ,
            )
            _run(
                ["docker", "tag", storage_image_id, plugin_image],
                cwd=repo_root,
                environ=environ,
            )
            _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    plugin_container,
                    "--network",
                    runtime_spec.internal_network,
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--memory",
                    f"{runtime_spec.memory_mb}m",
                    "--cpus",
                    str(runtime_spec.cpus),
                    "--pids-limit",
                    str(runtime_spec.pids),
                    "--tmpfs",
                    runtime_spec.tmpfs[0],
                    "--user",
                    runtime_spec.user,
                    runtime_spec.image_digest,
                    "--package-digest",
                    validated_v1.manifest.package_digest,
                    "--package-version",
                    validated_v1.manifest.version,
                ],
                cwd=repo_root,
                environ=environ,
            )
            _run(
                [
                    "docker",
                    "create",
                    "--name",
                    probe_container,
                    "--network",
                    f"{project}_storage",
                    "--entrypoint",
                    "sleep",
                    "-e",
                    "ROEHUB_STORAGE_POSTGRES_DSN",
                    "-v",
                    f"{temp_root}:/proof:ro",
                    storage_image_id,
                    "300",
                ],
                cwd=repo_root,
                environ=environ,
            )
            _run(
                ["docker", "network", "connect", plugin_network, probe_container],
                cwd=repo_root,
                environ=environ,
            )
            _run(
                ["docker", "start", probe_container],
                cwd=repo_root,
                environ=environ,
            )
            time.sleep(1)
            probe_result = _run(
                [
                    "docker",
                    "exec",
                    probe_container,
                    "python",
                    "-m",
                    "apps.migrations.extensions_plugin_runtime_probe",
                    "--bundle-v1",
                    "/proof/bundle-v1",
                    "--bundle-v2",
                    "/proof/bundle-v2",
                    "--publisher-keys",
                    "/proof/publisher-keys.json",
                    "--signing-key-file",
                    "/proof/gateway-signing.pem",
                    "--schema-path",
                    "/proof/plugin-manifest.schema.json",
                    "--plugin-base-url",
                    f"http://{plugin_container}:8080",
                    "--installation-id",
                    str(installation_id),
                    "--organization-id",
                    str(organization_id),
                    "--foreign-organization-id",
                    str(foreign_organization_id),
                    "--user-id",
                    str(user_id),
                    "--instance-id",
                    str(instance_id),
                ],
                cwd=repo_root,
                environ=environ,
            )
            proof_payload = _proof_json(probe_result.stdout)
            _assert_plugin_proof(proof_payload)
            _assert_container_policy(
                container_name=plugin_container,
                network_name=plugin_network,
                expected_image_digest=runtime_spec.image_digest,
                cwd=repo_root,
                environ=environ,
            )
        finally:
            _best_effort(["docker", "rm", "-f", probe_container], cwd=repo_root, environ=environ)
            _best_effort(["docker", "rm", "-f", plugin_container], cwd=repo_root, environ=environ)
            _best_effort(
                _compose(project, compose_path, "down", "-v", "--remove-orphans"),
                cwd=repo_root,
                environ=environ,
            )
            _best_effort(
                ["docker", "network", "rm", plugin_network],
                cwd=repo_root,
                environ=environ,
            )
            if plugin_image_id:
                _best_effort(
                    ["docker", "image", "rm", plugin_image],
                    cwd=repo_root,
                    environ=environ,
                )
                _best_effort(
                    ["docker", "image", "rm", plugin_image_id],
                    cwd=repo_root,
                    environ=environ,
                )
            if storage_image_id:
                _best_effort(
                    ["docker", "image", "rm", storage_image_id],
                    cwd=repo_root,
                    environ=environ,
                )
            cleanup_complete = _resources_absent(
                containers=(plugin_container, probe_container),
                networks=(plugin_network, f"{project}_storage"),
                images=tuple(
                    image
                    for image in (plugin_image, plugin_image_id, storage_image_id)
                    if image
                ),
                compose_project=project,
                cwd=repo_root,
                environ=environ,
            )

    return {
        "schema": "io.roehub.plugin-container-proof/v1alpha1",
        "docker": "passed",
        "signed_fixture": proof_payload,
        "image_digest_binding": "passed",
        "container_policy": "passed",
        "cleanup": "passed" if cleanup_complete else "failed",
    }


def _build_bundle(
    *,
    root: Path,
    version: str,
    image_reference: str,
    image_digest: str,
    architecture: str,
    signing_key: Ed25519PrivateKey,
) -> Path:
    root.mkdir()
    config_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {"dataset": {"type": "string"}},
        "required": ["dataset"],
    }
    (root / "config.schema.json").write_text(
        json.dumps(config_schema, sort_keys=True), encoding="utf-8"
    )
    (root / "LICENSE").write_text("Apache-2.0 fixture declaration\n", encoding="utf-8")
    (root / "sbom.spdx.json").write_text(
        json.dumps(
            {
                "spdxVersion": "SPDX-2.3",
                "dataLicense": "CC0-1.0",
                "SPDXID": "SPDXRef-DOCUMENT",
                "name": f"roehub-stage12-fixture-{version}",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "apiVersion": "roehub.io/v1alpha1",
        "kind": "Plugin",
        "metadata": {
            "id": "stage12.data-source",
            "version": version,
            "publisher": "stage12.publisher",
        },
        "spec": {
            "type": "data-source",
            "pluginApi": "v1alpha1",
            "rpc": {"version": "roehub.plugin.rpc/v1alpha1", "port": 8080},
            "image": {
                "reference": image_reference,
                "digest": image_digest,
                "architectures": [architecture],
            },
            "compatibility": {"roehubMin": "0.1.0", "roehubMaxExclusive": "0.2.0"},
            "permissions": [
                {"capability": "data.read"},
                {"capability": "panel.describe"},
            ],
            "configSchema": _artifact(root / "config.schema.json"),
            "license": {**_artifact(root / "LICENSE"), "spdx": "Apache-2.0"},
            "sbom": _artifact(root / "sbom.spdx.json"),
            "runtime": {
                "nonRootUid": 10001,
                "readOnlyRootFilesystem": True,
                "noNewPrivileges": True,
                "resources": {"cpus": 0.5, "memoryMb": 128, "pids": 64},
                "egress": [],
            },
        },
        "signature": {
            "algorithm": "Ed25519",
            "keyId": "stage12-publisher",
            "value": "placeholder",
        },
    }
    signature = manifest["signature"]
    assert isinstance(signature, dict)
    signature["value"] = sign_package_digest(
        private_key=signing_key,
        package_digest=canonical_package_digest(manifest),
    )
    (root / "roehub.plugin.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return root


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _docker_json(
    command: Sequence[str], *, cwd: Path, environ: Mapping[str, str]
) -> list[dict[str, object]]:
    result = _run(command, cwd=cwd, environ=environ)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PluginRuntimeProofError("Docker returned invalid inspection JSON") from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise PluginRuntimeProofError("Docker inspection JSON has invalid shape")
    return payload


def _proof_json(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    if start < 0:
        raise PluginRuntimeProofError("plugin runtime probe returned no JSON")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as error:
        raise PluginRuntimeProofError("plugin runtime probe returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise PluginRuntimeProofError("plugin runtime probe JSON has invalid shape")
    return payload


def _assert_plugin_proof(payload: Mapping[str, object]) -> None:
    expected = {
        "schema": "io.roehub.extensions-plugin-runtime-proof/v1alpha1",
        "signed_bundle": "passed",
        "package_instance_separation": "passed",
        "management_idempotency": "passed",
        "concurrent_idempotency": "passed",
        "permission_expansion_recent_auth": "rejected_when_stale",
        "foreign_organization_admin": "rejected",
        "short_lived_identity": "passed",
        "protocol_negotiation": "passed",
        "identity_full_scope": "passed",
        "identity_replay": "rejected",
        "capability_denial": "passed",
        "filesystem_write": "denied",
        "platform_database": "denied",
        "external_egress": "denied",
        "health": "ready",
        "metrics": "ready",
        "config_revision": 2,
        "rollback": "restored_previous_package",
        "rollback_revoked_publisher": "rejected",
        "publisher_trust_bootstrap": "passed",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PluginRuntimeProofError("plugin runtime proof matrix is incomplete")
    audit_events = payload.get("audit_events")
    if not isinstance(audit_events, int) or audit_events < 7:
        raise PluginRuntimeProofError("plugin runtime audit evidence is incomplete")


def _assert_container_policy(
    *,
    container_name: str,
    network_name: str,
    expected_image_digest: str,
    cwd: Path,
    environ: Mapping[str, str],
) -> None:
    inspect = _docker_json(
        ["docker", "container", "inspect", container_name],
        cwd=cwd,
        environ=environ,
    )[0]
    config = inspect.get("Config")
    host_config = inspect.get("HostConfig")
    mounts = inspect.get("Mounts")
    network_settings = inspect.get("NetworkSettings")
    if not isinstance(config, dict) or not isinstance(host_config, dict):
        raise PluginRuntimeProofError("plugin container inspection is incomplete")
    if inspect.get("Image") != expected_image_digest:
        raise PluginRuntimeProofError(
            "plugin container image does not match the signed digest"
        )
    if config.get("User") != "10001:10001":
        raise PluginRuntimeProofError("plugin container user is not the required non-root uid")
    if host_config.get("ReadonlyRootfs") is not True:
        raise PluginRuntimeProofError("plugin root filesystem is not read-only")
    if "ALL" not in (host_config.get("CapDrop") or []):
        raise PluginRuntimeProofError("plugin Linux capabilities are not dropped")
    if "no-new-privileges" not in (host_config.get("SecurityOpt") or []):
        raise PluginRuntimeProofError("plugin no-new-privileges policy is absent")
    if (
        host_config.get("Memory") != 134217728
        or host_config.get("NanoCpus") != 500000000
        or host_config.get("PidsLimit") != 64
    ):
        raise PluginRuntimeProofError("plugin resource limits do not match the signed manifest")
    if mounts not in ([], None):
        raise PluginRuntimeProofError("plugin container has unexpected mounts")
    if not isinstance(network_settings, dict):
        raise PluginRuntimeProofError("plugin network inspection is absent")
    networks = network_settings.get("Networks")
    if not isinstance(networks, dict) or set(networks) != {network_name}:
        raise PluginRuntimeProofError("plugin container is attached to an unexpected network")
    environment_names = {
        str(entry).split("=", 1)[0] for entry in config.get("Env", []) if isinstance(entry, str)
    }
    forbidden = {
        name
        for name in environment_names
        if any(marker in name for marker in ("PASSWORD", "TOKEN", "SECRET", "DSN", "DATABASE"))
    }
    if forbidden:
        raise PluginRuntimeProofError("plugin container received a forbidden environment key")
    rendered = json.dumps(inspect, sort_keys=True).lower()
    if "docker.sock" in rendered or "extensions_pg_dsn" in rendered:
        raise PluginRuntimeProofError("plugin container received a forbidden runtime boundary")


def _best_effort(
    command: Sequence[str], *, cwd: Path, environ: Mapping[str, str]
) -> None:
    subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environ),
        text=True,
        capture_output=True,
        check=False,
    )


def _resources_absent(
    *,
    containers: tuple[str, ...],
    networks: tuple[str, ...],
    images: tuple[str, ...],
    compose_project: str,
    cwd: Path,
    environ: Mapping[str, str],
) -> bool:
    for container in containers:
        result = subprocess.run(
            ["docker", "container", "inspect", container],
            cwd=cwd,
            env=dict(environ),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return False
    for network in networks:
        result = subprocess.run(
            ["docker", "network", "inspect", network],
            cwd=cwd,
            env=dict(environ),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return False
    for image in images:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=cwd,
            env=dict(environ),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return False
    volumes = subprocess.run(
        [
            "docker",
            "volume",
            "ls",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={compose_project}",
        ],
        cwd=cwd,
        env=dict(environ),
        text=True,
        capture_output=True,
        check=False,
    )
    return volumes.returncode == 0 and not volumes.stdout.strip()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        payload = run_runtime_proof(repo_root)
    except Exception as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "io.roehub.plugin-container-proof/v1alpha1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
